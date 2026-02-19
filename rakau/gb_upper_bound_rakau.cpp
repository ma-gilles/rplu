// C API for certified factor-C upper bounds on Cauchy-like row norms.
//
// Computes, for each target i,
//   ub_i = sum_{nodes} g_i^* (sum_{j in node} b_j^* b_j) g_i / d_min(node, i)^2
// with a certified factor-C guarantee by only aggregating a node when
//   (d_max / d_min)^2 <= C.
//
// This version is optimized for r=2; r>2 is not supported.
//
// Implementation notes:
// - Build a Rakau quadtree for sources (q) and a second quadtree for targets (c).
// - Dual-tree traversal over (source node, target node) pairs to amortize traversal work.
// - Node aggregation uses bounding-box distances only; no multipole/COM evaluation.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <limits>
#include <thread>
#include <vector>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <xsimd/xsimd.hpp>

#include <rakau/tree.hpp>

using rakau::quadtree;
using rakau::kwargs::coords;
using rakau::kwargs::masses;
using rakau::kwargs::max_leaf_n;

namespace {

#if defined(__GNUC__) || defined(__clang__)
#define RAKAUGB_RESTRICT __restrict__
#else
#define RAKAUGB_RESTRICT
#endif

struct NodeData {
    std::size_t begin = 0;
    std::size_t end = 0;
    std::size_t n_children = 0;  // number of descendants in depth-first order
    double xmin = 0.0;
    double xmax = 0.0;
    double ymin = 0.0;
    double ymax = 0.0;
    double s00 = 0.0;
    double s11 = 0.0;
    double s01_re = 0.0;
    double s01_im = 0.0;
};

struct Pair {
    std::size_t src = 0;
    std::size_t tgt = 0;
};

inline void update_gram(NodeData &node, double b0r, double b0i, double b1r, double b1i)
{
    node.s00 += b0r * b0r + b0i * b0i;
    node.s11 += b1r * b1r + b1i * b1i;
    node.s01_re += b0r * b1r + b0i * b1i;
    node.s01_im += b0r * b1i - b0i * b1r;
}

inline double quad_form_r2(const NodeData &node, double g0r, double g0i, double g1r, double g1i)
{
    const double s00 = node.s00;
    const double s11 = node.s11;
    const double s01r = node.s01_re;
    const double s01i = node.s01_im;

    const double Sg0_r = s00 * g0r + (s01r * g1r - s01i * g1i);
    const double Sg0_i = s00 * g0i + (s01r * g1i + s01i * g1r);
    const double Sg1_r = (s01r * g0r + s01i * g0i) + s11 * g1r;
    const double Sg1_i = (s01r * g0i - s01i * g0r) + s11 * g1i;

    return g0r * Sg0_r + g0i * Sg0_i + g1r * Sg1_r + g1i * Sg1_i;
}

template <typename Tree>
void build_node_data_src(
    const Tree &tree,
    const double *x_morton,
    const double *y_morton,
    const std::vector<double> &b0r,
    const std::vector<double> &b0i,
    const std::vector<double> &b1r,
    const std::vector<double> &b1i,
    std::vector<NodeData> &out)
{
    const auto &nodes = tree.nodes();
    const std::size_t n_nodes = nodes.size();
    for (std::size_t idx = n_nodes; idx-- > 0;) {
        const auto &node = nodes[idx];
        NodeData &nd = out[idx];
        nd.begin = node.begin;
        nd.end = node.end;
        nd.n_children = node.n_children;

        if (nd.n_children == 0) {
            double xmin = std::numeric_limits<double>::infinity();
            double xmax = -std::numeric_limits<double>::infinity();
            double ymin = std::numeric_limits<double>::infinity();
            double ymax = -std::numeric_limits<double>::infinity();
            for (std::size_t k = nd.begin; k < nd.end; ++k) {
                update_gram(nd, b0r[k], b0i[k], b1r[k], b1i[k]);
                const double x = x_morton[k];
                const double y = y_morton[k];
                xmin = std::min(xmin, x);
                xmax = std::max(xmax, x);
                ymin = std::min(ymin, y);
                ymax = std::max(ymax, y);
            }
            nd.xmin = xmin;
            nd.xmax = xmax;
            nd.ymin = ymin;
            nd.ymax = ymax;
            continue;
        }

        std::size_t child_idx = idx + 1;
        bool first = true;
        while (child_idx <= idx + nd.n_children) {
            const NodeData &child = out[child_idx];
            nd.s00 += child.s00;
            nd.s11 += child.s11;
            nd.s01_re += child.s01_re;
            nd.s01_im += child.s01_im;
            if (first) {
                nd.xmin = child.xmin;
                nd.xmax = child.xmax;
                nd.ymin = child.ymin;
                nd.ymax = child.ymax;
                first = false;
            } else {
                nd.xmin = std::min(nd.xmin, child.xmin);
                nd.xmax = std::max(nd.xmax, child.xmax);
                nd.ymin = std::min(nd.ymin, child.ymin);
                nd.ymax = std::max(nd.ymax, child.ymax);
            }
            child_idx += child.n_children + 1;
        }
    }
}

template <typename Tree>
void build_node_bbox(
    const Tree &tree,
    const double *x_morton,
    const double *y_morton,
    std::vector<NodeData> &out)
{
    const auto &nodes = tree.nodes();
    const std::size_t n_nodes = nodes.size();
    for (std::size_t idx = n_nodes; idx-- > 0;) {
        const auto &node = nodes[idx];
        NodeData &nd = out[idx];
        nd.begin = node.begin;
        nd.end = node.end;
        nd.n_children = node.n_children;

        if (nd.n_children == 0) {
            double xmin = std::numeric_limits<double>::infinity();
            double xmax = -std::numeric_limits<double>::infinity();
            double ymin = std::numeric_limits<double>::infinity();
            double ymax = -std::numeric_limits<double>::infinity();
            for (std::size_t k = nd.begin; k < nd.end; ++k) {
                const double x = x_morton[k];
                const double y = y_morton[k];
                xmin = std::min(xmin, x);
                xmax = std::max(xmax, x);
                ymin = std::min(ymin, y);
                ymax = std::max(ymax, y);
            }
            nd.xmin = xmin;
            nd.xmax = xmax;
            nd.ymin = ymin;
            nd.ymax = ymax;
            continue;
        }

        std::size_t child_idx = idx + 1;
        bool first = true;
        while (child_idx <= idx + nd.n_children) {
            const NodeData &child = out[child_idx];
            if (first) {
                nd.xmin = child.xmin;
                nd.xmax = child.xmax;
                nd.ymin = child.ymin;
                nd.ymax = child.ymax;
                first = false;
            } else {
                nd.xmin = std::min(nd.xmin, child.xmin);
                nd.xmax = std::max(nd.xmax, child.xmax);
                nd.ymin = std::min(nd.ymin, child.ymin);
                nd.ymax = std::max(nd.ymax, child.ymax);
            }
            child_idx += child.n_children + 1;
        }
    }
}

inline void dmin_dmax_bb_sq(const NodeData &src, const NodeData &tgt, double &dmin2, double &dmax2)
{
    double dx_min = 0.0;
    if (tgt.xmin > src.xmax) {
        dx_min = tgt.xmin - src.xmax;
    } else if (src.xmin > tgt.xmax) {
        dx_min = src.xmin - tgt.xmax;
    }

    double dy_min = 0.0;
    if (tgt.ymin > src.ymax) {
        dy_min = tgt.ymin - src.ymax;
    } else if (src.ymin > tgt.ymax) {
        dy_min = src.ymin - tgt.ymax;
    }

    const double dx_max = std::max(std::abs(tgt.xmin - src.xmax), std::abs(tgt.xmax - src.xmin));
    const double dy_max = std::max(std::abs(tgt.ymin - src.ymax), std::abs(tgt.ymax - src.ymin));

    dmin2 = dx_min * dx_min + dy_min * dy_min;
    dmax2 = dx_max * dx_max + dy_max * dy_max;
}

inline void accumulate_node_r2_const(
    const NodeData &node,
    const double *RAKAUGB_RESTRICT g0r,
    const double *RAKAUGB_RESTRICT g0i,
    const double *RAKAUGB_RESTRICT g1r,
    const double *RAKAUGB_RESTRICT g1i,
    std::size_t t_begin,
    std::size_t t_end,
    double inv_dmin2,
    double *RAKAUGB_RESTRICT out)
{
    const std::size_t t_size = t_end - t_begin;
    if (t_size == 0) {
        return;
    }

    const double s00 = node.s00;
    const double s11 = node.s11;
    const double s01r = node.s01_re;
    const double s01i = node.s01_im;

    const double *g0r_ptr = g0r + t_begin;
    const double *g0i_ptr = g0i + t_begin;
    const double *g1r_ptr = g1r + t_begin;
    const double *g1i_ptr = g1i + t_begin;
    double *out_ptr = out + t_begin;

    using batch = xsimd::simd_type<double>;
    constexpr std::size_t kBatch = batch::size;

    std::size_t i = 0;
    if constexpr (kBatch > 1) {
        const batch bs00(s00);
        const batch bs11(s11);
        const batch bs01r(s01r);
        const batch bs01i(s01i);
        const batch binv(inv_dmin2);

        for (; i + kBatch <= t_size; i += kBatch) {
            const auto bg0r = xsimd::load_unaligned(g0r_ptr + i);
            const auto bg0i = xsimd::load_unaligned(g0i_ptr + i);
            const auto bg1r = xsimd::load_unaligned(g1r_ptr + i);
            const auto bg1i = xsimd::load_unaligned(g1i_ptr + i);

            const auto Sg0_r = bs00 * bg0r + (bs01r * bg1r - bs01i * bg1i);
            const auto Sg0_i = bs00 * bg0i + (bs01r * bg1i + bs01i * bg1r);
            const auto Sg1_r = (bs01r * bg0r + bs01i * bg0i) + bs11 * bg1r;
            const auto Sg1_i = (bs01r * bg0i - bs01i * bg0r) + bs11 * bg1i;

            const auto quad = bg0r * Sg0_r + bg0i * Sg0_i + bg1r * Sg1_r + bg1i * Sg1_i;

            auto bout = xsimd::load_unaligned(out_ptr + i);
            bout += quad * binv;
            bout.store_unaligned(out_ptr + i);
        }
    }

    for (; i < t_size; ++i) {
        const double g0r_i = g0r_ptr[i];
        const double g0i_i = g0i_ptr[i];
        const double g1r_i = g1r_ptr[i];
        const double g1i_i = g1i_ptr[i];
        const double quad = quad_form_r2(node, g0r_i, g0i_i, g1r_i, g1i_i);
        out_ptr[i] += quad * inv_dmin2;
    }
}

inline void accumulate_node_weights_const(
    const NodeData &node,
    std::size_t t_begin,
    std::size_t t_end,
    double inv_dmin2,
    double *RAKAUGB_RESTRICT out)
{
    const std::size_t t_size = t_end - t_begin;
    if (t_size == 0) {
        return;
    }

    const double add = node.s00 * inv_dmin2;
    double *out_ptr = out + t_begin;

    using batch = xsimd::simd_type<double>;
    constexpr std::size_t kBatch = batch::size;

    std::size_t i = 0;
    if constexpr (kBatch > 1) {
        const batch badd(add);
        for (; i + kBatch <= t_size; i += kBatch) {
            auto bout = xsimd::load_unaligned(out_ptr + i);
            bout += badd;
            bout.store_unaligned(out_ptr + i);
        }
    }

    for (; i < t_size; ++i) {
        out_ptr[i] += add;
    }
}

inline void eval_leaf_exact_block(
    const NodeData &node,
    const double *RAKAUGB_RESTRICT qx,
    const double *RAKAUGB_RESTRICT qy,
    const double *RAKAUGB_RESTRICT b0r,
    const double *RAKAUGB_RESTRICT b0i,
    const double *RAKAUGB_RESTRICT b1r,
    const double *RAKAUGB_RESTRICT b1i,
    const double *RAKAUGB_RESTRICT cx,
    const double *RAKAUGB_RESTRICT cy,
    const double *RAKAUGB_RESTRICT g0r,
    const double *RAKAUGB_RESTRICT g0i,
    const double *RAKAUGB_RESTRICT g1r,
    const double *RAKAUGB_RESTRICT g1i,
    std::size_t t_begin,
    std::size_t t_end,
    double *RAKAUGB_RESTRICT out)
{
    const std::size_t t_size = t_end - t_begin;
    if (t_size == 0) {
        return;
    }

    const double *cx_ptr = cx + t_begin;
    const double *cy_ptr = cy + t_begin;
    const double *g0r_ptr = g0r + t_begin;
    const double *g0i_ptr = g0i + t_begin;
    const double *g1r_ptr = g1r + t_begin;
    const double *g1i_ptr = g1i + t_begin;
    double *out_ptr = out + t_begin;

    using batch = xsimd::simd_type<double>;
    constexpr std::size_t kBatch = batch::size;

    for (std::size_t k = node.begin; k < node.end; ++k) {
        const double qx_k = qx[k];
        const double qy_k = qy[k];
        const double b0r_k = b0r[k];
        const double b0i_k = b0i[k];
        const double b1r_k = b1r[k];
        const double b1i_k = b1i[k];

        std::size_t i = 0;
        if constexpr (kBatch > 1) {
            const batch bqx(qx_k);
            const batch bqy(qy_k);
            const batch bb0r(b0r_k);
            const batch bb0i(b0i_k);
            const batch bb1r(b1r_k);
            const batch bb1i(b1i_k);
            const batch bzero(0.0);

            for (; i + kBatch <= t_size; i += kBatch) {
                const auto bx = xsimd::load_unaligned(cx_ptr + i);
                const auto by = xsimd::load_unaligned(cy_ptr + i);
                const auto bg0r = xsimd::load_unaligned(g0r_ptr + i);
                const auto bg0i = xsimd::load_unaligned(g0i_ptr + i);
                const auto bg1r = xsimd::load_unaligned(g1r_ptr + i);
                const auto bg1i = xsimd::load_unaligned(g1i_ptr + i);

                const auto dx = bx - bqx;
                const auto dy = by - bqy;
                const auto dist2 = dx * dx + dy * dy;
                if (xsimd::any(dist2 == bzero)) {
                    throw std::runtime_error("Zero distance encountered between c and q.");
                }

                const auto dot_r = bg0r * bb0r - bg0i * bb0i + bg1r * bb1r - bg1i * bb1i;
                const auto dot_i = bg0r * bb0i + bg0i * bb0r + bg1r * bb1i + bg1i * bb1r;
                const auto contrib = (dot_r * dot_r + dot_i * dot_i) / dist2;

                auto bout = xsimd::load_unaligned(out_ptr + i);
                bout += contrib;
                bout.store_unaligned(out_ptr + i);
            }
        }

        for (; i < t_size; ++i) {
            const double dx = cx_ptr[i] - qx_k;
            const double dy = cy_ptr[i] - qy_k;
            const double dist2 = dx * dx + dy * dy;
            if (dist2 == 0.0) {
                throw std::runtime_error("Zero distance encountered between c and q.");
            }

            const double dot_r = g0r_ptr[i] * b0r_k - g0i_ptr[i] * b0i_k
                + g1r_ptr[i] * b1r_k - g1i_ptr[i] * b1i_k;
            const double dot_i = g0r_ptr[i] * b0i_k + g0i_ptr[i] * b0r_k
                + g1r_ptr[i] * b1i_k + g1i_ptr[i] * b1r_k;
            out_ptr[i] += (dot_r * dot_r + dot_i * dot_i) / dist2;
        }
    }
}

inline void eval_leaf_exact_block_weights(
    const NodeData &node,
    const double *RAKAUGB_RESTRICT qx,
    const double *RAKAUGB_RESTRICT qy,
    const double *RAKAUGB_RESTRICT w,
    const double *RAKAUGB_RESTRICT cx,
    const double *RAKAUGB_RESTRICT cy,
    std::size_t t_begin,
    std::size_t t_end,
    double *RAKAUGB_RESTRICT out)
{
    const std::size_t t_size = t_end - t_begin;
    if (t_size == 0) {
        return;
    }

    const double *cx_ptr = cx + t_begin;
    const double *cy_ptr = cy + t_begin;
    double *out_ptr = out + t_begin;

    using batch = xsimd::simd_type<double>;
    constexpr std::size_t kBatch = batch::size;

    for (std::size_t k = node.begin; k < node.end; ++k) {
        const double qx_k = qx[k];
        const double qy_k = qy[k];
        const double w_k = w[k];

        std::size_t i = 0;
        if constexpr (kBatch > 1) {
            const batch bqx(qx_k);
            const batch bqy(qy_k);
            const batch bw(w_k);
            const batch bzero(0.0);

            for (; i + kBatch <= t_size; i += kBatch) {
                const auto bx = xsimd::load_unaligned(cx_ptr + i);
                const auto by = xsimd::load_unaligned(cy_ptr + i);
                const auto dx = bx - bqx;
                const auto dy = by - bqy;
                const auto dist2 = dx * dx + dy * dy;
                if (xsimd::any(dist2 == bzero)) {
                    throw std::runtime_error("Zero distance encountered between c and q.");
                }

                auto bout = xsimd::load_unaligned(out_ptr + i);
                bout += bw / dist2;
                bout.store_unaligned(out_ptr + i);
            }
        }

        for (; i < t_size; ++i) {
            const double dx = cx_ptr[i] - qx_k;
            const double dy = cy_ptr[i] - qy_k;
            const double dist2 = dx * dx + dy * dy;
            if (dist2 == 0.0) {
                throw std::runtime_error("Zero distance encountered between c and q.");
            }
            out_ptr[i] += w_k / dist2;
        }
    }
}

inline void push_children_pairs(
    const std::vector<NodeData> &nodes,
    std::size_t parent_idx,
    std::vector<Pair> &stack,
    bool split_target,
    std::size_t other_idx)
{
    std::size_t child_idx = parent_idx + 1;
    const std::size_t end = parent_idx + nodes[parent_idx].n_children;
    while (child_idx <= end) {
        if (split_target) {
            stack.push_back({other_idx, child_idx});
        } else {
            stack.push_back({child_idx, other_idx});
        }
        child_idx += nodes[child_idx].n_children + 1;
    }
}

inline void process_pair_stack(
    const std::vector<NodeData> &src_nodes,
    const std::vector<NodeData> &tgt_nodes,
    const double *RAKAUGB_RESTRICT qx,
    const double *RAKAUGB_RESTRICT qy,
    const double *RAKAUGB_RESTRICT b0r,
    const double *RAKAUGB_RESTRICT b0i,
    const double *RAKAUGB_RESTRICT b1r,
    const double *RAKAUGB_RESTRICT b1i,
    const double *RAKAUGB_RESTRICT cx,
    const double *RAKAUGB_RESTRICT cy,
    const double *RAKAUGB_RESTRICT g0r,
    const double *RAKAUGB_RESTRICT g0i,
    const double *RAKAUGB_RESTRICT g1r,
    const double *RAKAUGB_RESTRICT g1i,
    std::size_t tgt_root,
    double *RAKAUGB_RESTRICT out,
    double C)
{
    std::vector<Pair> stack;
    stack.reserve(256);
    stack.push_back({0, tgt_root});

    while (!stack.empty()) {
        const Pair cur = stack.back();
        stack.pop_back();

        const NodeData &src = src_nodes[cur.src];
        const NodeData &tgt = tgt_nodes[cur.tgt];

        double dmin2 = 0.0;
        double dmax2 = 0.0;
        dmin_dmax_bb_sq(src, tgt, dmin2, dmax2);

        if (dmin2 > 0.0 && dmax2 <= C * dmin2) {
            const double inv_dmin2 = 1.0 / dmin2;
            accumulate_node_r2_const(src, g0r, g0i, g1r, g1i, tgt.begin, tgt.end, inv_dmin2, out);
            continue;
        }

        const bool src_leaf = (src.n_children == 0);
        const bool tgt_leaf = (tgt.n_children == 0);

        if (src_leaf && tgt_leaf) {
            eval_leaf_exact_block(src,
                                  qx, qy,
                                  b0r, b0i, b1r, b1i,
                                  cx, cy,
                                  g0r, g0i, g1r, g1i,
                                  tgt.begin, tgt.end,
                                  out);
            continue;
        }

        if (src_leaf) {
            push_children_pairs(tgt_nodes, cur.tgt, stack, true, cur.src);
            continue;
        }
        if (tgt_leaf) {
            push_children_pairs(src_nodes, cur.src, stack, false, cur.tgt);
            continue;
        }

        const std::size_t src_size = src.end - src.begin;
        const std::size_t tgt_size = tgt.end - tgt.begin;
        if (tgt_size >= src_size) {
            push_children_pairs(tgt_nodes, cur.tgt, stack, true, cur.src);
        } else {
            push_children_pairs(src_nodes, cur.src, stack, false, cur.tgt);
        }
    }
}

inline std::vector<std::size_t> build_target_cut(const std::vector<NodeData> &tgt_nodes)
{
    std::vector<std::size_t> roots;
    roots.push_back(0);

    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) {
        hw = 1;
    }
    const std::size_t desired = std::max<std::size_t>(1, static_cast<std::size_t>(hw) * 4);

    std::size_t i = 0;
    while (roots.size() < desired && i < roots.size()) {
        const std::size_t idx = roots[i];
        if (tgt_nodes[idx].n_children == 0) {
            ++i;
            continue;
        }

        roots[i] = roots.back();
        roots.pop_back();

        std::size_t child_idx = idx + 1;
        const std::size_t end = idx + tgt_nodes[idx].n_children;
        while (child_idx <= end) {
            roots.push_back(child_idx);
            child_idx += tgt_nodes[child_idx].n_children + 1;
        }
    }

    return roots;
}

struct LeafInfo {
    std::size_t begin = 0;
    std::size_t end = 0;
    std::size_t agg_offset = 0;
    std::size_t agg_count = 0;
    std::size_t exact_offset = 0;
    std::size_t exact_count = 0;
};

struct AggEntry {
    std::size_t src = 0;
    double inv_dmin2 = 0.0;
};

struct GbUbPlan {
    std::size_t n = 0;
    std::size_t m = 0;
    double C = 1.0;
    std::size_t max_leaf = 0;

    std::vector<std::size_t> perm_src;
    std::vector<std::size_t> perm_tgt;

    std::vector<double> qx_morton;
    std::vector<double> qy_morton;
    std::vector<double> cx_morton;
    std::vector<double> cy_morton;

    std::vector<NodeData> src_nodes;
    std::vector<NodeData> tgt_nodes;
    std::vector<LeafInfo> leaves;
    std::vector<std::size_t> agg_src_nodes;
    std::vector<double> agg_inv_dmin2;
    std::vector<std::size_t> exact_src_leaves;

    std::vector<double> b0r;
    std::vector<double> b0i;
    std::vector<double> b1r;
    std::vector<double> b1i;
    std::vector<double> g0r;
    std::vector<double> g0i;
    std::vector<double> g1r;
    std::vector<double> g1i;
    std::vector<double> w_morton;
    std::vector<double> out_morton;
};

inline void compute_node_gram(
    std::vector<NodeData> &nodes,
    const double *RAKAUGB_RESTRICT b0r,
    const double *RAKAUGB_RESTRICT b0i,
    const double *RAKAUGB_RESTRICT b1r,
    const double *RAKAUGB_RESTRICT b1i)
{
    const std::size_t n_nodes = nodes.size();
    for (std::size_t idx = n_nodes; idx-- > 0;) {
        NodeData &nd = nodes[idx];
        nd.s00 = 0.0;
        nd.s11 = 0.0;
        nd.s01_re = 0.0;
        nd.s01_im = 0.0;

        if (nd.n_children == 0) {
            for (std::size_t k = nd.begin; k < nd.end; ++k) {
                update_gram(nd, b0r[k], b0i[k], b1r[k], b1i[k]);
            }
            continue;
        }

        std::size_t child_idx = idx + 1;
        while (child_idx <= idx + nd.n_children) {
            const NodeData &child = nodes[child_idx];
            nd.s00 += child.s00;
            nd.s11 += child.s11;
            nd.s01_re += child.s01_re;
            nd.s01_im += child.s01_im;
            child_idx += child.n_children + 1;
        }
    }
}

inline void compute_node_weights(
    std::vector<NodeData> &nodes,
    const double *RAKAUGB_RESTRICT w)
{
    const std::size_t n_nodes = nodes.size();
    for (std::size_t idx = n_nodes; idx-- > 0;) {
        NodeData &nd = nodes[idx];
        nd.s00 = 0.0;
        nd.s11 = 0.0;
        nd.s01_re = 0.0;
        nd.s01_im = 0.0;

        if (nd.n_children == 0) {
            for (std::size_t k = nd.begin; k < nd.end; ++k) {
                nd.s00 += w[k];
            }
            continue;
        }

        std::size_t child_idx = idx + 1;
        while (child_idx <= idx + nd.n_children) {
            const NodeData &child = nodes[child_idx];
            nd.s00 += child.s00;
            child_idx += child.n_children + 1;
        }
    }
}

inline void append_src_to_tgt_subtree(
    std::size_t src_idx,
    const NodeData &src,
    std::size_t tgt_idx,
    const std::vector<NodeData> &tgt_nodes,
    const std::vector<int> &leaf_id,
    std::vector<std::vector<AggEntry>> &agg_lists)
{
    std::vector<std::size_t> stack;
    stack.push_back(tgt_idx);
    while (!stack.empty()) {
        const std::size_t cur = stack.back();
        stack.pop_back();
        const NodeData &tgt = tgt_nodes[cur];
        if (tgt.n_children == 0) {
            double dmin2 = 0.0;
            double dmax2 = 0.0;
            dmin_dmax_bb_sq(src, tgt, dmin2, dmax2);
            if (dmin2 == 0.0) {
                throw std::runtime_error("Zero distance encountered in precomputed aggregation.");
            }
            const int lid = leaf_id[cur];
            if (lid >= 0) {
                agg_lists[static_cast<std::size_t>(lid)].push_back({src_idx, 1.0 / dmin2});
            }
            continue;
        }

        std::size_t child_idx = cur + 1;
        const std::size_t end = cur + tgt.n_children;
        while (child_idx <= end) {
            stack.push_back(child_idx);
            child_idx += tgt_nodes[child_idx].n_children + 1;
        }
    }
}

inline void build_interaction_lists(
    const std::vector<NodeData> &src_nodes,
    const std::vector<NodeData> &tgt_nodes,
    double C,
    std::vector<LeafInfo> &leaves,
    std::vector<std::size_t> &agg_src_nodes,
    std::vector<double> &agg_inv_dmin2,
    std::vector<std::size_t> &exact_src_leaves)
{
    std::vector<std::size_t> tgt_leaf_nodes;
    tgt_leaf_nodes.reserve(tgt_nodes.size());
    for (std::size_t i = 0; i < tgt_nodes.size(); ++i) {
        if (tgt_nodes[i].n_children == 0) {
            tgt_leaf_nodes.push_back(i);
        }
    }
    std::sort(tgt_leaf_nodes.begin(), tgt_leaf_nodes.end(), [&](std::size_t a, std::size_t b) {
        return tgt_nodes[a].begin < tgt_nodes[b].begin;
    });

    leaves.assign(tgt_leaf_nodes.size(), {});

    std::vector<int> leaf_id(tgt_nodes.size(), -1);
    for (std::size_t i = 0; i < tgt_leaf_nodes.size(); ++i) {
        leaf_id[tgt_leaf_nodes[i]] = static_cast<int>(i);
    }

    std::vector<std::vector<AggEntry>> agg_lists(tgt_leaf_nodes.size());
    std::vector<std::vector<std::size_t>> exact_lists(tgt_leaf_nodes.size());

    std::vector<Pair> stack;
    stack.reserve(256);
    stack.push_back({0, 0});

    while (!stack.empty()) {
        const Pair cur = stack.back();
        stack.pop_back();

        const NodeData &src = src_nodes[cur.src];
        const NodeData &tgt = tgt_nodes[cur.tgt];

        double dmin2 = 0.0;
        double dmax2 = 0.0;
        dmin_dmax_bb_sq(src, tgt, dmin2, dmax2);

        if (dmin2 > 0.0 && dmax2 <= C * dmin2) {
            if (tgt.n_children == 0) {
                const int lid = leaf_id[cur.tgt];
                if (lid >= 0) {
                    agg_lists[static_cast<std::size_t>(lid)].push_back({cur.src, 1.0 / dmin2});
                }
            } else {
                append_src_to_tgt_subtree(cur.src, src, cur.tgt, tgt_nodes, leaf_id, agg_lists);
            }
            continue;
        }

        const bool src_leaf = (src.n_children == 0);
        const bool tgt_leaf = (tgt.n_children == 0);
        if (src_leaf && tgt_leaf) {
            const int lid = leaf_id[cur.tgt];
            if (lid >= 0) {
                exact_lists[static_cast<std::size_t>(lid)].push_back(cur.src);
            }
            continue;
        }

        if (src_leaf) {
            push_children_pairs(tgt_nodes, cur.tgt, stack, true, cur.src);
            continue;
        }
        if (tgt_leaf) {
            push_children_pairs(src_nodes, cur.src, stack, false, cur.tgt);
            continue;
        }

        const std::size_t src_size = src.end - src.begin;
        const std::size_t tgt_size = tgt.end - tgt.begin;
        if (tgt_size >= src_size) {
            push_children_pairs(tgt_nodes, cur.tgt, stack, true, cur.src);
        } else {
            push_children_pairs(src_nodes, cur.src, stack, false, cur.tgt);
        }
    }

    std::size_t agg_total = 0;
    std::size_t exact_total = 0;
    for (std::size_t i = 0; i < tgt_leaf_nodes.size(); ++i) {
        const std::size_t node_idx = tgt_leaf_nodes[i];
        const NodeData &tgt = tgt_nodes[node_idx];
        leaves[i].begin = tgt.begin;
        leaves[i].end = tgt.end;
        leaves[i].agg_offset = agg_total;
        leaves[i].agg_count = agg_lists[i].size();
        agg_total += leaves[i].agg_count;
        leaves[i].exact_offset = exact_total;
        leaves[i].exact_count = exact_lists[i].size();
        exact_total += leaves[i].exact_count;
    }

    agg_src_nodes.resize(agg_total);
    agg_inv_dmin2.resize(agg_total);
    exact_src_leaves.resize(exact_total);
    for (std::size_t i = 0; i < leaves.size(); ++i) {
        const auto &agg = agg_lists[i];
        const auto &exact = exact_lists[i];
        for (std::size_t j = 0; j < agg.size(); ++j) {
            agg_src_nodes[leaves[i].agg_offset + j] = agg[j].src;
            agg_inv_dmin2[leaves[i].agg_offset + j] = agg[j].inv_dmin2;
        }
        std::copy(exact.begin(), exact.end(), exact_src_leaves.begin() + leaves[i].exact_offset);
    }
}

} // namespace

extern "C" int rakau_gb_upper_bound_2d_r2(
    const double *cx,
    const double *cy,
    std::size_t n,
    const double *qx,
    const double *qy,
    std::size_t m,
    const double *g_re,
    const double *g_im,
    const double *b_re,
    const double *b_im,
    std::size_t r,
    double C,
    std::size_t max_leaf,
    double *out)
{
    try {
        if (!cx || !cy || !qx || !qy || !g_re || !g_im || !b_re || !b_im || !out) {
            return 1;
        }
        if (n == 0 || m == 0) {
            return 0;
        }
        if (!std::isfinite(C) || C < 1.0) {
            return 2;
        }
        if (r != 2) {
            return 3;
        }

        std::vector<double> xs(qx, qx + m);
        std::vector<double> ys(qy, qy + m);
        std::vector<double> ws(m, 1.0);

        std::vector<double> xt(cx, cx + n);
        std::vector<double> yt(cy, cy + n);
        std::vector<double> wt(n, 1.0);

        if (max_leaf == 0) {
            max_leaf = 64;
        }

        quadtree<double> t_src{coords<0> = xs, coords<1> = ys, masses = ws, max_leaf_n = max_leaf};
        quadtree<double> t_tgt{coords<0> = xt, coords<1> = yt, masses = wt, max_leaf_n = max_leaf};

        const auto &perm_src = t_src.perm();
        const auto p_its_src = t_src.p_its_u();
        const double *x_morton = p_its_src[0];
        const double *y_morton = p_its_src[1];

        const auto &perm_tgt = t_tgt.perm();
        const auto p_its_tgt = t_tgt.p_its_u();
        const double *cx_morton = p_its_tgt[0];
        const double *cy_morton = p_its_tgt[1];

        std::vector<double> b0r(m), b0i(m), b1r(m), b1i(m);
        for (std::size_t i = 0; i < m; ++i) {
            const std::size_t orig = perm_src[i];
            b0r[i] = b_re[orig * r + 0];
            b0i[i] = b_im[orig * r + 0];
            b1r[i] = b_re[orig * r + 1];
            b1i[i] = b_im[orig * r + 1];
        }

        std::vector<double> g0r(n), g0i(n), g1r(n), g1i(n);
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t orig = perm_tgt[i];
            g0r[i] = g_re[orig * r + 0];
            g0i[i] = g_im[orig * r + 0];
            g1r[i] = g_re[orig * r + 1];
            g1i[i] = g_im[orig * r + 1];
        }

        const auto &src_nodes = t_src.nodes();
        std::vector<NodeData> src_data(src_nodes.size());
        build_node_data_src(t_src, x_morton, y_morton, b0r, b0i, b1r, b1i, src_data);

        const auto &tgt_nodes = t_tgt.nodes();
        std::vector<NodeData> tgt_data(tgt_nodes.size());
        build_node_bbox(t_tgt, cx_morton, cy_morton, tgt_data);

        std::vector<double> out_morton(n, 0.0);
        const auto tgt_roots = build_target_cut(tgt_data);

        tbb::parallel_for(tbb::blocked_range<std::size_t>(0, tgt_roots.size()), [&](const auto &range) {
            for (std::size_t i = range.begin(); i != range.end(); ++i) {
                process_pair_stack(src_data, tgt_data,
                                   x_morton, y_morton,
                                   b0r.data(), b0i.data(), b1r.data(), b1i.data(),
                                   cx_morton, cy_morton,
                                   g0r.data(), g0i.data(), g1r.data(), g1i.data(),
                                   tgt_roots[i],
                                   out_morton.data(),
                                   C);
            }
        });

        tbb::parallel_for(tbb::blocked_range<std::size_t>(0, n), [&](const auto &range) {
            for (std::size_t i = range.begin(); i != range.end(); ++i) {
                out[perm_tgt[i]] = out_morton[i];
            }
        });

        return 0;
    } catch (const std::exception &) {
        return 4;
    } catch (...) {
        return 5;
    }
}

extern "C" void *rakau_gb_ub_plan_create_2d_r2(
    const double *cx,
    const double *cy,
    std::size_t n,
    const double *qx,
    const double *qy,
    std::size_t m,
    double C,
    std::size_t max_leaf,
    int *err)
{
    try {
        if (!cx || !cy || !qx || !qy) {
            if (err) {
                *err = 1;
            }
            return nullptr;
        }
        if (!std::isfinite(C) || C < 1.0) {
            if (err) {
                *err = 2;
            }
            return nullptr;
        }

        if (max_leaf == 0) {
            max_leaf = 64;
        }

        GbUbPlan *plan = new GbUbPlan();
        plan->n = n;
        plan->m = m;
        plan->C = C;
        plan->max_leaf = max_leaf;

        if (n == 0 || m == 0) {
            if (err) {
                *err = 0;
            }
            return plan;
        }

        std::vector<double> xs(qx, qx + m);
        std::vector<double> ys(qy, qy + m);
        std::vector<double> ws(m, 1.0);

        std::vector<double> xt(cx, cx + n);
        std::vector<double> yt(cy, cy + n);
        std::vector<double> wt(n, 1.0);

        quadtree<double> t_src{coords<0> = xs, coords<1> = ys, masses = ws, max_leaf_n = max_leaf};
        quadtree<double> t_tgt{coords<0> = xt, coords<1> = yt, masses = wt, max_leaf_n = max_leaf};

        const auto &perm_src = t_src.perm();
        plan->perm_src.assign(perm_src.begin(), perm_src.end());
        const auto p_its_src = t_src.p_its_u();
        const double *x_morton = p_its_src[0];
        const double *y_morton = p_its_src[1];
        plan->qx_morton.assign(x_morton, x_morton + m);
        plan->qy_morton.assign(y_morton, y_morton + m);

        const auto &perm_tgt = t_tgt.perm();
        plan->perm_tgt.assign(perm_tgt.begin(), perm_tgt.end());
        const auto p_its_tgt = t_tgt.p_its_u();
        const double *cx_morton = p_its_tgt[0];
        const double *cy_morton = p_its_tgt[1];
        plan->cx_morton.assign(cx_morton, cx_morton + n);
        plan->cy_morton.assign(cy_morton, cy_morton + n);

        plan->src_nodes.resize(t_src.nodes().size());
        build_node_bbox(t_src, plan->qx_morton.data(), plan->qy_morton.data(), plan->src_nodes);

        plan->tgt_nodes.resize(t_tgt.nodes().size());
        build_node_bbox(t_tgt, plan->cx_morton.data(), plan->cy_morton.data(), plan->tgt_nodes);

        build_interaction_lists(plan->src_nodes, plan->tgt_nodes, C,
                                plan->leaves, plan->agg_src_nodes, plan->agg_inv_dmin2,
                                plan->exact_src_leaves);

        if (err) {
            *err = 0;
        }
        return plan;
    } catch (const std::exception &) {
        if (err) {
            *err = 4;
        }
        return nullptr;
    } catch (...) {
        if (err) {
            *err = 5;
        }
        return nullptr;
    }
}

extern "C" int rakau_gb_ub_plan_compute_2d_r2(
    void *plan_ptr,
    const double *g_re,
    const double *g_im,
    const double *b_re,
    const double *b_im,
    double *out)
{
    try {
        if (!plan_ptr || !g_re || !g_im || !b_re || !b_im || !out) {
            return 1;
        }

        GbUbPlan *plan = static_cast<GbUbPlan *>(plan_ptr);
        const std::size_t n = plan->n;
        const std::size_t m = plan->m;
        if (n == 0) {
            return 0;
        }
        if (m == 0) {
            std::fill(out, out + n, 0.0);
            return 0;
        }

        plan->b0r.resize(m);
        plan->b0i.resize(m);
        plan->b1r.resize(m);
        plan->b1i.resize(m);
        for (std::size_t i = 0; i < m; ++i) {
            const std::size_t orig = plan->perm_src[i];
            plan->b0r[i] = b_re[orig * 2 + 0];
            plan->b0i[i] = b_im[orig * 2 + 0];
            plan->b1r[i] = b_re[orig * 2 + 1];
            plan->b1i[i] = b_im[orig * 2 + 1];
        }

        compute_node_gram(plan->src_nodes,
                          plan->b0r.data(), plan->b0i.data(),
                          plan->b1r.data(), plan->b1i.data());

        plan->g0r.resize(n);
        plan->g0i.resize(n);
        plan->g1r.resize(n);
        plan->g1i.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t orig = plan->perm_tgt[i];
            plan->g0r[i] = g_re[orig * 2 + 0];
            plan->g0i[i] = g_im[orig * 2 + 0];
            plan->g1r[i] = g_re[orig * 2 + 1];
            plan->g1i[i] = g_im[orig * 2 + 1];
        }

        if (plan->out_morton.size() != n) {
            plan->out_morton.assign(n, 0.0);
        } else {
            std::fill(plan->out_morton.begin(), plan->out_morton.end(), 0.0);
        }

        const double *cx = plan->cx_morton.data();
        const double *cy = plan->cy_morton.data();
        const double *b0r = plan->b0r.data();
        const double *b0i = plan->b0i.data();
        const double *b1r = plan->b1r.data();
        const double *b1i = plan->b1i.data();
        const double *g0r = plan->g0r.data();
        const double *g0i = plan->g0i.data();
        const double *g1r = plan->g1r.data();
        const double *g1i = plan->g1i.data();

        tbb::parallel_for(tbb::blocked_range<std::size_t>(0, plan->leaves.size()), [&](const auto &range) {
            for (std::size_t i = range.begin(); i != range.end(); ++i) {
                const LeafInfo &leaf = plan->leaves[i];
                for (std::size_t j = 0; j < leaf.agg_count; ++j) {
                    const std::size_t src_idx = plan->agg_src_nodes[leaf.agg_offset + j];
                    const double inv_dmin2 = plan->agg_inv_dmin2[leaf.agg_offset + j];
                    accumulate_node_r2_const(plan->src_nodes[src_idx],
                                             g0r, g0i, g1r, g1i,
                                             leaf.begin, leaf.end,
                                             inv_dmin2,
                                             plan->out_morton.data());
                }
                for (std::size_t j = 0; j < leaf.exact_count; ++j) {
                    const std::size_t src_idx = plan->exact_src_leaves[leaf.exact_offset + j];
                    eval_leaf_exact_block(plan->src_nodes[src_idx],
                                          plan->qx_morton.data(), plan->qy_morton.data(),
                                          b0r, b0i, b1r, b1i,
                                          cx, cy,
                                          g0r, g0i, g1r, g1i,
                                          leaf.begin, leaf.end,
                                          plan->out_morton.data());
                }
            }
        });

        tbb::parallel_for(tbb::blocked_range<std::size_t>(0, n), [&](const auto &range) {
            for (std::size_t i = range.begin(); i != range.end(); ++i) {
                out[plan->perm_tgt[i]] = plan->out_morton[i];
            }
        });

        return 0;
    } catch (const std::exception &) {
        return 4;
    } catch (...) {
        return 5;
    }
}

extern "C" int rakau_gb_ub_plan_compute_weights_2d_r2(
    void *plan_ptr,
    const double *w,
    double *out)
{
    try {
        if (!plan_ptr || !w || !out) {
            return 1;
        }

        GbUbPlan *plan = static_cast<GbUbPlan *>(plan_ptr);
        const std::size_t n = plan->n;
        const std::size_t m = plan->m;
        if (n == 0) {
            return 0;
        }
        if (m == 0) {
            std::fill(out, out + n, 0.0);
            return 0;
        }

        if (plan->w_morton.size() != m) {
            plan->w_morton.resize(m);
        }
        for (std::size_t i = 0; i < m; ++i) {
            const std::size_t orig = plan->perm_src[i];
            plan->w_morton[i] = w[orig];
        }

        compute_node_weights(plan->src_nodes, plan->w_morton.data());

        if (plan->out_morton.size() != n) {
            plan->out_morton.assign(n, 0.0);
        } else {
            std::fill(plan->out_morton.begin(), plan->out_morton.end(), 0.0);
        }

        const double *cx = plan->cx_morton.data();
        const double *cy = plan->cy_morton.data();
        const double *qx = plan->qx_morton.data();
        const double *qy = plan->qy_morton.data();
        const double *w_morton = plan->w_morton.data();

        tbb::parallel_for(tbb::blocked_range<std::size_t>(0, plan->leaves.size()), [&](const auto &range) {
            for (std::size_t i = range.begin(); i != range.end(); ++i) {
                const LeafInfo &leaf = plan->leaves[i];
                for (std::size_t j = 0; j < leaf.agg_count; ++j) {
                    const std::size_t src_idx = plan->agg_src_nodes[leaf.agg_offset + j];
                    const double inv_dmin2 = plan->agg_inv_dmin2[leaf.agg_offset + j];
                    accumulate_node_weights_const(plan->src_nodes[src_idx],
                                                  leaf.begin, leaf.end,
                                                  inv_dmin2,
                                                  plan->out_morton.data());
                }
                for (std::size_t j = 0; j < leaf.exact_count; ++j) {
                    const std::size_t src_idx = plan->exact_src_leaves[leaf.exact_offset + j];
                    eval_leaf_exact_block_weights(plan->src_nodes[src_idx],
                                                  qx, qy,
                                                  w_morton,
                                                  cx, cy,
                                                  leaf.begin, leaf.end,
                                                  plan->out_morton.data());
                }
            }
        });

        tbb::parallel_for(tbb::blocked_range<std::size_t>(0, n), [&](const auto &range) {
            for (std::size_t i = range.begin(); i != range.end(); ++i) {
                out[plan->perm_tgt[i]] = plan->out_morton[i];
            }
        });

        return 0;
    } catch (const std::exception &) {
        return 4;
    } catch (...) {
        return 5;
    }
}

extern "C" void rakau_gb_ub_plan_destroy(void *plan_ptr)
{
    if (!plan_ptr) {
        return;
    }
    GbUbPlan *plan = static_cast<GbUbPlan *>(plan_ptr);
    delete plan;
}
