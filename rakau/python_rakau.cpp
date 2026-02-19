// Minimal C API wrapper for rakau 2D potentials (source-weighted 1/r^2).
// Exposes a single function callable via ctypes/cffi:
//
//   int rakau_potential_2d(const double* x,
//                          const double* y,
//                          const double* w,
//                          std::size_t n,
//                          double theta,
//                          double eps,
//                          int use_gpu,
//                          double* out)
//
// Returns 0 on success, non-zero on error.

#include <cstddef>
#include <exception>
#include <vector>

#include <rakau/tree.hpp>

using namespace rakau;
using namespace rakau::kwargs;

extern "C" int rakau_potential_2d(const double *x, const double *y, const double *w, std::size_t n, double theta,
                                  double eps_value, int use_gpu, double *out)
{
    try {
        if (!x || !y || !w || !out) {
            return 1;
        }
        // Copy inputs into std::vector for quadtree.
        std::vector<double> xs(x, x + n), ys(y, y + n), ws(w, w + n);
        using quadtree_uint64 = tree<2, double, std::uint64_t, mac::bh>;
        quadtree_uint64 t{coords<0> = xs, coords<1> = ys, masses = ws};

        std::vector<double> split_weights;
#if defined(RAKAU_WITH_CUDA)
        if (use_gpu) {
            // Route all work to the accelerator.
            split_weights = {0.0, 1.0};
        }
#endif
        std::vector<double> res;
        if (split_weights.empty()) {
            t.pots_o(res, theta, eps = eps_value);
        } else {
            t.pots_o(res, theta, eps = eps_value, split = split_weights);
        }

        // Copy back to output buffer.
        std::copy(res.begin(), res.end(), out);
        return 0;
    } catch (const std::exception &) {
        return 2;
    } catch (...) {
        return 3;
    }
}
