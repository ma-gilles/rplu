#!/usr/bin/env python3
"""
Paper experiments for Loewner Cauchy-like matrices (sin and tan cases).

Produces clean plots comparing (row-norm Frobenius residuals for CUR methods):
- C2PLU (greedy) vs RPLU (random) vs SVD
- Timing to tolerance for fixed d (varying n)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
# os.environ.setdefault("JAX_PLATFORMS", "gpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
# Set default path for pyfmmlib2d if not already set
# The path should point to the directory containing pyfmmlib2d package
if "PYFMMLIB2D_PATH" not in os.environ:
    vendor_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor", "pyfmmlib2d")
    if os.path.exists(vendor_path):
        # Add to sys.path so pyfmmlib2d can be imported
        import sys
        if vendor_path not in sys.path:
            sys.path.insert(0, vendor_path)
        os.environ["PYFMMLIB2D_PATH"] = vendor_path

import jax
import jax.numpy as jnp

from cauchy.loewner_utils import build_loewner_cauchy_matrix
from cauchy.cauchy_jax import frob_norm_from_uv
from cauchy.unified_cur import build_cur_unified

# Try to import FMM support, but make it optional
# Set path before importing if vendor directory exists
if "PYFMMLIB2D_PATH" not in os.environ:
    vendor_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor", "pyfmmlib2d")
    if os.path.exists(vendor_path):
        os.environ["PYFMMLIB2D_PATH"] = vendor_path

from cauchy.fmm_cauchy import CauchyFMMOperator, randomized_svd_operator
HAS_FMM = True
# except ImportError:
#     HAS_FMM = False
#     CauchyFMMOperator = None
#     randomized_svd_operator = None

jax.config.update("jax_enable_x64", True)


METHOD_STYLE = {
    "r2plu_ub": {"label": "RPLU", "color": "#FF6600", "marker": "s", "linestyle": "-"},
    "c2plu": {"label": "C2PLU", "color": "#0066CC", "marker": "o", "linestyle": "-"},
    "rsvd_fmm": {"label": "RSVD (FMM)", "color": "#AA00AA", "marker": "D", "linestyle": ":"},
    "svd": {"label": "SVD", "color": "#9467bd", "marker": "v", "linestyle": "-"},
}

# Match `test_baryrat_cur_half_paper.py`: stop CUR when the Frobenius upper-bound
# residual reaches this relative tolerance (vs. the initial residual).
DEFAULT_CUR_TOL = 1e-11


def method_label(method: str, *, rsvd_rank: int | None = None) -> str:
    base = METHOD_STYLE.get(method, {}).get("label", method)
    if method == "rsvd_fmm" and rsvd_rank is not None:
        return f"{base} (r={int(rsvd_rank)})"
    return base


def set_plot_style():
    try:
        plt.style.use("seaborn-v0_8-paper")
    except Exception:
        try:
            plt.style.use("seaborn-paper")
        except Exception:
            plt.style.use("default")

    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "Liberation Serif"],
            "font.size": 14,
            "axes.labelsize": 15,
            "axes.titlesize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.titlesize": 16,
            "axes.linewidth": 2.0,
            "grid.linewidth": 1.2,
            "lines.linewidth": 3.5,
            "lines.markersize": 10,
            "patch.linewidth": 1.2,
            "xtick.major.width": 1.8,
            "ytick.major.width": 1.8,
            "xtick.minor.width": 1.2,
            "ytick.minor.width": 1.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.25,
            "text.usetex": False,
        }
    )


def ensure_dir(path_str: str) -> Path:
    path = Path(path_str)
    path.mkdir(parents=True, exist_ok=True)
    return path


def sample_real_interval(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-1.0, 1.0, size=n)


def sample_unit_circle(n: int, rng: np.random.Generator) -> np.ndarray:
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    return np.exp(1j * theta)


def f_sin_factory(d: float):
    def f(x: np.ndarray) -> np.ndarray:
        return np.sin( d * x)
    return f


def f_tan_factory(d: float):
    sqrt_d = float(np.sqrt(d))

    def f(z: np.ndarray) -> np.ndarray:
        # For unit circle points, define z**sqrt_d via angle to avoid branch issues.
        angles = np.angle(z)
        z_power = np.exp(1j * sqrt_d * angles)
        return np.tan(sqrt_d * z_power)

    return f


def build_full_cauchy_like_matrix(
    c: np.ndarray, q: np.ndarray, g: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """
    Build the explicit Cauchy-like matrix

        A[i,j] = sum_k g[i,k] * b[j,k] / (c[i] - q[j]).

    This is used only for small-n experiments where forming A is feasible.
    """
    c = np.asarray(c, dtype=np.complex128)
    q = np.asarray(q, dtype=np.complex128)
    g = np.asarray(g, dtype=np.complex128)
    b = np.asarray(b, dtype=np.complex128)
    if g.ndim == 1:
        g = g[:, None]
    if b.ndim == 1:
        b = b[:, None]
    numer = g @ b.T
    denom = c[:, None] - q[None, :]
    A = numer / denom
    return np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)


def make_ranks(max_rank: int) -> list[int]:
    max_rank = int(max_rank)
    base = list(range(1, min(20, max_rank) + 1))
    if max_rank > 20:
        step = 5 if max_rank <= 200 else 10
        base.extend(list(range(25, max_rank + 1, step)))
    return sorted(set(base))




def svd_errors(A: np.ndarray, max_rank: int) -> list[float]:
    """
    Compute SVD residual Frobenius norms for ranks 0, 1, 2, ..., max_rank.
    
    Returns errors where errs[k] = ||A - A_k||_F where A_k is the rank-k SVD approximation.
    """
    A = np.asarray(A)
    U, s, Vh = jnp.linalg.svd(A, full_matrices=False)
    # Convert JAX array to numpy for compatibility
    s = np.asarray(s)
    max_k = int(min(max_rank, len(s)))

    # Compute residual norms using cumsum: ||A - A_k||_F^2 = sum_{i=k}^{n-1} s[i]^2
    s_squared = s ** 2
    # Cumulative sum from the end: remaining[k] = sum_{i=k}^{n-1} s[i]^2
    # Pad with 0 at the end for rank == len(s) case
    remaining_squared = np.concatenate([np.cumsum(s_squared[::-1])[::-1], [0.0]])
    # Take square roots and convert to list
    errs = [float(np.sqrt(remaining_squared[k])) for k in range(max_k + 1)]

    if max_rank > len(s):
        errs.extend([0.0] * (max_rank - len(s)))
    return errs


def build_indices(
    c: np.ndarray,
    q: np.ndarray,
    g: np.ndarray,
    b: np.ndarray,
    max_rank: int,
    block_size: int,
    use_exact_norm: bool,
    barnes_hut_theta: float,
    rng_seed: int,
    method: str,
    gb_tolerance: float | None = None,
    frob_norm_relative_tolerance: float | None = None,
    return_true_norms: bool = False,
    fmm_precision: float = 5.0,
    explicit_matrix: np.ndarray | None = None,
    rsvd_full_error: bool = False,
    rsvd_n_iter: int = 0,
):
    
    if method == "c2plu":
        # Use upper_bound method with rakau backend and greedy sampling
        row_sampling = 'greedy'
        i_indices, j_indices, g_b_norms, _, metadata = build_cur_unified(
            c, q, g, b, max_rank,
            sampling_method='upper_bound',
            block_size=block_size,
            rng_seed=rng_seed,
            time_blocks=False,
            allow_extra_after_tol=False,
            frob_norm_relative_tolerance=frob_norm_relative_tolerance,
            column_sampling=row_sampling,
            row_sampling=row_sampling,
            upper_bound_backend='rakau',
            return_true_norms=return_true_norms,
        )
        # Extract Frobenius norms from metadata (actual norms)
        frobenius_norms = metadata.get('frobenius_norms', np.full(max_rank, np.nan))
        return i_indices, j_indices, frobenius_norms
    
    if method == "r2plu_ub":
        # Use upper_bound method with rakau backend and random sampling
        row_sampling = 'random'
        i_indices, j_indices, g_b_norms, _, metadata = build_cur_unified(
            c, q, g, b, max_rank,
            sampling_method='upper_bound',
            block_size=block_size,
            rng_seed=rng_seed,
            time_blocks=False,
            allow_extra_after_tol=False,
            frob_norm_relative_tolerance=frob_norm_relative_tolerance,
            column_sampling=row_sampling,
            row_sampling=row_sampling,
            upper_bound_backend='rakau',
            return_true_norms=return_true_norms,
        )
        # Extract Frobenius norms from metadata (actual norms)
        frobenius_norms = metadata.get('frobenius_norms', np.full(max_rank, np.nan))
        return i_indices, j_indices, frobenius_norms
    
    if method == "rsvd_fmm":
        if not HAS_FMM:
            raise ImportError("pyfmmlib2d is required for rsvd_fmm method. Install it or set PYFMMLIB2D_PATH.")
        # Use randomized SVD with FMM matvecs
        # Pass vendor path explicitly
        vendor_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor", "pyfmmlib2d")
        fmm_op = CauchyFMMOperator(c, q, g, b, precision=int(fmm_precision), vendor_path=vendor_path if os.path.exists(vendor_path) else None)
        use_explicit_matrix = False
        if use_explicit_matrix:
            A = build_full_cauchy_like_matrix(c, q, g, b)
            print("Using explicit matrix")
            matvec = lambda x: A @ x
            rmatvec = lambda x: A.conj().T @ x
        else:
            matvec = fmm_op.matvec
            rmatvec = fmm_op.rmatvec
        U, s, Vh = randomized_svd_operator(
            matvec,
            rmatvec,
            fmm_op.shape,
            max_rank,
            oversample=10,
            n_iter=rsvd_n_iter,
            seed=rng_seed,
        )
        if rsvd_full_error and explicit_matrix is not None:
            # Convert to JAX arrays for GPU computation
            A = build_full_cauchy_like_matrix(c, q, g, b)
            # A = jnp.asarray(explicit_matrix, dtype=explicit_matrix.dtype)
            U_jax = jnp.asarray(U)
            s_jax = jnp.asarray(s)
            Vh_jax = jnp.asarray(Vh)

            # Compute residual norms using a full orthonormal basis from U.
            U_jax_complete, _ = jnp.linalg.qr(U_jax, mode="complete")
            UTA = U_jax_complete.conj().T @ A
            row_norms_sq = jnp.sum(jnp.abs(UTA) ** 2, axis=1)
            residual_sq = jnp.cumsum(row_norms_sq[::-1])[::-1]
            residual_sq = jnp.concatenate([residual_sq, jnp.zeros((1,), dtype=residual_sq.dtype)])
            residual_sq = jnp.maximum(residual_sq, 0.0)
            max_k = min(max_rank, residual_sq.shape[0] - 1)
            residual_norms = jnp.sqrt(residual_sq[: max_k + 1])
            if max_rank > max_k and residual_norms.size > 0:
                pad = jnp.full((max_rank - max_k,), residual_norms[-1])
                residual_norms = jnp.concatenate([residual_norms, pad])
            frobenius_norms = [float(x) for x in np.asarray(residual_norms)]
        else:
            # Approximate: ||A - A_k||_F ≈ sqrt(sum(s[k:]^2)).
            # (Randomized SVD doesn't provide exact singular values.)
            frobenius_norms = []
        # Return dummy indices (not used for SVD)
        dummy_indices = np.arange(max_rank, dtype=np.int32)
        return dummy_indices, dummy_indices, np.array(frobenius_norms)
    
    raise ValueError(f"Unknown method: {method}")


def run_case_data(case: str, n: int, d: float, rng: np.random.Generator):
    # Here `n` denotes the matrix dimension, so we build an n×n Loewner matrix.
    n_rows = n
    n_cols = n
    if case == "sin":
        x = sample_real_interval(n_rows, rng)
        y = sample_real_interval(n_cols, rng)
        f = f_sin_factory(d)
        label = f"sin(pi*{d}*x)"
    elif case == "tan":
        x = sample_unit_circle(n_rows, rng)
        y = sample_unit_circle(n_cols, rng)
        f = f_tan_factory(d)
        label = f"tan(sqrt({d})*z**sqrt({d}))"
    else:
        raise ValueError(f"Unknown case: {case}")

    c, q, g, b = build_loewner_cauchy_matrix(x, y, f)
    return {
        "x": x,
        "y": y,
        "f": f,
        "c": c,
        "q": q,
        "g": g,
        "b": b,
        "label": label,
    }


def compare_methods(
    n: int,
    d_sin: float,
    d_tan: float,
    max_rank: int,
    block_size: int,
    barnes_hut_theta: float,
    num_seeds: int,
    svd_mode: str,
    cur_tol: float,
    output_dir: Path,
    plot_norm_bounds: bool = False,
    include_rsvd_fmm: bool = False,
    rsvd_rank: int | None = None,
    fmm_precision: float = 5.0,
    rsvd_n_iter: int = 3,
):
    if include_rsvd_fmm and not HAS_FMM:
        raise ImportError(
            "RSVD (FMM) requested but unavailable. Install `pyfmmlib2d` (or set "
            "`PYFMMLIB2D_PATH`) so `from cauchy.fmm_cauchy import CauchyFMMOperator` works."
        )
    rng = np.random.default_rng(42)
    cases = ["sin", "tan"]
    d_map = {"sin": d_sin, "tan": d_tan}
    results = {}

    for case in cases:
        d = d_map[case]
        data = run_case_data(case, n, d, rng)
        max_rank_case = min(max_rank, len(data["x"]), len(data["y"]))
        ranks = make_ranks(max_rank_case)
        initial_norm = float(frob_norm_from_uv(jnp.asarray(data["g"]), jnp.asarray(data["b"])))
        svd_errs = [np.nan] * (max_rank_case + 1)
        A_full = None
        if svd_mode == "full" or svd_mode == "auto":
            A_full = build_full_cauchy_like_matrix(data["c"], data["q"], data["g"], data["b"])
            svd_errs = svd_errors(A_full, max_rank_case)
            # Ensure length matches
            if len(svd_errs) < max_rank_case + 1:
                svd_errs.extend([0.0] * (max_rank_case + 1 - len(svd_errs)))
            elif len(svd_errs) > max_rank_case + 1:
                svd_errs = svd_errs[:max_rank_case + 1]
        # import ipdb; ipdb.set_trace()
        method_results = {}
        methods_to_test = ["c2plu", "r2plu_ub"]
        if include_rsvd_fmm:
            methods_to_test.append("rsvd_fmm")
        for method in methods_to_test:
            seeds = [42 + i for i in range(num_seeds)] if method.startswith("r2plu") else [42]
            all_errs = []
            for seed in seeds:
                rsvd_full_error = (method == "rsvd_fmm") and (A_full is not None)
                _, _, frobenius_norms = build_indices(
                    data["c"],
                    data["q"],
                    data["g"],
                    data["b"],
                    int(min(max_rank_case, rsvd_rank)) if (method == "rsvd_fmm" and rsvd_rank is not None) else max_rank_case,
                    block_size=block_size,
                    use_exact_norm=False,
                    barnes_hut_theta=1.0,
                    rng_seed=seed,
                    method=method,
                    frob_norm_relative_tolerance=cur_tol,
                    return_true_norms=not plot_norm_bounds,  # Turn off true norms when plot_norm_bounds is True
                    fmm_precision=fmm_precision,
                    explicit_matrix=A_full,
                    rsvd_full_error=rsvd_full_error,
                    rsvd_n_iter=rsvd_n_iter,
                )
                errs = frobenius_norms.tolist() if isinstance(frobenius_norms, np.ndarray) else list(frobenius_norms)
                # With CUR early stopping enabled, different seeds can stop at different
                # ranks. Pad out to a common length so we can compute mean/std.
                target_len = max_rank_case + 1  # includes rank-0 residual at index 0
                if len(errs) < target_len:
                    pad_val = errs[-1] if errs else np.nan
                    errs = errs + [pad_val] * (target_len - len(errs))
                all_errs.append(errs)
            err_arr = np.array(all_errs)
            method_results[method] = {
                "mean": np.nanmean(err_arr, axis=0).tolist(),
                "std": np.nanstd(err_arr, axis=0).tolist(),
                "min": np.nanmin(err_arr, axis=0).tolist(),
                "max": np.nanmax(err_arr, axis=0).tolist(),
            }
            if method.startswith("r2plu") and num_seeds > 1:
                method_results[method]["all"] = all_errs
        # Convert svd_errs properly - handle both np.nan and None
        svd_errors_final = []
        for e in svd_errs:
            if e is None:
                svd_errors_final.append(None)
            elif isinstance(e, (int, float)) and np.isfinite(e):
                svd_errors_final.append(float(e))
            else:
                svd_errors_final.append(None)
        
        results[case] = {
            "ranks": [int(r) for r in ranks],
            "svd_errors": svd_errors_final,
            "methods": method_results,
            "label": data["label"],
            "error_mode": "frob_row_norm_residual",
            "initial_norm": initial_norm,
        }

    set_plot_style()

    # Collect all handles and labels for separate legend file (deduplicated)
    all_handles = []
    all_labels = []
    seen_labels = set()
    
    # Create separate plot for each case
    for case in cases:
        d = d_map[case]
        res = results[case]
        ranks = res["ranks"]
        
        # Create single subplot figure
        fig, ax = plt.subplots(1, 1, figsize=(6, 4.8))
        
        svd_errs_arr = np.array([e if e is not None else np.nan for e in res["svd_errors"]])
        if len(svd_errs_arr) > 0 and np.isfinite(svd_errs_arr).any():
            # Calculate marker spacing
            marker_every = max(1, len(svd_errs_arr) // 10)
            line = ax.semilogy(
                np.arange(len(svd_errs_arr)),
                svd_errs_arr,
                linestyle=METHOD_STYLE["svd"]["linestyle"],
                color=METHOD_STYLE["svd"]["color"],
                label=METHOD_STYLE["svd"]["label"],
                linewidth=5.0,
                marker=METHOD_STYLE["svd"]["marker"],
                markersize=10,
                markevery=marker_every,
                markerfacecolor='white',
                markeredgewidth=3.0,
                markeredgecolor=METHOD_STYLE["svd"]["color"],
                alpha=0.9,
                zorder=3,
            )
            # Add to legend only once
            if METHOD_STYLE["svd"]["label"] not in seen_labels:
                all_handles.append(line[0])
                all_labels.append(METHOD_STYLE["svd"]["label"])
                seen_labels.add(METHOD_STYLE["svd"]["label"])
        
        for method in res["methods"].keys():
            style = METHOD_STYLE[method]
            mean = np.array(res["methods"][method]["mean"])
            std = np.array(res["methods"][method]["std"])
            min_vals = np.array(res["methods"][method]["min"])
            max_vals = np.array(res["methods"][method]["max"])
            # Calculate marker spacing
            marker_every = max(1, len(mean) // 10)
            line = ax.semilogy(
                np.arange(len(mean)),
                mean,
                linestyle=style["linestyle"],
                color=style["color"],
                marker=style["marker"],
                label=style["label"],
                linewidth=5.0,
                markersize=10,
                markevery=marker_every,
                markerfacecolor='white',
                markeredgewidth=3.0,
                markeredgecolor=style["color"],
                alpha=0.9,
                zorder=3,
            )
            # Add to legend only once
            if style["label"] not in seen_labels:
                all_handles.append(line[0])
                all_labels.append(style["label"])
                seen_labels.add(style["label"])
            
            if num_seeds > 1 and method.startswith("r2plu"):
                ax.fill_between(
                    np.arange(len(mean)),
                    np.maximum(min_vals, 1e-18),
                    max_vals,
                    color=style["color"],
                    alpha=0.15,
                    linewidth=0,
                    zorder=1,
                )
        
        # Remove labels and title
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
        
        # Enhanced grid (matching experiment3 style)
        ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.8, color='gray', which='major')
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, color='gray', which='minor')
        ax.set_axisbelow(True)
        ax.minorticks_on()
        
        # Enhanced spines and ticks (matching experiment3 style)
        for spine in ax.spines.values():
            spine.set_linewidth(2.2)
            spine.set_color('black')
        ax.tick_params(axis='both', labelsize=18, width=2.0, length=6, 
                       which='major', color='black')
        ax.tick_params(axis='both', labelsize=18, width=1.2, length=3, which='minor', color='gray')
        
        # Save individual plot with case and d in filename
        plot_path = output_dir / f"loewner_compare_methods_n{n}_{case}_{int(d)}.png"
        fig.patch.set_facecolor('white')
        fig.tight_layout()
        fig.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.35)
        print(f"Saved plot: {plot_path}")
        plt.close(fig)
        
        # Plot Frobenius norm decay
        fig_frob, ax_frob = plt.subplots(1, 1, figsize=(6, 4.8))
        fig_frob.patch.set_facecolor('white')
        
        for method in methods_to_test:
            if method not in res["methods"]:
                continue
            style = METHOD_STYLE[method]
            method_data = res["methods"][method]
            mean_norms = np.array(method_data["mean"])
            std_norms = np.array(method_data["std"])
            min_norms = np.array(method_data["min"])
            max_norms = np.array(method_data["max"])
            x_vals = np.arange(len(mean_norms))
            
            marker_every = max(1, len(mean_norms) // 10)
            line = ax_frob.semilogy(
                x_vals,
                mean_norms,
                linestyle=style["linestyle"],
                color=style["color"],
                marker=style["marker"],
                label=style["label"],
                linewidth=5.0,
                markersize=10,
                markevery=marker_every,
                markerfacecolor='white',
                markeredgewidth=3.0,
                markeredgecolor=style["color"],
                alpha=0.9,
                zorder=3,
            )
            
            if num_seeds > 1 and method.startswith("r2plu"):
                ax_frob.fill_between(
                    x_vals,
                    np.maximum(min_norms, 1e-20),
                    max_norms,
                    color=style["color"],
                    alpha=0.15,
                    linewidth=0,
                    zorder=1,
                )
        
        ax_frob.set_xlabel("")
        ax_frob.set_ylabel("")
        ax_frob.set_title("")
        ax_frob.grid(True, alpha=0.4, linestyle='-', linewidth=0.8, color='gray', which='major')
        ax_frob.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, color='gray', which='minor')
        ax_frob.set_axisbelow(True)
        ax_frob.minorticks_on()
        
        for spine in ax_frob.spines.values():
            spine.set_linewidth(2.2)
            spine.set_color('black')
        ax_frob.tick_params(axis='both', labelsize=18, width=2.0, length=6, 
                           which='major', color='black')
        ax_frob.tick_params(axis='both', labelsize=18, width=1.2, length=3, which='minor', color='gray')
        
        frob_plot_path = output_dir / f"loewner_frobenius_norm_n{n}_{case}_{int(d)}.png"
        fig_frob.tight_layout()
        fig_frob.savefig(frob_plot_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.35)
        print(f"Saved Frobenius norm plot: {frob_plot_path}")
        plt.close(fig_frob)
    
    # Save legend separately (matching experiment3 style)
    if all_handles:
        fig_legend, legend_ax = plt.subplots(figsize=(4, 6))
        legend_ax.axis('off')
        fig_legend.patch.set_facecolor('white')
        legend = fig_legend.legend(all_handles, all_labels, loc='center', ncol=1,
                                  frameon=True, fancybox=False, shadow=False,
                                  framealpha=1.0, facecolor='white', edgecolor='black',
                                  fontsize=15, handlelength=2.5, handletextpad=0.5, columnspacing=1.5)
        legend.get_frame().set_linewidth(2.0)
        legend.get_frame().set_boxstyle('round', pad=0.5)
        legend_path = output_dir / f"loewner_compare_methods_n{n}_d{d_sin}and{d_tan}_legend.png"
        fig_legend.savefig(legend_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.2)
        print(f"Saved legend: {legend_path}")
        plt.close(fig_legend)

    out_json = output_dir / f"loewner_compare_methods_n{n}_d{d_sin}and{d_tan}.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Return list of plot paths
    plot_paths = [output_dir / f"loewner_compare_methods_n{n}_{case}_{int(d_map[case])}.png" for case in cases]
    if all_handles:
        plot_paths.append(legend_path)
    return plot_paths, out_json


def time_to_tolerance(
    c: np.ndarray,
    q: np.ndarray,
    g: np.ndarray,
    b: np.ndarray,
    max_rank: int,
    block_size: int,
    barnes_hut_theta: float,
    tol: float,
    tol_mode: str,
    method: str,
    rng_seed: int,
):
    initial_norm = float(frob_norm_from_uv(jnp.asarray(g), jnp.asarray(b)))
    if tol_mode == "relative":
        frob_norm_relative_tolerance = tol
    else:
        frob_norm_relative_tolerance = tol / initial_norm if initial_norm > 0 else None
    start = time.perf_counter()
    theta_used = 1.0 if method == "c2plu" else barnes_hut_theta
    i_idx, _, frobenius_norms = build_indices(
        c,
        q,
        g,
        b,
        max_rank,
        block_size,
        use_exact_norm=False,
        barnes_hut_theta=theta_used,
        rng_seed=rng_seed,
        method=method,
        frob_norm_relative_tolerance=frob_norm_relative_tolerance,
    )
    elapsed = time.perf_counter() - start
    actual_rank = int(len(i_idx))
    final_norm = float(frobenius_norms[-1]) if len(frobenius_norms) > 0 else np.nan
    frobenius_norms_list = (
        frobenius_norms.tolist()
        if isinstance(frobenius_norms, np.ndarray)
        else list(frobenius_norms)
    )
    return {
        "time": elapsed,
        "rank": actual_rank,
        "initial_norm": initial_norm,
        "final_norm": final_norm,
        "frobenius_norms": frobenius_norms_list,
    }


def timing_fixed_d(
    d_sin: float,
    d_tan: float,
    n_values: list[int],
    max_rank: int,
    block_size: int,
    barnes_hut_theta: float,
    tol: float,
    tol_mode: str,
    num_seeds: int,
    output_dir: Path,
    include_rsvd_fmm: bool = False,
    rsvd_rank: int = 600,
    fmm_precision: float = 5.0,
):
    if include_rsvd_fmm and not HAS_FMM:
        raise ImportError(
            "RSVD (FMM) requested but unavailable. Install `pyfmmlib2d` (or set "
            "`PYFMMLIB2D_PATH`) so `from cauchy.fmm_cauchy import CauchyFMMOperator` works."
        )
    rng = np.random.default_rng(17)
    cases = ["sin", "tan"]
    d_map = {"sin": d_sin, "tan": d_tan}
    results = {}

    for case in cases:
        d = d_map[case]
        print(f"\nProcessing case: {case} (d={d})")
        case_results = {"n_values": n_values, "r2plu_ub": [], "c2plu": []}
        if include_rsvd_fmm:
            case_results["rsvd_fmm"] = []
        for n in n_values:
            data = run_case_data(case, n, d, rng)
            max_rank_eff = int(min(n // 2, max_rank))
            for method in ("c2plu","r2plu_ub"):
                seeds = [202 + i for i in range(num_seeds)] if method.startswith("r2plu") else [202]
                times = []
                ranks = []
                frobenius_norms_list = []
                for seed in seeds:
                    res = time_to_tolerance(
                        data["c"],
                        data["q"],
                        data["g"],
                        data["b"],
                        max_rank_eff,
                        block_size,
                        barnes_hut_theta,
                        tol,
                        tol_mode,
                        method,
                        seed,
                    )
                    times.append(res["time"])
                    ranks.append(res["rank"])
                    frobenius_norms_list.append(res["frobenius_norms"])
                    print(f"  {method_label(method)} | case={case}, n={n}, seed={seed} | time: {res['time']:.3f}s, rank: {res['rank']}, final_norm: {res['final_norm']:.2e}")
                case_results[method].append(
                    {
                        "time_mean": float(np.mean(times)),
                        "time_std": float(np.std(times)),
                        "rank_mean": float(np.mean(ranks)),
                        "rank_std": float(np.std(ranks)),
                        "frobenius_norms": frobenius_norms_list,
                    }
                )
            if include_rsvd_fmm:
                rsvd_rank_eff = int(min(rsvd_rank, len(data["x"]), len(data["y"])))
                if rsvd_rank_eff < 1:
                    case_results["rsvd_fmm"].append(
                        {
                            "time_mean": float("nan"),
                            "time_std": float("nan"),
                            "rank_mean": 0.0,
                            "rank_std": 0.0,
                            "frobenius_norms": [],
                        }
                    )
                else:
                    seed = 303
                    start = time.perf_counter()
                    _, _, frobenius_norms = build_indices(
                        data["c"],
                        data["q"],
                        data["g"],
                        data["b"],
                        rsvd_rank_eff,
                        block_size=block_size,
                        use_exact_norm=False,
                        barnes_hut_theta=1.0,
                        rng_seed=seed,
                        method="rsvd_fmm",
                        fmm_precision=fmm_precision,
                    )
                    elapsed = time.perf_counter() - start
                    final_norm = float(frobenius_norms[-1]) if len(frobenius_norms) > 0 else np.nan
                    print(f"  {method_label('rsvd_fmm', rsvd_rank=rsvd_rank_eff)} | case={case}, n={n} | time: {elapsed:.3f}s, rank: {rsvd_rank_eff}, final_norm: {final_norm:.2e}")
                    frob_list = (
                        frobenius_norms.tolist()
                        if isinstance(frobenius_norms, np.ndarray)
                        else list(frobenius_norms)
                    )
                    case_results["rsvd_fmm"].append(
                        {
                            "time_mean": float(elapsed),
                            "time_std": 0.0,
                            "rank_mean": float(rsvd_rank_eff),
                            "rank_std": 0.0,
                            "frobenius_norms": [frob_list],
                        }
                    )
        results[case] = case_results

    set_plot_style()
    
    # Collect all handles and labels for separate legend file (deduplicated)
    all_handles = []
    all_labels = []
    seen_labels = set()
    
    # Create separate plot for each case
    for case in cases:
        res = results[case]
        n_vals = res["n_values"]
        methods_to_plot = ["c2plu", "r2plu_ub"]
        if "rsvd_fmm" in res:
            methods_to_plot.append("rsvd_fmm")
        
        # Create single subplot figure
        fig, ax = plt.subplots(1, 1, figsize=(6, 4.8))
        fig.patch.set_facecolor('white')
        
        for method in methods_to_plot:
            style = METHOD_STYLE[method]
            mean = np.array([v["time_mean"] for v in res[method]])
            std = np.array([v["time_std"] for v in res[method]])
            rank_mean = np.array([v["rank_mean"] for v in res[method]])
            
            # Calculate marker spacing
            marker_every = max(1, len(n_vals) // 10)
            line = ax.plot(
                n_vals,
                mean,
                linestyle='-',  # Full/solid line for time
                color=style["color"],
                marker=style["marker"],
                label=method_label(method, rsvd_rank=rsvd_rank),
                linewidth=5.0,
                markersize=10,
                markevery=marker_every,
                markerfacecolor='white',
                markeredgewidth=3.0,
                markeredgecolor=style["color"],
                alpha=0.9,
                zorder=3,
            )
            
            # Add to legend only once
            if method_label(method, rsvd_rank=rsvd_rank) not in seen_labels:
                all_handles.append(line[0])
                all_labels.append(method_label(method, rsvd_rank=rsvd_rank))
                seen_labels.add(method_label(method, rsvd_rank=rsvd_rank))
            
            if num_seeds > 1 and method.startswith("r2plu"):
                ax.fill_between(
                    n_vals,
                    np.maximum(mean - std, 1e-12),
                    mean + std,
                    color=style["color"],
                    alpha=0.15,
                    linewidth=0,
                    zorder=1,
                )
        
        # Create dual-axis plot with rank on right y-axis
        ax2 = ax.twinx()
        
        for method in methods_to_plot:
            style = METHOD_STYLE[method]
            rank_mean = np.array([v["rank_mean"] for v in res[method]])
            rank_std = np.array([v["rank_std"] for v in res[method]])
            
            marker_every = max(1, len(n_vals) // 10)
            # Rank uses same marker and color as time, but dashed line
            line2 = ax2.plot(
                n_vals,
                rank_mean,
                linestyle='--',  # Dashed line for rank
                color=style["color"],
                marker=style["marker"],  # Same marker as time
                label=f"{method_label(method, rsvd_rank=rsvd_rank)} (rank)",
                linewidth=5.0,
                markersize=10,
                markevery=marker_every,
                markerfacecolor='white',
                markeredgewidth=3.0,
                markeredgecolor=style["color"],
                alpha=0.9,
                zorder=2,
            )
        
        # Remove labels and title
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax2.set_ylabel("")
        ax.set_title("")
        
        # Enhanced grid and styling
        ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.8, color='gray', which='major')
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, color='gray', which='minor')
        ax.set_axisbelow(True)
        ax.minorticks_on()
        
        for spine in ax.spines.values():
            spine.set_linewidth(2.2)
            spine.set_color('black')
        ax.tick_params(axis='both', labelsize=18, width=2.0, length=6, 
                       which='major', color='black')
        ax.tick_params(axis='both', labelsize=18, width=1.2, length=3, which='minor', color='gray')
        
        # Right axis styling - make it bold and visible like left axis
        ax2.spines['right'].set_linewidth(2.2)
        ax2.spines['right'].set_color('black')
        ax2.spines['right'].set_visible(True)
        ax2.tick_params(axis='y', labelsize=18, width=2.0, length=6, 
                       which='major', color='black', labelcolor='black')
        # Set nice integer ticks for rank
        all_ranks = []
        for method in methods_to_plot:
            all_ranks.extend([v["rank_mean"] for v in res[method]])
        if all_ranks:
            rank_min = min(all_ranks)
            rank_max = max(all_ranks)
            # Round to nice values
            rank_min = max(0, int(np.floor(rank_min / 10) * 10))
            rank_max = int(np.ceil(rank_max / 10) * 10)
            rank_range = rank_max - rank_min
            if rank_range > 0:
                rank_step = max(10, int(np.ceil(rank_range / 5) / 10) * 10)
                ticks = np.arange(rank_min, rank_max + rank_step, rank_step)
                ax2.set_yticks(ticks)
                ax2.set_ylim(bottom=max(0, rank_min - rank_step), top=rank_max + rank_step)
            else:
                # If all ranks are the same, just set a small range
                ax2.set_yticks([int(rank_min)])
                ax2.set_ylim(bottom=max(0, rank_min - 5), top=rank_max + 5)
        
        ax.set_xscale("log")
        ax.set_yscale("log")
        
        # Save dual-axis plot
        plot_path = output_dir / f"loewner_timing_fixed_d{int(d_map[case])}_{case}.png"
        fig.tight_layout()
        fig.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.35)
        print(f"Saved plot: {plot_path}")
        plt.close(fig)
        
        # Create separate rank plot
        fig_rank, ax_rank = plt.subplots(1, 1, figsize=(6, 4.8))
        fig_rank.patch.set_facecolor('white')
        
        methods_to_plot_rank = ["r2plu_ub", "c2plu"]
        if "rsvd_fmm" in res:
            methods_to_plot_rank.append("rsvd_fmm")

        for method in methods_to_plot_rank:
            style = METHOD_STYLE[method]
            rank_mean = np.array([v["rank_mean"] for v in res[method]])
            rank_std = np.array([v["rank_std"] for v in res[method]])
            
            marker_every = max(1, len(n_vals) // 10)
            line_rank = ax_rank.plot(
                n_vals,
                rank_mean,
                linestyle=style["linestyle"],
                color=style["color"],
                marker=style["marker"],
                label=method_label(method, rsvd_rank=rsvd_rank),
                linewidth=5.0,
                markersize=10,
                markevery=marker_every,
                markerfacecolor='white',
                markeredgewidth=3.0,
                markeredgecolor=style["color"],
                alpha=0.9,
                zorder=3,
            )
            
            if num_seeds > 1 and method.startswith("r2plu"):
                ax_rank.fill_between(
                    n_vals,
                    np.maximum(rank_mean - rank_std, 0),
                    rank_mean + rank_std,
                    color=style["color"],
                    alpha=0.15,
                    linewidth=0,
                    zorder=1,
                )
        
        ax_rank.set_xlabel("")
        ax_rank.set_ylabel("")
        ax_rank.set_title("")
        
        ax_rank.grid(True, alpha=0.4, linestyle='-', linewidth=0.8, color='gray', which='major')
        ax_rank.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, color='gray', which='minor')
        ax_rank.set_axisbelow(True)
        ax_rank.minorticks_on()
        
        for spine in ax_rank.spines.values():
            spine.set_linewidth(2.2)
            spine.set_color('black')
        ax_rank.tick_params(axis='both', labelsize=18, width=2.0, length=6, 
                           which='major', color='black')
        ax_rank.tick_params(axis='both', labelsize=18, width=1.2, length=3, which='minor', color='gray')
        
        ax_rank.set_xscale("log")
        
        rank_plot_path = output_dir / f"loewner_rank_fixed_d{int(d_map[case])}_{case}.png"
        fig_rank.tight_layout()
        fig_rank.savefig(rank_plot_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.35)
        print(f"Saved rank plot: {rank_plot_path}")
        plt.close(fig_rank)
        
        # Create subplots for Frobenius norm decay (one subplot per n-value)
        num_n = len(n_vals)
        ncols = min(4, num_n)  # Max 4 columns
        nrows = (num_n + ncols - 1) // ncols
        
        fig_frob, axes_frob = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.8 * nrows))
        fig_frob.patch.set_facecolor('white')
        if num_n == 1:
            axes_frob = [axes_frob]
        else:
            axes_frob = axes_frob.flatten()
        
        for idx, n_val in enumerate(n_vals):
            ax_frob = axes_frob[idx]
            
            for method in methods_to_plot:
                style = METHOD_STYLE[method]
                method_data = res[method][idx]
                frob_norms_list = method_data.get("frobenius_norms", [])
                
                if not frob_norms_list:
                    continue
                
                # Average across seeds if multiple seeds
                if len(frob_norms_list) > 1:
                    max_len = max(len(norms) for norms in frob_norms_list)
                    padded = [norms + [norms[-1]] * (max_len - len(norms)) if len(norms) < max_len else norms 
                              for norms in frob_norms_list]
                    mean_norms = np.nanmean(padded, axis=0)
                    std_norms = np.nanstd(padded, axis=0)
                else:
                    mean_norms = np.array(frob_norms_list[0])
                    std_norms = np.zeros_like(mean_norms)
                
                ranks_plot = np.arange(len(mean_norms))
                marker_every = max(1, len(mean_norms) // 10)
                
                ax_frob.semilogy(
                    ranks_plot,
                    mean_norms,
                    linestyle=style["linestyle"],
                    color=style["color"],
                    marker=style["marker"],
                    label=method_label(method, rsvd_rank=rsvd_rank),
                    linewidth=5.0,
                    markersize=10,
                    markevery=marker_every,
                    markerfacecolor='white',
                    markeredgewidth=3.0,
                    markeredgecolor=style["color"],
                    alpha=0.9,
                    zorder=3,
                )
                
                if len(frob_norms_list) > 1 and method.startswith("r2plu"):
                    ax_frob.fill_between(
                        ranks_plot,
                        np.maximum(mean_norms - std_norms, 1e-20),
                        mean_norms + std_norms,
                        color=style["color"],
                        alpha=0.15,
                        linewidth=0,
                        zorder=1,
                    )
            
            ax_frob.set_xlabel("")
            ax_frob.set_ylabel("")
            ax_frob.set_title(f"n = {n_val}", fontsize=18, fontweight='bold')
            ax_frob.grid(True, alpha=0.4, linestyle='-', linewidth=0.8, color='gray', which='major')
            ax_frob.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, color='gray', which='minor')
            ax_frob.set_axisbelow(True)
            ax_frob.minorticks_on()
            
            for spine in ax_frob.spines.values():
                spine.set_linewidth(2.2)
                spine.set_color('black')
            ax_frob.tick_params(axis='both', labelsize=18, width=2.0, length=6, 
                               which='major', color='black')
            ax_frob.tick_params(axis='both', labelsize=18, width=1.2, length=3, which='minor', color='gray')
        
        # Hide unused subplots
        for idx in range(num_n, len(axes_frob)):
            axes_frob[idx].axis('off')
        
        frob_plot_path = output_dir / f"loewner_frobenius_norm_fixed_d{int(d_map[case])}_{case}.png"
        fig_frob.tight_layout()
        fig_frob.savefig(frob_plot_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.35)
        print(f"Saved Frobenius norm plot: {frob_plot_path}")
        plt.close(fig_frob)

    # Combined plots showing both sin and tan on the same axes
    case_styles = {
        "sin": {"linestyle": "-", "case_label": "sin"},
        "tan": {"linestyle": "--", "case_label": "tan"},
    }
    case_text = {"sin": "sin(dz)", "tan": "tan(dz^d)"}
    shared_n = results[cases[0]]["n_values"]
    consistent_n = all(results[c]["n_values"] == shared_n for c in cases[1:])
    combined_handles = []
    combined_labels = []
    combined_seen = set()
    legend_path_combined = None

    if consistent_n:
        # Combined time plot
        fig_time, ax_time = plt.subplots(1, 1, figsize=(7, 5))
        fig_time.patch.set_facecolor('white')

        method_offsets = {"r2plu_ub": 0.012, "c2plu": -0.012, "rsvd_fmm": 0.0}

        for case in cases:
            res = results[case]
            n_vals = np.array(res["n_values"])
            methods_to_plot = ["c2plu", "r2plu_ub"]
            if "rsvd_fmm" in res:
                methods_to_plot.append("rsvd_fmm")

            for method in methods_to_plot:
                style = METHOD_STYLE[method]
                case_style = case_styles.get(case, {"linestyle": "-"})
                case_label_text = case_text.get(case, case)
                mean = np.array([v["time_mean"] for v in res[method]])
                std = np.array([v["time_std"] for v in res[method]])
                marker_every = max(1, len(n_vals) // 10)
                label = f"{method_label(method, rsvd_rank=rsvd_rank)} - {case_label_text}"

                offset_factor = 1.0 + method_offsets.get(method, 0.0)
                x_vals = n_vals * offset_factor

                line = ax_time.plot(
                    x_vals,
                    mean,
                    linestyle=case_style["linestyle"],
                    color=style["color"],
                    marker=style["marker"],
                    label=label,
                    linewidth=4.5,
                    markersize=9,
                    markevery=marker_every,
                    markerfacecolor='white',
                    markeredgewidth=2.5,
                    markeredgecolor=style["color"],
                    alpha=0.9,
                    zorder=3,
                )

                if label not in combined_seen:
                    combined_handles.append(line[0])
                    combined_labels.append(label)
                    combined_seen.add(label)

                if num_seeds > 1 and method.startswith("r2plu"):
                    ax_time.fill_between(
                        x_vals,
                        np.maximum(mean - std, 1e-12),
                        mean + std,
                        color=style["color"],
                        alpha=0.12,
                        linewidth=0,
                        zorder=1,
                    )

        ax_time.set_xlabel("")
        ax_time.set_ylabel("")
        ax_time.set_xscale("log")
        ax_time.set_yscale("log")
        ax_time.grid(True, alpha=0.4, linestyle='-', linewidth=0.8, color='gray', which='major')
        ax_time.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, color='gray', which='minor')
        ax_time.set_axisbelow(True)
        ax_time.minorticks_on()
        for spine in ax_time.spines.values():
            spine.set_linewidth(2.0)
            spine.set_color('black')
        ax_time.tick_params(axis='both', labelsize=18, width=2.0, length=6,
                            which='major', color='black')
        ax_time.tick_params(axis='both', labelsize=18, width=1.2, length=3, which='minor', color='gray')
        if len(shared_n) > 1:
            ax_time.set_xlim(left=min(shared_n) * 0.92, right=max(shared_n) * 1.08)

        combined_time_path = output_dir / f"loewner_timing_fixed_d{int(d_sin)}and{int(d_tan)}_time.png"
        fig_time.tight_layout()
        fig_time.savefig(combined_time_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.3)
        print(f"Saved combined time plot: {combined_time_path}")
        plt.close(fig_time)

        # Combined rank plot
        fig_rank_combo, ax_rank_combo = plt.subplots(1, 1, figsize=(7, 5))
        fig_rank_combo.patch.set_facecolor('white')

        for case in cases:
            res = results[case]
            n_vals = np.array(res["n_values"])
            methods_to_plot = ["c2plu", "r2plu_ub"]
            if "rsvd_fmm" in res:
                methods_to_plot.append("rsvd_fmm")

            for method in methods_to_plot:
                style = METHOD_STYLE[method]
                case_style = case_styles.get(case, {"linestyle": "-"})
                case_label_text = case_text.get(case, case)
                rank_mean = np.array([v["rank_mean"] for v in res[method]])
                rank_std = np.array([v["rank_std"] for v in res[method]])
                marker_every = max(1, len(n_vals) // 10)
                label = f"{method_label(method, rsvd_rank=rsvd_rank)} - {case_label_text}"

                offset_factor = 1.0 + method_offsets.get(method, 0.0)
                x_vals = n_vals * offset_factor

                ax_rank_combo.plot(
                    x_vals,
                    rank_mean,
                    linestyle=case_style["linestyle"],
                    color=style["color"],
                    marker=style["marker"],
                    label=label,
                    linewidth=4.5,
                    markersize=9,
                    markevery=marker_every,
                    markerfacecolor='white',
                    markeredgewidth=2.5,
                    markeredgecolor=style["color"],
                    alpha=0.9,
                    zorder=3,
                )

                if label not in combined_seen:
                    combined_handles.append(ax_rank_combo.lines[-1])
                    combined_labels.append(label)
                    combined_seen.add(label)

                if num_seeds > 1 and method.startswith("r2plu"):
                    ax_rank_combo.fill_between(
                        x_vals,
                        np.maximum(rank_mean - rank_std, 0),
                        rank_mean + rank_std,
                        color=style["color"],
                        alpha=0.12,
                        linewidth=0,
                        zorder=1,
                    )

        ax_rank_combo.set_xlabel("")
        ax_rank_combo.set_ylabel("")
        ax_rank_combo.set_xscale("log")
        ax_rank_combo.grid(True, alpha=0.4, linestyle='-', linewidth=0.8, color='gray', which='major')
        ax_rank_combo.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, color='gray', which='minor')
        ax_rank_combo.set_axisbelow(True)
        ax_rank_combo.minorticks_on()
        for spine in ax_rank_combo.spines.values():
            spine.set_linewidth(2.0)
            spine.set_color('black')
        ax_rank_combo.tick_params(axis='both', labelsize=18, width=2.0, length=6,
                                  which='major', color='black')
        ax_rank_combo.tick_params(axis='both', labelsize=18, width=1.2, length=3, which='minor', color='gray')
        if len(shared_n) > 1:
            ax_rank_combo.set_xlim(left=min(shared_n) * 0.92, right=max(shared_n) * 1.08)

        combined_rank_path = output_dir / f"loewner_rank_fixed_d{int(d_sin)}and{int(d_tan)}.png"
        fig_rank_combo.tight_layout()
        fig_rank_combo.savefig(combined_rank_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.3)
        print(f"Saved combined rank plot: {combined_rank_path}")
        plt.close(fig_rank_combo)

        # Separate legend for combined plots
        if combined_handles:
            fig_leg_c, ax_leg_c = plt.subplots(figsize=(5, 3.5))
            ax_leg_c.axis('off')
            leg = ax_leg_c.legend(
                combined_handles,
                combined_labels,
                loc='center',
                ncol=2,
                frameon=True,
                framealpha=1.0,
                edgecolor='black',
                facecolor='white',
                handlelength=2.5,
                handletextpad=0.6,
                columnspacing=1.5,
            )
            leg.get_frame().set_linewidth(2.0)
            legend_path_combined = output_dir / f"loewner_timing_fixed_d{int(d_sin)}and{int(d_tan)}_combined_legend.png"
            fig_leg_c.savefig(legend_path_combined, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.2)
            print(f"Saved combined legend: {legend_path_combined}")
            plt.close(fig_leg_c)

    # Save legend separately
    if all_handles:
        fig_legend, legend_ax = plt.subplots(figsize=(4, 6))
        legend_ax.axis('off')
        fig_legend.patch.set_facecolor('white')
        legend = fig_legend.legend(all_handles, all_labels, loc='center', ncol=1,
                                  frameon=True, fancybox=False, shadow=False,
                                  framealpha=1.0, facecolor='white', edgecolor='black',
                                  fontsize=15, handlelength=2.5, handletextpad=0.5, columnspacing=1.5)
        legend.get_frame().set_linewidth(2.0)
        legend.get_frame().set_boxstyle('round', pad=0.5)
        legend_path = output_dir / f"loewner_timing_fixed_d{int(d_sin)}and{int(d_tan)}_legend.png"
        fig_legend.savefig(legend_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.2)
        print(f"Saved legend: {legend_path}")
        plt.close(fig_legend)

    out_json = output_dir / f"loewner_timing_fixed_d{d_sin}and{d_tan}.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Return list of plot paths
    plot_paths = [output_dir / f"loewner_timing_fixed_d{int(d_map[case])}_{case}.png" for case in cases]
    if consistent_n:
        if legend_path_combined:
            plot_paths.extend([combined_time_path, combined_rank_path, legend_path_combined])
        else:
            plot_paths.extend([combined_time_path, combined_rank_path])
    if all_handles:
        plot_paths.append(legend_path)
    return plot_paths, out_json


def parse_list(values: str, cast=float):
    return [cast(v) for v in values.split(",") if v.strip()]


def main():
    parser = argparse.ArgumentParser(description="Loewner Cauchy-like paper experiments.")
    sub = parser.add_subparsers(dest="command", required=True)

    cmp_p = sub.add_parser("compare_methods", help="C2PLU/RPLU vs SVD.")
    cmp_p.add_argument("--n", type=int, default=2000)
    cmp_p.add_argument("--d-sin", type=float, default=1000)
    cmp_p.add_argument("--d-tan", type=float, default=400)
    cmp_p.add_argument("--max-rank", type=int, default=1000)
    cmp_p.add_argument("--block-size", type=int, default=3)
    cmp_p.add_argument("--theta", type=float, default=1.0)
    cmp_p.add_argument("--num-seeds", type=int, default=10)
    cmp_p.add_argument("--svd-mode", choices=["auto", "full", "skip"], default="auto")
    cmp_p.add_argument("--cur-tol", type=float, default=None)
    cmp_p.add_argument("--output-dir", type=str, default="plots/loewner_cauchy")
    cmp_p.add_argument("--plot_norm_bounds", action="store_true", default=False)
    cmp_p.add_argument("--include-rsvd-fmm", action="store_true", default=True)
    cmp_p.add_argument("--no-include-rsvd-fmm", dest="include_rsvd_fmm", action="store_false")
    cmp_p.add_argument("--rsvd-rank", type=int, default=None)
    cmp_p.add_argument("--rsvd-n-iter", type=int, default=0, help="Number of power iterations for randomized SVD (default: 3)")
    cmp_p.add_argument("--fmm-precision", type=float, default=5.0, help="FMM precision parameter (int from -2 to 5, default: 5.0 for high accuracy)")

    td_p = sub.add_parser("timing_fixed_d", help="Timing to tolerance for fixed d.")
    td_p.add_argument("--d-sin", type=float, default=1000)
    td_p.add_argument("--d-tan", type=float, default=400)
    td_p.add_argument("--n-values", type=str, default="1000,5000,10000,50000,100000,500000,1000000")
    # td_p.add_argument("--n-values", type=str, default="100000,100000,100000")#,500000,1000000")

    td_p.add_argument("--max-rank", type=int, default=100)
    td_p.add_argument("--block-size", type=int, default=3)
    td_p.add_argument("--theta", type=float, default=1.0)
    td_p.add_argument("--cur-tol", "--tol", dest="cur_tol", type=float, default=-1)
    td_p.add_argument("--tol-mode", choices=["relative", "absolute"], default="relative")
    td_p.add_argument("--include-rsvd-fmm", action="store_true", default=True)
    td_p.add_argument("--no-include-rsvd-fmm", dest="include_rsvd_fmm", action="store_false")
    td_p.add_argument("--rsvd-rank", type=int, default=100)
    td_p.add_argument("--fmm-precision", type=float, default=5.0, help="FMM precision parameter (int from -2 to 5, default: 5.0 for high accuracy)")
    td_p.add_argument("--num-seeds", type=int, default=1)
    td_p.add_argument("--output-dir", type=str, default="plots/loewner_cauchy")

    args = parser.parse_args()
    out_dir = ensure_dir(args.output_dir)

    if args.command == "compare_methods":
        compare_methods(
            n=args.n,
            d_sin=args.d_sin,
            d_tan=args.d_tan,
            max_rank=args.max_rank,
            block_size=args.block_size,
            barnes_hut_theta=args.theta,
            num_seeds=args.num_seeds,
            svd_mode=args.svd_mode,
            cur_tol=args.cur_tol,
            output_dir=out_dir,
            plot_norm_bounds=args.plot_norm_bounds,
            include_rsvd_fmm=args.include_rsvd_fmm,
            rsvd_rank=args.rsvd_rank,
            fmm_precision=args.fmm_precision,
            rsvd_n_iter=args.rsvd_n_iter,
        )
    elif args.command == "timing_fixed_d":
        n_values = parse_list(args.n_values, int)
        timing_fixed_d(
            d_sin=args.d_sin,
            d_tan=args.d_tan,
            n_values=n_values,
            max_rank=args.max_rank,
            block_size=args.block_size,
            barnes_hut_theta=args.theta,
            tol=args.cur_tol,
            tol_mode=args.tol_mode,
            num_seeds=args.num_seeds,
            include_rsvd_fmm=args.include_rsvd_fmm,
            rsvd_rank=args.rsvd_rank,
            fmm_precision=args.fmm_precision,
            output_dir=out_dir,
        )
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
