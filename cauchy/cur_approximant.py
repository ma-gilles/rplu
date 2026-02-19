"""
CUR-based rational approximant builder.

This module provides functionality to build rational approximants using
CauchyCUR decomposition until a tolerance criterion is met.
"""

import numpy as np
import time
import warnings
import jax.numpy as jnp
from cauchy.cauchy_jax import frob_norm_from_uv
from cauchy.barycentric import (
    evaluate_barycentric,
    fit_weights_svd,
    fit_weights_svd_jax,
    fit_weights_fmm_iterative,
    fit_weights_fmm_iterative_jax,
    barycentric_poles,
)
from cauchy.loewner_utils import build_loewner_cauchy_matrix_generator
from cauchy.unified_cur import build_cur_unified


def _fit_weights_standard(z_all, f_all, support_indices, remaining_indices, use_jax_svd=False):
    """
    Fit weights using standard single weight fitting.
    
    Parameters
    ----------
    z_all : array
        All points (pre-computed)
    f_all : array
        All function values (pre-computed)
    support_indices : array
        Indices into z_all for support points
    remaining_indices : array
        Indices into z_all for remaining points (for error minimization)
    
    Returns
    -------
    weights : array
        Barycentric weights
    """
    z_support = z_all[support_indices]
    f_support = f_all[support_indices]
    z_remaining = z_all[remaining_indices]
    f_remaining = f_all[remaining_indices]
    
    # Fit weights using all remaining points
    if use_jax_svd:
        weights = fit_weights_svd_jax(z_support, f_support, z_remaining, f_remaining)
    else:
        weights = fit_weights_svd(z_support, f_support, z_remaining, f_remaining)
    
    return weights


def _fit_weights_dispatch(z_all, f_all, support_indices, remaining_indices, method='svd_jax'):
    """
    Fit barycentric weights using the requested solver.

    method options:
      - 'svd_jax' (default): JAX SVD on device
      - 'svd_np' : NumPy/Scipy SVD on CPU
      - 'fmm_lanczos' : CPU Barnes–Hut + scipy eigsh (smallest singular vector)
      - 'fmm_lanczos_jax' : JAX Barnes–Hut + iterative Rayleigh steps
    """
    method = (method or 'svd_jax').lower()
    use_jax = method == 'svd_jax'
    if method == 'svd_jax':
        return _fit_weights_standard(z_all, f_all, support_indices, remaining_indices, use_jax_svd=True)
    if method == 'svd_np':
        return _fit_weights_standard(z_all, f_all, support_indices, remaining_indices, use_jax_svd=False)
    if method == 'fmm_lanczos':
        z_support = z_all[support_indices]
        f_support = f_all[support_indices]
        z_remaining = z_all[remaining_indices]
        f_remaining = f_all[remaining_indices]
        return fit_weights_fmm_iterative(z_support, f_support, z_remaining, f_remaining)
    if method == 'fmm_lanczos_jax':
        z_support = jnp.asarray(z_all[support_indices])
        f_support = jnp.asarray(f_all[support_indices])
        z_remaining = jnp.asarray(z_all[remaining_indices])
        f_remaining = jnp.asarray(f_all[remaining_indices])
        return np.asarray(fit_weights_fmm_iterative_jax(z_support, f_support, z_remaining, f_remaining))
    # Fallback to JAX SVD
    return _fit_weights_standard(z_all, f_all, support_indices, remaining_indices, use_jax_svd=True)


def _barycentric_prz(z_support, f_support, weights):
    """
    Compute poles, residues, and zeros for a barycentric rational function.

    Matches Chebfun's AAA pencil (see cauchy/aaa/aaa.m).
    """
    z_support = np.asarray(z_support)
    f_support = np.asarray(f_support)
    weights = np.asarray(weights)

    m = len(weights)
    if m <= 1:
        return np.array([]), np.array([]), np.array([])

    from scipy.linalg import eigvals

    B = np.eye(m + 1, dtype=complex)
    B[0, 0] = 0.0

    E = np.zeros((m + 1, m + 1), dtype=complex)
    E[0, 1:] = weights
    E[1:, 0] = 1.0
    E[1:, 1:] = np.diag(z_support)
    poles = eigvals(E, B)
    poles = poles[np.isfinite(poles)]

    if len(poles) > 0:
        diff = poles[:, np.newaxis] - z_support[np.newaxis, :]
        diff = np.where(np.abs(diff) < 1e-14, 1e-14 + 0.0j, diff)
        numerator = (1.0 / diff) @ (f_support * weights)
        denom_deriv = -((1.0 / diff) ** 2) @ weights
        denom_deriv = np.where(np.abs(denom_deriv) < 1e-14, 1e-14 + 0.0j, denom_deriv)
        residues = numerator / denom_deriv
    else:
        residues = np.array([], dtype=complex)

    E = np.zeros((m + 1, m + 1), dtype=complex)
    E[0, 1:] = weights * f_support
    E[1:, 0] = 1.0
    E[1:, 1:] = np.diag(z_support)
    zeros = eigvals(E, B)
    zeros = zeros[np.isfinite(zeros)]

    return poles, residues, zeros


def _chebfun_froissart_cleanup(
    z_all,
    f_all,
    support_indices,
    weights,
    cleanup_tol=1e-13,
    weight_method='svd_jax',
    exclude_indices_for_fit=None,
    support_tol=1e-14,
):
    """
    Chebfun AAA-style cleanup: remove Froissart doublets (aaa.m cleanup).

    Identifies spurious poles via small residues, removes the closest support points,
    then refits barycentric weights on remaining sample points.
    """
    support_indices = np.asarray(support_indices, dtype=int)
    weights = np.asarray(weights)
    if len(support_indices) <= 1:
        return support_indices, weights, 0

    z_support = np.asarray(z_all)[support_indices]
    f_support = np.asarray(f_all)[support_indices]

    try:
        poles, residues, _ = _barycentric_prz(z_support, f_support, weights)
    except Exception:
        return support_indices, weights, 0

    if len(poles) == 0 or len(residues) != len(poles):
        return support_indices, weights, 0

    z_all_arr = np.asarray(z_all)
    f_all_arr = np.asarray(f_all)

    # Chebfun's cleanup uses the sample set (Z,F) excluding current support points.
    sample_mask = np.ones(len(z_all_arr), dtype=bool)
    sample_mask[support_indices] = False
    if exclude_indices_for_fit is not None:
        sample_mask[np.asarray(exclude_indices_for_fit, dtype=int)] = False
    Z = z_all_arr[sample_mask]
    F = f_all_arr[sample_mask]

    if len(Z) == 0:
        return support_indices, weights, 0

    absF = np.abs(F)
    finite_nonzero = (absF > 0) & np.isfinite(absF)
    if not np.any(finite_nonzero):
        geometric_mean_of_absF = 0.0
    else:
        geometric_mean_of_absF = float(np.exp(np.mean(np.log(absF[finite_nonzero]))))
    if geometric_mean_of_absF == 0.0:
        return support_indices, weights, 0

    # Compute minimum distance from each pole to any sample point
    Zdistances = np.min(np.abs(poles[:, np.newaxis] - Z[np.newaxis, :]), axis=1)
    Zdistances = np.where(Zdistances == 0, np.finfo(float).tiny, Zdistances)
    # Identify Froissart doublets: poles with residues small relative to distance and function scale
    # Formula: |residue| / (distance * geometric_mean(|F|)) < cleanup_tol
    # This is equivalent to: |residue| / distance < cleanup_tol * geometric_mean(|F|)
    ii = np.where((np.abs(residues) / (Zdistances * geometric_mean_of_absF)) < cleanup_tol)[0]
    ni = int(len(ii))
    if ni == 0:
        return support_indices, weights, 0
    if ni == 1:
        warnings.warn("AAA:Froissart: 1 Froissart doublet")
    else:
        warnings.warn(f"AAA:Froissart: {ni} Froissart doublets")

    support_list = list(support_indices.tolist())
    for pole_idx in ii:
        if not support_list:
            break
        z_curr = np.asarray(z_all)[support_list]
        remove_pos = int(np.argmin(np.abs(z_curr - poles[pole_idx])))
        support_list.pop(remove_pos)

    support_new = np.asarray(support_list, dtype=int)
    removed = int(len(support_indices) - len(support_new))
    if removed <= 0:
        return support_indices, weights, 0

    exclude = []
    if exclude_indices_for_fit is not None:
        exclude.append(np.asarray(exclude_indices_for_fit, dtype=int))
    exclude.append(support_new)
    exclude_all = np.unique(np.concatenate(exclude)) if exclude else support_new

    # Weight fitting needs points away from all excluded support points to avoid singularities.
    mask = np.ones(len(z_all), dtype=bool)
    for z0 in np.asarray(z_all)[exclude_all]:
        mask &= np.abs(np.asarray(z_all) - z0) > support_tol
    remaining_indices = np.where(mask)[0]

    if len(support_new) == 0:
        return support_new, np.array([], dtype=complex), removed
    if len(remaining_indices) == 0:
        return support_new, np.ones(len(support_new), dtype=complex), removed

    weights_new = _fit_weights_dispatch(z_all, f_all, support_new, remaining_indices, method=weight_method)
    return support_new, weights_new, removed


def evaluate_barycentric_outside_support(z, z_support, f_support, weights, tol=1e-12, return_mask=False):
    """
    Evaluate barycentric interpolant away from support points.

    Returns NaN at points within `tol` of any support point to avoid trivial interpolation.
    """
    z = np.asarray(z)
    scalar_output = z.ndim == 0
    if scalar_output:
        z = np.array([z])

    if len(z_support) == 0:
        result = np.full(len(z), np.nan, dtype=complex)
        mask = np.zeros(len(z), dtype=bool)
        if scalar_output:
            return (result[0], mask[0]) if return_mask else result[0]
        return (result, mask) if return_mask else result

    diffs = z[:, np.newaxis] - np.asarray(z_support)[np.newaxis, :]
    outside_mask = np.all(np.abs(diffs) > tol, axis=1)
    result = evaluate_barycentric(z, z_support, f_support, weights)
    result = np.where(outside_mask, result, np.nan + 0.0j)

    if scalar_output:
        return (result[0], outside_mask[0]) if return_mask else result[0]
    return (result, outside_mask) if return_mask else result


class CURApproximant:
    """A rational approximant built using CUR decomposition."""
    def __init__(self, z_support, f_support, weights):
        """
        Parameters
        ----------
        z_support : array
            Support points (complex)
        f_support : array
            Function values at support points
        weights : array
            Barycentric weights
        """
        self.z_support = np.asarray(z_support)
        self.f_support = np.asarray(f_support)
        self.weights = np.asarray(weights)
    
    def __call__(self, z):
        """Evaluate the approximant at points z."""
        return evaluate_barycentric(z, self.z_support, self.f_support, self.weights)
    
    def compute_error(self, z_test, f_test):
        """Compute maximum error on test points."""
        if len(z_test) == 0:
            return 0.0
        f_approx = self(z_test)
        errors = np.abs(f_test - f_approx)
        valid_errors = errors[np.isfinite(errors)]
        return np.max(valid_errors) if len(valid_errors) > 0 else np.nan

    def evaluate_outside_support(self, z, tol=1e-12, return_mask=False):
        """Evaluate the approximant while excluding support points."""
        return evaluate_barycentric_outside_support(
            z, self.z_support, self.f_support, self.weights, tol=tol, return_mask=return_mask
        )

    def poles(self, weight_tol=1e-12):
        """Return poles of the barycentric interpolant."""
        return barycentric_poles(self.z_support, self.weights, weight_tol=weight_tol)


class CURHalfApproximant(CURApproximant):
    """CUR approximant built from only one side (rows or columns)."""

    def __init__(self, z_support, f_support, weights, side):
        super().__init__(z_support, f_support, weights)
        self.side = side  # 'x' or 'y'


def build_cur_half_approximant(z_points, f_points, f_func=None, relative_gb_tolerance=1e-13, frob_norm_relative_tolerance=1e-13, block_size=1,
                               use_exact_norm=False, barnes_hut_theta=1.0, max_rank=None, rng_seed=42,
                               sampling_method='rejection', allow_extra_after_tol=False,
                               row_sampling='random', column_sampling='random', use_jax_svd=True,
                               weight_method='svd_jax', cleanup=False, relative_gb_tolerance_cleanup=1e-14,
                               chebfun_cleanup=False, chebfun_cleanup_tol=None,
                               residual_norm_kind='frobenius',
                               split_percent=50.0,
                               force_side='auto'):
    """
    Build a CUR-based rational approximant using only one side of the CUR support.

    The algorithm:
    1. Split points into x/y sets and run CUR to get i_idx/j_idx.
    2. Fit barycentric weights using only y[j_idx] (columns) against all non-support points.
    3. Fit barycentric weights using only x[i_idx] (rows) against all non-support points.
    4. Pick the side with smaller max error on non-support points.
    5. (Optional) run cleanup pass to prune the chosen support points.

    split_percent controls the x/y split (x gets split_percent%, y gets the remainder).
    """
    z_all = np.asarray(z_points)
    if f_func is not None:
        f_all = f_func(z_all)
    else:
        f_all = np.asarray(f_points)

    rng = np.random.default_rng(rng_seed)
    n = len(z_all)
    all_indices = np.arange(n)
    rng.shuffle(all_indices)

    split_percent = float(split_percent)
    if not np.isfinite(split_percent) or split_percent <= 0.0 or split_percent >= 100.0:
        raise ValueError("split_percent must be in (0, 100)")
    force_side = str(force_side).lower()
    if force_side not in {"auto", "x", "y"}:
        raise ValueError("force_side must be 'auto', 'x', or 'y'")
    n1 = int(np.floor(n * split_percent / 100.0))
    n1 = max(1, min(n - 1, n1))
    index_set_1 = all_indices[:n1]
    index_set_2 = all_indices[n1:]

    x = z_all[index_set_1]
    y = z_all[index_set_2]
    f_x = f_all[index_set_1]
    f_y = f_all[index_set_2]

    c, q, g, b = build_loewner_cauchy_matrix_generator(x, y, f_x, f_y)
    initial_gb_norm = float(frob_norm_from_uv(g, b))

    if max_rank is None:
        max_rank = min(len(index_set_1), len(index_set_2)) - 1

    build_start = time.time()
    # Convert 'greedy' to 'upper_bound' (like c2plu in test_loewner_cauchy_paper.py).
    # Treat 'greedy' as a user-facing alias that implies greedy row/col selection.
    half_sampling_method = sampling_method
    half_row_sampling = row_sampling
    half_column_sampling = column_sampling
    if half_sampling_method == 'greedy':
        half_sampling_method = 'upper_bound'
        half_row_sampling = 'greedy'
        half_column_sampling = 'greedy'
    
    i_idx_all, j_idx_all, _, _, metadata = build_cur_unified(
        c, q, g, b, rank=max_rank,
        sampling_method=half_sampling_method,
        block_size=block_size,
        use_exact_norm=use_exact_norm,
        barnes_hut_theta=barnes_hut_theta,
        rng_seed=rng_seed,
        relative_gb_tolerance=relative_gb_tolerance,
        frob_norm_relative_tolerance=frob_norm_relative_tolerance,
        residual_norm_kind=residual_norm_kind,
        allow_extra_after_tol=allow_extra_after_tol,
        row_sampling=half_row_sampling,
        column_sampling=half_column_sampling,
        upper_bound_backend='rakau',
        C=5.0,
        max_leaf=64,
    )
    actual_rank = len(i_idx_all)
    i_idx = i_idx_all[:actual_rank]
    j_idx = j_idx_all[:actual_rank]
    build_time = time.time() - build_start
    final_gb_norm = metadata.get('final_gb_norm', initial_gb_norm)

    # Map indices into global z_all
    support_y = index_set_2[j_idx]
    support_x = index_set_1[i_idx]

    fit_start = time.time()
    # Fit weights using all points except the candidate's support points.
    remaining_x = np.setdiff1d(np.arange(len(z_all)), support_x)
    remaining_y = np.setdiff1d(np.arange(len(z_all)), support_y)

    # Candidate using y-support
    weights_y = _fit_weights_dispatch(z_all, f_all, support_y, remaining_y, method=weight_method)
    approx_y = CURHalfApproximant(z_all[support_y], f_all[support_y], weights_y, side='y')
    err_y = approx_y.compute_error(z_all[remaining_y], f_all[remaining_y]) if len(remaining_y) else 0.0

    # Candidate using x-support
    weights_x = _fit_weights_dispatch(z_all, f_all, support_x, remaining_x, method=weight_method)
    approx_x = CURHalfApproximant(z_all[support_x], f_all[support_x], weights_x, side='x')
    err_x = approx_x.compute_error(z_all[remaining_x], f_all[remaining_x]) if len(remaining_x) else 0.0
    fit_time = time.time() - fit_start

    if force_side == 'x':
        choose_x = True
    elif force_side == 'y':
        choose_x = False
    else:
        choose_x = np.isnan(err_y) or (not np.isnan(err_x) and err_x < err_y)

    if choose_x:
        chosen_side = 'x'
        support_indices = support_x
        weights = weights_x
        max_error = err_x
    else:
        chosen_side = 'y'
        support_indices = support_y
        weights = weights_y
        max_error = err_y
    suffix = "forced" if force_side in {"x", "y"} else "auto"
    print(
        f"CUR-Half chose {chosen_side}-side support ({suffix}; "
        f"x candidates: {len(index_set_1)}, y candidates: {len(index_set_2)})"
    )

    cleanup_time = 0.0
    cleanup_gb_norm = None



    # Optional Chebfun AAA-style cleanup: remove Froissart doublets and refit weights.
    chebfun_cleanup_time = 0.0
    chebfun_cleanup_removed = 0
    if chebfun_cleanup:
        chebfun_tol = 1e-13 if chebfun_cleanup_tol is None else float(chebfun_cleanup_tol)
        chebfun_start = time.time()
        support_indices, weights, chebfun_cleanup_removed = _chebfun_froissart_cleanup(
            z_all,
            f_all,
            support_indices,
            weights,
            cleanup_tol=chebfun_tol,
            weight_method=weight_method,
        )
        chebfun_cleanup_time = time.time() - chebfun_start
        build_time += chebfun_cleanup_time
        remaining = np.setdiff1d(np.arange(len(z_all)), support_indices)
        max_error = (
            CURHalfApproximant(z_all[support_indices], f_all[support_indices], weights, side=chosen_side)
            .compute_error(z_all[remaining], f_all[remaining])
            if len(remaining)
            else 0.0
        )

    chosen = CURHalfApproximant(z_all[support_indices], f_all[support_indices], weights, side=chosen_side)

    return chosen, {
        'rank': len(support_indices),
        'side': chosen_side,
        'build_time': build_time,
        'fit_time': fit_time,
        'cleanup': bool(cleanup),
        'cleanup_time': cleanup_time,
        'cleanup_gb_norm': cleanup_gb_norm,
        'chebfun_cleanup': bool(chebfun_cleanup),
        'chebfun_cleanup_tol': (None if chebfun_cleanup_tol is None else float(chebfun_cleanup_tol)),
        'chebfun_cleanup_time': chebfun_cleanup_time,
        'chebfun_cleanup_removed': chebfun_cleanup_removed,
        'gb_norm': final_gb_norm,
        'initial_gb_norm': initial_gb_norm,
        'frobenius_norms': metadata.get('frobenius_norms', None),
        'max_row_norms': metadata.get('max_row_norms', None),
        'residual_norms': metadata.get('residual_norms', None),
        'residual_norm_kind': metadata.get('residual_norm_kind', residual_norm_kind),
        'split_percent': split_percent,
        'force_side': force_side,
        'support_indices': support_indices,
        'max_error': max_error,
        'i_idx': i_idx,
        'j_idx': j_idx,
        'index_set_1': index_set_1,
        'index_set_2': index_set_2,
        'weights': weights,
    }
