"""
Barycentric interpolation utilities.

Provides functions for fitting barycentric weights and evaluating barycentric forms,
in both NumPy and JAX versions.
"""

import numpy as np
from scipy.linalg import svd
from scipy.sparse.linalg import LinearOperator, eigsh
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

_bh_backend = None
_bh_2d = None
# Prefer Rakau (GPU-capable).
try:
    from .Barnes_hut.barnes_hut_rakau import barnes_hut_2d as _bh_2d
    _bh_backend = "rakau"
except Exception:
    _bh_2d = None
    _bh_backend = None

# Expose availability for tests
BARNES_HUT_AVAILABLE = _bh_2d is not None


def _require_barnes_hut():
    if _bh_2d is None:
        raise RuntimeError(
            "Barnes-Hut backend is unavailable. "
            "Build the cauchy/Barnes_hut libraries or install Rakau to enable FMM acceleration."
        )


def _bh_potential(sources, targets, masses, theta=0.6, eps2=1e-14, use_gpu=None, backend=None):
    """
    Compute sum_j masses[j] / (|targets_i - sources_j|^2 + eps2) using Barnes–Hut.
    Returns potentials at the target locations only.
    """
    _require_barnes_hut()
    masses = jnp.asarray(masses, dtype=jnp.float64)
    src_x = jnp.asarray(jnp.real(sources), dtype=jnp.float64)
    src_y = jnp.asarray(jnp.imag(sources), dtype=jnp.float64)
    tgt_x = jnp.asarray(jnp.real(targets), dtype=jnp.float64)
    tgt_y = jnp.asarray(jnp.imag(targets), dtype=jnp.float64)

    zeros_tgt = jnp.zeros_like(tgt_x)
    all_x = jnp.concatenate([src_x, tgt_x])
    all_y = jnp.concatenate([src_y, tgt_y])
    all_mass = jnp.concatenate([masses, zeros_tgt])
    use_gpu_flag = bool(use_gpu) if use_gpu is not None else jax.default_backend() == "gpu"

    del backend  # Backward-compatible argument; only Rakau backend is supported.
    potentials = _bh_2d(all_x, all_y, all_mass, eps2=eps2, theta=theta, use_gpu=use_gpu_flag, backend="rakau")
    return potentials[len(src_x):]


def _cauchy_potential(sources, targets, charges, theta=0.6, eps2=1e-14, use_gpu=None, backend=None):
    """
    Fast approximation of sum_j charges[j] / (targets_i - sources_j) using the identity
    1/(z) = conj(z) / |z|^2 and the Barnes–Hut 1/|.|^2 potential.
    """
    charges = jnp.asarray(charges)
    # potentials with real/imag charges
    pot0 = _bh_potential(sources, targets, jnp.real(charges), theta=theta, eps2=eps2, use_gpu=use_gpu, backend=backend)
    pot0 = pot0 + 1j * _bh_potential(sources, targets, jnp.imag(charges), theta=theta, eps2=eps2, use_gpu=use_gpu, backend=backend)

    weighted = charges * jnp.conj(jnp.asarray(sources))
    pot1 = _bh_potential(sources, targets, jnp.real(weighted), theta=theta, eps2=eps2, use_gpu=use_gpu, backend=backend)
    pot1 = pot1 + 1j * _bh_potential(sources, targets, jnp.imag(weighted), theta=theta, eps2=eps2, use_gpu=use_gpu, backend=backend)

    return jnp.conj(jnp.asarray(targets)) * pot0 - pot1


def _apply_cauchy_like(q, c, b, g, vec, theta=0.8, eps2=1e-14, use_gpu=None, backend="rakau"):
    """
    Apply the Cauchy-like matrix C(q,c,b,g) to vec without forming C explicitly.
    C_ij = <g_i, b_j> / (c_i - q_j)
    """
    vec = jnp.asarray(vec)
    contributions = []
    for r in range(b.shape[1]):
        charges = b[:, r] * vec
        pot = _cauchy_potential(q, c, charges, theta=theta, eps2=eps2, use_gpu=use_gpu, backend=backend)
        contributions.append(pot * g[:, r])
    return jnp.sum(jnp.stack(contributions, axis=1), axis=1)


def _apply_cauchy_like_adj(q, c, b, g, vec, theta=0.8, eps2=1e-14, use_gpu=None, backend="rakau"):
    """Apply C^* to vec using the same fast potentials."""
    vec = jnp.asarray(vec)
    contributions = []
    for r in range(g.shape[1]):
        charges = jnp.conj(g[:, r]) * vec
        pot = _cauchy_potential(c, q, charges, theta=theta, eps2=eps2, use_gpu=use_gpu, backend=backend)
        contributions.append(pot * jnp.conj(b[:, r]))
    return jnp.sum(jnp.stack(contributions, axis=1), axis=1)


def fit_weights_svd(z_support, f_support, z_remaining, f_remaining):
    """
    Fit barycentric weights using SVD (NumPy version).
    
    Parameters
    ----------
    z_support : array
        Support points (complex)
    f_support : array
        Function values at support points
    z_remaining : array
        Remaining points (for error minimization)
    f_remaining : array
        Function values at remaining points
    
    Returns
    -------
    weights : array
        Barycentric weights (normalized)
    """
    m = len(z_support)
    n_remaining = len(z_remaining)
    
    if m == 0:
        return np.array([])
    if n_remaining == 0:
        return np.ones(m, dtype=complex)
    
    # Build Cauchy-like matrix C where C @ w = 0 (same as AAA)
    z_rem_col = z_remaining[:, np.newaxis]
    z_sup_row = z_support[np.newaxis, :]
    diffs = z_rem_col - z_sup_row
    
    f_rem_col = f_remaining[:, np.newaxis]
    f_sup_row = f_support[np.newaxis, :]
    f_diffs = f_rem_col - f_sup_row
    
    small_diff_mask = np.abs(diffs) < 1e-14
    C = np.where(small_diff_mask, 0.0, f_diffs / diffs)
    
    # Solve using SVD
    try:
        U, s, Vh = svd(C, full_matrices=False)
        if len(s) == 0:
            weights = np.ones(m, dtype=complex)
        else:
            weights = Vh[-1, :].conj()  # Take conjugate for proper complex handling
            norm_w = np.linalg.norm(weights)
            if norm_w > 1e-14:
                weights = weights / norm_w
            else:
                weights = np.ones(m, dtype=complex)
    except Exception as e:
        weights = np.ones(m, dtype=complex)
    
    return weights


def fit_weights_fmm_iterative(
    z_support,
    f_support,
    z_remaining,
    f_remaining,
    theta=0.6,
    eps2=1e-14,
    use_gpu=None,
    backend=None,
    max_iter=200,
    tol=1e-10,
):
    """
    Fit barycentric weights using an iterative smallest-singular-vector solve
    accelerated by the Barnes–Hut FMM.

    Parameters
    ----------
    z_support : array
        Support points (complex)
    f_support : array
        Function values at support points
    z_remaining : array
        Remaining points (for error minimization)
    f_remaining : array
        Function values at remaining points
    theta : float
        Barnes–Hut opening angle (smaller = more accurate, larger = faster)
    eps2 : float
        Softening for the 1/|.|^2 kernel
    use_gpu : bool or None
        Force GPU/CPU for the Barnes–Hut call. None = follow JAX default backend.
    backend : str
        Backward-compatible argument. Only 'rakau' backend is supported.
    max_iter : int
        Maximum Arnoldi iterations for eigensolve.
    tol : float
        Convergence tolerance for the eigensolver.

    Returns
    -------
    weights : array
        Barycentric weights (normalized)
    """
    m = len(z_support)
    n_remaining = len(z_remaining)
    if m == 0:
        return np.array([])
    if n_remaining == 0:
        return np.ones(m, dtype=complex)

    if not BARNES_HUT_AVAILABLE:
        return fit_weights_svd(z_support, f_support, z_remaining, f_remaining)

    # Generators for the Cauchy-like matrix
    c = jnp.asarray(z_remaining, dtype=jnp.complex128)
    q = jnp.asarray(z_support, dtype=jnp.complex128)
    g = jnp.stack([jnp.asarray(f_remaining, dtype=jnp.complex128), -jnp.ones(n_remaining, dtype=jnp.complex128)], axis=1)
    b = jnp.stack([jnp.ones(m, dtype=jnp.complex128), jnp.asarray(f_support, dtype=jnp.complex128)], axis=1)

    backend = backend or _bh_backend

    def normal_matvec_np(v):
        v_c = jnp.asarray(v, dtype=jnp.complex128)
        Cv = _apply_cauchy_like(q, c, b, g, v_c, theta=theta, eps2=eps2, use_gpu=use_gpu, backend=backend)
        CtCv = _apply_cauchy_like_adj(q, c, b, g, Cv, theta=theta, eps2=eps2, use_gpu=use_gpu, backend=backend)
        return np.asarray(CtCv)

    op = LinearOperator(
        dtype=np.complex128,
        shape=(m, m),
        matvec=normal_matvec_np,
        rmatvec=normal_matvec_np,  # Hermitian PSD
    )

    try:
        eigvals, eigvecs = eigsh(op, k=1, which="SM", maxiter=max_iter, tol=tol)
        weights = eigvecs[:, 0]
    except Exception:
        return fit_weights_svd(z_support, f_support, z_remaining, f_remaining)

    norm_w = np.linalg.norm(weights)
    if norm_w > 1e-14:
        weights = weights / norm_w
    else:
        weights = np.ones(m, dtype=complex)
    return weights


def fit_weights_fmm_iterative_jax(
    z_support,
    f_support,
    z_remaining,
    f_remaining,
    theta=0.6,
    eps2=1e-14,
    use_gpu=None,
    backend=None,
    max_iter=200,
    tol=1e-10,
    step_scale=0.9,
    rng_key=None,
):
    """
    JAX version of fit_weights_fmm_iterative that keeps the matvec on the JAX device.
    Uses projected gradient steps on the Rayleigh quotient of C^* C.
    """
    m = z_support.shape[0]
    n_remaining = z_remaining.shape[0]
    if m == 0:
        return jnp.array([])
    if n_remaining == 0:
        return jnp.ones(m, dtype=jnp.complex128)

    if not BARNES_HUT_AVAILABLE:
        return fit_weights_svd_jax(z_support, f_support, z_remaining, f_remaining)

    key = jax.random.PRNGKey(0) if rng_key is None else rng_key
    c = jnp.asarray(z_remaining, dtype=jnp.complex128)
    q = jnp.asarray(z_support, dtype=jnp.complex128)
    g = jnp.stack([jnp.asarray(f_remaining, dtype=jnp.complex128), -jnp.ones(n_remaining, dtype=jnp.complex128)], axis=1)
    b = jnp.stack([jnp.ones(m, dtype=jnp.complex128), jnp.asarray(f_support, dtype=jnp.complex128)], axis=1)

    def normal_matvec(v):
        Cv = _apply_cauchy_like(q, c, b, g, v, theta=theta, eps2=eps2, use_gpu=use_gpu, backend=backend)
        return _apply_cauchy_like_adj(q, c, b, g, Cv, theta=theta, eps2=eps2, use_gpu=use_gpu, backend=backend)

    # Rough estimate of largest eigenvalue for step size
    v = jax.random.normal(key, (m,), dtype=jnp.float64) + 1j * jax.random.normal(key, (m,), dtype=jnp.float64)
    v = v / jnp.linalg.norm(v)
    u = v
    for _ in range(6):
        u = normal_matvec(u)
        u_norm = jnp.linalg.norm(u)
        u = u / jnp.where(u_norm < 1e-30, 1.0, u_norm)
    lmax = jnp.real(jnp.vdot(u, normal_matvec(u)))
    step = step_scale / (lmax + 1e-12)

    prev = v
    for _ in range(max_iter):
        w = normal_matvec(prev)
        v_new = prev - step * w
        v_new = v_new / jnp.linalg.norm(v_new)
        if jnp.linalg.norm(v_new - prev) < tol:
            prev = v_new
            break
        prev = v_new

    weights = prev
    norm_w = jnp.linalg.norm(weights)
    weights = jnp.where(norm_w > 1e-14, weights / norm_w, jnp.ones_like(weights))
    return weights


def evaluate_barycentric(z, z_support, f_support, weights):
    """
    Evaluate barycentric form: r(z) = sum(w_j * f_j / (z - z_j)) / sum(w_j / (z - z_j))
    
    NumPy version.
    
    Parameters
    ----------
    z : array
        Points to evaluate at
    z_support : array
        Support points
    f_support : array
        Function values at support points
    weights : array
        Barycentric weights
    
    Returns
    -------
    result : array
        Evaluated function values
    """
    z = np.asarray(z)
    if z.ndim == 0:
        z = np.array([z])
        scalar_output = True
    else:
        scalar_output = False
    
    n_points = len(z)
    n_support = len(z_support)
    
    if n_support == 0:
        return np.full(n_points, np.nan, dtype=complex)
    
    # Vectorized evaluation
    z_col = z[:, np.newaxis]
    z_sup_row = z_support[np.newaxis, :]
    diffs = z_col - z_sup_row
    
    # Handle exact matches
    exact_mask = np.abs(diffs) < 1e-14
    exact_indices = np.any(exact_mask, axis=1)
    
    result = np.zeros(n_points, dtype=complex)
    if np.any(exact_indices):
        exact_rows = exact_mask[exact_indices]
        exact_support_idx = np.argmax(exact_rows, axis=1)
        result[exact_indices] = f_support[exact_support_idx]
    
    # Compute barycentric form for non-exact points
    non_exact = ~exact_indices
    if np.any(non_exact):
        diffs_non_exact = diffs[non_exact, :]
        small_diff_mask = np.abs(diffs_non_exact) < 1e-14
        phase = np.where(np.abs(diffs_non_exact) > 1e-20,
                        diffs_non_exact / np.abs(diffs_non_exact),
                        1.0 + 0.0j)
        diffs_safe = np.where(small_diff_mask, 1e-14 * phase, diffs_non_exact)
        
        weights_row = weights[np.newaxis, :]
        terms = weights_row / diffs_safe
        
        numerator = np.sum(terms * f_support[np.newaxis, :], axis=1)
        denominator = np.sum(terms, axis=1)
        
        valid = np.abs(denominator) > 1e-12
        result[non_exact] = np.where(valid, numerator / denominator, np.nan)
    
    return result[0] if scalar_output else result


def barycentric_poles(z_support, weights, weight_tol=1e-12):
    """
    Compute poles of a barycentric rational interpolant using the AAA pencil.

    Parameters
    ----------
    z_support : array
        Support points (complex)
    weights : array
        Barycentric weights
    weight_tol : float
        Threshold to discard near-zero/NaN weights

    Returns
    -------
    poles : ndarray
        Poles of the interpolant.
    """
    z_support = np.asarray(z_support)
    weights = np.asarray(weights)

    if len(z_support) == 0:
        return np.array([])

    finite_mask = np.isfinite(weights)
    weight_mask = np.abs(weights) > weight_tol
    keep = finite_mask & weight_mask
    z_support = z_support[keep]
    weights = weights[keep]

    m = len(z_support)
    if m <= 1:
        return np.array([])

    from scipy.linalg import eig

    a = np.zeros((m + 1, m + 1), dtype=complex)
    b = np.zeros((m + 1, m + 1), dtype=complex)
    a[:m, :m] = np.diag(z_support)
    a[:m, m] = weights
    a[m, :m] = 1.0
    b[:m, :m] = np.eye(m)

    eigvals = eig(a, b, left=False, right=False)
    poles = eigvals[np.isfinite(eigvals)]
    return poles


def fit_weights_svd_jax(z_support, f_support, z_remaining, f_remaining):
    """
    Fit barycentric weights using SVD (JAX version for GPU).
    
    Note: Not using JIT because dynamic shapes cause expensive recompilation.
    
    Parameters
    ----------
    z_support : array
        Support points (complex)
    f_support : array
        Function values at support points
    z_remaining : array
        Remaining points (for error minimization)
    f_remaining : array
        Function values at remaining points
    
    Returns
    -------
    weights : array
        Barycentric weights
    """
    m = z_support.shape[0]
    n_remaining = z_remaining.shape[0]
    
    if m == 0:
        return jnp.array([])
    if n_remaining == 0:
        return jnp.ones(m, dtype=complex)
    
    z_rem_col = z_remaining[:, jnp.newaxis]
    z_sup_row = z_support[jnp.newaxis, :]
    diffs = z_rem_col - z_sup_row
    
    f_rem_col = f_remaining[:, jnp.newaxis]
    f_sup_row = f_support[jnp.newaxis, :]
    f_diffs = f_rem_col - f_sup_row
    
    small_diff_mask = jnp.abs(diffs) < 1e-14
    C = jnp.where(small_diff_mask, 0.0, f_diffs / diffs)
    
    U, s, Vh = jnp.linalg.svd(C, full_matrices=False)
    # Right singular vector corresponding to smallest singular value
    weights = jnp.conj(Vh[-1, :])
    
    return weights
