"""
Compute Frobenius norm error ||A - X||_F efficiently using matvecs.

Key formula: ||A - CUR||_F^2 = ||A||_F^2 + ||CUR||_F^2 - 2*Re(trace(A^H @ CUR))

Complexity:
- General: O(k * lmatvec_cost + k^2) for CUR error
- SVD: O(k * lmatvec_cost) for SVD error

Memory: O(k^2 + n + m) - only k×k matrices plus O(n) or O(m) temporaries
"""

import jax
import jax.numpy as jnp
import jax.scipy.signal as jss
import numpy as np
import time
from typing import Optional
from functools import partial


@jax.jit
def _compute_error_sq_from_grams(
    G_col: jnp.ndarray,
    G_row: jnp.ndarray, 
    B: jnp.ndarray,
    U: jnp.ndarray,
    A_norm_sq: float
) -> jnp.ndarray:
    """
    JIT-compiled final error computation from Gram matrices.
    
    Computes: ||A||_F^2 + ||CUR||_F^2 - 2*Re(trace(A^H @ CUR))
    
    where:
    - U is the core matrix (W^{-1} for CURIncremental, T for PivotedQRCUR)
    - ||CUR||_F^2 = trace(U^H @ G_col @ U @ G_row)
    - trace(A^H @ CUR) = trace(B @ U)  where B = R @ A^H @ C
    
    Derivation:
    CUR = C @ U @ R
    ||CUR||_F^2 = trace(R^H @ U^H @ C^H @ C @ U @ R)
               = trace(U^H @ G_col @ U @ G_row)
    
    trace(A^H @ CUR) = trace(A^H @ C @ U @ R)
                     = trace(R @ A^H @ C @ U)  (cycle)
                     = trace(B @ U)
    """
    # trace(B @ U) = sum of element-wise product of B and U.T
    cross_trace = jnp.real(jnp.sum(B * U.T))
    
    # Q = U^H @ G_col @ U = conj(U).T @ G_col @ U
    Q = jnp.conj(U).T @ G_col @ U
    
    # ||CUR||_F^2 = trace(Q @ G_row)
    cur_norm_sq = jnp.real(jnp.trace(Q @ G_row))
    
    # Final error squared
    error_sq = A_norm_sq + cur_norm_sq - 2 * cross_trace
    return jnp.maximum(error_sq, 0.0)



def _get_U_matrix(cur_obj, k: int) -> jnp.ndarray:
    """
    Extract the U matrix (core inverse) from a CUR object.
    
    For CURIncremental: U_core (in 'inverse' mode) or reconstructed from SVD/QR
    For PivotedQRCUR: T_core or reconstructed from SVD factors
    
    The matrix U satisfies: CUR = C @ U @ R where C = A[:, J] and R = A[I, :].
    U_core_lmatvec(v) computes v @ U (left multiply).
    
    Parameters
    ----------
    cur_obj : CURIncremental or PivotedQRCUR
        CUR decomposition object
    k : int
        Rank of decomposition
        
    Returns
    -------
    U : array of shape (k, k)
        The core matrix U = W^{-1}
    """
    # Check for PivotedQRCUR (has T_core attribute)
    if hasattr(cur_obj, 'T_core'):
        if hasattr(cur_obj, 'T_use_svd') and cur_obj.T_use_svd:
            # Reconstruct from SVD factors: T = T_u @ diag(T_s) @ T_vt
            # This matrix multiplication should be fast for k×k, but ensure it's executed
            uT = cur_obj.T_u[:k, :k]
            sT = cur_obj.T_s[:k]
            vT = cur_obj.T_vt[:k, :k]
            # Compute T = uT @ (sT[:, None] * vT) efficiently
            # First compute diag(sT) @ vT = sT[:, None] * vT (broadcast multiply)
            T = uT @ (sT[:, None] * vT)
            # Block until ready to ensure computation completes
            T.block_until_ready()
            return T
        else:
            T = cur_obj.T_core[:k, :k]
            T.block_until_ready()  # Ensure it's ready
            return T
    
    # CURIncremental
    if hasattr(cur_obj, 'update_U_option'):
        if cur_obj.update_U_option == 'inverse':
            # U_core is directly the inverse matrix
            return cur_obj.U_core[:k, :k]
        elif cur_obj.update_U_option == 'svd':
            # U_core = (U, inv_s, Vh), reconstruct: W^{-1} = Vh^H @ diag(inv_s) @ U^H
            U_svd, inv_s, Vh = cur_obj.U_core
            return Vh[:k, :k].conj().T @ (inv_s[:k, None] * U_svd[:k, :k].conj().T)
        elif cur_obj.update_U_option == 'qr':
            # Need to reconstruct from QR - use lmatvec to build row by row
            # U_core_lmatvec(e_j) = e_j @ U = U[j, :] (j-th row)
            dtype = cur_obj.A.dtype if hasattr(cur_obj, 'A') else jnp.float64
            eye_k = jnp.eye(k, dtype=dtype)
            U_rows = [cur_obj.U_core_lmatvec(eye_k[j, :]) for j in range(k)]
            return jnp.vstack(U_rows)
    
    # Fallback: build from U_core_lmatvec (row by row)
    # U_core_lmatvec(e_j) = e_j @ U = U[j, :] (j-th row)
    dtype = cur_obj.A.dtype if hasattr(cur_obj, 'A') else jnp.float64
    eye_k = jnp.eye(k, dtype=dtype)
    U_rows = [cur_obj.U_core_lmatvec(eye_k[j, :]) for j in range(k)]
    return jnp.vstack(U_rows)


def compute_frobenius_error_cur(
    Aop,
    cur_obj,
    row_norms_squared: Optional[jnp.ndarray] = None,
    verbose: bool = False
) -> float:
    """
    Compute ||A - CUR||_F exactly using O(k) lmatvecs.
    
    Uses: ||A - CUR||_F^2 = ||A||_F^2 + ||CUR||_F^2 - 2*Re(trace(A^H @ CUR))
    
    Complexity:
        Time: O(k * lmatvec_cost + k^2) - k lmatvecs, k get_col, k get_row
        Memory: O(k^2 + n + m) - k×k Gram matrices plus O(n) or O(m) temporaries
    
    Parameters
    ----------
    Aop : AOperator
        Matrix operator with matvec, lmatvec, get_col, get_row methods
    cur_obj : CURIncremental or PivotedQRCUR
        CUR decomposition object with U matrix precomputed
    row_norms_squared : array, optional
        Precomputed sum of |A[i,:]|^2 for each row. If None, computed from Aop.
    verbose : bool
        Print progress
        
    Returns
    -------
    float
        ||A - CUR||_F
    """
    n, m = Aop.shape
    k = int(cur_obj.k)
    
    if k == 0:
        # Error is just ||A||_F
        if row_norms_squared is None:
            if hasattr(Aop, 'compute_all_row_norms_squared'):
                row_norms_squared = Aop.compute_all_row_norms_squared()
            else:
                row_norms_squared = jnp.array([jnp.sum(jnp.abs(Aop.get_row(i))**2) for i in range(n)])
        return float(jnp.sqrt(jnp.sum(row_norms_squared)))
    
    I = cur_obj.I_array[:k] if hasattr(cur_obj, 'I_array') else cur_obj.I[:k]
    J = cur_obj.J_array[:k] if hasattr(cur_obj, 'J_array') else cur_obj.J[:k]
    I = jnp.asarray(I)
    J = jnp.asarray(J)
    
    # 1. ||A||_F^2 from precomputed row norms
    if row_norms_squared is None:
        if hasattr(Aop, 'compute_all_row_norms_squared'):
            row_norms_squared = Aop.compute_all_row_norms_squared()
        else:
            row_norms_squared = jnp.array([jnp.sum(jnp.abs(Aop.get_row(i))**2) for i in range(n)])
    
    A_norm_sq = float(jnp.sum(row_norms_squared))
    if verbose:
        print(f"  ||A||_F^2 = {A_norm_sq:.6e}")
    
    # 2. Get U matrix (precomputed in cur_obj)
    if verbose:
        print(f"  Getting U matrix from cur_obj...")
    U = _get_U_matrix(cur_obj, k)
    if verbose:
        print(f"    U matrix shape: {U.shape}, blocking until ready...", flush=True)
    U.block_until_ready()  # Ensure U matrix computation completes
    
    # 3. Compute G_col and B in a single loop (both need col_j)
    # G_col = C^H @ C, B = R @ A^H @ C
    if verbose:
        print(f"  Computing G_col and B ({k} get_col + {k} lmatvec)...")
    
    G_col = jnp.zeros((k, k), dtype=Aop.dtype)
    B = jnp.zeros((k, k), dtype=Aop.dtype)
    
    # Track timing for diagnostics
    get_col_times = []
    lmatvec_C_times = []
    lmatvec_times = []
    matvec_R_times = []
    
    for j in range(k):
        if verbose and (j % 20 == 0 or j == k - 1):
            print(f"    Processing column {j+1}/{k}...", flush=True)
        
        # get_col
        t0 = time.time()
        col_j = Aop.get_col(int(J[j]))
        col_j.block_until_ready()
        get_col_times.append(time.time() - t0)
        
        # lmatvec_with_C
        t0 = time.time()
        G_col_j = Aop.lmatvec_with_C(col_j, J)
        G_col_j.block_until_ready()
        lmatvec_C_times.append(time.time() - t0)
        G_col = G_col.at[:, j].set(G_col_j)
        
        # lmatvec
        t0 = time.time()
        M_col_j = Aop.lmatvec(col_j)
        M_col_j.block_until_ready()
        lmatvec_times.append(time.time() - t0)
        
        # matvec_with_R
        t0 = time.time()
        B_j = Aop.matvec_with_R(M_col_j, I)
        B_j.block_until_ready()
        matvec_R_times.append(time.time() - t0)
        B = B.at[:, j].set(B_j)
    
    if verbose:
        print(f"    Timing breakdown (average per column):")
        print(f"      get_col: {np.mean(get_col_times):.3f}s")
        print(f"      lmatvec_with_C: {np.mean(lmatvec_C_times):.3f}s")
        print(f"      lmatvec: {np.mean(lmatvec_times):.3f}s")
        print(f"      matvec_with_R: {np.mean(matvec_R_times):.3f}s")
        print(f"    Total for G_col and B: {np.sum(get_col_times) + np.sum(lmatvec_C_times) + np.sum(lmatvec_times) + np.sum(matvec_R_times):.3f}s")
    
    # 4. Compute G_row = R @ R^H
    if verbose:
        print(f"  Computing G_row ({k} get_row)...")
    
    G_row = jnp.zeros((k, k), dtype=Aop.dtype)
    get_row_times = []
    matvec_R_row_times = []
    
    for j in range(k):
        if verbose and (j % 20 == 0 or j == k - 1):
            print(f"    Processing row {j+1}/{k}...", flush=True)
        
        # get_row
        t0 = time.time()
        row_j = Aop.get_row(int(I[j]))
        row_j.block_until_ready()
        get_row_times.append(time.time() - t0)
        
        # matvec_with_R
        t0 = time.time()
        G_row_j = Aop.matvec_with_R(jnp.conj(row_j), I)
        G_row_j.block_until_ready()
        matvec_R_row_times.append(time.time() - t0)
        G_row = G_row.at[:, j].set(G_row_j)
    
    if verbose:
        print(f"    Timing breakdown (average per row):")
        print(f"      get_row: {np.mean(get_row_times):.3f}s")
        print(f"      matvec_with_R: {np.mean(matvec_R_row_times):.3f}s")
        print(f"    Total for G_row: {np.sum(get_row_times) + np.sum(matvec_R_row_times):.3f}s")
    
    # 5. Compute final error using JIT'd helper
    if verbose:
        print(f"  Computing final error...")
    
    # Ensure all Gram matrices are ready before final computation
    G_col.block_until_ready()
    G_row.block_until_ready()
    B.block_until_ready()
    
    error_sq = _compute_error_sq_from_grams(G_col, G_row, B, U, A_norm_sq)
    error_sq.block_until_ready()  # Ensure computation completes
    error_sq = float(error_sq)
    
    if verbose:
        cross_trace = float(jnp.real(jnp.sum(B * U.T)))
        Q = jnp.conj(U).T @ G_col @ U
        cur_norm_sq = float(jnp.real(jnp.trace(Q @ G_row)))
        print(f"  trace(A^H @ CUR) = {cross_trace:.6e}")
        print(f"  ||CUR||_F^2 = {cur_norm_sq:.6e}")
        print(f"  ||A - CUR||_F = {jnp.sqrt(error_sq):.6e}")
    
    return float(jnp.sqrt(error_sq))


@jax.jit 
def _compute_svd_error_sq(
    M: jnp.ndarray,
    s: jnp.ndarray,
    Vh: jnp.ndarray,
    A_norm_sq: float
) -> jnp.ndarray:
    """
    JIT-compiled final SVD error computation.
    
    M[:, i] = A^H @ U[:, i] (precomputed via lmatvecs)
    
    Computes: ||A||_F^2 + ||USVh||_F^2 - 2*Re(trace(A^H @ USVh))
    """
    # ||USVh||_F^2 = sum(s^2)
    svd_norm_sq = jnp.sum(s**2)
    
    # trace(A^H @ USVh) = sum_i s_i * M[:, i]^H @ Vh[i, :]^H
    # = sum_i s_i * conj(<M[:, i], Vh[i, :]>)
    # Vectorized: sum(s * conj(diag(M^H @ Vh^H))) = sum(s * conj(sum(conj(M) * Vh.T, axis=0)))
    # Using vdot for each: cross_terms[i] = conj(vdot(M[:, i], Vh[i, :]))
    cross_terms = jnp.conj(jnp.sum(jnp.conj(M) * Vh.T, axis=0))
    cross_trace = jnp.real(jnp.sum(s * cross_terms))
    
    error_sq = A_norm_sq + svd_norm_sq - 2 * cross_trace
    return jnp.maximum(error_sq, 0.0)


def _build_svd_M_loop(lmatvec_fn, U, k, m, dtype):
    """
    Build M matrix where M[:, i] = A^H @ U[:, i] using fori_loop.
    """
    def body_fn(i, M):
        M = M.at[:, i].set(lmatvec_fn(U[:, i]))
        return M
    
    M_init = jnp.zeros((m, k), dtype=dtype)
    M = jax.lax.fori_loop(0, k, body_fn, M_init)
    return M


def compute_frobenius_error_svd(
    Aop,
    U: jnp.ndarray,
    s: jnp.ndarray,
    Vh: jnp.ndarray,
    row_norms_squared: Optional[jnp.ndarray] = None,
    verbose: bool = False,
    use_jit_loops: bool = True
) -> float:
    """
    Compute ||A - U @ diag(s) @ Vh||_F exactly.
    
    Uses: ||A - USVh||_F^2 = ||A||_F^2 + ||USVh||_F^2 - 2*Re(trace(A^H @ USVh))
    
    Complexity:
        Time: O(k * lmatvec_cost)
        Memory: O(nk + km) for U, s, Vh (already allocated by caller) + O(mk) for M
        
    Parameters
    ----------
    use_jit_loops : bool
        If True, use jax.lax.fori_loop for JIT-compiled loops (faster).
    """
    n, m = Aop.shape
    k = len(s)
    
    # 1. ||A||_F^2
    if row_norms_squared is None:
        if hasattr(Aop, 'compute_all_row_norms_squared'):
            row_norms_squared = Aop.compute_all_row_norms_squared()
        else:
            row_norms_squared = jnp.array([jnp.sum(jnp.abs(Aop.get_row(i))**2) for i in range(n)])
    
    A_norm_sq = float(jnp.sum(row_norms_squared))
    if verbose:
        print(f"  ||A||_F^2 = {A_norm_sq:.6e}")
        print(f"  ||USVh||_F^2 = {float(jnp.sum(s**2)):.6e}")
        print(f"  Computing trace(A^H @ USVh) via {k} lmatvecs...")
    
    # 2. Compute M[:, i] = A^H @ U[:, i] for all i (k lmatvecs)
    # For large sparse matrices, use Python loops to avoid capturing matrix as constant
    # in JIT-compiled fori_loop. The operator's matvec methods are already JIT-compiled.
    if use_jit_loops and not hasattr(Aop, 'bcoo_matrix'):
        # Only use JIT loops for dense/small operators to avoid constant capture
        M = _build_svd_M_loop(Aop.lmatvec, U, k, m, Aop.dtype)
    else:
        # Python loop - each matvec call is separate, no constant capture
        M_cols = [Aop.lmatvec(U[:, i]) for i in range(k)]
        M = jnp.column_stack(M_cols)  # (m, k)
    
    # 3. Compute final error using JIT'd helper
    error_sq = _compute_svd_error_sq(M, s, Vh, A_norm_sq)
    error_sq = float(error_sq)
    
    if verbose:
        cross_terms = jnp.conj(jnp.sum(jnp.conj(M) * Vh.T, axis=0))
        cross_trace = float(jnp.real(jnp.sum(s * cross_terms)))
        print(f"  trace(A^H @ USVh) = {cross_trace:.6e}")
        print(f"  ||A - USVh||_F = {jnp.sqrt(error_sq):.6e}")
    
    return float(jnp.sqrt(error_sq))


if __name__ == '__main__':
    import jax.numpy as jnp
    from cur import CURIncremental
    from matrix_classes import AOperator
    
    class DenseMatrixOperator(AOperator):
        def __init__(self, A):
            self.A = jnp.asarray(A)
            self._shape = self.A.shape
            self._dtype = self.A.dtype
            
        @property
        def shape(self):
            return self._shape
            
        @property
        def dtype(self):
            return self._dtype
            
        def get_row(self, i):
            return self.A[i, :]
            
        def get_col(self, j):
            return self.A[:, j]
            
        def matvec(self, v):
            return self.A @ v
            
        def lmatvec(self, v):
            return self.A.conj().T @ v
            
        def matvec_with_C(self, v, J):
            return self.A[:, J] @ v
            
        def lmatvec_with_C(self, v, J):
            return self.A[:, J].conj().T @ v
            
        def lmatvec_with_R(self, v, I):
            return self.A[I, :].conj().T @ v
            
        def matvec_with_R(self, v, I):
            return self.A[I, :] @ v
    
    print("=" * 60)
    print("Testing Frobenius error (exact, batched)")
    print("=" * 60)
    
    key = jax.random.PRNGKey(42)
    n, m, rank = 50, 40, 8
    
    key, k1, k2, k3 = jax.random.split(key, 4)
    A_lowrank = jax.random.normal(k1, (n, 15)) @ jax.random.normal(k2, (15, m))
    A_noise = 0.1 * jax.random.normal(k3, (n, m))
    A_dense = (A_lowrank + A_noise).astype(jnp.float32)
    
    print(f"\nMatrix: ({n}, {m}), CUR rank: {rank}")
    
    Aop = DenseMatrixOperator(A_dense)
    row_norms = jnp.sum(jnp.abs(A_dense)**2, axis=1)
    
    # Test CUR
    print("\n--- CUR (batched) ---")
    cur = CURIncremental(Aop, row_norms, pivot_sampling='random', random_seed=42)
    cur.build(rank)
    
    # Ground truth
    k = int(cur.k)
    I, J = cur.I_array[:k], cur.J_array[:k]
    R = A_dense[I, :]
    CUR_dense = jnp.zeros_like(A_dense)
    for i in range(n):
        z = A_dense[i, J]
        v = cur.U_core_lmatvec(z)
        CUR_dense = CUR_dense.at[i, :].set(R.conj().T @ v)
    true_cur = float(jnp.linalg.norm(A_dense - CUR_dense, ord='fro'))
    
    computed_cur = compute_frobenius_error_cur(Aop, cur, row_norms, verbose=True)
    
    print(f"\n  Ground truth: {true_cur:.10e}")
    print(f"  Computed:     {computed_cur:.10e}")
    rel_err = abs(true_cur - computed_cur) / true_cur
    print(f"  Rel error:    {rel_err:.2e}")
    assert rel_err < 1e-3, f"CUR error too large: {rel_err}"
    print("  ✓ Passed")
    
    # Test SVD
    print("\n--- SVD ---")
    from randomized_svd import randomized_svd
    U, s, Vh = randomized_svd(Aop, rank, random_state=42)
    
    SVD_dense = U @ jnp.diag(s) @ Vh
    true_svd = float(jnp.linalg.norm(A_dense - SVD_dense, ord='fro'))
    
    computed_svd = compute_frobenius_error_svd(Aop, U, s, Vh, row_norms, verbose=True)
    
    print(f"\n  Ground truth: {true_svd:.10e}")
    print(f"  Computed:     {computed_svd:.10e}")
    rel_err = abs(true_svd - computed_svd) / true_svd
    print(f"  Rel error:    {rel_err:.2e}")
    assert rel_err < 1e-3, f"SVD error too large: {rel_err}"
    print("  ✓ Passed")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
