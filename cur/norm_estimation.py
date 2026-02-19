"""
Norm estimation functions using JAX.

This module provides memory-efficient methods for computing matrix norms
using iterative algorithms like Lanczos.
"""

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def lanczos_2norm(A_op, max_iter=50, tol=1e-6, random_state=42, verbose=False):
    """
    Compute the 2-norm (largest singular value) of an operator using Lanczos algorithm.
    
    Memory efficient: only requires O(k) memory for k iterations, not O(n*k) like randomized SVD.
    Converges faster than power iteration for well-separated eigenvalues.
    
    The algorithm applies Lanczos to A^T A (or A A^T) to find the largest eigenvalue,
    then takes the square root to get the largest singular value.
    
    Parameters
    ----------
    A_op : object with matvec and lmatvec methods
        Matrix operator. Must have:
        - shape: tuple (n, m)
        - dtype: data type
        - matvec(x): computes A @ x
        - lmatvec(y): computes A^T @ y (or A^H @ y for complex)
    max_iter : int
        Maximum number of Lanczos iterations. Default: 50
    tol : float
        Tolerance for convergence (checks relative change in estimated eigenvalue).
        Default: 1e-6
    random_state : int
        Random seed for initial vector. Default: 42
    verbose : bool
        If True, print progress updates. Default: False
        
    Returns
    -------
    float
        Largest singular value (2-norm) of the operator
        
    Notes
    -----
    The Lanczos algorithm builds a tridiagonal matrix T such that the eigenvalues
    of T approximate the eigenvalues of A^T A (or A A^T). The largest eigenvalue
    of A^T A is σ_max^2, where σ_max is the largest singular value of A.
    
    For numerical stability, we use the smaller of A^T A or A A^T depending on
    the matrix dimensions.
    
    Examples
    --------
    >>> from norm_estimation import lanczos_2norm
    >>> # Assuming A_op is a matrix operator with matvec/lmatvec methods
    >>> sigma_max = lanczos_2norm(A_op, max_iter=50, verbose=True)
    """
    n, m = A_op.shape
    key = jax.random.PRNGKey(random_state)
    
    # Choose smaller dimension for efficiency
    if m <= n:
        # Apply Lanczos to A^T A (m x m)
        def matvec_ATA(v):
            Av = A_op.matvec(v)
            return A_op.lmatvec(Av)
        dim = m
    else:
        # Apply Lanczos to A A^T (n x n)
        def matvec_ATA(v):
            ATv = A_op.lmatvec(v)
            return A_op.matvec(ATv)
        dim = n
    
    # Initialize Lanczos vectors
    v = jax.random.normal(key, (dim,), dtype=A_op.dtype)
    v = v / jnp.linalg.norm(v)
    
    # Lanczos tridiagonal matrix elements
    alpha = []  # Diagonal elements
    beta = []   # Off-diagonal elements
    
    v_prev = jnp.zeros_like(v)
    beta_prev = 0.0
    prev_max_eigval = 0.0
    
    for i in range(max_iter):
        # w = A^T A v - beta_{i-1} v_{i-1}
        w = matvec_ATA(v) - beta_prev * v_prev
        
        # alpha_i = v^T w
        alpha_i_arr = jnp.dot(v, w)
        alpha_i_arr.block_until_ready()
        alpha_i = float(alpha_i_arr)
        alpha.append(alpha_i)
        
        # w = w - alpha_i v
        w = w - alpha_i * v
        
        # Note: No reorthogonalization to keep memory O(1) instead of O(nk)
        # This may cause some loss of orthogonality for very large iterations,
        # but is acceptable for computing the largest eigenvalue.
        
        # beta_i = ||w||
        beta_i_arr = jnp.linalg.norm(w)
        beta_i_arr.block_until_ready()
        beta_i = float(beta_i_arr)
        
        # Check for convergence (small beta means we've found an invariant subspace)
        if beta_i < 1e-14:
            if verbose:
                print(f"      Lanczos: invariant subspace found at iteration {i+1} (beta={beta_i:.2e})", flush=True)
            break
        
        beta.append(beta_i)
        
        # Compute largest eigenvalue of tridiagonal matrix to check convergence
        # Note: For a k x k tridiagonal matrix, we need k diagonal elements (alpha)
        # and k-1 off-diagonal elements (beta). Slice beta to ensure correct size.
        k = len(alpha)
        T = np.diag(alpha)
        if beta:
            # beta should have exactly k-1 elements for a k x k tridiagonal matrix
            beta_for_T = beta[:k-1] if len(beta) >= k else beta
            if beta_for_T:
                T = T + np.diag(beta_for_T, 1) + np.diag(beta_for_T, -1)
        eigvals = np.linalg.eigvalsh(T)
        max_eigval = np.max(eigvals)
        
        # Check eigenvalue convergence
        if i > 0:
            rel_change = abs(max_eigval - prev_max_eigval) / max(abs(max_eigval), 1e-14)
            if verbose and (i % 10 == 0 or i == max_iter - 1):
                print(f"      Lanczos iteration {i+1}/{max_iter}, max eigenvalue: {max_eigval:.6e}, rel_change: {rel_change:.2e}", flush=True)
            if rel_change < tol:
                if verbose:
                    print(f"      Lanczos converged at iteration {i+1} (rel_change={rel_change:.2e})", flush=True)
                break
        elif verbose:
            print(f"      Lanczos iteration {i+1}/{max_iter}, max eigenvalue: {max_eigval:.6e}", flush=True)
        
        prev_max_eigval = max_eigval
        
        # Update for next iteration
        v_prev = v
        v = w / beta_i
        beta_prev = beta_i
    
    # Build final tridiagonal matrix and compute largest eigenvalue
    # For a k x k tridiagonal matrix, we need k diagonal elements (alpha)
    # and k-1 off-diagonal elements (beta)
    k = len(alpha)
    if k == 0:
        return 0.0
    
    T = np.diag(alpha)
    if beta:
        # beta should have exactly k-1 elements for a k x k tridiagonal matrix
        beta_for_T = beta[:k-1] if len(beta) >= k else beta
        if beta_for_T:
            T = T + np.diag(beta_for_T, 1) + np.diag(beta_for_T, -1)
    
    # Compute eigenvalues of tridiagonal matrix
    eigvals = np.linalg.eigvalsh(T)
    max_eigval = np.max(eigvals)
    
    # The 2-norm is sqrt(largest eigenvalue of A^T A)
    # Handle potential numerical issues with negative eigenvalues
    sigma = np.sqrt(max(0.0, max_eigval))
    
    return float(sigma)
