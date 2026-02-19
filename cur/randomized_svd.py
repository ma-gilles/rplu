"""
Randomized SVD implementation using only matrix-vector products.

This module provides a GPU-accelerated randomized SVD implementation that works
with linear operators and only requires matvec/lmatvec operations, making it
memory-efficient for large matrices.
"""

import time
import numpy as np
import jax
import jax.numpy as jnp
from typing import Tuple, Optional, Dict


def randomized_svd(
    A,
    n_components: int,
    n_oversamples: int = 10,
    n_iter: int = 0,
    random_state: Optional[int] = None,
    return_timings: bool = False,
    batch_matvec: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Compute a truncated randomized SVD using only matrix-vector products.
    
    This implementation uses the randomized SVD algorithm which only requires
    matvec and lmatvec operations, making it suitable for large matrices that
    cannot be stored densely. The computation is GPU-accelerated using JAX.
    
    Algorithm:
        1. Generate random Gaussian matrix Omega of size (m, l)
        2. Compute Y = A @ Omega via matvec operations
        3. Compute Q, R = qr(Y) to get orthonormal basis
        4. Compute B = Q.T @ A via lmatvec operations
        5. Compute SVD of small matrix B
        6. Reconstruct U = Q @ U_small
    
    Parameters
    ----------
    A : LinearOperator or ConvNDOperator
        Linear operator with .shape, .dtype, .matvec(x), and .lmatvec(x) methods.
        For batch operations, matvec should accept (N, B) arrays and return (N, B).
    n_components : int
        Number of singular values and vectors to compute (k).
    n_oversamples : int, optional (default=10)
        Number of additional random vectors to sample for better accuracy.
        The algorithm will use l = n_components + n_oversamples random vectors.
    n_iter : int, optional (default=0)
        Number of power iterations to improve the quality of the approximation.
        Power iterations help when the singular values decay slowly.
        Typically 2-5 iterations are sufficient.
    random_state : int, optional (default=None)
        Random seed for reproducibility.
    return_timings : bool, optional (default=False)
        If True, return detailed timing breakdown as a fourth return value.
        Useful for performance analysis but disabled by default for efficiency.
    
    Returns
    -------
    U : jax.Array, shape (n, n_components)
        Left singular vectors (columns of U). GPU array (JAX).
    s : jax.Array, shape (n_components,)
        Singular values in descending order. GPU array (JAX).
    Vt : jax.Array, shape (n_components, m)
        Right singular vectors (rows of Vt). GPU array (JAX).
    timings : dict, optional
        Only returned if return_timings=True. Dictionary with timing breakdown:
        - 'random_matrix': Time to generate random matrix
        - 'A_times_Omega': Time for A @ Omega
        - 'QR': Time for QR decomposition
        - 'Q_transpose_A': Time for Q.T @ A
        - 'small_SVD': Time for SVD of small matrix
        - 'reconstruct_U': Time to reconstruct U
        - 'total': Total time
    
    Notes
    -----
    All outputs are JAX arrays on GPU. To convert to numpy, use np.asarray(output).
    
    Examples
    --------
    >>> from conv_nd_operator import ConvNDOperator
    >>> import numpy as np
    >>> import jax.numpy as jnp
    >>> 
    >>> # Create a 3D convolution operator
    >>> kernel = np.random.randn(3, 3, 3).astype(np.float32)
    >>> shape = (32, 32, 32)
    >>> op = ConvNDOperator(kernel, shape, dtype=jnp.float32)
    >>> 
    >>> # Compute rank-50 approximation (returns JAX arrays on GPU)
    >>> U, s, Vt = randomized_svd(op, n_components=50)
    >>> 
    >>> # Convert to numpy if needed
    >>> U_np = np.asarray(U)
    >>> 
    >>> # With timing information
    >>> U, s, Vt, timings = randomized_svd(op, n_components=50, return_timings=True)
    >>> print(f"Total time: {timings['total']:.3f}s")
    >>> print(f"A @ Omega: {timings['A_times_Omega']:.3f}s")
    
    References
    ----------
    .. [1] Halko, N., Martinsson, P. G., & Tropp, J. A. (2011).
           Finding structure with randomness: Probabilistic algorithms for
           constructing approximate matrix decompositions.
           SIAM review, 53(2), 217-288.
    """
    total_start = time.time() if return_timings else None
    timings = {} if return_timings else None
    
    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)
    
    n, m = A.shape
    l = min(n_components + n_oversamples, min(m, n))
    
    # Step 1: Generate random matrix on GPU
    if return_timings:
        start = time.time()
    key = jax.random.PRNGKey(random_state if random_state is not None else 42)
    Omega = jax.random.normal(key, (m, l), dtype=A.dtype)
    if return_timings:
        Omega = Omega.block_until_ready()
        timings['random_matrix'] = time.time() - start
    
    # Step 2: Y = A @ Omega using batch matvec (keep on GPU)
    if return_timings:
        start = time.time()
    
    if batch_matvec:
        Y = A.matvec(Omega)  # Shape: (n, l) - stays on GPU
    else:
        Y = jnp.zeros((n, l), dtype=A.dtype)
        for i in range(l):
            Y = Y.at[:, i].set(A.matvec(Omega[:, i]))

    if return_timings:
        Y = Y.block_until_ready()
        timings['A_times_Omega'] = time.time() - start
    
    # Free memory: Omega no longer needed
    del Omega, key
    
    # Step 3: Power iterations to improve subspace quality
    if n_iter > 0:
        if return_timings:
            start = time.time()
        
        # Power iteration: Y = (A @ A^H)^q @ Y
        # For q iterations, we compute: Y -> A @ (A^H @ Y) -> ... (q times)
        # After each iteration, we orthonormalize to prevent numerical issues
        for i in range(n_iter):
            # Y = A @ (A^H @ Y)
            Y, _ = jnp.linalg.qr(Y)
            if batch_matvec:
                Y = A.matvec(A.lmatvec(Y))
            else:
                # Sequential version
                temp = jnp.zeros((m, l), dtype=A.dtype)
                for j in range(l):
                    temp = temp.at[:, j].set(A.lmatvec(Y[:, j]))
                Y_new = jnp.zeros((n, l), dtype=A.dtype)
                for j in range(l):
                    Y_new = Y_new.at[:, j].set(A.matvec(temp[:, j]))
                Y = Y_new
            
        
        if return_timings:
            Y = Y.block_until_ready()
            timings['power_iterations'] = time.time() - start
    
    # Step 4: QR decomposition on GPU
    if return_timings:
        start = time.time()
    Q, _ = jnp.linalg.qr(Y)
    if return_timings:
        Q = Q.block_until_ready()
        timings['QR'] = time.time() - start
    
    # Free memory: Y no longer needed
    del Y
    
    # Step 5: B = Q.T @ A using batch lmatvec (keep on GPU)
    if return_timings:
        start = time.time()
    
    if batch_matvec:
        B = A.lmatvec(Q)  # Shape: (m, l) - stays on GPU
    else:
        B = jnp.zeros((m, l), dtype=A.dtype)
        for i in range(l):
            B = B.at[:, i].set(A.lmatvec(Q[:, i]))
    B = B.T  # Transpose to get (l, m)
    if return_timings:
        B = B.block_until_ready()
        timings['Q_transpose_A'] = time.time() - start
    
    # Step 6: SVD of small matrix B on GPU
    if return_timings:
        start = time.time()
        B = B.block_until_ready()
    U_small, s, Vt = jnp.linalg.svd(B, full_matrices=False)
    if return_timings:
        U_small = U_small.block_until_ready()
        s = s.block_until_ready()
        Vt = Vt.block_until_ready()
        timings['small_SVD'] = time.time() - start
    
    # Free memory: B no longer needed
    del B
    
    # Step 7: Reconstruct U on GPU
    if return_timings:
        start = time.time()
    U = Q @ U_small
    if return_timings:
        U = U.block_until_ready()
        timings['reconstruct_U'] = time.time() - start
    
    # Free memory: Q and U_small no longer needed
    del Q, U_small
    
    # Truncate to requested rank (still on GPU)
    U_out = U[:, :n_components]
    s_out = s[:n_components]
    Vt_out = Vt[:n_components, :]
    
    # Free memory: full U, s, Vt no longer needed
    del U, s, Vt
    
    if return_timings:
        U_out = U_out.block_until_ready()
        timings['total'] = time.time() - total_start
        return U_out, s_out, Vt_out, timings
    else:
        return U_out, s_out, Vt_out

class WoodburyPreconditioner:
    """(D + U V^T)^{-1} using Woodbury. U,V are JAX arrays (N x r) on GPU."""
    def __init__(self, D_inv_matvec, U, V):
        # Assume U and V are already JAX arrays on GPU - don't copy
        self.U_jax = U
        self.V_jax = V
        self.D_inv_matvec = D_inv_matvec
        r = self.U_jax.shape[1]
        # Preallocate K matrix (k x k) - no O(kn) memory needed
        # Compute K = I + V^T D^{-1} U column by column to avoid storing DU (N x k)
        K = jnp.eye(r, dtype=self.U_jax.dtype)
        for i in range(r):
            # Compute (D^{-1} U)[:, i] = D^{-1} @ U[:, i]
            ui = self.U_jax[:, i]
            dui = self.D_inv_matvec(ui)
            # K[:, i] += V^T @ (D^{-1} U)[:, i] = V^T @ dui
            K = K.at[:, i].add(self.V_jax.T @ dui)
        self.K_jax = K

    def __call__(self, z):
        invD_z = self.D_inv_matvec(z)
        rhs = self.V_jax.T @ invD_z
        t = jnp.linalg.solve(self.K_jax, rhs)
        return invD_z - self.D_inv_matvec(self.U_jax @ t)
