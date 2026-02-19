# conv_nd_operator.py
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.scipy.signal as jss
from functools import partial
from matrix_classes import AOperator
Array = jax.Array

# --------- Static jittable functions (no self) ---------
@partial(jax.jit, static_argnums=(2, 3, 4))
def _matvec_jit(x: Array, kernel: Array, N: int, shape_vol: tuple, method: str = 'auto') -> Array:
    """JIT-compiled matvec: x (N,) or (N,B) -> y with same shape"""
    if x.ndim == 1:
        # Single vector case
        V = jnp.reshape(x, shape_vol, order='C')
        Y = jss.convolve(V, kernel, mode='same', method=method)
        return jnp.reshape(Y, (N,), order='C')
    else:
        # Batch case: x (N, B) -> (B, *shape_vol)
        B = x.shape[1]
        VB = jnp.reshape(x.T, (B, *shape_vol), order='C')
        # Direct batch convolution - no vmap needed
        YB = jss.convolve(VB, kernel[None, ...], mode='same', method=method)
        return jnp.reshape(YB, (B, N), order='C').T

@partial(jax.jit, static_argnums=(2, 3, 4))
def _lmatvec_jit(y: Array, kernel: Array, N: int, shape_vol: tuple, method: str = 'auto') -> Array:
    """
    JIT-compiled lmatvec: y (N,) or (N,B) -> z with same shape
    The adjoint of convolution is convolution with the conjugated flipped kernel.
    """
    if y.ndim == 1:
        # Single vector case
        V = jnp.reshape(y, shape_vol, order='C')
        # For adjoint: convolve y with conj(flip(k))
        k_conj_flipped = jnp.conj(kernel[tuple(slice(None, None, -1) for _ in range(kernel.ndim))])
        Z = jss.convolve(V, k_conj_flipped, mode='same', method=method)
        return jnp.reshape(Z, (N,), order='C')
    else:
        # Batch case: y (N, B) -> (B, *shape_vol)
        B = y.shape[1]
        VB = jnp.reshape(y.T, (B, *shape_vol), order='C')
        # For adjoint: convolve y with conj(flip(k))
        k_conj_flipped = jnp.conj(kernel[tuple(slice(None, None, -1) for _ in range(kernel.ndim))])
        ZB = jss.convolve(VB, k_conj_flipped[None, ...], mode='same', method=method)
        return jnp.reshape(ZB, (B, N), order='C').T

class ConvNDOperator(AOperator):
    """
    Dimension-invariant linear convolution operator on a volume of shape S = (n1, n2, ..., nd).
    Forward uses zero-padded *linear* convolution (cval=0), with output cropped to 'same' shape.

    Forward (matvec):   y = K (*) x   implemented as   convolve(x, K, mode='same')
    Adjoint (lmatvec):  z = A^H y     implemented as   convolve(y, conj(flip(K)), mode='same')

    - Supports real/complex dtypes.
    - REQUIRES odd-sized kernels in all dimensions for proper adjoint with 'same' mode.
    - Batch inputs (N, B) are reshaped to (B, *shape), run in one call (no Python loops), then reshaped back.
    - get_row / get_col (dense): done via single forward/adjoint application to a delta image (O(N) but loop-free).
    - Row norms squared: one convolution of |K|^2 with an all-ones array (same boundary rule).

    NOTE: This operator is *linear* and the adjoint identity holds exactly because we use zero padding (cval=0).
          Periodic/reflect/mirror/nearest BCs are NOT modeled here; for those, use a custom padding + lax,
          or a circular-convolution FFT backend.
    """

    def __init__(self, kernel: Array, shape: tuple[int, ...], dtype=jnp.complex128, method: str = 'fft'):
        k = jnp.asarray(kernel, dtype=dtype)
        if k.ndim < 1:
            raise ValueError("kernel must be at least 1D")
        
        # Force kernel to be odd-sized in all dimensions for proper adjoint with 'same' mode
        if any(s % 2 == 0 for s in k.shape):
            raise ValueError(f"kernel must have odd size in all dimensions for proper adjoint with 'same' mode. Got shape: {k.shape}")
        
        self.kernel = k
        self.shape_vol = tuple(int(s) for s in shape)
        if any(s <= 0 for s in self.shape_vol):
            raise ValueError("all dimensions of shape must be positive integers")
        self.dtype = jnp.dtype(dtype)
        self.method = method
        # Precompute helpful bits
        self.N = int(jnp.prod(jnp.array(self.shape_vol)))
        self.shape = (self.N, self.N)  # linear operator view
        # No need to precompute flipped kernel - done in _lmatvec_jit

    # --------- helpers: reshape with C-order convention ---------
    def _vec(self, V: Array) -> Array:
        return jnp.reshape(V, (self.N,), order='C')

    # --------- public API ---------
    def matvec(self, x: Array) -> Array:
        """
        x: shape (N,) or (N,B) -> y with same shape.
        """
        x = jnp.asarray(x, dtype=self.dtype)
        if x.ndim == 1:
            if x.size != self.N:
                raise ValueError(f"x must have length {self.N}")
        elif x.ndim == 2:
            if x.shape[0] != self.N:
                raise ValueError(f"x first dimension must be {self.N}")
        else:
            raise ValueError("x must be 1D or 2D (batch)")
        
        return _matvec_jit(x, self.kernel, self.N, self.shape_vol, self.method)

    def lmatvec(self, y: Array) -> Array:
        """
        y: shape (N,) or (N,B) -> z with same shape. Exact Hilbert-space adjoint under <·,·>_2.
        """
        y = jnp.asarray(y, dtype=self.dtype)
        if y.ndim == 1:
            if y.size != self.N:
                raise ValueError(f"y must have length {self.N}")
        elif y.ndim == 2:
            if y.shape[0] != self.N:
                raise ValueError(f"y first dimension must be {self.N}")
        else:
            raise ValueError("y must be 1D or 2D (batch)")
        
        return _lmatvec_jit(y, self.kernel, self.N, self.shape_vol, self.method)

    # Dense row/column via single application (no Python loops).
    def get_col(self, linear_index: int, dense: bool = True) -> Array:
        """
        Column j of A: A e_j.  Construct delta at input position and apply forward once.
        Returns (N,) dense vector (set dense=False if you prefer (indices, values) later).
        JIT-compatible (no bounds checking).
        """
        N = int(self.N)  # Convert to Python int to avoid tracing issues
        delta = jnp.zeros((N,), dtype=self.dtype).at[linear_index].set(1.0)
        return self.matvec(delta)

    def get_row(self, linear_index: int, dense: bool = True) -> Array:
        """
        Row i of A: e_i^T A = (A^H e_i)^* .
        Compute adjoint on a delta at output position, then conjugate.
        Returns (N,) dense vector.
        JIT-compatible (no bounds checking).
        """
        N = int(self.N)  # Convert to Python int to avoid tracing issues
        delta = jnp.zeros((N,), dtype=self.dtype).at[linear_index].set(1.0)
        return jnp.conj(self.lmatvec(delta))

    def compute_all_row_norms_squared(self) -> Array:
        """
        r[i] = ||A[i, :]||_2^2 for each output position i (length N).
        For zero-padded linear convolution, this equals conv_same(|K|^2, ones(shape)).
        """
        Ksq = jnp.abs(self.kernel) ** 2
        ones = jnp.ones(self.shape_vol, dtype=self.dtype)
        R = jss.convolve(ones, Ksq, mode='same', method=self.method)  # same shape
        return self._vec(R)


    # “Adjoint handle”
    @property
    def H(self):
        class _Adj:
            def __init__(self, op: "ConvNDOperator"):
                self._op = op
                self.shape = op.shape
                self.dtype = op.dtype
            def matvec(self, y):   # (op.H) @ y
                return self._op.lmatvec(y)
            def lmatvec(self, x):  # (op.H).H @ x
                return self._op.matvec(x)
            @property
            def H(self):
                return self._op
            
            def compute_all_row_norms_squared(self) -> Array:
                """
                Compute row norms of A^H (which equal column norms of A).
                
                For A^H, row i corresponds to column i of A.
                So: ||A^H[i, :]||^2 = ||A[:, i]||^2
                
                Column norms of A are computed by: ||A[:, j]||^2 = ||A @ e_j||^2
                where e_j is a delta at input position j.
                
                For convolution, A @ e_j is the convolution of delta at position j with kernel K.
                The squared norm is computed by convolving |K|^2 with a delta, but we need
                to account for the fact that column norms use the forward kernel while
                the adjoint uses the flipped kernel.
                
                Actually, column norms should be computed using the same approach as row norms
                but with the kernel in its original orientation (not flipped). However, for
                non-symmetric kernels, we need to use the flipped kernel approach similar
                to the adjoint computation.
                
                The correct formula: column norms = conv_same(|K_flipped|^2, ones) where
                K_flipped is the flipped kernel (same as used in adjoint).
                """
                # Row norms of A^H = Column norms of A
                # Use the flipped kernel approach (similar to adjoint computation)
                # Column j norm = ||A @ e_j||^2 where e_j is delta at input position j
                # For convolution, this is computed using the flipped kernel
                k_flipped = self._op.kernel[tuple(slice(None, None, -1) for _ in range(self._op.kernel.ndim))]
                Ksq_flipped = jnp.abs(k_flipped) ** 2
                ones = jnp.ones(self._op.shape_vol, dtype=self._op.dtype)
                # Compute column norms using flipped kernel (matching adjoint computation)
                R = jss.convolve(ones, Ksq_flipped, mode='same', method=self._op.method)
                return self._op._vec(R)
        return _Adj(self)
    
    @property
    def T(self):
        """Transpose property (same as adjoint for real convolution operators)."""
        return self.H


# class SparseMatrixOperator(AOperator):
#     """
#     Efficient AOperator for JAX sparse matrices.
    
#     This class wraps a JAX sparse matrix and provides the AOperator interface,
#     enabling efficient matrix-vector operations, row/column access, and norm computations.
    
#     Supports:
#     - CSR (Compressed Sparse Row) format
#     - COO (Coordinate) format  
#     - BCOO (Block Compressed) format
    
#     Features:
#     - Efficient matvec/lmatvec operations using JAX sparse primitives
#     - Batch processing support
#     - Row/column extraction via sparse indexing
#     - Row norm squared computation
#     - Exact adjoint identity preservation
#     """
    
#     def __init__(self, sparse_matrix, dtype=jnp.complex128):
#         """
#         Initialize with a JAX sparse matrix.
        
#         Args:
#             sparse_matrix: JAX sparse matrix (CSR, COO, or BCOO format)
#             dtype: Target dtype for operations (default: complex128)
#         """
#         if not isinstance(sparse_matrix, (jax_sparse.CSR, jax_sparse.COO, jax_sparse.BCOO)):
#             raise ValueError("sparse_matrix must be a JAX sparse matrix (CSR, COO, or BCOO)")
        
#         self.sparse_matrix = sparse_matrix
#         self.dtype = jnp.dtype(dtype)
#         self.shape = sparse_matrix.shape
#         self.N = self.shape[0]
#         self.M = self.shape[1]
        
#         # Convert to dense for base class compatibility if needed
#         # This is only used for some methods that require dense access
#         self._dense_matrix = None
        
#     def _get_dense_matrix(self):
#         """Lazily convert sparse matrix to dense for operations that need it."""
#         if self._dense_matrix is None:
#             self._dense_matrix = self.sparse_matrix.todense().astype(self.dtype)
#         return self._dense_matrix
    
#     @partial(jax.jit, static_argnums=(1,))
#     def matvec(self, x: Array, batch: bool = False) -> Array:
#         """
#         Matrix-vector multiplication: y = A @ x
        
#         Args:
#             x: Input vector of shape (M,) or (M, B) for batch processing
#             batch: Whether x is batched (ignored, inferred from x.ndim)
            
#         Returns:
#             y: Output vector of shape (N,) or (N, B)
#         """
#         x = jnp.asarray(x, dtype=self.dtype)
        
#         if x.ndim == 1:
#             if x.size != self.M:
#                 raise ValueError(f"x must have length {self.M}")
#             return self.sparse_matrix @ x
            
#         elif x.ndim == 2:
#             if x.shape[0] != self.M:
#                 raise ValueError(f"x first dimension must be {self.M}")
#             # Batch matrix-vector multiplication
#             # For sparse matrices, we need to do this column by column
#             B = x.shape[1]
#             results = []
#             for b in range(B):
#                 results.append(self.sparse_matrix @ x[:, b])
#             return jnp.column_stack(results)
#         else:
#             raise ValueError("x must be 1D or 2D")
    
#     @partial(jax.jit, static_argnums=(1,))
#     def lmatvec(self, y: Array, batch: bool = False) -> Array:
#         """
#         Adjoint matrix-vector multiplication: z = A^H @ y
        
#         Args:
#             y: Input vector of shape (N,) or (N, B) for batch processing
#             batch: Whether y is batched (ignored, inferred from y.ndim)
            
#         Returns:
#             z: Output vector of shape (M,) or (M, B)
#         """
#         y = jnp.asarray(y, dtype=self.dtype)
        
#         if y.ndim == 1:
#             if y.size != self.N:
#                 raise ValueError(f"y must have length {self.N}")
#             # For sparse matrices, A^H @ y = (y^H @ A)^H
#             # We compute this by taking the conjugate of A^T @ y^*
#             return jnp.conj(self.sparse_matrix.T @ jnp.conj(y))
            
#         elif y.ndim == 2:
#             if y.shape[0] != self.N:
#                 raise ValueError(f"y first dimension must be {self.N}")
#             # Batch adjoint matrix-vector multiplication
#             B = y.shape[1]
#             results = []
#             for b in range(B):
#                 results.append(jnp.conj(self.sparse_matrix.T @ jnp.conj(y[:, b])))
#             return jnp.column_stack(results)
#         else:
#             raise ValueError("y must be 1D or 2D")
    
#     def get_row(self, i: int, dense: bool = True) -> Array:
#         """
#         Get row i of the sparse matrix.
        
#         Args:
#             i: Row index
#             dense: Whether to return dense array (ignored for sparse matrices)
            
#         Returns:
#             Row i as a dense vector of shape (M,)
#         """
#         if not (0 <= i < self.N):
#             raise IndexError(f"Row index {i} out of bounds [0, {self.N})")
        
#         # Create a unit vector at position i
#         e_i = jnp.zeros(self.N, dtype=self.dtype).at[i].set(1.0)
#         return self.lmatvec(e_i)
    
#     def get_col(self, j: int, dense: bool = True) -> Array:
#         """
#         Get column j of the sparse matrix.
        
#         Args:
#             j: Column index
#             dense: Whether to return dense array (ignored for sparse matrices)
            
#         Returns:
#             Column j as a dense vector of shape (N,)
#         """
#         if not (0 <= j < self.M):
#             raise IndexError(f"Column index {j} out of bounds [0, {self.M})")
        
#         # Create a unit vector at position j
#         e_j = jnp.zeros(self.M, dtype=self.dtype).at[j].set(1.0)
#         return self.matvec(e_j)
    
#     def compute_all_row_norms_squared(self) -> Array:
#         """
#         Compute squared L2 norms of all rows: ||A[i, :]||^2 for all i.
        
#         Returns:
#             Array of shape (N,) containing row norm squares
#         """
#         # For sparse matrices, we can compute this more efficiently
#         # by using the fact that ||A[i, :]||^2 = sum(|A[i, j]|^2) for all j
        
#         # Convert to dense temporarily for this computation
#         # In practice, you might want to implement this more efficiently
#         # using sparse-specific operations
#         dense_A = self._get_dense_matrix()
#         return jnp.sum(jnp.abs(dense_A) ** 2, axis=1)
    
#     def matvec_with_R(self, v: Array, I: list) -> Array:
#         """
#         Compute A[I, :] @ v by selecting specific rows from matvec result.
        
#         Args:
#             v: Vector of shape (M,)
#             I: List of row indices to select
            
#         Returns:
#             Selected rows of A @ v
#         """
#         if not I:
#             return jnp.zeros(0, dtype=self.dtype)
        
#         full_result = self.matvec(v)
#         return full_result[jnp.array(I)]
    
#     def lmatvec_with_C(self, v: Array, J: list) -> Array:
#         """
#         Compute A[:, J]^T @ v by selecting specific columns from lmatvec result.
        
#         Args:
#             v: Vector of shape (N,)
#             J: List of column indices to select
            
#         Returns:
#             Selected columns of A^H @ v
#         """
#         if not J:
#             return jnp.zeros(0, dtype=self.dtype)
        
#         full_result = self.lmatvec(v)
#         return full_result[jnp.array(J)]
    
#     def matvec_with_C(self, v: Array, J: list) -> Array:
#         """
#         Compute A[:, J] @ v by zero-padding v and applying matvec.
        
#         Args:
#             v: Vector of shape (len(J),)
#             J: List of column indices
            
#         Returns:
#             A[:, J] @ v of shape (N,)
#         """
#         if not J:
#             return jnp.zeros(self.N, dtype=self.dtype)
        
#         # Pad v with zeros
#         v_padded = jnp.zeros(self.M, dtype=self.dtype)
#         v_padded = v_padded.at[jnp.array(J)].set(jnp.asarray(v, dtype=self.dtype))
#         return self.matvec(v_padded)
    
#     def lmatvec_with_R(self, v: Array, I: list) -> Array:
#         """
#         Compute A[I, :]^T @ v by zero-padding v and applying lmatvec.
        
#         Args:
#             v: Vector of shape (len(I),)
#             I: List of row indices
            
#         Returns:
#             A[I, :]^T @ v of shape (M,)
#         """
#         if not I:
#             return jnp.zeros(self.M, dtype=self.dtype)
        
#         # Pad v with zeros
#         v_padded = jnp.zeros(self.N, dtype=self.dtype)
#         v_padded = v_padded.at[jnp.array(I)].set(jnp.asarray(v, dtype=self.dtype))
#         return self.lmatvec(v_padded)
    
#     # Adjoint handle
#     @property
#     def H(self):
#         """Return the adjoint (Hermitian transpose) of this operator."""
#         class _Adj:
#             def __init__(self, op: "SparseMatrixOperator"):
#                 self._op = op
#                 self.shape = (op.M, op.N)  # Swap dimensions
#                 self.dtype = op.dtype
#                 self.N = op.M
#                 self.M = op.N
            
#             def matvec(self, y, batch=False):
#                 return self._op.lmatvec(y, batch=batch)
            
#             def lmatvec(self, x, batch=False):
#                 return self._op.matvec(x, batch=batch)
            
#             @property
#             def H(self):
#                 return self._op
            
#             def compute_all_row_norms_squared(self) -> Array:
#                 """
#                 Compute row norms of A^T (column norms of A).
#                 Column norms of A = row norms of A^T.
#                 """
#                 # Delegate to the transpose operator if it exists
#                 if hasattr(self._op, 'T') and hasattr(self._op.T, 'compute_all_row_norms_squared'):
#                     return self._op.T.compute_all_row_norms_squared()
                
#                 # Otherwise compute directly using BCOO structure
#                 data = self._op.bcoo_matrix.data
#                 indices = self._op.bcoo_matrix.indices
#                 data_sq = jnp.abs(data) ** 2
#                 if data_sq.ndim > 1:
#                     data_sq = jnp.sum(data_sq, axis=tuple(range(1, data_sq.ndim)))
                
#                 if indices.ndim == 2 and indices.shape[1] == 2:
#                     from jax import lax
#                     col_indices = indices[:, 1]
#                     col_norms_sq = jnp.zeros(self._op.M, dtype=data_sq.dtype)
#                     col_norms_sq = lax.scatter_add(
#                         col_norms_sq,
#                         col_indices[:, None],
#                         data_sq,
#                         lax.ScatterDimensionNumbers(
#                             update_window_dims=(),
#                             inserted_window_dims=(0,),
#                             scatter_dims_to_operand_dims=(0,)
#                         )
#                     )
#                     return col_norms_sq
#                 else:
#                     dense_A = self._op.bcoo_matrix.todense()
#                     return jnp.sum(jnp.abs(dense_A) ** 2, axis=0)
                
#         return _Adj(self)
    
#     @property
#     def T(self):
#         """Transpose property (same as adjoint for real sparse matrices)."""
#         return self.H


# def _self_test():
#     key = jax.random.key(0)

#     def run_one(shape, kshape, complex_kernel: bool):
#         n = int(jnp.prod(jnp.array(shape)))
#         kkey, xkey, ykey = jax.random.split(key, 3)
#         K = jax.random.normal(kkey, kshape)
#         if complex_kernel:
#             K = K + 1j * jax.random.normal(kkey, kshape)
#         K = K / jnp.maximum(1, jnp.prod(jnp.array(kshape)))  # mild scaling

#         A = ConvNDOperator(K, shape, dtype=jnp.complex128 if complex_kernel else jnp.float64)

#         # 1) Adjoint identity (single, batched)
#         # Use separate keys to avoid correlation
#         xkey2, ykey2 = jax.random.split(xkey, 2)
#         if complex_kernel:
#             x = jax.random.normal(xkey2, (n,)) + 1j * jax.random.normal(xkey2, (n,))
#             y = jax.random.normal(ykey2, (n,)) + 1j * jax.random.normal(ykey2, (n,))
#         else:
#             x = jax.random.normal(xkey2, (n,))
#             y = jax.random.normal(ykey2, (n,))
#         lhs = jnp.vdot(A.matvec(x), y)          # conj(Ax)·y
#         rhs = jnp.vdot(x, A.lmatvec(y))
#         error = jnp.abs(lhs - rhs) / jnp.maximum(jnp.abs(lhs), 1e-20)
#         print(f"Shape: {shape}, Kernel shape: {kshape}, Complex: {complex_kernel}")
#         print(f"lhs: {lhs}, rhs: {rhs}, rel_error: {error:.2e}")
#         assert jnp.allclose(lhs, rhs, rtol=1e-6, atol=1e-6), f"Adjoint property failed: lhs={lhs}, rhs={rhs}, error={error:.2e}"

#         B = 4
#         # Use separate keys for batch test
#         xkey3, ykey3 = jax.random.split(xkey2, 2)
#         if complex_kernel:
#             xb = jax.random.normal(xkey3, (n, B)) + 1j * jax.random.normal(xkey3, (n, B))
#             yb = jax.random.normal(ykey3, (n, B)) + 1j * jax.random.normal(ykey3, (n, B))
#         else:
#             xb = jax.random.normal(xkey3, (n, B))
#             yb = jax.random.normal(ykey3, (n, B))
#         lhsb = jnp.vdot(A.matvec(xb), yb)
#         rhsb = jnp.vdot(xb, A.lmatvec(yb))
#         error_batch = jnp.abs(lhsb - rhsb) / jnp.maximum(jnp.abs(lhsb), 1e-20)
#         print(f"Batch test - lhs: {lhsb}, rhs: {rhsb}, rel_error: {error_batch:.2e}")
#         assert jnp.allclose(lhsb, rhsb, rtol=1e-6, atol=1e-6), f"Batch adjoint property failed: error={error_batch:.2e}"

#         # 2) Dense A via a single batched call
#         I = jnp.eye(n, dtype=A.dtype)
#         Adense = A.matvec(I)  # (n, n)
#         AdenseH = A.lmatvec(I)
#         assert jnp.allclose(AdenseH, jnp.conj(Adense).T, rtol=1e-12, atol=1e-12)

#         # 3) get_row / get_col vs dense A
#         i = int(n // 3)
#         j = int(n // 2)
#         row_i = A.get_row(i, dense=True)
#         col_j = A.get_col(j, dense=True)
#         assert jnp.allclose(row_i, Adense[i, :], rtol=1e-12, atol=1e-12)
#         assert jnp.allclose(col_j, Adense[:, j], rtol=1e-12, atol=1e-12)

#         # 4) Row norms
#         r_sq = A.compute_all_row_norms_squared()
#         r_sq_ref = jnp.sum(jnp.abs(Adense) ** 2, axis=1)
#         assert jnp.allclose(r_sq, r_sq_ref, rtol=1e-12, atol=1e-12)

#     # Try comprehensive shapes/kernels - both even and odd sizes
#     print("Testing ConvNDOperator with various shapes and kernels...")
    
#     # 1D tests - only odd-sized kernels for 'same' mode
#     run_one((16,), (3,), False)           # 1D even, odd kernel
#     run_one((15,), (3,), False)           # 1D odd, odd kernel
#     run_one((17,), (5,), True)            # 1D odd, odd kernel, complex
#     run_one((14,), (3,), True)            # 1D even, odd kernel, complex
    
#     # 2D tests - only odd-sized kernels for 'same' mode
#     run_one((8, 8), (3, 3), False)        # 2D even-even, odd-odd kernel
#     run_one((7, 9), (3, 3), False)        # 2D odd-odd, odd-odd kernel
#     run_one((6, 10), (5, 3), True)        # 2D even-even, odd-odd kernel, complex
#     run_one((9, 7), (3, 3), True)         # 2D odd-odd, odd-odd kernel, complex
    
#     # 3D tests - only odd-sized kernels for 'same' mode
#     run_one((4, 5, 6), (3, 3, 3), False)  # 3D mixed, odd kernel
#     run_one((6, 6, 6), (3, 3, 3), False)  # 3D even-even-even, odd-odd-odd kernel
#     run_one((5, 7, 9), (3, 3, 3), True)   # 3D odd-odd-odd, odd kernel, complex
#     run_one((8, 4, 6), (3, 3, 3), True)  # 3D even-even-even, odd-odd-odd kernel, complex
    
#     # Edge cases
#     run_one((2,), (1,), False)            # Very small 1D
#     run_one((3, 3), (1, 1), False)        # Very small 2D
#     run_one((2, 2, 2), (1, 1, 1), True)  # Very small 3D
    
#     print("All ConvNDOperator tests passed.")


# def _sparse_matrix_self_test():
#     """Comprehensive tests for SparseMatrixOperator."""
#     key = jax.random.key(42)
    
#     def run_sparse_test(n, m, sparsity=0.1, complex_matrix=True):
#         """Test SparseMatrixOperator with given dimensions and sparsity."""
#         print(f"Testing SparseMatrixOperator: {n}x{m}, sparsity={sparsity}, complex={complex_matrix}")
        
#         # Create a random sparse matrix
#         key1, key2 = jax.random.split(key)
        
#         # Generate random values
#         if complex_matrix:
#             values = (jax.random.normal(key1, (n * m,)) + 
#                      1j * jax.random.normal(key2, (n * m,)))
#         else:
#             values = jax.random.normal(key1, (n * m,))
        
#         # Create sparse pattern
#         nnz = int(n * m * sparsity)
#         indices = jax.random.choice(key1, n * m, shape=(nnz,), replace=False)
        
#         # Create sparse matrix in COO format
#         rows = indices // m
#         cols = indices % m
#         data = values[indices]
        
#         sparse_matrix = jax_sparse.COO((data, rows, cols), shape=(n, m))
#         A = SparseMatrixOperator(sparse_matrix, dtype=jnp.complex128)
        
#         # Test 1: Adjoint identity (single vectors)
#         xkey, ykey = jax.random.split(key1, 2)
#         x = jax.random.normal(xkey, (m,)) + 1j * jax.random.normal(xkey, (m,))
#         y = jax.random.normal(ykey, (n,)) + 1j * jax.random.normal(ykey, (n,))
        
#         lhs = jnp.vdot(A.matvec(x), y)  # <Ax, y>
#         rhs = jnp.vdot(x, A.lmatvec(y))  # <x, A^H y>
        
#         print(f"  Adjoint identity (single): lhs={lhs}, rhs={rhs}")
#         assert jnp.allclose(lhs, rhs, rtol=1e-10, atol=1e-10), f"Adjoint identity failed: {lhs} != {rhs}"
        
#         # Test 2: Adjoint identity (batched)
#         B = 3
#         xb = jax.random.normal(xkey, (m, B)) + 1j * jax.random.normal(xkey, (m, B))
#         yb = jax.random.normal(ykey, (n, B)) + 1j * jax.random.normal(ykey, (n, B))
        
#         lhsb = jnp.vdot(A.matvec(xb, batch=True), yb)
#         rhsb = jnp.vdot(xb, A.lmatvec(yb, batch=True))
        
#         print(f"  Adjoint identity (batch): lhs={lhsb}, rhs={rhsb}")
#         assert jnp.allclose(lhsb, rhsb, rtol=1e-10, atol=1e-10), f"Batched adjoint identity failed"
        
#         # Test 3: Dense matrix comparison
#         dense_A = A._get_dense_matrix()
#         I = jnp.eye(m, dtype=A.dtype)
#         Adense_via_matvec = A.matvec(I, batch=True)
#         Adense_via_lmatvec = A.lmatvec(jnp.eye(n, dtype=A.dtype), batch=True)
        
#         print(f"  Dense matrix consistency check")
#         assert jnp.allclose(Adense_via_matvec, dense_A, rtol=1e-10, atol=1e-10)
#         assert jnp.allclose(Adense_via_lmatvec, jnp.conj(dense_A).T, rtol=1e-10, atol=1e-10)
        
#         # Test 4: get_row / get_col vs dense matrix
#         i = n // 3
#         j = m // 2
#         row_i = A.get_row(i, dense=True)
#         col_j = A.get_col(j, dense=True)
        
#         print(f"  Row/column extraction test")
#         assert jnp.allclose(row_i, dense_A[i, :], rtol=1e-10, atol=1e-10)
#         assert jnp.allclose(col_j, dense_A[:, j], rtol=1e-10, atol=1e-10)
        
#         # Test 5: Row norms squared
#         r_sq = A.compute_all_row_norms_squared()
#         r_sq_ref = jnp.sum(jnp.abs(dense_A) ** 2, axis=1)
        
#         print(f"  Row norms squared test")
#         assert jnp.allclose(r_sq, r_sq_ref, rtol=1e-10, atol=1e-10)
        
#         # Test 6: Adjoint operator (H property)
#         AH = A.H
#         assert AH.shape == (m, n)
#         assert jnp.allclose(AH.matvec(y), A.lmatvec(y), rtol=1e-10, atol=1e-10)
#         assert jnp.allclose(AH.lmatvec(x), A.matvec(x), rtol=1e-10, atol=1e-10)
#         assert AH.H is A  # Double adjoint should return original
        
#         print(f"  ✓ All tests passed for {n}x{m} sparse matrix")
    
#     # Run tests with different configurations
#     run_sparse_test(10, 8, sparsity=0.2, complex_matrix=True)
#     run_sparse_test(15, 12, sparsity=0.1, complex_matrix=False)
#     run_sparse_test(20, 20, sparsity=0.05, complex_matrix=True)
#     run_sparse_test(5, 15, sparsity=0.3, complex_matrix=True)
    
#     print("All SparseMatrixOperator tests passed.")


# if __name__ == "__main__":
#     _self_test()
#     _sparse_matrix_self_test()
