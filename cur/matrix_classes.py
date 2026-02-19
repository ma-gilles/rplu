
import numpy as np
import jax
import jax.numpy as jnp
from scipy.linalg import toeplitz, matmul_toeplitz
try:
    from jax.experimental import sparse as jax_sparse
except ImportError:
    try:
        import jax.sparse as jax_sparse
    except ImportError:
        raise ImportError("JAX sparse is required for SparseMatrixOperator. Please install jax with sparse support.")


class AOperator:
    """
    Abstract interface for A. Provide:
      - shape: (n, m)
      - dtype: np.dtype
      - get_row(i) -> (m,) ndarray
      - get_col(j) -> (n,) ndarray
      - matvec(x)  -> A @ x           (x shape (m,))
      - lmatvec(y) -> A^* @ y         (y shape (n,))
      - matvec_with_R(v, I) -> A[I, :] @ v  (v shape (m,), I list of row indices)
      - lmatvec_with_C(v, J) -> A[:, J]^T @ v  (v shape (n,), J list of col indices)
    """
    def __init__(self, A):
        self.A = A
        self.shape = A.shape
        self.dtype = A.dtype
    def get_row(self, i): return jnp.asarray(self.A[i, :])
    def get_col(self, j): return jnp.asarray(self.A[:, j])
    def matvec(self, x):  return jnp.asarray(self.A @ x)
    def lmatvec(self, y): return jnp.asarray(self.A.conj().T @ y)
    
    def matvec_with_R(self, v, I):
        """Compute A[I, :] @ v by doing matvec, then selecting rows.
        I should be an array of indices. Indices out of bounds will be filled with 0.
        E.g., if A is 10x10 and I is [0, 1, 2, 10, 11], then the result will be:
        [A[0, :], A[1, :], A[2, :], 0, 0]
        """
        # v may be (m,) or (m, B); matvec will return (n,) or (n, B)
        result_full = self.matvec(v)
        # Select rows in I along the first axis; supports batched outputs as well
        return result_full.at[I].get(mode='fill', fill_value=0.0)
    
    def lmatvec_with_C(self, v, J):
        """Compute A[:, J]^T @ v by doing lmatvec, then selecting columns."""
        # v may be (n,) or (n, B); lmatvec will return (m,) or (m, B)
        result_full = self.lmatvec(v)
        # Select entries at indices J along the first axis; supports batched outputs
        return result_full.at[J].get(mode='fill', fill_value=0.0)
    
    def matvec_with_C(self, v, J):
        """Compute A[:, J] @ v by doing matvec with zero-padded v."""
        # Handle both Python lists and JAX arrays
        J_array = J
        v_arr = jnp.asarray(v, dtype=self.dtype)
        # Vector case: (m,)
        if v_arr.ndim == 1:
            v_padded = jnp.zeros(self.shape[1], dtype=self.dtype)
            # Out-of-bounds indices will be dropped
            v_padded = v_padded.at[J_array].set(v_arr, mode='drop')
            return self.matvec(v_padded)
        # Batched case: (m, B)
        elif v_arr.ndim == 2:
            B = v_arr.shape[1]
            v_padded = jnp.zeros((self.shape[1], B), dtype=self.dtype)
            v_padded = v_padded.at[J_array, :].set(v_arr, mode='drop')
            return self.matvec(v_padded)
        else:
            raise ValueError("v must be 1D or 2D for matvec_with_C")
    
    def lmatvec_with_R(self, v, I):
        """Compute A[I, :]^T @ v by doing lmatvec with zero-padded v."""
        # Handle both Python lists and JAX arrays
        v_arr = jnp.asarray(v, dtype=self.dtype)
        # Vector case: (n,)
        if v_arr.ndim == 1:
            v_padded = jnp.zeros(self.shape[0], dtype=self.dtype)
            v_padded = v_padded.at[I].set(v_arr, mode='drop')
            return self.lmatvec(v_padded)
        # Batched case: (n, B)
        elif v_arr.ndim == 2:
            B = v_arr.shape[1]
            v_padded = jnp.zeros((self.shape[0], B), dtype=self.dtype)
            v_padded = v_padded.at[I, :].set(v_arr, mode='drop')
            return self.lmatvec(v_padded)
        else:
            raise ValueError("v must be 1D or 2D for lmatvec_with_R")
        
    def compute_all_row_norms_squared(self):
        """Compute all row norms ||A[i, :]||^2 efficiently."""
        return jnp.sum(jnp.abs(self.A)**2, axis=1)
    
    def __matmul__(self, other):
        """Support matrix multiplication with @ operator."""
        if not (isinstance(other, np.ndarray) or isinstance(other, jnp.ndarray)):
            return NotImplemented
        return self.matvec(other)
    
    def __rmatmul__(self, other):
        """Support right matrix multiplication with @ operator."""
        if not (isinstance(other, np.ndarray) or isinstance(other, jnp.ndarray)):
            return NotImplemented
        return self.lmatvec(other)
    @property
    def T(self):
        """Return the transpose operator."""
        # Construct a new AOperator of the same type, with matrix transposed and shape reversed.
        # Assumes self.A is the underlying 2D matrix (NumPy/JAX array).
        # Other attributes (e.g. dtype) should be preserved if needed.
        return type(self)(self.A.T)
    
    @property
    def H(self):
        """Return the adjoint (conjugate transpose) operator."""
        # For real matrices, adjoint is the same as transpose
        return type(self)(self.A.conj().T)


class ToeplitzOperator(AOperator):
    """
    Efficient AOperator for Toeplitz matrices.
    A Toeplitz matrix has constant diagonals: A[i,j] = t[i-j] for some vector t.
    """
    def __init__(self, t, n, m):
        """
        Initialize with Toeplitz vector t.
        For an n x m Toeplitz matrix, t should have length n + m - 1.
        t[0] is the main diagonal, t[1] is the first superdiagonal, etc.
        t[-1] is the first subdiagonal, etc.
        """
        self.t = np.asarray(t, dtype=np.complex128)
        self.n = n
        self.m = m
        self.shape = (n, m)
        self.dtype = self.t.dtype
        
        # Validate t length
        expected_len = n + m - 1
        if len(self.t) != expected_len:
            raise ValueError(f"Toeplitz vector t must have length {expected_len}, got {len(self.t)}")
    
    def get_row(self, i):
        """Get row i of the Toeplitz matrix."""
        j_indices = np.arange(self.m)
        diag_indices = i - j_indices
        valid_mask = (-self.m + 1 <= diag_indices) & (diag_indices <= self.n - 1)
        
        # Map diagonal indices to actual t indices and return directly
        t_indices = diag_indices[valid_mask] + (self.m - 1)
        return self.t[t_indices]
    
    def get_col(self, j):
        """Get column j of the Toeplitz matrix."""
        i_indices = np.arange(self.n)
        diag_indices = i_indices - j
        valid_mask = (-self.m + 1 <= diag_indices) & (diag_indices <= self.n - 1)
        
        # Map diagonal indices to actual t indices and return directly
        t_indices = diag_indices[valid_mask] + (self.m - 1)
        return self.t[t_indices]
    
    def matvec(self, x, batch=False):
        """Compute A @ x using scipy.linalg.matmul_toeplitz for efficient Toeplitz matrix-vector product.
        
        Args:
            x: Input vector(s). Can be 1D (single vector) or 2D (batch of vectors).
            batch: If True, treat 2D input as batch and process all columns efficiently.
                  If False, flatten 2D input to 1D (for scipy compatibility).
        
        Returns:
            Result of A @ x. Shape (n,) for single vector, (n, batch_size) for batch.
        """
        # Ensure x is 2D for consistent processing
        if x.ndim == 1:
            x = x[:, np.newaxis]
            squeeze_result = True
        else:
            squeeze_result = False
        
        # Validate input dimensions
        if x.shape[0] != self.m:
            raise ValueError(f"Input vectors must have length {self.m}")
        
        # Get first column and row for scipy's matmul_toeplitz
        c, r = toeplitz_first_col_row_from_t(self.t, self.n, self.m)
        
        # Use scipy's efficient matmul_toeplitz
        result = matmul_toeplitz((c, r), x)
        
        # Squeeze back to 1D if input was 1D and not using batch processing
        if squeeze_result and not batch:
            result = result.squeeze(axis=-1)
        
        return result
    
    def lmatvec(self, y, batch=False):
        """Compute A^H @ y using scipy.linalg.matmul_toeplitz for efficient Toeplitz matrix-vector product.
        
        Args:
            y: Input vector(s). Can be 1D (single vector) or 2D (batch of vectors).
            batch: If True, treat 2D input as batch and process all columns efficiently.
                  If False, flatten 2D input to 1D (for scipy compatibility).
        
        Returns:
            Result of A^H @ y. Shape (m,) for single vector, (m, batch_size) for batch.
        """
        # Ensure y is 2D for consistent processing
        if y.ndim == 1:
            y = y[:, np.newaxis]
            squeeze_result = True
        else:
            squeeze_result = False
        
        # Validate input dimensions
        if y.shape[0] != self.n:
            raise ValueError(f"Input vectors must have length {self.n}")
        
        # Get first column and row for the original matrix
        c, r = toeplitz_first_col_row_from_t(self.t, self.n, self.m)
        
        # For A^H, we need the conjugate transpose
        # A^H[j,i] = A[i,j]^H = c[i-j]^H if i >= j, else r[j-i]^H
        # The first column of A^H is [c[0]^H, r[1]^H, ..., r[m-1]^H]
        # The first row of A^H is [c[0]^H, c[1]^H, ..., c[n-1]^H]
        c_H = np.concatenate([c[:1].conj(), r[1:].conj()])  # First column of A^H (length m)
        r_H = c[:self.n].conj()  # First row of A^H (length n)
        
        # Use scipy's efficient matmul_toeplitz
        result = matmul_toeplitz((c_H, r_H), y)
        
        # Squeeze back to 1D if input was 1D and not using batch processing
        if squeeze_result and not batch:
            result = result.squeeze(axis=-1)
        
        return result

    def compute_all_row_norms_squared(self):
        """Compute all row norms ||A[i, :]||^2 efficiently."""
        return toeplitz_row_norms_sq_from_t(self.t, self.n, self.m)
    
    def __matmul__(self, other):
        """Support matrix multiplication with @ operator."""
        if not isinstance(other, np.ndarray):
            return NotImplemented
        
        # Use batch processing for matrices, single vector processing for vectors
        return self.matvec(other, batch=(other.ndim == 2))
    
    def __rmatmul__(self, other):
        """Support right matrix multiplication with @ operator."""
        if not isinstance(other, np.ndarray):
            return NotImplemented
        
        # Use batch processing for matrices, single vector processing for vectors
        return self.lmatvec(other, batch=(other.ndim == 2))
    
    def __array__(self, dtype=None):
        """Support numpy array conversion and operations like .T"""
        # Convert to dense matrix for array operations
        c, r = toeplitz_first_col_row_from_t(self.t, self.n, self.m)
        A_dense = toeplitz(c, r)
        if dtype is not None:
            return A_dense.astype(dtype)
        return A_dense
    
    @property
    def T(self):
        """Transpose property for matrix operations."""
        return ToeplitzOperatorTranspose(self)
    
    def __getitem__(self, key):
        """Support indexing like A[i, j]"""
        if isinstance(key, tuple) and len(key) == 2:
            i, j = key
            if isinstance(i, int) and isinstance(j, int):
                # Single element access
                if 0 <= i < self.n and 0 <= j < self.m:
                    diag_idx = i - j
                    if -self.m + 1 <= diag_idx <= self.n - 1:
                        t_idx = diag_idx + (self.m - 1)
                        return self.t[t_idx]
                    else:
                        return 0
                else:
                    raise IndexError(f"Index ({i}, {j}) out of bounds for matrix shape {self.shape}")
            else:
                raise NotImplementedError("Only single element indexing is supported")
        else:
            raise NotImplementedError("Only 2D indexing is supported")


class ToeplitzOperatorTranspose:
    """Transpose wrapper for ToeplitzOperator to support .T @ operations."""
    
    def __init__(self, toeplitz_op):
        self.toeplitz_op = toeplitz_op
        self.shape = (toeplitz_op.m, toeplitz_op.n)  # Transposed shape
        self.dtype = toeplitz_op.dtype
    
    def __matmul__(self, other):
        """Support matrix multiplication with @ operator for transpose."""
        if not isinstance(other, np.ndarray):
            return NotImplemented
        
        # Use batch processing for matrices, single vector processing for vectors
        return self.toeplitz_op.lmatvec(other, batch=(other.ndim == 2))
    
    def __rmatmul__(self, other):
        """Support right matrix multiplication with @ operator for transpose."""
        if not isinstance(other, np.ndarray):
            return NotImplemented
        
        # Use batch processing for matrices, single vector processing for vectors
        return self.toeplitz_op.matvec(other, batch=(other.ndim == 2))
    
    def __array__(self, dtype=None):
        """Support numpy array conversion."""
        # Convert to dense matrix for array operations
        # For transpose, we need to construct the transpose of the original matrix
        # The transpose of a Toeplitz matrix is also Toeplitz
        # We need to construct the transpose by swapping the first column and row
        c, r = toeplitz_first_col_row_from_t(self.toeplitz_op.t, self.toeplitz_op.n, self.toeplitz_op.m)
        A_dense = toeplitz(r, c)  # Swap c and r for transpose
        if dtype is not None:
            return A_dense.astype(dtype)
        return A_dense
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Support numpy ufuncs by converting to dense matrix."""
        if ufunc is np.matmul:
            # Handle matmul by converting to dense matrix
            dense = self.__array__()
            inputs = list(inputs)
            for i, inp in enumerate(inputs):
                if inp is self:
                    inputs[i] = dense
            return getattr(ufunc, method)(*inputs, **kwargs)
        else:
            # For other ufuncs, convert to dense matrix
            dense = self.__array__()
            inputs = list(inputs)
            for i, inp in enumerate(inputs):
                if inp is self:
                    inputs[i] = dense
            return getattr(ufunc, method)(*inputs, **kwargs)
    
    def __array_function__(self, func, types, args, kwargs):
        """Support numpy functions by converting to dense matrix."""
        # Convert to dense matrix for numpy functions
        dense = self.__array__()
        new_args = []
        for arg in args:
            if arg is self:
                new_args.append(dense)
            else:
                new_args.append(arg)
        return func(*new_args, **kwargs)
    
    @property
    def T(self):
        """Double transpose returns original operator."""
        return self.toeplitz_op


def toeplitz_first_col_row_from_t(t, n, m):
    """
    Recover (c, r) from the “Toeplitz vector” t of length n+m-1.
    A[i,j] = t[offset + i - j],  offset = m-1.
    c: first column (len n), r: first row (len m).
    """
    t = np.asarray(t)
    if t.size != n + m - 1:
        raise ValueError(f"len(t)={t.size} but expected {n+m-1}")
    offset = m - 1
    c = t[offset : offset + n]
    # robust for m=1: walk left from offset and take first m entries
    r = t[offset::-1][:m]
    return c, r

def toeplitz_row_norms_sq_from_t(t, n, m):
    """
    Return the vector of row 2-norms squared for the n×m Toeplitz matrix
    defined by t (len n+m-1). Runs in O(n).
    """
    c, r = toeplitz_first_col_row_from_t(t, n, m)
    c = np.asarray(c); r = np.asarray(r)
    if c.size == 0:
        return np.array([], dtype=np.result_type(c, r, float))
    if r[0] != c[0]:
        # Shouldn’t happen if c,r come from t; keep the guard.
        raise ValueError("Inconsistent c[0] and r[0] for Toeplitz.")

    n_, m_ = c.size, r.size
    rt = np.result_type(c, r, float)

    # s0 = |c0|^2 + sum_{k=1}^{m-1} |r_k|^2
    s0 = np.abs(c[0])**2 + np.sum(np.abs(r[1:])**2)
    if n_ == 1:
        return np.array([s0], dtype=rt)

    # Recurrence (vectorized):
    # For i=1..m-1: s_i = s_{i-1} + |c_i|^2 - |r_{m-i}|^2
    # For i>=m:     s_i = s_{i-1} + |c_i|^2 - |c_{i-m}|^2
    i1 = min(n_-1, m_-1)  # last index where the r-term exists

    # Part A: i = 1..i1
    # r indices: (m-1), (m-2), ..., (m-i1)
    partA = np.abs(c[1:i1+1])**2 - np.abs(r[m_-1 : m_-1 - i1 : -1])**2

    # Part B: i = m..n-1 (only if n-1 >= m)
    if n_-1 >= m_:
        add_terms = np.abs(c[m_:])**2      # |c_m|^2, ..., |c_{n-1}|^2
        sub_terms = np.abs(c[:n_-m_])**2   # |c_0|^2, ..., |c_{n-m-1}|^2
        partB = add_terms - sub_terms
        deltas = np.concatenate([np.zeros(1, dtype=rt), partA, partB])
    else:
        deltas = np.concatenate([np.zeros(1, dtype=rt), partA])

    return s0 + np.cumsum(deltas, dtype=rt)


# ============================================================================
# Module-level JIT-compiled functions for sparse matrix operations
# These are defined at module level to avoid capturing the BCOO matrix as
# constants in the JIT compilation (which would cause memory issues for large
# sparse matrices). The BCOO matrix is passed as a traced argument instead.
# ============================================================================

@jax.jit
def _sparse_matvec_jit(bcoo_matrix: jax_sparse.BCOO, x: jnp.ndarray) -> jnp.ndarray:
    """JIT-compiled sparse matrix-vector product: A @ x"""
    return bcoo_matrix @ x


@jax.jit  
def _sparse_lmatvec_jit(bcoo_matrix: jax_sparse.BCOO, y: jnp.ndarray) -> jnp.ndarray:
    """JIT-compiled sparse adjoint matrix-vector product: A^H @ y = conj(A^T @ conj(y))"""
    # Compute transpose inside JIT - avoids storing two copies of the matrix
    return jnp.conj(bcoo_matrix.T @ jnp.conj(y))


class SparseMatrixOperator(AOperator):
    """
    Efficient AOperator for sparse matrices using JAX sparse BCOO format.
    
    This class wraps a sparse matrix in JAX BCOO (Block COO) format and provides
    the AOperator interface, enabling efficient matrix-vector operations, row/column
    access, and norm computations.
    
    Features:
    - Efficient matvec/lmatvec operations using JAX sparse primitives
    - Batch processing support
    - Row/column extraction
    - Row norm squared computation
    - Support for conversion from SciPy sparse matrices
    
    NOTE: The matvec/lmatvec operations use module-level JIT functions that take
    the BCOO matrix as a traced argument, avoiding constant capture issues for
    large sparse matrices.
    """
    
    def __init__(self, sparse_matrix, dtype=None):
        """
        Initialize with a sparse matrix.
        
        Args:
            sparse_matrix: Can be:
                - JAX sparse BCOO matrix
                - SciPy sparse matrix (CSR, CSC, COO) - will be converted to BCOO
            dtype: Target dtype for operations (default: inferred from sparse_matrix)
        """
        # Convert SciPy sparse to JAX BCOO if needed
        if hasattr(sparse_matrix, 'toarray'):  # SciPy sparse matrix
            import scipy.sparse as sp
            if isinstance(sparse_matrix, (sp.csr_matrix, sp.csc_matrix, sp.coo_matrix)):
                if dtype is None:
                    dtype = sparse_matrix.dtype
                self.bcoo_matrix = jax_sparse.BCOO.from_scipy_sparse(sparse_matrix)
            else:
                raise ValueError(f"Unsupported SciPy sparse format: {type(sparse_matrix)}")
        elif isinstance(sparse_matrix, jax_sparse.BCOO):
            self.bcoo_matrix = sparse_matrix
            if dtype is None:
                dtype = sparse_matrix.dtype
        else:
            raise ValueError(f"sparse_matrix must be a JAX BCOO or SciPy sparse matrix, got {type(sparse_matrix)}")
        # self.bcoo_matrix = self.bcoo_matrix.astype(dtype)
        self.shape = self.bcoo_matrix.shape
        self.dtype = jnp.dtype(dtype) if dtype is not None else jnp.dtype(self.bcoo_matrix.dtype)
        
        # Store for base class compatibility
        self.A = None  # We don't store dense matrix by default
        
    def matvec(self, x, batch=False):
        """
        Compute A @ x using JAX sparse BCOO matrix-vector product.
        Uses module-level JIT function to avoid capturing matrix as constant.
        
        Args:
            x: Input vector(s). Can be 1D (single vector) or 2D (batch of vectors).
            batch: If True, treat 2D input as batch and process all columns efficiently.
        
        Returns:
            Result of A @ x. Shape (n,) for single vector, (n, batch_size) for batch.
        """
        x = jnp.asarray(x, dtype=self.dtype)
        
        if x.ndim == 1:
            if x.shape[0] != self.shape[1]:
                raise ValueError(f"Input vector must have length {self.shape[1]}, got {x.shape[0]}")
        elif x.ndim == 2:
            if x.shape[0] != self.shape[1]:
                raise ValueError(f"Input first dimension must be {self.shape[1]}, got {x.shape[0]}")
        else:
            raise ValueError("x must be 1D or 2D")
        
        # Use module-level JIT function with BCOO as traced argument
        return _sparse_matvec_jit(self.bcoo_matrix, x)
    
    def lmatvec(self, y, batch=False):
        """
        Compute A^H @ y using JAX sparse BCOO adjoint matrix-vector product.
        Uses module-level JIT function with cached transpose to avoid capturing
        matrix as constant and avoid recomputing transpose each call.
        
        Args:
            y: Input vector(s). Can be 1D (single vector) or 2D (batch of vectors).
            batch: If True, treat 2D input as batch and process all columns efficiently.
        
        Returns:
            Result of A^H @ y. Shape (m,) for single vector, (m, batch_size) for batch.
        """
        y = jnp.asarray(y, dtype=self.dtype)
        
        if y.ndim == 1:
            if y.shape[0] != self.shape[0]:
                raise ValueError(f"Input vector must have length {self.shape[0]}, got {y.shape[0]}")
        elif y.ndim == 2:
            if y.shape[0] != self.shape[0]:
                raise ValueError(f"Input first dimension must be {self.shape[0]}, got {y.shape[0]}")
        else:
            raise ValueError("y must be 1D or 2D")
        
        # Use module-level JIT function with original BCOO as traced argument
        # Transpose is computed inside JIT to avoid storing two copies
        # A^H @ y = conj(A^T @ conj(y))
        return _sparse_lmatvec_jit(self.bcoo_matrix, y)
    
    def get_row(self, i):
        """
        Get row i of the sparse matrix efficiently using direct BCOO access.
        
        Args:
            i: Row index
        
        Returns:
            Row i as a JAX array of shape (m,)
        """
        # Use unit vector approach - BCOO doesn't have direct row access
        e_i = jnp.zeros(self.shape[0], dtype=self.dtype).at[i].set(1.0)
        return self.lmatvec(e_i)
    
    def get_col(self, j):
        """
        Get column j of the sparse matrix efficiently using direct BCOO access.
        
        Args:
            j: Column index
        
        Returns:
            Column j as a JAX array of shape (n,)
        """
        if not (0 <= j < self.shape[1]):
            raise IndexError(f"Column index {j} out of bounds [0, {self.shape[1]})")
        
        # For efficiency, use direct BCOO indexing
        # BCOO doesn't have direct column access, so we use the unit vector approach
        e_j = jnp.zeros(self.shape[1], dtype=self.dtype).at[j].set(1.0)
        return self.matvec(e_j)
    
    def compute_all_row_norms_squared(self):
        """
        Compute all row norms ||A[i, :]||^2 efficiently using BCOO structure.
        
        Returns:
            Array of shape (n,) containing row norm squares
        """
        # For sparse matrices, we compute row norms by:
        # ||A[i, :]||^2 = sum(|A[i, j]|^2) for all j
        # We do this efficiently using the BCOO structure with scatter_add
        
        # Get the data and indices from BCOO
        data = self.bcoo_matrix.data
        indices = self.bcoo_matrix.indices
        
        # Compute |data|^2
        data_sq = jnp.abs(data) ** 2
        
        # Flatten data_sq if it has extra dimensions (BCOO can have block structure)
        if data_sq.ndim > 1:
            # For block BCOO, sum over the block dimensions
            data_sq = jnp.sum(data_sq, axis=tuple(range(1, data_sq.ndim)))
        
        # Handle different BCOO index structures
        if indices.ndim == 2 and indices.shape[1] == 2:
            # Standard 2D case: indices is (nse, 2)
            row_indices = indices[:, 0]
            
            # Use jax.lax.scatter_add for efficient accumulation (most efficient method)
            from jax import lax
            # Use the same dtype as data_sq to avoid dtype mismatches
            row_norms_sq = jnp.zeros(self.shape[0], dtype=data_sq.dtype)
            row_norms_sq = lax.scatter_add(
                row_norms_sq,
                row_indices[:, None],
                data_sq,
                lax.ScatterDimensionNumbers(
                    update_window_dims=(),
                    inserted_window_dims=(0,),
                    scatter_dims_to_operand_dims=(0,)
                )
            )
            return row_norms_sq
        else:
            # Fallback: convert to dense and compute (works for any BCOO structure)
            # This is less efficient but correct for complex BCOO structures
            dense_A = self.bcoo_matrix.todense()
            return jnp.sum(jnp.abs(dense_A) ** 2, axis=1)
    
    def __matmul__(self, other):
        """Support matrix multiplication with @ operator."""
        if not (isinstance(other, np.ndarray) or isinstance(other, jnp.ndarray)):
            return NotImplemented
        return self.matvec(other, batch=(other.ndim == 2))
    
    def __rmatmul__(self, other):
        """Support right matrix multiplication with @ operator."""
        if not (isinstance(other, np.ndarray) or isinstance(other, jnp.ndarray)):
            return NotImplemented
        return self.lmatvec(other, batch=(other.ndim == 2))
    
    @property
    def T(self):
        """Transpose property for matrix operations."""
        return SparseMatrixOperator(self.bcoo_matrix.transpose(), dtype=self.dtype)


# class SparseMatrixOperatorTranspose:
#     """Transpose wrapper for SparseMatrixOperator to support .T @ operations."""
    
#     def __init__(self, sparse_op):
#         self.sparse_op = sparse_op
#         self.shape = (sparse_op.shape[1], sparse_op.shape[0])  # Transposed shape
#         self.dtype = sparse_op.dtype
    
#     def matvec(self, x, batch=False):
#         """Support matrix multiplication with @ operator for transpose."""
#         return self.sparse_op.lmatvec(x, batch=batch)
    
#     def lmatvec(self, y, batch=False):
#         """Support right matrix multiplication with @ operator for transpose."""
#         return self.sparse_op.matvec(y, batch=batch)
    
#     def compute_all_row_norms_squared(self):
#         """
#         Compute row norms of A^T (column norms of A).
#         Column norms of A = row norms of A^T.
#         """
#         # Column norms = row norms of transpose
#         # Use BCOO structure: ||A[:, j]||^2 = sum(|A[i, j]|^2) for all i
#         data = self.sparse_op.bcoo_matrix.data
#         indices = self.sparse_op.bcoo_matrix.indices
#         data_sq = jnp.abs(data) ** 2
#         if data_sq.ndim > 1:
#             data_sq = jnp.sum(data_sq, axis=tuple(range(1, data_sq.ndim)))
        
#         if indices.ndim == 2 and indices.shape[1] == 2:
#             from jax import lax
#             col_indices = indices[:, 1]
#             col_norms_sq = jnp.zeros(self.sparse_op.shape[1], dtype=data_sq.dtype)
#             col_norms_sq = lax.scatter_add(
#                 col_norms_sq,
#                 col_indices[:, None],
#                 data_sq,
#                 lax.ScatterDimensionNumbers(
#                     update_window_dims=(),
#                     inserted_window_dims=(0,),
#                     scatter_dims_to_operand_dims=(0,)
#                 )
#             )
#             return col_norms_sq
#         else:
#             dense_A = self.sparse_op.bcoo_matrix.todense()
#             return jnp.sum(jnp.abs(dense_A) ** 2, axis=0)
    
#     def __matmul__(self, other):
#         """Support matrix multiplication with @ operator."""
#         if not (isinstance(other, np.ndarray) or isinstance(other, jnp.ndarray)):
#             return NotImplemented
#         return self.matvec(other, batch=(other.ndim == 2))
    
#     def __rmatmul__(self, other):
#         """Support right matrix multiplication with @ operator."""
#         if not (isinstance(other, np.ndarray) or isinstance(other, jnp.ndarray)):
#             return NotImplemented
#         return self.lmatvec(other, batch=(other.ndim == 2))
    
#     @property
#     def T(self):
#         """Double transpose returns original operator."""
#         return self.sparse_op
