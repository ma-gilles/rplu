"""
CUR-type sparse approximation using pivoted QR.

This module implements the algorithm from Section 4 of:
G.W. Stewart, "Four algorithms for the efficient computation of truncated 
pivoted QR approximations to a sparse matrix", Numer. Math. 83, 313-323 (1999).

The algorithm produces an approximation of the form Y T Z^T where:
- Y consists of selected columns of X (from pivoted QR on A)
- Z consists of selected rows of X (from pivoted QR on A^T)
- T is a core matrix computed using matvecs

Algorithm:
1. Compute pivoted QR on A to select k columns J
2. Compute pivoted QR on A^T to select k rows I (columns of A^T)
3. Compute T = (Y^T Y)^{-1} Y^T X Z^T (Z Z^T)^{-1} using matvecs only
   where Y = A[:, J], Z = A[I, :]
"""

import jax
import jax.numpy as jnp
from matrix_classes import AOperator
from pivoted_qr import pivoted_qr
import time
from typing import Any, Optional, Tuple, Union


class AOperatorTranspose(AOperator):
    """
    Transpose wrapper for AOperator.
    Allows running pivoted QR on A^T by swapping matvec/lmatvec and get_row/get_col.
    """
    def __init__(self, Aop: AOperator):
        """
        Initialize transpose operator.
        
        Parameters
        ----------
        Aop : AOperator
            Original operator with shape (n, p)
        """
        # Store original array if available (for operators like DenseOperator)
        # For operators like ConvNDOperator, this will be None
        self.A = getattr(Aop, 'A', None)
        self.Aop = Aop  # Store original operator
        # Transpose shape: (p, n)
        self.shape = (Aop.shape[1], Aop.shape[0])
        self.dtype = Aop.dtype
        self.n, self.p = self.shape
    
    def get_row(self, i):
        """Row i of A^T is column i of A"""
        return self.Aop.get_col(i)
    
    def get_col(self, j):
        """Column j of A^T is row j of A"""
        return self.Aop.get_row(j)
    
    def matvec(self, x):
        """A^T @ x = A^H @ x (conjugate transpose)"""
        return self.Aop.lmatvec(x)
    
    def lmatvec(self, y):
        """A^T^H @ y = A @ y"""
        return self.Aop.matvec(y)
    
    def matvec_with_R(self, v, I):
        """A^T[I, :] @ v = A[:, I]^T @ v"""
        return self.Aop.lmatvec_with_C(v, I)
    
    def lmatvec_with_C(self, v, J):
        """A^T[:, J]^T @ v = A[J, :] @ v"""
        return self.Aop.matvec_with_R(v, J)
    
    def matvec_with_C(self, v, J):
        """A^T[:, J] @ v = A[J, :]^T @ v"""
        return self.Aop.lmatvec_with_R(v, J)
    
    def lmatvec_with_R(self, v, I):
        """A^T[I, :]^T @ v = A[:, I] @ v"""
        return self.Aop.matvec_with_C(v, I)
    
    def compute_all_row_norms_squared(self):
        """Row norms of A^T are column norms of A"""
        # This should be provided, but if not, we can compute from columns
        # For now, raise error - should be provided
        raise NotImplementedError("Row norms of A^T should be provided as column norms of A")


class PivotedQRCUR:
    """
    CUR-type approximation using pivoted QR.
    
    Computes an approximation X ≈ Y T Z^T where:
    - Y = X[:, J] (selected columns)
    - Z = X[I, :] (selected rows)
    - T is a core matrix (k x k)
    
    Uses pivoted QR to select columns, then selects rows and computes T.
    """
    
    def __init__(self, Aop: AOperator, col_norms_squared: jnp.ndarray,
                 row_norms_squared: Optional[jnp.ndarray] = None,
                 max_rank: Optional[int] = None, tol: Optional[float] = None,
                 debug: bool = False, profile: bool = False, use_jit: bool = True,
                 random_seed: Optional[int] = None,
                 use_svd_for_T: bool = True, compute_T_via_pinv: bool = False,
                 column_sampling: str = 'greedy',
                 pivot_sampling: Optional[str] = None,  # Alias for column_sampling (for API consistency with CURIncremental)
                 store_preconditioner_matrices: bool = False,
                 M_inv_matvec: Optional[callable] = None):
        """
        Initialize pivoted QR-based CUR approximation.
        
        Parameters
        ----------
        Aop : AOperator
            The matrix operator (n x p)
        col_norms_squared : jnp.ndarray, shape (p,)
            Squared column norms ||A[:, j]||^2 (assumed to be provided for free)
        row_norms_squared : jnp.ndarray, optional, shape (n,)
            Squared row norms ||A[i, :]||^2 (computed if not provided)
        max_rank : int, optional
            Maximum rank to compute (default: min(n, p))
        tol : float, optional
            Tolerance for early termination in QR step
        debug : bool
            Enable debug output
        profile : bool
            Enable profiling
        use_jit : bool
            Use JIT-compiled functions
        random_seed : int, optional
            Random seed for reproducibility when column_sampling is 'random' or 'uniform'. Default: None.
        use_svd_for_T : bool, optional
            If True, use SVD to compute the core matrix T (more numerically stable).
            If False, use triangular solves (faster but may be less stable). Default: True.
            Only used if compute_T_via_pinv=False.
        compute_T_via_pinv : bool, optional
            If True, compute T = pinv(A[I, J]) directly (similar to CUR.py).
            If False, use normal equations method (via R factors). Default: False.
        column_sampling : str, optional
            Column/pivot sampling mode: 'greedy' (deterministic, argmax), 
            'random' (probability proportional to norm squared), 
            or 'uniform' (uniform random sampling from remaining columns). Default: 'greedy'.
        pivot_sampling : str, optional
            Alias for column_sampling (for API consistency with CURIncremental).
            If provided, overrides column_sampling.
        store_preconditioner_matrices : bool, optional
            If True, precompute and store matrices needed for Woodbury preconditioner:
            W_squared_core = R @ D^{-1} @ C = A[I, :] @ D^{-1} @ A[:, J]
            and the Schur complement (T^{-1} + W_squared_core).
            This allows efficient Woodbury solver via get_woodbury_solver(). Default: False.
        M_inv_matvec : callable, optional
            Function to apply D^{-1} (diagonal preconditioner inverse).
            Required if store_preconditioner_matrices=True.
            If None and store_preconditioner_matrices=True, uses identity.
        """
        self.A = Aop
        n, p = Aop.shape
        self.n, self.p = n, p
        
        if col_norms_squared.shape != (p,):
            raise ValueError(f"col_norms_squared must have shape ({p},), got {col_norms_squared.shape}")
        
        self.max_rank = max_rank if max_rank is not None else min(n, p)
        self.max_rank = min(self.max_rank, n, p)
        
        self.debug = debug
        self.profile = profile
        self.use_jit = use_jit
        self.random_seed = random_seed
        self.use_svd_for_T = use_svd_for_T
        self.compute_T_via_pinv = compute_T_via_pinv
        # pivot_sampling/column_sampling: 'greedy', 'random', or 'uniform'
        # pivot_sampling is the unified name (for API consistency with CURIncremental)
        self.pivot_sampling = pivot_sampling if pivot_sampling is not None else column_sampling
        self.column_sampling = self.pivot_sampling  # Alias for backward compatibility
        
        # Track when all norms become negative (for API consistency with CURIncremental)
        self.iter_when_all_norms_are_negative = -1
        
        # Preconditioner matrices for Woodbury solver
        self.store_preconditioner_matrices = store_preconditioner_matrices
        self.M_inv_matvec = M_inv_matvec if M_inv_matvec is not None else (lambda x: x)
        
        # Disable JIT if debug or profile
        if debug and use_jit:
            if self.debug:
                print("DEBUG: use_jit set to False because debug is True")
            use_jit = False
        if profile and use_jit:
            if self.debug:
                print("DEBUG: use_jit set to False because profile is True")
            use_jit = False
        self.use_jit = use_jit
        
        # Column norms (provided)
        # Preserve dtype from AOperator to maintain precision (use A.dtype if input is compatible)
        target_dtype = self.A.dtype
        self.col_norms_sq = jnp.asarray(col_norms_squared, dtype=target_dtype)
        
        # Row norms (compute if not provided)
        if row_norms_squared is None:
            if self.debug:
                print("Computing row norms...")
            row_norms_squared = self.A.compute_all_row_norms_squared()
        self.row_norms_sq = jnp.asarray(row_norms_squared, dtype=target_dtype)
        
        # Set tolerance for QR
        if tol is None:
            eps = jnp.finfo(self.A.dtype).eps
            max_initial_norm_sq = jnp.max(self.col_norms_sq)
            self.tol = eps * max_initial_norm_sq
        else:
            max_initial_norm_sq = jnp.max(self.col_norms_sq)
            self.tol = tol * max_initial_norm_sq
        
        # Pre-allocate arrays (fixed size to avoid recompilation)
        fixed_capacity = max(128, self.max_rank)
        self.array_capacity = fixed_capacity
        
        # Column indices (selected by QR)
        self.J_array = jnp.ones(fixed_capacity, dtype=jnp.int64) * (p + 1)
        
        # Row indices (selected after columns)
        self.I_array = jnp.ones(fixed_capacity, dtype=jnp.int64) * (n + 1)
        
        # Core matrix T (k x k)
        self.T_core = jnp.zeros((fixed_capacity, fixed_capacity), dtype=self.A.dtype)
        
        # SVD factors for T (for thresholded representation)
        self.T_u = jnp.zeros((fixed_capacity, fixed_capacity), dtype=self.A.dtype)
        self.T_s = jnp.zeros(fixed_capacity, dtype=self.A.dtype)
        self.T_vt = jnp.zeros((fixed_capacity, fixed_capacity), dtype=self.A.dtype)
        self.T_use_svd = False  # Whether to use SVD factors instead of T_core
        
        # Preconditioner matrices (for Woodbury solver)
        # W_squared_core = R @ D^{-1} @ C = A[I, :] @ D^{-1} @ A[:, J]
        self.W_squared_core = None
        # Factors for direct T^{-1} computation: T = X^{-1} @ V @ Y^{-1}
        # where X = R_cols^T @ R_cols, Y = R_rows^T @ R_rows, V = YtXZt
        # So T^{-1} = Y @ V^{-1} @ X
        self.R_cols = None  # Upper triangular from QR on A
        self.R_rows = None  # Upper triangular from QR on A^T
        self.YtXZt = None   # Middle matrix V = Y^T X Z^T
        # Schur complement SVD factors: (T^{-1} + W_squared_core) = schur_U @ diag(schur_s) @ schur_Vh
        self.schur_U = None
        self.schur_s = None
        self.schur_inv_s = None
        self.schur_Vh = None
        self._woodbury_ready = False  # Whether preconditioner matrices are precomputed
        
        # Current rank
        self.k = 0
        
        # QR result
        self.qr_result = None
        self.qr_result_T = None  # QR result for A^T (for accessing R_rows)
    
    def build(self, rank: Optional[int] = None, timing: bool = False, 
              use_svd_for_T: Optional[bool] = None,
              compute_T_via_pinv: Optional[bool] = None) -> dict:
        """
        Build the CUR approximation.
        
        Parameters
        ----------
        rank : int, optional
            Target rank (default: self.max_rank)
        timing : bool
            Return timing information
        use_svd_for_T : bool, optional
            If provided, override self.use_svd_for_T for this build call.
            If None, use self.use_svd_for_T. Default: None.
            Only used if compute_T_via_pinv=False.
        compute_T_via_pinv : bool, optional
            If provided, override self.compute_T_via_pinv for this build call.
            If None, use self.compute_T_via_pinv. Default: None.
            
        Returns
        -------
        dict
            Dictionary with keys:
            - 'I': row indices (k,)
            - 'J': column indices (k,)
            - 'T': core matrix (k x k)
            - 'k': actual rank achieved
            - 'total_time': total time
            - 'timings': (optional) timing breakdown
        """
        if rank is None:
            rank = self.max_rank
        else:
            rank = min(rank, self.max_rank)
        
        if timing:
            start_time = time.time()
            timings = {}
        
        # Step 1: Compute pivoted QR on A to select columns
        if self.debug:
            print(f"Step 1: Computing pivoted QR on A (rank={rank})...")
        
        self.qr_result = pivoted_qr(
            self.A, 
            self.col_norms_sq,
            rank=rank,
            tol=self.tol,
            debug=self.debug,
            profile=self.profile,
            use_jit=self.use_jit,
            return_timings=timing,
            random_seed=self.random_seed,
            column_sampling=self.column_sampling
        )
        
        # Extract column selection from QR
        # QR result now returns 'I' (chosen column indices) instead of 'P'
        I_cols = self.qr_result['I']  # Chosen column indices
        k = self.qr_result['k']
        R_cols = self.qr_result['R']  # k x k upper triangular
        
        # Store R_cols for direct T^{-1} computation (if preconditioner is enabled)
        if self.store_preconditioner_matrices:
            self.R_cols = R_cols
        
        if self.debug:
            print(f"QR on A selected {k} columns: J = {I_cols[:k]}")
        
        # Store column indices
        self.J_array = I_cols
        self.k = k
        
        # Step 2: Compute pivoted QR on A^T to select rows
        # Row norms of A^T are column norms of A (already computed as row_norms_sq)
        if self.debug:
            print(f"Step 2: Computing pivoted QR on A^T (rank={k})...")
        
        # Create transpose operator
        A_T = AOperatorTranspose(self.A)
        
        # Check if A^T satisfies n >= p requirement for pivoted QR
        # A^T has shape (p, n), so we need p >= n
        # If not, we can't use pivoted QR directly, so we'll select rows based on norms
        qr_result_T = pivoted_qr(
            A_T,
            self.row_norms_sq,  # Column norms of A^T = row norms of A
            rank=k,
            tol=self.tol,
            debug=self.debug,
            profile=self.profile,
            use_jit=self.use_jit,
            return_timings=timing,
            random_seed=self.random_seed,
            column_sampling=self.column_sampling
        )
        
        # Extract row selection from QR on A^T
        # QR result now returns 'I' (chosen column indices) instead of 'P'
        I_rows = qr_result_T['I']  # These are row indices of A (columns of A^T)
        R_rows = qr_result_T['R']  # k x k upper triangular
        
        # Store R_rows for direct T^{-1} computation (if preconditioner is enabled)
        if self.store_preconditioner_matrices:
            self.R_rows = R_rows
        
        # Store qr_result_T for later access
        self.qr_result_T = qr_result_T
        
        # Update iter_when_all_norms_are_negative from QR results
        # Use the minimum (earliest occurrence) from columns and rows QR
        qr_iter_neg = self.qr_result.get('iter_when_all_norms_are_negative', -1)
        qr_T_iter_neg = qr_result_T.get('iter_when_all_norms_are_negative', -1)
        if qr_iter_neg >= 0:
            self.iter_when_all_norms_are_negative = qr_iter_neg
        if qr_T_iter_neg >= 0:
            if self.iter_when_all_norms_are_negative < 0 or qr_T_iter_neg < self.iter_when_all_norms_are_negative:
                self.iter_when_all_norms_are_negative = qr_T_iter_neg
        
        if self.debug:
            print(f"Selected {k} rows: I = {I_rows[:k]}")
        
        # Store row indices
        self.I_array = I_rows
        
        # Step 3: Compute core matrix T using matvecs
        # T should satisfy: Y T Z^T ≈ X
        # where Y = A[:, J] = A[:, I_cols[:k]], Z = A[I, :] = A[I_rows[:k], :]
        # Optimal T = (Y^T Y)^{-1} Y^T X Z^T (Z Z^T)^{-1}
        # Since Y^T Y = R_cols^T R_cols and Z Z^T = R_rows^T R_rows (from QR), we can use R directly
        
        if self.debug:
            print(f"Step 3: Computing core matrix T using matvecs...")
        
        # Determine which method to use (allow override in build call)
        use_pinv = compute_T_via_pinv if compute_T_via_pinv is not None else self.compute_T_via_pinv
        use_svd = use_svd_for_T if use_svd_for_T is not None else self.use_svd_for_T
        
        if use_pinv:
            # Method 1: Compute T = pinv(A[I, J]) directly (similar to CUR.py)
            if self.debug:
                print(f"Computing T = pinv(A[I, J]) directly...")
            
            # Compute A[I, J] by extracting rows and columns
            W = jnp.zeros((k, k), dtype=self.A.dtype)
            
            for i, i_idx in enumerate(self.I_array[:k]):
                row_i = self.A.get_row(int(i_idx))  # (p,)
                # Extract columns J from row_i
                W = W.at[i, :].set(row_i[self.J_array[:k]])
            
            # W_squared_core will be computed later in _precompute_woodbury_matrices() if needed
            # No need to compute it in the main loop
            
            # Compute SVD of W and use it directly for T
            # W = U @ diag(s) @ Vh, so pinv(W) = Vh^T @ diag(1/s) @ U^T
            # We can represent T = pinv(W) as: T = uT @ diag(sT) @ vT^T
            # where uT = Vh^T, sT = 1/s (thresholded), vT = U^T
            U, s, Vh = jnp.linalg.svd(W, full_matrices=False)
            eps = jnp.finfo(W.dtype).eps
            rcond = 10.0 * eps * max(W.shape)
            tol = rcond * jnp.max(s)
            # Threshold 1/s (inverse singular values)
            sT_thresholded = jnp.where(s > tol, 1.0 / s, 0.0)
            
            # Store SVD factors for matvec operations
            # T = Vh^T @ diag(1/s) @ U^T, so:
            # uT = Vh^T (right singular vectors of W)
            # sT = 1/s (inverse singular values, thresholded)
            # vT = U^T (left singular vectors of W)
            self.T_u = self.T_u.at[:k, :k].set(Vh.conj().T)
            self.T_s = self.T_s.at[:k].set(sT_thresholded)
            self.T_vt = self.T_vt.at[:k, :k].set(U.conj().T)
            self.T_use_svd = True
            
            # Don't store full T when using SVD factors - use factors directly
            # T_core will be reconstructed from SVD factors in result dictionary if needed
        
        else:
            # Method 2: Compute T via normal equations (original method)
            # Compute Y^T X Z^T using matvecs as in the paper (Section 4)
            # Y^T X Z = (A[:, J])^T @ A @ (A[I, :])^T
            YtXZt = jnp.zeros((k, k), dtype=self.A.dtype)
            
            # NOTE: Matvec calls are done OUTSIDE JIT to avoid capturing
            # the entire matrix data as constants. The operator's own matvec methods
            # are already JIT-compiled internally.
            I_to_k = self.I_array[:k]
            J_to_k = self.J_array[:k]
            for i, i_idx in enumerate(I_to_k):  # Iterate over selected rows
                # Get row outside JIT to avoid recompilation
                row_i = self.A.get_row(int(i_idx))  # (p,)
                # Compute matvec OUTSIDE JIT - operator methods are already JIT-compiled
                A_row_i = self.A.matvec(row_i)  # (n,)
                AA_row_i = self.A.lmatvec_with_C(A_row_i, J_to_k)  # (k,)
                YtXZt = YtXZt.at[:, i].set(AA_row_i)
            
            # Store YtXZt for direct T^{-1} computation (if preconditioner is enabled)
            # T = X^{-1} @ YtXZt @ Y^{-1}, so T^{-1} = Y @ pinv(YtXZt) @ X
            if self.store_preconditioner_matrices:
                self.YtXZt = YtXZt
            
            # W_squared_core will be computed later in _precompute_woodbury_matrices() if needed
            # No need to compute it in the main loop

            if use_svd:
                # Step 1: Solve R_cols^T R_cols @ temp = Y^T X Z^T using SVD
                # (R_cols^T R_cols)^{-1} = (R_cols)^{-1} (R_cols^T)^{-1}
                # Use SVD: R_cols = U_cols @ S_cols @ V_cols^T
                # Then (R_cols^T R_cols)^{-1} = V_cols @ diag(1/s_cols^2) @ V_cols^T
                U_cols, s_cols, Vt_cols = jnp.linalg.svd(R_cols, full_matrices=False)
                # Filter small singular values (use sqrt(eps) since we compute 1/s^2)
                eps = jnp.finfo(R_cols.dtype).eps
                rcond_cols = jnp.sqrt(10.0 * eps) * max(R_cols.shape)
                tol_cols = rcond_cols * jnp.max(s_cols)
                inv_s_cols_sq = jnp.where(s_cols > tol_cols, 1.0 / (s_cols ** 2), 0.0)
                # temp = (R_cols^T R_cols)^{-1} @ Y^T X Z^T
                # (R_cols^T R_cols)^{-1} = Vt^T @ diag(1/s^2) @ Vt
                # So: temp = Vt^T @ diag(1/s^2) @ Vt @ YtXZt
                # where diag(1/s^2) @ (Vt @ YtXZt) = (1/s^2)[:, None] * (Vt @ YtXZt)
                temp = Vt_cols.T @ (inv_s_cols_sq[:, None] * (Vt_cols @ YtXZt))
                
                # Step 2: Solve R_rows^T R_rows @ T = temp^T using SVD
                # Use SVD: R_rows = U_rows @ S_rows @ Vt_rows^T
                # Then (R_rows^T R_rows)^{-1} = V_rows @ diag(1/s_rows^2) @ V_rows^T
                U_rows, s_rows, Vt_rows = jnp.linalg.svd(R_rows, full_matrices=False)
                # Filter small singular values (use sqrt(eps) since we compute 1/s^2)
                eps = jnp.finfo(R_rows.dtype).eps
                rcond_rows = jnp.sqrt(10.0 * eps) * max(R_rows.shape)
                tol_rows = rcond_rows * jnp.max(s_rows)
                inv_s_rows_sq = jnp.where(s_rows > tol_rows, 1.0 / (s_rows ** 2), 0.0)
                # T = (R_rows^T R_rows)^{-1} @ temp^T
                # (R_rows^T R_rows)^{-1} = Vt^T @ diag(1/s^2) @ Vt
                # So: T = Vt^T @ diag(1/s^2) @ Vt @ temp^T
                # where diag(1/s^2) @ (Vt @ temp^T) = (1/s^2)[:, None] * (Vt @ temp^T)
                T = Vt_rows.T @ (inv_s_rows_sq[:, None] * (Vt_rows @ temp.T))
                T = T.T  # Transpose back to get correct T

                # SVD Threshold T:
                uT, sT, vT = jnp.linalg.svd(T, full_matrices=False)
                # Threshold singular values (use eps since we're not inverting)
                eps = jnp.finfo(T.dtype).eps
                rcond = 10.0 * eps * max(T.shape)
                tol = rcond * jnp.max(sT)
                sT_thresholded = jnp.where(sT > tol, sT, 0.0)
                
                # Store SVD factors for matvec operations
                self.T_u = self.T_u.at[:k, :k].set(uT)
                self.T_s = self.T_s.at[:k].set(sT_thresholded)
                self.T_vt = self.T_vt.at[:k, :k].set(vT)
                self.T_use_svd = True
                
                # Also store T_core for fallback T^{-1} computation
                # T = uT @ diag(sT_thresholded) @ vT
                T_reconstructed = uT @ (sT_thresholded[:, None] * vT)
                self.T_core = self.T_core.at[:k, :k].set(T_reconstructed)

            else:
                # Step 1: Solve R_cols^T R_cols @ temp = Y^T X Z^T using triangular solves
                # This is: R_cols^T @ (R_cols @ temp) = Y^T X Z^T
                # Solve R_cols @ temp1 = Y^T X Z^T (transpose solve)
                # Then solve R_cols^T @ temp = temp1
                temp1 = jax.scipy.linalg.solve_triangular(R_cols, YtXZt, lower=False, trans=1)  # k x k
                temp = jax.scipy.linalg.solve_triangular(R_cols, temp1, lower=False)  # k x k
                
                # Step 2: Solve R_rows^T R_rows @ T = temp^T using triangular solves
                # This is: R_rows^T @ (R_rows @ T) = temp^T
                # Solve R_rows @ T1 = temp^T (transpose solve)
                # Then solve R_rows^T @ T = T1
                T1 = jax.scipy.linalg.solve_triangular(R_rows, temp.T, lower=False, trans=1)  # k x k
                T = jax.scipy.linalg.solve_triangular(R_rows, T1, lower=False)  # k x k
                T = T.T  # Transpose back to get correct T
                
                # Store T (no SVD thresholding for triangular solve path)
                self.T_core = self.T_core.at[:k, :k].set(T)
                self.T_use_svd = False

                
        # Complete Woodbury precomputation if requested
        # W_squared_core will be computed in _precompute_woodbury_matrices()
        if self.store_preconditioner_matrices:
            self._precompute_woodbury_matrices()
        
        # Prepare result
        # When using SVD factors, don't reconstruct T - use matvec operations instead
        # Only return T if using T_core directly
        result = {
            'I': self.I_array[:k],
            'J': self.J_array[:k],
            'k': k,
        }
        
        # Only include T in result if not using SVD factors
        if not self.T_use_svd:
            result['T'] = self.T_core[:k, :k]
        
        if timing:
            total_time = time.time() - start_time
            timings['total_time'] = total_time
            result['total_time'] = total_time
            result['timings'] = timings
        
        if self.debug:
            total_time = time.time() - start_time
            print(f"CUR approximation complete: k={k}, time={total_time:.3f}s")
        
        return result
    
    def matvec(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Apply the CUR approximation: Y T Z^T @ x
        
        Parameters
        ----------
        x : jnp.ndarray, shape (p,)
            Input vector
            
        Returns
        -------
        jnp.ndarray, shape (n,)
            Output vector
        """
        if self.k == 0:
            raise ValueError("CUR approximation not built yet. Call build() first.")
        
        k = self.k
        J = self.J_array[:k]
        I = self.I_array[:k]
        
        # Compute Z^T @ x
        # Z = A[I, :] is (k, p), so Z^T is (p, k)
        # Z^T @ x = (p, k) @ (p,) = (k,)
        # This is: A[I, :].T @ x = A[I, :] @ x (since we want the result to be (k,))
        # Actually: Z^T @ x where Z^T = (p, k) means we compute each element as sum_j Z^T[j, i] * x[j]
        # But Z^T[j, i] = Z[i, j] = A[I[i], j], so Z^T @ x = A[I, :] @ x
        # Use matvec_with_R to compute A[I, :] @ x
        Zt_x = self.A.matvec_with_R(x, I)  # (k, p) @ (p,) = (k,)
        
        # Compute T @ (Z^T @ x) using SVD factors if available
        if self.T_use_svd:
            # T @ v = uT @ (sT * (vT @ v))
            uT = self.T_u[:k, :k]
            sT = self.T_s[:k]
            vT = self.T_vt[:k, :k]
            T_Zt_x = uT @ (sT * (vT @ Zt_x))
        else:
            T = self.T_core[:k, :k]
            T_Zt_x = T @ Zt_x  # k x 1
        
        # Compute Y @ (T @ Z^T @ x)
        # Y = A[:, J], so Y @ v = A[:, J] @ v
        result = self.A.matvec_with_C(T_Zt_x, J)  # n x 1
        
        return result
    
    def lmatvec(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Apply the CUR approximation from the left: x @ Y T Z^T
        
        Parameters
        ----------
        x : jnp.ndarray, shape (n,)
            Input vector
            
        Returns
        -------
        jnp.ndarray, shape (p,)
            Output vector
        """
        if self.k == 0:
            raise ValueError("CUR approximation not built yet. Call build() first.")
        
        k = self.k
        J = self.J_array[:k]
        I = self.I_array[:k]
        
        # Compute x @ Y
        # Y = A[:, J], so x @ Y = x @ A[:, J] = x^T @ A[:, J]
        # This is: matvec(x) then select columns J
        x_Y = self.A.lmatvec_with_C(x, J)  # k x 1
        
        # Compute (x @ Y) @ T using SVD factors if available
        if self.T_use_svd:
            # T = uT @ diag(sT) @ vT
            # v @ T = v @ (uT @ diag(sT) @ vT) = (v @ uT) @ diag(sT) @ vT
            uT = self.T_u[:k, :k]
            sT = self.T_s[:k]
            vT = self.T_vt[:k, :k]
            # (v @ uT) @ diag(sT) = (v @ uT) * sT (element-wise)
            # Then multiply by vT
            x_Y_T = ((x_Y @ uT) * sT) @ vT  # k x 1
        else:
            T = self.T_core[:k, :k]
            x_Y_T = x_Y @ T  # k x 1
        
        # Compute (x @ Y @ T) @ Z^T
        # Z = A[I, :], so (x @ Y @ T) @ Z^T = (x @ Y @ T) @ A[I, :]^T
        # This is: sum_i (x @ Y @ T)[i] * A[I[i], :]
        # We need to compute this as a linear combination of rows A[I[i], :]
        # Use lmatvec_with_R which expects a vector of length k (number of rows in I)
        result = self.A.lmatvec_with_R(x_Y_T, I)  # p x 1
        
        return result
    
    def _precompute_schur_complement(self):
        """
        Compute the Schur complement (T^{-1} + W_squared_core) and its SVD.
        
        Assumes W_squared_core is already computed and stored in self.W_squared_core.
        This is called from build() after W_squared_core is computed in the main loop.
        
        If R_cols, R_rows, and YtXZt are stored (from store_preconditioner_matrices=True),
        computes T^{-1} directly without going through T:
            T = X^{-1} @ V @ Y^{-1}  where X = R_cols^T R_cols, Y = R_rows^T R_rows, V = YtXZt
            T^{-1} = Y @ V^{-1} @ X
        """
        if self.k == 0:
            raise ValueError("CUR approximation not built yet. Call build() first.")
        if self.W_squared_core is None:
            raise ValueError("W_squared_core not computed. Call _precompute_woodbury_matrices() instead.")
        
        k = self.k
        
        if self.debug:
            print(f"Computing Schur complement and SVD (k={k})...")
        
        # Compute T^{-1} using T_use_svd method (FORCED - other methods disabled)
        # T_use_svd gives best Woodbury solver performance for ill-conditioned CUR
        # Other methods (YtXZt, T_core_direct) are disabled for now
        
        # Method 1: T_use_svd (ONLY method enabled)
        if self.T_use_svd:
            # Fallback: T = T_u @ diag(T_s) @ T_vt, so T^{-1} = T_vt^H @ diag(1/T_s) @ T_u^H
            # NOTE: When compute_T_via_pinv=True, T_s stores 1/s (inverse singular values)
            #       When use_svd_for_T=True, T_s stores s (singular values)
            #       We need to handle both cases correctly.
            uT = self.T_u[:k, :k]
            sT = self.T_s[:k]
            vT = self.T_vt[:k, :k]
            
            # Check if T_s contains singular values or inverse singular values
            # If compute_T_via_pinv=True, T_s = 1/s, so we need to invert again
            # If use_svd_for_T=True, T_s = s, so we can use directly
            # We can detect this by checking if T_core is available and comparing
            if self.T_core is not None and not jnp.allclose(self.T_core[:k, :k], 0):
                # T_core is available, use it directly (more accurate)
                T = self.T_core[:k, :k]
                U, s, Vh = jnp.linalg.svd(T, full_matrices=False)
                eps = jnp.finfo(T.dtype).eps
                tol = 10.0 * eps * max(T.shape) * jnp.max(s)
                inv_s = jnp.where(s > tol, 1.0 / s, 0.0)
                T_inv = Vh.conj().T @ (inv_s[:, None] * U.conj().T)
            else:
                # T_core not available, use SVD factors
                # For compute_T_via_pinv=True: T_s = 1/s, so T^{-1} = U @ diag(s) @ Vh = T_vt^H @ diag(1/T_s) @ T_u^H
                # For use_svd_for_T=True: T_s = s, so T^{-1} = Vh^H @ diag(1/s) @ U^H = vT^H @ diag(1/sT) @ uT^H
                eps = jnp.finfo(sT.dtype).eps
                tol = 10.0 * eps * k * jnp.max(sT)
                inv_sT = jnp.where(sT > tol, 1.0 / sT, 0.0)
                T_inv = vT.conj().T @ (inv_sT[:, None] * uT.conj().T)
        else:
            # T_use_svd is required - other methods are disabled
            raise ValueError("T_use_svd is required but not available. "
                           "Please use compute_T_via_pinv=True to enable T_use_svd method.")
            # Should not reach here if T was computed correctly
            raise ValueError("Cannot compute T^{-1}: missing required matrices")
        
        # Schur complement: T^{-1} + W_squared_core
        schur = T_inv + self.W_squared_core
        
        # Precompute SVD of Schur complement for stable solves
        U, s, Vh = jnp.linalg.svd(schur, full_matrices=False)
        eps = jnp.finfo(schur.dtype).eps
        tol = 10.0 * eps * k * jnp.max(s)
        inv_s = jnp.where(s > tol, 1.0 / s, 0.0)
        
        self.schur_U = U
        self.schur_s = s
        self.schur_inv_s = inv_s
        self.schur_Vh = Vh
        self._woodbury_ready = True
        
        if self.debug:
            print(f"Schur complement precomputed.")
    
    def _compute_W_squared_core(self, I, J):
        """
        Compute W_squared_core = R @ D^{-1} @ C = A[I, :] @ D^{-1} @ A[:, J].
        
        This is a helper method to avoid code duplication.
        
        Parameters
        ----------
        I : jnp.ndarray, shape (k,)
            Selected row indices
        J : jnp.ndarray, shape (k,)
            Selected column indices
            
        Returns
        -------
        jnp.ndarray, shape (k, k)
            W_squared_core matrix
        """
        k = len(I)
        W_squared_core = jnp.zeros((k, k), dtype=self.A.dtype)
        for j_idx in range(k):
            col_j = self.A.get_col(int(J[j_idx]))
            Dinv_col_j = self.M_inv_matvec(col_j)
            W_squared_core = W_squared_core.at[:, j_idx].set(
                self.A.matvec_with_R(Dinv_col_j, I)
            )
        return W_squared_core
    
    def _precompute_woodbury_matrices(self):
        """
        Precompute matrices needed for Woodbury solver:
        - W_squared_core = R @ D^{-1} @ C = A[I, :] @ D^{-1} @ A[:, J]
        - Schur complement = T^{-1} + W_squared_core
        - SVD of Schur complement for stable solves
        
        This is called automatically in build() if store_preconditioner_matrices=True,
        or can be called manually with prepare_woodbury_solver().
        
        Note: If build() was called with store_preconditioner_matrices=True, W_squared_core
        is already computed and this method only needs to compute the Schur complement.
        """
        if self.k == 0:
            raise ValueError("CUR approximation not built yet. Call build() first.")
        
        k = self.k
        
        # Check if W_squared_core was already computed in build()
        if self.W_squared_core is not None:
            if self.debug:
                print(f"W_squared_core already computed, just computing Schur complement...")
            self._precompute_schur_complement()
            return
        
        # W_squared_core not yet computed - compute it now
        I = self.I_array[:k]
        J = self.J_array[:k]
        
        if self.debug:
            print(f"Precomputing Woodbury matrices (k={k})...")
        
        # Compute W_squared_core = R @ D^{-1} @ C = A[I, :] @ D^{-1} @ A[:, J]
        # This is a k x k matrix
        self.W_squared_core = self._compute_W_squared_core(I, J)
        
        # Now compute Schur complement
        self._precompute_schur_complement()
    
    def prepare_woodbury_solver(self, M_inv_matvec: Optional[callable] = None):
        """
        Prepare the Woodbury solver by precomputing necessary matrices.
        
        Call this after build() if you didn't set store_preconditioner_matrices=True
        in __init__, or if you want to use a different M_inv_matvec.
        
        Parameters
        ----------
        M_inv_matvec : callable, optional
            Function to apply D^{-1} (diagonal preconditioner inverse).
            If None, uses the M_inv_matvec from __init__ (or identity if not set).
        """
        if M_inv_matvec is not None:
            self.M_inv_matvec = M_inv_matvec
        self._precompute_woodbury_matrices()
    
    def _solve_schur(self, b: jnp.ndarray) -> jnp.ndarray:
        """Solve (T^{-1} + W_squared_core) x = b using precomputed SVD."""
        return self.schur_Vh.conj().T @ (self.schur_inv_s * (self.schur_U.conj().T @ b))
    
    def get_woodbury_solver(self, M_inv_matvec: Optional[callable] = None):
        """
        Get a Woodbury solver object for use with GMRES or other iterative solvers.
        
        Parameters
        ----------
        M_inv_matvec : callable, optional
            Function to apply D^{-1}. If None, uses precomputed M_inv_matvec.
            
        Returns
        -------
        PivotedQRCURWoodburySolver
            Solver object with solve(b) method
        """
        if M_inv_matvec is not None and M_inv_matvec != self.M_inv_matvec:
            # Need to recompute with new M_inv_matvec
            self.M_inv_matvec = M_inv_matvec
            self._precompute_woodbury_matrices()
        elif not self._woodbury_ready:
            self._precompute_woodbury_matrices()
        
        return PivotedQRCURWoodburySolver(self)
    
    def to_unified(self):
        """
        Convert to CUR wrapper for consistent interface.
        
        Returns
        -------
        CUR
            CUR wrapper
        """
        from unified_cur import CUR
        return CUR(self, implementation_type='pivoted_qr')


class PivotedQRCURWoodburySolver:
    """
    Woodbury solver wrapper for PivotedQRCUR.
    
    This class wraps a PivotedQRCUR object and provides a solve(b) method
    compatible with JAX GMRES and other iterative solvers.
    
    The solver uses precomputed matrices from PivotedQRCUR for efficiency.
    All heavy computation is done during initialization (in PivotedQRCUR),
    so solve() is fast and JAX-tracing compatible.
    """
    
    def __init__(self, pivoted_qr_cur_obj: PivotedQRCUR):
        """
        Initialize the Woodbury solver.
        
        Parameters
        ----------
        pivoted_qr_cur_obj : PivotedQRCUR
            The PivotedQRCUR object with precomputed Woodbury matrices.
            Should have _woodbury_ready=True (call prepare_woodbury_solver() if not).
        """
        self.cur = pivoted_qr_cur_obj
        
        if not pivoted_qr_cur_obj._woodbury_ready:
            pivoted_qr_cur_obj._precompute_woodbury_matrices()
        
        # Cache references to precomputed matrices for fast access
        self._k = int(pivoted_qr_cur_obj.k)
        self._I = pivoted_qr_cur_obj.I_array[:self._k]
        self._J = pivoted_qr_cur_obj.J_array[:self._k]
        self._M_inv_matvec = pivoted_qr_cur_obj.M_inv_matvec
        self._schur_U = pivoted_qr_cur_obj.schur_U
        self._schur_inv_s = pivoted_qr_cur_obj.schur_inv_s
        self._schur_Vh = pivoted_qr_cur_obj.schur_Vh
    
    def solve(self, b: jnp.ndarray) -> jnp.ndarray:
        """
        Solve (D + C T R) x = b using precomputed Woodbury matrices.
        
        Parameters
        ----------
        b : jnp.ndarray, shape (n,)
            Right-hand side vector
            
        Returns
        -------
        jnp.ndarray, shape (n,)
            Solution vector x
        """
        # Step 1: D^{-1} b
        Dinv_b = self._M_inv_matvec(b)
        
        # Step 2: R D^{-1} b = A[I, :] @ (D^{-1} b)
        R_Dinv_b = self.cur.A.matvec_with_R(Dinv_b, self._I)
        
        # Step 3: Solve (T^{-1} + R D^{-1} C) t = R D^{-1} b using precomputed SVD
        t = self._schur_Vh.conj().T @ (self._schur_inv_s * (self._schur_U.conj().T @ R_Dinv_b))
        
        # Step 4: C @ t = A[:, J] @ t
        C_t = self.cur.A.matvec_with_C(t, self._J)
        
        # Step 5: Final result
        x = Dinv_b - self._M_inv_matvec(C_t)
        return x


def pivoted_qr_cur(Aop: AOperator, col_norms_squared: jnp.ndarray,
                   row_norms_squared: Optional[jnp.ndarray] = None,
                   rank: Optional[int] = None, tol: Optional[float] = None,
                   debug: bool = False, profile: bool = False,
                   use_jit: bool = True, return_timings: bool = False,
                   return_unified: bool = False, use_svd_for_T: bool = True,
                   compute_T_via_pinv: bool = False) -> Union[dict, 'CUR']:
    """
    Compute CUR-type approximation using pivoted QR.
    
    Parameters
    ----------
    Aop : AOperator
        Matrix operator (n x p)
    col_norms_squared : jnp.ndarray, shape (p,)
        Squared column norms ||A[:, j]||^2 (assumed to be provided for free)
    row_norms_squared : jnp.ndarray, optional, shape (n,)
        Squared row norms ||A[i, :]||^2 (computed if not provided)
    rank : int, optional
        Maximum rank to compute (default: min(n, p))
    tol : float, optional
        Tolerance for early termination in QR step
    debug : bool
        Enable debug output
    profile : bool
        Enable profiling
    use_jit : bool
        Use JIT-compiled functions
    return_timings : bool
        Return timing information
    return_unified : bool
        If True, return CUR object instead of dict
        
    Returns
    -------
    dict or CUR
        If return_unified=False: Dictionary with 'I', 'J', 'T', 'k', and optionally 'timings'
        If return_unified=True: CUR object
    """
    cur = PivotedQRCUR(Aop, col_norms_squared, row_norms_squared=row_norms_squared,
                       max_rank=rank, tol=tol, debug=debug, profile=profile, use_jit=use_jit,
                       use_svd_for_T=use_svd_for_T, compute_T_via_pinv=compute_T_via_pinv)
    result = cur.build(rank=rank, timing=return_timings)
    
    if return_unified:
        return cur.to_unified()
    else:
        return result

