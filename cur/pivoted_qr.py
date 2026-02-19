"""
Q-less truncated pivoted QR factorization for sparse matrices.

This module implements the algorithm from:
G.W. Stewart, "Four algorithms for the efficient computation of truncated 
pivoted QR approximations to a sparse matrix", Numer. Math. 83, 313-323 (1999).

The algorithm computes R^(k)_11 from the truncated pivoted QR decomposition
without storing Q, making it memory-efficient for sparse matrices.
"""

import jax
import jax.numpy as jnp
from jax import lax
from matrix_classes import AOperator
import time
from typing import Optional, Tuple


# JIT-compiled helper functions
@jax.jit
def _solve_triangular_transpose_jit(R_sub: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """
    Solve R^T @ x = b where R is upper triangular.
    Uses jax.scipy.linalg.solve_triangular.
    """
    # R is upper triangular, so R^T is lower triangular
    # Solve R^T @ x = b using trans=1 (which solves A^T @ x = b)
    return jax.scipy.linalg.solve_triangular(R_sub, b, lower=False, trans=1)


@jax.jit
def _solve_triangular_jit(R_sub: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """
    Solve R @ x = b where R is upper triangular.
    Uses jax.scipy.linalg.solve_triangular.
    """
    return jax.scipy.linalg.solve_triangular(R_sub, b, lower=False)


class PivotedQR:
    """
    Q-less truncated pivoted QR factorization.
    
    Computes R from the pivoted QR decomposition X @ P = Q @ R where:
    - X is the input matrix (n x p, n >= p)
    - P is a permutation (tracked, not applied)
    - Q is orthogonal (not stored)
    - R is upper triangular (k x k, where k is the truncation rank) - R^(k)_11 from the algorithm
    
    The algorithm uses column pivoting based on column norms and terminates
    early when the pivot becomes sufficiently small.
    
    Stores:
      - R: upper triangular factor (k x p)
      - P: permutation array - P[i] is the original column index at position i
      - col_norms_sq: squared column norms (updated during factorization)
    """
    
    def __init__(self, Aop: AOperator, col_norms_squared: jnp.ndarray,
                 max_rank: Optional[int] = None, tol: Optional[float] = None,
                 debug: bool = False, profile: bool = False, use_jit: bool = True,
                 random_seed: Optional[int] = None,
                 column_sampling: str = 'greedy'):
        """
        Initialize pivoted QR factorization.
        
        Parameters
        ----------
        Aop : AOperator
            The matrix operator (n x p, where n >= p)
        col_norms_squared : jnp.ndarray, shape (p,)
            Squared column norms ||A[:, j]||^2 (assumed to be provided for free)
        max_rank : int, optional
            Maximum rank to compute (default: p)
        tol : float, optional
            Tolerance for early termination. If the pivot norm squared
            is less than tol * max_initial_norm, stop. Default: machine epsilon.
        debug : bool
            Enable debug output
        profile : bool
            Enable profiling
        use_jit : bool
            Use JIT-compiled step function
        random_seed : int, optional
            Random seed for reproducibility when column_sampling is 'random' or 'uniform'. Default: None.
        column_sampling : str, optional
            Column sampling mode: 'greedy' (deterministic, argmax), 
            'random' (probability proportional to norm squared), 
            or 'uniform' (uniform random sampling from remaining columns). Default: 'greedy'.
        """
        self.A = Aop
        n, p = Aop.shape
        self.n, self.p = n, p

        
        if col_norms_squared.shape != (p,):
            raise ValueError(f"col_norms_squared must have shape ({p},), got {col_norms_squared.shape}")
        
        # Max rank is limited by min(n, p) since we can only compute up to min(n, p) columns
        self.max_rank = max_rank if max_rank is not None else min(n, p)
        self.max_rank = min(self.max_rank, n, p)
        
        self.debug = debug
        self.profile = profile
        self.use_jit = use_jit
        
        # Set column sampling mode (default: greedy)
        # 'greedy': Deterministic selection of column with maximum norm
        # 'random': Randomized selection with probability proportional to norm squared
        # 'uniform': Uniform random sampling from remaining columns
        self.column_sampling = column_sampling
        
        # Initialize random number generator if any random sampling is enabled
        if self.column_sampling in ['random', 'uniform']:
            if random_seed is None:
                random_seed = 42
            self.rng_key = jax.random.PRNGKey(random_seed)
        else:
            self.rng_key = None
        
        # Track when all norms become negative (early termination due to numerical issues)
        self.iter_when_all_norms_are_negative = -1
        
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
        
        # Initialize column norms squared (provided for free)
        # Preserve dtype from AOperator to maintain precision (use A.dtype to preserve 64-bit if available)
        target_dtype = self.A.dtype
        self.col_norms_sq = jnp.asarray(col_norms_squared, dtype=target_dtype)
        
        # Store initial max norm for tolerance check
        self.max_initial_norm_sq = jnp.max(self.col_norms_sq)
        
        # Set tolerance
        # if tol is not None:
        #     print("WARNING: TOL IN PIVOTED QR DOES NOT DO ANYTHING")
        #     print("WARNING: TOL IN PIVOTED QR DOES NOT DO ANYTHING")
        #     print("WARNING: TOL IN PIVOTED QR DOES NOT DO ANYTHING")
        
        # Pre-allocate R matrix (upper triangular, capacity x capacity)
        # R^(k)_11 is k x k, so we only store the leading block
        # Use fixed size to avoid recompilation, starting with 128
        fixed_capacity = 128
        self.R = jnp.identity(fixed_capacity, dtype=self.A.dtype)
        self.array_capacity = fixed_capacity
        
        # Permutation array: P[i] = original column index at position i
        # Initially identity: P[i] = i
        # P is fixed-size (p elements), so no recompilation issues
        self.chosen_columns = jnp.ones(fixed_capacity, dtype=jnp.int64) * (p + 1)
        
        # Current rank (number of columns processed)
        self.k = jnp.int64(0)
        
        self.remaining_cols_mask = jnp.ones(p, dtype=bool)
        
        # Create JIT-compiled step function if use_jit is True
        self._step_jit_fn = None
        # if self.use_jit:
        self._create_step_jit()
    
    def _ensure_capacity(self):
        """Adaptively double capacity if needed"""
        if self.k >= self.array_capacity:
            # Double the capacity
            new_capacity = self.array_capacity * 2
            if self.debug:
                print(f"  [PivotedQR] Increasing capacity from {self.array_capacity} to {new_capacity} (recompilation will occur)")
            
            # Expand chosen_columns array
            self.chosen_columns = jnp.concatenate([
                self.chosen_columns, 
                jnp.ones(new_capacity - self.chosen_columns.shape[0], dtype=jnp.int64) * (self.p + 1)
            ])
            
            # Expand R matrix (pad with identity)
            old_R = self.R
            self.R = jnp.identity(new_capacity, dtype=self.A.dtype)
            # Copy old R into new R (upper triangular block)
            old_size = old_R.shape[0]
            self.R = self.R.at[:old_size, :old_size].set(old_R[:old_size, :old_size])
            
            self.array_capacity = new_capacity
            # Recreate JIT function after capacity expansion since R shape changed
            if self.use_jit:
                self._create_step_jit()
    
    def _create_step_jit(self):
        """
        Create JIT-compiled helper functions for the step, with matvecs OUTSIDE.
        
        This avoids capturing the matrix data as constants in the JIT compilation,
        which prevents memory issues for large sparse matrices.
        
        The step is split into phases with matvecs called between them.
        """
        @jax.jit
        def compute_w(R, v):
            """Compute w from triangular solves."""
            R_col_k = _solve_triangular_transpose_jit(R, v)
            w = _solve_triangular_jit(R, R_col_k)
            return w, R_col_k
        
        @jax.jit
        def compute_w_new(R, v_new):
            """Compute w_new from triangular solves in reorthogonalization."""
            r = _solve_triangular_transpose_jit(R, v_new)
            w_new = _solve_triangular_jit(R, r)
            return w_new, r
        
        @jax.jit  
        def finalize_step(R, k, q, R_col_k, r, r_values, remaining_cols_mask, col_norms_sq):
            """Finalize R and column norms update."""
            # R[0:k, k] = R_col_k + r
            R = R.at[:, k].set(R_col_k + r)
            
            # R[k, k] = ||q||
            R_kk = jnp.linalg.norm(q)
            R = R.at[k, k].set(R_kk)
            
            # Normalize q for returning (used in norm update already)
            q_normalized = q / R_kk
            
            # Update norms
            r_values_masked = jnp.where(remaining_cols_mask, r_values, 0.0)
            col_norms_sq = col_norms_sq - jnp.abs(r_values_masked)**2
            
            return R, col_norms_sq, q_normalized
        
        self._compute_w = compute_w
        self._compute_w_new = compute_w_new
        self._finalize_step = finalize_step

    
    def step(self) -> Tuple[bool, float]:
        """
        Perform one step of the pivoted QR factorization.
        
        Returns
        -------
        bool
            True if factorization should continue, False if early termination
        float
            Pivot norm squared (for monitoring)
        """
        
        # Find pivot column (using current permutation)
        # Among unprocessed columns (positions >= k), find the one with max norm
        # col_norms_sq is in original ordering, so we need to map positions to original indices
        # Use masking to avoid creating new arrays of different sizes
        # Create mask for unprocessed positions (>= k and not in processed_mask)
        
        cols_norms = jnp.where(self.remaining_cols_mask, self.col_norms_sq, -jnp.inf)

        if jnp.max(cols_norms) <= 0 and self.iter_when_all_norms_are_negative == -1:
            self.iter_when_all_norms_are_negative = self.k
            print(f"PivotedQR - ALL COLUMNS NORMS ARE NEGATIVE at step {self.k}!")
            self.column_sampling = 'uniform'
            print(f"PivotedQR - Switching to uniform sampling because all columns norms are negative")
            # Initialize RNG key if not already initialized
            if self.rng_key is None:
                self.rng_key = jax.random.PRNGKey(42)
        
            # return False, 0.0

        if self.column_sampling == 'random':
            # Random sampling: sample column with probability proportional to norm squared
            # Get probabilities for unselected columns
            col_probs = jnp.where(self.remaining_cols_mask, cols_norms, 0.0)
            col_probs = jnp.maximum(col_probs, 0.0)  # Ensure non-negative
            col_probs_sum = jnp.sum(col_probs)
            
            if col_probs_sum > 0:
                col_probs = col_probs / col_probs_sum  # Normalize to probabilities
                
                # Sample column index using JAX random
                self.rng_key, subkey = jax.random.split(self.rng_key)
                pivot_idx = int(jax.random.choice(subkey, self.p, p=col_probs))
            else:
                # Fallback to argmax if all probabilities are zero
                pivot_idx = int(jnp.argmax(cols_norms))
        elif self.column_sampling == 'greedy':
            # Deterministic: select column with maximum norm
            pivot_idx = int(jnp.argmax(cols_norms))
        elif self.column_sampling == 'uniform':
            # Uniform sampling from remaining columns only
            # Use mask-based approach to avoid dynamic array sizes (jnp.where creates variable-size arrays)
            # Create uniform probabilities over remaining columns (fixed-size array)
            uniform_probs = jnp.where(self.remaining_cols_mask, 1.0, 0.0)
            remaining_count = jnp.sum(uniform_probs)
            # Normalize to probabilities
            uniform_probs = uniform_probs / (remaining_count + 1e-10)
            
            # Sample from all columns using masked probabilities (avoids jnp.where + dynamic indexing)
            self.rng_key, subkey = jax.random.split(self.rng_key)
            pivot_idx = int(jax.random.choice(subkey, self.p, p=uniform_probs))

        pivot_norm_sq = cols_norms.at[pivot_idx].get()
        
        
        # Ensure capacity BEFORE writing to arrays at index self.k
        self._ensure_capacity()
        
        # Fetch column (outside JIT)
        col_k = self.A.get_col(pivot_idx)
        
        # ========================================
        # Step computation with matvecs OUTSIDE JIT to avoid capturing matrix as constants
        # ========================================
        
        # Step 1: v = X[:, 0:k]^T @ X[:, k] (MATVEC outside JIT)
        v = self.A.lmatvec_with_C(col_k, self.chosen_columns)
        
        # Steps 2-3: Triangular solves (JIT)
        w, R_col_k = self._compute_w(self.R, v)
        
        # Step 4: prev_cols_w = X[:, 0:k] @ w (MATVEC outside JIT)
        prev_cols_w = self.A.matvec_with_C(w, self.chosen_columns)
        q = col_k - prev_cols_w
        
        # Step 5: v_new = X[:, 0:k]^T @ q (MATVEC outside JIT - reorthogonalization)
        v_new = self.A.lmatvec_with_C(q, self.chosen_columns)
        
        # Steps 6-7: Triangular solves (JIT)
        w_new, r = self._compute_w_new(self.R, v_new)
        
        # Step 8: prev_cols_w_new = X[:, 0:k] @ w_new (MATVEC outside JIT)
        prev_cols_w_new = self.A.matvec_with_C(w_new, self.chosen_columns)
        q = q - prev_cols_w_new
        
        # Normalize q for lmatvec
        q_norm = jnp.linalg.norm(q)
        q_normalized = q / q_norm
        
        # Step 12: r_values = X^T @ q_normalized (MATVEC outside JIT)
        r_values = self.A.lmatvec(q_normalized)
        
        # Steps 9-11, 13: Finalize R and update norms (JIT)
        self.R, self.col_norms_sq, _ = self._finalize_step(
            self.R, self.k, q, R_col_k, r, r_values, 
            self.remaining_cols_mask, self.col_norms_sq
        )

        self.remaining_cols_mask = self.remaining_cols_mask.at[pivot_idx].set(False)
        self.chosen_columns = self.chosen_columns.at[self.k].set(pivot_idx)
        self.k += 1
        
        if self.debug and (self.k % 10 == 0 or self.k == 1):
            if self.k > 0:
                print(f"\nStep {self.k}: pivot_norm_sq={pivot_norm_sq:.2e}, R[{self.k-1},{self.k-1}]={self.R[self.k-1, self.k-1]:.2e}")
            else:
                print(f"\nStep {self.k}: pivot_norm_sq={pivot_norm_sq:.2e}")
        
        return True, pivot_norm_sq
    
    
    def build(self, rank: Optional[int] = None, timing: bool = False) -> dict:
        """
        Build the pivoted QR factorization up to the specified rank.
        
        Parameters
        ----------
        rank : int, optional
            Target rank (default: self.max_rank)
        timing : bool
            Return timing information
            
        Returns
        -------
        dict
            Dictionary with keys:
            - 'R': upper triangular factor (k x p)
            - 'P': permutation indices (p,)
            - 'k': actual rank achieved
            - 'timings': (optional) array of step timings
        """
        if rank is None:
            rank = self.max_rank
        else:
            rank = min(rank, self.max_rank)
        
        start_time = time.time()
        timings = []
        
        for step_idx in range(rank):
            step_start = time.time()
            continue_flag, pivot_norm_sq = self.step()
            
            step_time = time.time() - step_start
            timings.append(step_time)
            
            if not continue_flag:
                if self.debug:
                    print(f"Early termination at step {step_idx + 1}")
                break
        
        total_time = time.time() - start_time
        
        # Trim R to actual size (k x k upper triangular)
        R_final = self.R[:self.k, :self.k]
        
        result = {
            'R': R_final,
            'I': self.chosen_columns[:self.k],
            'k': self.k,
            'total_time': total_time,
            'iter_when_all_norms_are_negative': self.iter_when_all_norms_are_negative
        }
        
        if timing:
            result['timings'] = jnp.array(timings)
            result['avg_step_time'] = jnp.mean(jnp.array(timings)) if timings else 0.0
        
        if self.debug:
            print(f"QR factorization complete: k={self.k}, time={total_time:.3f}s")
        
        return result
    
def pivoted_qr(Aop: AOperator, col_norms_squared: jnp.ndarray,
                rank: Optional[int] = None, tol: Optional[float] = None,
                debug: bool = False, profile: bool = False,
                use_jit: bool = True, return_timings: bool = False,
                random_seed: Optional[int] = None,
                column_sampling: str = 'greedy') -> dict:
    """
    Compute Q-less truncated pivoted QR factorization.
    
    Parameters
    ----------
    Aop : AOperator
        Matrix operator (n x p, n >= p)
    col_norms_squared : jnp.ndarray, shape (p,)
        Squared column norms ||A[:, j]||^2 (assumed to be provided for free)
    rank : int, optional
        Maximum rank to compute (default: p)
    tol : float, optional
        Tolerance for early termination (default: machine epsilon)
    debug : bool
        Enable debug output
    profile : bool
        Enable profiling
    use_jit : bool
        Use JIT-compiled step function
    return_timings : bool
        Return timing information
    random_seed : int, optional
        Random seed for reproducibility when column_sampling is 'random' or 'uniform'. Default: None.
    column_sampling : str, optional
        Column sampling mode: 'greedy' (deterministic, argmax), 
        'random' (probability proportional to norm squared), 
        or 'uniform' (uniform random sampling from remaining columns). Default: 'greedy'.
        
    Returns
    -------
    dict
        Dictionary with 'R', 'P', 'k', and optionally 'timings'
    """
    qr = PivotedQR(Aop, col_norms_squared, max_rank=rank, tol=tol,
                   debug=debug, profile=profile, use_jit=use_jit,
                   random_seed=random_seed,
                   column_sampling=column_sampling)
    return qr.build(rank=rank, timing=return_timings)

