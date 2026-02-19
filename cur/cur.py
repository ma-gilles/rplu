import jax
import jax.numpy as jnp
try:
    from .matrix_classes import AOperator
except ImportError:
    # Fallback for when cur is not a package
    from matrix_classes import AOperator
import time

# ============================================================================
# Module-level JIT-compiled helper functions for pivot selection
# ============================================================================

@jax.jit
def _select_max_unmasked(values, mask):
    """Select argmax where mask is False."""
    return jnp.argmax(jnp.where(mask, -jnp.inf, values))

@jax.jit
def _mask_selected(values, mask):
    """Set masked entries to -inf."""
    return jnp.where(mask, -jnp.inf, values)

@jax.jit
def _compute_probs_proportional(values, mask):
    """Compute probabilities proportional to values (masked to 0)."""
    probs = jnp.where(mask, 0.0, jnp.maximum(values, 0.0))
    probs_sum = jnp.sum(probs)
    n_unmasked = jnp.sum(~mask)
    return jnp.where(probs_sum > 0, probs / probs_sum, 
                     jnp.where(mask, 0.0, 1.0 / (n_unmasked + 1e-10)))

@jax.jit
def _compute_probs_uniform(mask):
    """Compute uniform probabilities over unmasked entries."""
    probs = jnp.where(mask, 0.0, 1.0)
    return probs / (jnp.sum(probs) + 1e-10)


USE_CPQR = True

# # Auto-detect GPU and MAGMA availability
# def _setup_magma_if_available():
#     """
#     Try to enable MAGMA if available. Returns True if successful.
    
#     This function only uses MAGMA if JAX_GPU_MAGMA_PATH is explicitly set by the user.
#     We do NOT auto-detect conda MAGMA as it has known accuracy issues.
    
#     To use MAGMA, set JAX_GPU_MAGMA_PATH before importing this module:
#         export JAX_GPU_MAGMA_PATH=/path/to/libmagma.so
#     """
#     import os
    
#     # Check if we're on GPU
#     try:
#         backend = jax.default_backend()
#         if backend != 'gpu':
#             return False
#     except:
#         return False
    
#     # Only use MAGMA if explicitly set by user (skip conda MAGMA due to accuracy issues)
#     if 'JAX_GPU_MAGMA_PATH' in os.environ:
#         try:
#             jax.config.update('jax_use_magma', 'on')
#             return True
#         except:
#             return False
    
#     return False

# Try to enable MAGMA at module import
# _MAGMA_AVAILABLE = _setup_magma_if_available()

class CUR_woodbury_solver:
    def __init__(self, CUR, M_inv_matvec=None):
        ## Solves (\alphaI + CUR)x = b using Woodbury:
        # For now take Dinv_op as identity * alpha
        self.CUR = CUR
        self.M_inv_matvec = M_inv_matvec
        # self._precompute_core_inverse()
        self._precompute_core_solver()


    def _precompute_core_solver(self, rcond: float | None = None):
        """
        Precompute a numerically stable right-inverse for
        M = W_core + (1/alpha) W_squared_core via SVD with cutoff.

        Stores U, inv_s, Vh, and M for accurate solves through core_matvec.
        """
        M = self.CUR.W_core + self.CUR.W_squared_core 
        # SVD factorization
        U, s, Vh = jnp.linalg.svd(M, full_matrices=True)
        # Robust cutoff
        if rcond is None:
            # Scale with size and machine epsilon; slightly conservative factor
            eps = jnp.finfo(M.dtype).eps
            rcond = 10.0 * eps * max(M.shape)
        tol = rcond * jnp.max(s)
        inv_s = jnp.where(s > tol, 1.0 / s, 0.0)
        # Cache for fast apply
        self._core_U = U
        self._core_inv_s = inv_s
        self._core_Vh = Vh
        self._core_M = M
        
    def core_matvec(self, b: jnp.ndarray) -> jnp.ndarray:
        """
        Apply the stable pseudoinverse of M to b using SVD:
        x = V diag(inv_s) U^H b. Supports b with shape (k,) or (k, B).
        """
        U = self._core_U
        inv_s = self._core_inv_s
        Vh = self._core_Vh
        if b.ndim == 1:
            return Vh.conj().T @ (inv_s * (U.conj().T @ b))
        if b.ndim == 2:
            return Vh.conj().T @ (inv_s[:, None] * (U.conj().T @ b))
        raise ValueError("b must be 1D or 2D")

    def solve(self, b):
        Dinv_b = self.M_inv_matvec(b) 
        R_Dinv_b = self.CUR.A.matvec_with_R(Dinv_b, self.CUR.I_array)
        # Use stable SVD-based inverse application
        core_inv_R_Dinv_b = self.core_matvec(R_Dinv_b)
        C_core_inv_R_Dinv_b = self.CUR.A.matvec_with_C(core_inv_R_Dinv_b, self.CUR.J_array)
        x = Dinv_b - self.M_inv_matvec(C_core_inv_R_Dinv_b) 
        return x

    
class CURIncremental:
    """
    Incremental CUR using outer-product update:
        C_{k+1}U_{k+1}R_{k+1} = C_k U_k R_k + h lambda^T
    Keeps:
      - I (list of row indices), J (list of col indices)
      - U_core = W^{-1} where W = A[I,J]
      - Exact residual column-norms n_col and row-norms n_row
    Interacts with A only via get_row/get_col/matvec/lmatvec and given row-norms of A.
    """
    def __init__(self, Aop: AOperator, row_norms_squared, debug=False, random_seed=None, store_preconditioner_matrices = False, profile=False, use_jit=True, update_U_option='qr', M_inv_matvec=None, pivot_sampling='greedy', column_sampling=None, auto_convert_to_svd=True):
        self.A = Aop
        n, m = Aop.shape
        self.n, self.m = n, m
        self.M_inv_matvec = M_inv_matvec
        # Allow JIT even with update_U_option; only disable for debug/profile
        if debug or profile:
            use_jit = False

        self.profile = profile  # Enable block_until_ready() for accurate timing
        self.use_jit = use_jit  # Use JIT-compiled step function
        
        # Validate update_U_option
        if update_U_option not in ['inverse', 'svd', 'qr']:
            raise ValueError(f"update_U_option must be 'inverse', 'svd', or 'qr', got '{update_U_option}'")
        self.update_U_option = update_U_option
        self.original_update_U_option = update_U_option  # Store original for reference (before auto conversion)
        self.auto_convert_to_svd = auto_convert_to_svd  # Default: convert QR/inverse to SVD at end for accuracy

        # index sets
        
        # Pre-allocated arrays - use fixed large size to avoid doubling/recompilation
        # For 64^3, ranks up to 2048 should cover most use cases
        fixed_capacity = 128
        # Initialize with out-of-bounds indices (n+1, m+1) so mode='fill' works correctly
        self.I_array = jnp.ones(fixed_capacity, dtype=jnp.int64) * (n + 1)
        self.J_array = jnp.ones(fixed_capacity, dtype=jnp.int64) * (m + 1)
        self.k = 0  # Current rank (number of elements used)
        self.array_capacity = fixed_capacity  # Fixed capacity

        # core inverse U_k (k x k)
        # For 'inverse': store as matrix
        # For 'svd': store as tuple (U, inv_s, Vh)
        # For 'qr': store as tuple (Q, R)
        # Initialize with zeros for all modes; will be properly set when first step is taken
        self.sigma1_estimate = None
        if self.update_U_option == 'qr':
            # Initialize with identity Q and R for QR mode (will be updated on first step)
            Q_init = jnp.eye(fixed_capacity, dtype=Aop.dtype)
            R_init = jnp.zeros((fixed_capacity, fixed_capacity), dtype=Aop.dtype)
            if USE_CPQR:
                self.U_core = (Q_init, R_init, jnp.arange(fixed_capacity), jnp.ones(fixed_capacity, dtype=bool))
            else:
                self.U_core = (Q_init, R_init)
                ## Estimate sigma 1 by sketching
                # Start in the column space of A: R^m (for power iteration on A^T A)
                self.rng_key = jax.random.PRNGKey(0 if random_seed is None else random_seed)
                v = jax.random.normal(self.rng_key, (m,), dtype=Aop.dtype)
                v = v / jnp.linalg.norm(v)
                # A couple iterations to get a good estimate
                for _ in range(3):
                    # v <- (A^T A) v, with normalization
                    # Reuse v: v = A^T (A v) / ||A^T (A v)||
                    v = Aop.lmatvec(Aop.matvec(v))  # v in R^m
                    v = v / jnp.linalg.norm(v)

                # Rayleigh quotient estimate: ||A v|| is an estimate of sigma_1
                # Reuse v by computing norm directly
                self.sigma1_estimate = jnp.linalg.norm(Aop.matvec(v))

            
        elif self.update_U_option == 'svd':
            self.U_core = (jnp.zeros((fixed_capacity, fixed_capacity), dtype=Aop.dtype), jnp.zeros(fixed_capacity, dtype=Aop.dtype), jnp.zeros((fixed_capacity, fixed_capacity), dtype=Aop.dtype))
        else:
            self.U_core = jnp.zeros((fixed_capacity, fixed_capacity), dtype=Aop.dtype)


        # Always allocate W_core; allocate W_squared_core only if requested
        self.store_preconditioner_matrices = store_preconditioner_matrices 
        # Initialize W_core to identity for QR mode (will be updated incrementally)
        if self.update_U_option == 'qr':
            self.W_core = jnp.eye(fixed_capacity, dtype=Aop.dtype)
        else:
            self.W_core = jnp.zeros((fixed_capacity, fixed_capacity), dtype=Aop.dtype)
        self.W_squared_core = None
        if self.store_preconditioner_matrices:
            self.W_squared_core = jnp.zeros((fixed_capacity, fixed_capacity), dtype=Aop.dtype)

        # Column residual norms bookkeeping removed

        # Row residual norms: initialize from provided ||A_{i,:}||^2
        assert row_norms_squared.shape == (n,)
        self.n_row = jnp.asarray(row_norms_squared).real.astype(jnp.float64)
        
        # Masks to track selected rows/columns
        self.selected_rows_mask = jnp.zeros(self.n, dtype=bool)
        self.selected_cols_mask = jnp.zeros(self.m, dtype=bool)
        
        # Debug option
        self.debug = debug
        
        # Pivot sampling option: 'greedy', 'random', or 'uniform'
        # column_sampling is an alias for pivot_sampling (for API consistency with PivotedQRCUR)
        self.pivot_sampling = column_sampling if column_sampling is not None else pivot_sampling
        self.column_sampling = self.pivot_sampling  # Alias for API consistency
        
        # JAX random key for sampling (only needed for 'random' or 'uniform')
        if self.pivot_sampling in ['random', 'uniform']:
            if random_seed is None:
                random_seed = 42
            self.rng_key = jax.random.PRNGKey(random_seed)
        else:
            self.rng_key = None
        

        self.iter_when_all_norms_are_negative = -1  

        # Create JIT-compiled step functions (split to avoid capturing matrix in JIT)
        self._step_jit_pre, self._step_jit_post = self._create_step_jit()
        
        # Create JIT-compiled sampling functions
        self._sample_row_jit = jax.jit(lambda key, probs: jax.random.choice(key, self.n, p=probs))
        self._sample_col_jit = jax.jit(lambda key, probs: jax.random.choice(key, self.m, p=probs))

    # ---------- basic helpers ----------
    def _trim_to_size(self):
        """Trim arrays to current rank size (called after build completes)."""
        k = self.k
        self.I_array = self.I_array[:k]
        self.J_array = self.J_array[:k]
        self.W_core = self.W_core[:k, :k]

        if self.update_U_option == 'svd' or self.update_U_option == 'qr':
            self.prepare_U_core_solver()
        else:  # 'inverse'
            self.U_core = self.U_core[:k, :k]
        # Always trim W_core; trim W_squared_core only if allocated
        if self.W_squared_core is not None:
            self.W_squared_core = self.W_squared_core[:k, :k]
        self.array_capacity = k
    
    def _ensure_capacity(self):
        """Double capacity if needed (triggers JIT recompilation)."""
        if self.k >= self.array_capacity:
            # Double the capacity
            new_capacity = self.array_capacity * 2
            print(f"  [CUR] Increasing capacity from {self.array_capacity} to {new_capacity} (recompilation will occur)")
            
            # Expand I_array and J_array
            # So that this works after trim
            self.I_array = jnp.concatenate([self.I_array, jnp.ones(new_capacity- self.I_array.shape[0], dtype=jnp.int64) * (self.n + 1)])
            self.J_array = jnp.concatenate([self.J_array, jnp.ones(new_capacity- self.J_array.shape[0], dtype=jnp.int64) * (self.m + 1)])
            
            # Expand U_core (pad with zeros)
            
            self.array_capacity = new_capacity

            old_W_core = self.W_core
            self.W_core = jnp.zeros((new_capacity, new_capacity), dtype=self.A.dtype)
            self.W_core = self.W_core.at[:old_W_core.shape[0], :old_W_core.shape[1]].set(old_W_core)

            if self.update_U_option == 'svd' or self.update_U_option == 'qr':
                self.prepare_U_core_solver()
            else:  # 'inverse'
                old_U = self.U_core
                self.U_core = jnp.zeros((new_capacity, new_capacity), dtype=self.A.dtype)
                self.U_core = self.U_core.at[:old_U.shape[0], :old_U.shape[1]].set(old_U)

            if self.store_preconditioner_matrices:
                old_W_squared_core = self.W_squared_core
                self.W_squared_core = jnp.zeros((new_capacity, new_capacity), dtype=self.A.dtype)
                self.W_squared_core = self.W_squared_core.at[:old_W_squared_core.shape[0], :old_W_squared_core.shape[1]].set(old_W_squared_core)


    def _create_step_jit(self):
        """
        Create JIT-compiled step functions with matvec operations OUTSIDE the JIT.
        
        This avoids capturing the matrix data as constants in the JIT compilation,
        which prevents memory issues for large sparse matrices. The step is split
        into two JIT-compiled functions:
        1. step_jit_pre: Compute inputs needed for matvecs (lam, u, U_R_lam)
        2. step_jit_post: Use matvec outputs to update state
        
        Matvecs are called between pre and post, outside of JIT.
        
        Returns:
            Tuple of (step_jit_pre, step_jit_post)
        """
        @jax.jit
        def step_jit_pre(I_array, J_array, rres, current_row, current_col, p):
            """
            First phase: compute inputs needed for matvecs.
            Returns (lam, w, z, omega, sigma) for external matvec calls.
            Note: u = U_core @ w is computed outside this JIT because U_core
            can be a tuple (for QR/SVD modes) not a simple matrix.
            """
            # Extract w and z from pre-fetched row/col
            w = current_col.at[I_array].get(mode='fill', fill_value=0.0)
            z = current_row.at[J_array].get(mode='fill', fill_value=0.0).conj()
            omega = current_row.at[p].get()
            
            # Compute sigma from residual
            sigma = rres.at[p].get()

            # Compute lambda
            lam = (rres / sigma).astype(current_col.dtype)
            
            return lam, w, z, omega, sigma
        
        def step_jit_post_fn(U_core, n_row, I_array, J_array, selected_rows_mask, selected_cols_mask, k,
                             W_core, update_u_option, q, p, current_col,
                             lam, w, z, omega, sigma, A_lam, C_u, CUR_lam, sigma1_estimate):
            """
            Second phase: use matvec outputs to update state.
            """
            # Compute h = c - C_u
            h = current_col - C_u
            
            # Row norm update
            lam_norm_sq = jnp.vdot(lam, lam).real
            n_row_new = n_row + ((-2.0 * ((A_lam - CUR_lam)) + lam_norm_sq * h).conj() * h).real 
            
            # Update W_core = A[I, J]
            W_core_new = W_core.at[k].set(z.conj())
            W_core_new = W_core_new.at[:,k].set(w)
            W_core_new = W_core_new.at[k,k].set(omega)

            # Update U_core based on update_u_option
            if update_u_option == 'svd':
                U_core_new = prepare_U_core_svd_solver(W_core_new, rcond=None)
            elif update_u_option == 'qr':
                U_core_new = prepare_U_core_qr_solver(W_core_new, rcond=None, sigma1_estimate=sigma1_estimate if not USE_CPQR else None)
            else:  # 'inverse'
                u_inv = U_core @ w
                sigma_inv = 1.0 / sigma
                Uw = u_inv
                zU = U_core.T.conj() @ z.conj()
                U_core_new = U_core.at[:, k].set(-Uw * sigma_inv)
                U_core_new = U_core_new.at[k, :].set(-zU * sigma_inv)
                U_core_new = U_core_new.at[k, k].set(sigma_inv)
                U_core_new = U_core_new + jnp.outer(Uw, zU) * sigma_inv
            
            # Update indices and masks
            I_array_new = I_array.at[k].set(q)
            J_array_new = J_array.at[k].set(p)
            k_new = k + 1
            selected_rows_mask_new = selected_rows_mask.at[q].set(True)
            selected_cols_mask_new = selected_cols_mask.at[p].set(True)
            
            return (U_core_new, n_row_new, I_array_new, J_array_new,
                    selected_rows_mask_new, selected_cols_mask_new, k_new, W_core_new)
        
        step_jit_post = jax.jit(step_jit_post_fn, static_argnames=('update_u_option',))
        
        return step_jit_pre, step_jit_post
    
    def _compute_residual(self, current_row):
        """
        Compute residual: r = A[q,:] - (z^T U_core) R_k
        Note: Can't be JIT-compiled because lmatvec_with_R may use scipy internally.
        """
        current_row = jnp.asarray(current_row)
        z = current_row.at[self.J_array].get(mode='fill', fill_value=0.0).conj()
        a = self.U_core_lmatvec(z)
        zUR = self.A.lmatvec_with_R(a, self.I_array)
        return current_row - zUR

    def _pick_pivot(self):
        """
        Pick row q and column p for next CUR step.
        
        Sampling modes:
        - 'greedy': argmax row norm, then argmax column entry
        - 'random': sample row ~ norm², sample column ~ entry²  
        - 'uniform': uniform row, greedy column
        
        Returns (q, p, current_row, rres).
        """
        # Mask already-selected rows
        self.n_row = _mask_selected(self.n_row, self.selected_rows_mask)
        
        # Check if all norms are negative (switch to uniform if so)
        if float(jnp.max(self.n_row)) <= 0 and self.iter_when_all_norms_are_negative == -1:
            print(f"CUR - ALL ROW NORMS ARE NEGATIVE at step {self.k}")
            self.iter_when_all_norms_are_negative = self.k
            self.pivot_sampling = 'uniform'
        
        # Select row index q
        if self.pivot_sampling == 'greedy':
            q = int(_select_max_unmasked(self.n_row, self.selected_rows_mask))
        elif self.pivot_sampling == 'random':
            row_probs = _compute_probs_proportional(self.n_row, self.selected_rows_mask)
            self.rng_key, subkey = jax.random.split(self.rng_key)
            q = int(self._sample_row_jit(subkey, row_probs))
        else:  # 'uniform'
            row_probs = _compute_probs_uniform(self.selected_rows_mask)
            self.rng_key, subkey = jax.random.split(self.rng_key)
            q = int(self._sample_row_jit(subkey, row_probs))
        
        # Fetch row and compute residual
        current_row = self.A.get_row(q)
        rres = self._compute_residual(current_row)
        
        # Select column index p
        if self.pivot_sampling == 'random':
            col_probs = _compute_probs_proportional(jnp.abs(rres)**2, self.selected_cols_mask)
            self.rng_key, subkey = jax.random.split(self.rng_key)
            p = int(self._sample_col_jit(subkey, col_probs))
        else:
            p = int(_select_max_unmasked(jnp.abs(rres), self.selected_cols_mask))
        
        return q, p, current_row, rres

    # ---------- public step ----------

    def step(self):
        """
        One CUR rank-1 append using rank-1 outer update and norm updates.
        Returns (q, p) indices chosen.
        
        NOTE: matvec calls are done OUTSIDE the JIT to avoid capturing the entire
        matrix data as constants. This prevents memory issues for large sparse matrices.
        """
        # 1) Pick pivot (returns row and residual)
        q, p, current_row, rres = self._pick_pivot()
        
        # 2) Ensure capacity
        self._ensure_capacity()
        
        # 3) Fetch column (outside JIT - different operators have different get_col)
        current_col = self.A.get_col(p)
        
        # 4) First JIT phase: compute inputs for matvecs
        lam, w, z, omega, sigma = self._step_jit_pre(
            self.I_array, self.J_array,
            rres, current_row, current_col, p
        )
        
        # 5) Compute u = U_core @ w (outside JIT because U_core can be tuple for QR/SVD)
        u = self.apply_U_core(w)
        
        # 6) Matvec operations OUTSIDE JIT to avoid capturing matrix as constants
        A_lam = self.A.matvec(lam)
        C_u = self.A.matvec_with_C(u, self.J_array)
        
        # CUR_lam = A[:, J] @ (U_core @ (A_lam[I]))
        R_lam = A_lam.at[self.I_array].get(mode='fill', fill_value=0.0)
        U_R_lam = self.apply_U_core(R_lam)
        CUR_lam = self.A.matvec_with_C(U_R_lam, self.J_array)
        
        # 6) Second JIT phase: update state using matvec results
        (self.U_core, self.n_row, self.I_array, self.J_array,
         self.selected_rows_mask, self.selected_cols_mask, self.k,
         self.W_core) = \
            self._step_jit_post(
                self.U_core, self.n_row, self.I_array, self.J_array,
                self.selected_rows_mask, self.selected_cols_mask, self.k,
                self.W_core, self.update_U_option, q, p, current_col,
                lam, w, z, omega, sigma, A_lam, C_u, CUR_lam,
                self.sigma1_estimate if self.sigma1_estimate is not None else 1.0
            )
        
        return q, p

    def build(self, rank, timing=False):
        """
        Run 'rank' steps.
        
        Returns
        -------
        dict
            Dictionary with keys:
            - 'I': row indices (k,)
            - 'J': column indices (k,)
            - 'U': core matrix (k x k) - equivalent to T in PivotedQRCUR
            - 'k': actual rank achieved
            - 'norm_residuals': residual norms at each step
            - 'total_time': total build time (if timing=True)
            - 'timings': timing at each step (if timing=True)
            - 'iter_when_all_norms_are_negative': iteration when all norms became negative
        """
        start_time = time.time()
        if timing:
            timings = jnp.zeros(rank+1)
        norm_residuals = jnp.zeros(rank+1)
        norm_residuals = norm_residuals.at[0].set(jnp.sum(self.n_row))
        for k in range(rank):
            self.step()
            if timing:
                timings = timings.at[k+1].set(time.time() - start_time)
            norm_residuals = norm_residuals.at[k+1].set(jnp.sum(self.n_row))
        self._trim_to_size()  # This also calls prepare_U_core_solver for svd/qr
        
        # Default behavior: if using QR or inverse updates, convert to SVD at the end for better accuracy
        if self.auto_convert_to_svd and (self.update_U_option == 'qr' or self.update_U_option == 'inverse'):
            self._convert_to_svd()
        
        # Compute W_squared_core after build completes (more efficient than incremental)
        # W_squared_core = A[I, :] @ D^{-1} @ A[:, J]
        if self.store_preconditioner_matrices:
            self._compute_W_squared_core()
        
        total_time = time.time() - start_time
        k = int(self.k)
        
        # Return dict for unified API (consistent with PivotedQRCUR)
        result = {
            'I': self.I_array,
            'J': self.J_array,
            'U': self.U_core,  # Core matrix (equivalent to T in PivotedQRCUR)
            'k': k,
            'norm_residuals': norm_residuals,
            'iter_when_all_norms_are_negative': self.iter_when_all_norms_are_negative,
            'total_time': total_time,
        }
        if timing:
            result['timings'] = timings
        
        return result
    
    def _compute_W_squared_core(self):
        """
        Compute W_squared_core = A[I, :] @ D^{-1} @ A[:, J] after build completes.
        
        This is more efficient than incremental computation because we can
        batch the matvec_with_R operations. Uses column-by-column computation
        to avoid allocating large (n, k) matrices.
        """
        k = int(self.k)
        if k == 0:
            return
        
        # Allocate W_squared_core if not already done
        if self.W_squared_core is None:
            self.W_squared_core = jnp.zeros((k, k), dtype=self.A.dtype)
        
        # Compute column-by-column: W_squared_core[:, j] = A[I, :] @ (D^{-1} @ A[:, J[j]])
        # This uses matvec_with_R which returns a k-vector (no large allocation)
        I_to_k = self.I_array[:k]
        J_to_k = self.J_array[:k]
        
        W_squared_core = jnp.zeros((k, k), dtype=self.A.dtype)
        for j_idx in range(k):
            col_j = self.A.get_col(int(J_to_k[j_idx]))
            Dinv_col_j = self.M_inv_matvec(col_j) if self.M_inv_matvec is not None else col_j
            W_squared_core = W_squared_core.at[:, j_idx].set(
                self.A.matvec_with_R(Dinv_col_j, I_to_k)
            )
        
        self.W_squared_core = W_squared_core


    def prepare_U_core_solver(self, rcond: float | None = None, sigma1_estimate: float | None = None):
        """
        Precompute a stable solver for the current core W = A[I, J].
        """
        if self.update_U_option == 'svd':
            U, s, Vh = prepare_U_core_svd_solver(self.W_core, rcond)
            self.U_core = (U,s,Vh)
        elif self.update_U_option == 'qr':
            U_core_result = prepare_U_core_qr_solver(self.W_core, rcond, self.sigma1_estimate)
            self.U_core = U_core_result  # Already a tuple (either 2 or 4 elements)
            
    def _convert_to_svd(self, rcond: float | None = None):
        """
        Internal method: Convert U_core from QR or inverse representation to SVD representation.
        
        This is the default behavior when auto_convert_to_svd=True: CUR will use
        QR/inverse updates during iteration (faster) but automatically convert to SVD
        at the end for better accuracy in error computation.
        
        Note: This method assumes it's only called for 'qr' or 'inverse' modes.
        The call sites check this before invoking.
        
        Parameters
        ----------
        rcond : float, optional
            Relative condition number cutoff for SVD. If None, uses default.
        """
        k = int(self.k)
        if k == 0:
            return
        
        # Use W_core (should already be trimmed to [:k, :k] if called after _trim_to_size)
        # But handle both cases: if not trimmed yet, extract the relevant part
        if self.W_core.shape[0] > k:
            W_core = self.W_core[:k, :k]
        else:
            W_core = self.W_core
        
        # Convert to SVD format
        U_svd, inv_s, Vh = prepare_U_core_svd_solver(W_core, rcond=rcond)
        
        # Replace U_core with SVD representation
        self.U_core = (U_svd, inv_s, Vh)
        
        # Update the update_U_option flag so matvec functions use SVD
        self.update_U_option = 'svd'

    

    # ---------- Unified apply for U_core depending on update_U_option ----------
    def apply_U_core(self, b: jnp.ndarray) -> jnp.ndarray:
        """
        Apply W^{-1} to b.
        - If update_U_option is 'svd': use SVD-based stable pseudoinverse.
        - If update_U_option is 'qr': use QR factorization with triangular solve.
        - If update_U_option is 'inverse': fast path using stored U_core matrix multiply.
        """
        result = U_core_matvec_jit(self.U_core, b, self.update_U_option)
        # Handle shape: JIT functions return 2D for CPQR, squeeze if input was 1D
        if self.update_U_option == 'qr' and USE_CPQR and b.ndim == 1:
            return result[:, 0] if result.ndim == 2 else result
        return result

    def U_core_lmatvec(self, a_row: jnp.ndarray) -> jnp.ndarray:
        """
        Left-apply row vector to U_core: returns a_row @ U_core.
        Uses appropriate path based on update_U_option.
        Supports a_row with shape (k,) or (B, k).
        
        NOTE: Do NOT conjugate a_row here - U_core_lmatvec_jit handles conjugation internally.
        """
        result = U_core_lmatvec_jit(self.U_core, a_row, self.update_U_option)
        # Handle shape: JIT functions return 2D for CPQR, squeeze if input was 1D
        if self.update_U_option == 'qr' and USE_CPQR and a_row.ndim == 1:
            return result[:, 0] if result.ndim == 2 else result
        return result


    def matvec(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Apply the current CUR approximation to x: CUR@x
        """
        return self.A.matvec_with_C(self.apply_U_core( self.A.matvec_with_R(x, self.I_array)), self.J_array)

    def lmatvec(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Apply the current CUR approximation to x from the left: x@CUR
        """
        return self.A.lmatvec_with_R(self.U_core_lmatvec(self.A.lmatvec_with_C(x, self.J_array)), self.I_array)

    def matvec_with_rank(self, x: jnp.ndarray, rank: int) -> jnp.ndarray:
        ## TODO: copy the current CUR to a new object ( so that the original CUR is not modified), truncate W to rank, re-compute the U_core, then apply the truncated CUR.
        # Copy the current CUR to a new object
        new_cur = self.truncate_to_rank(rank)
        return new_cur.matvec(x)

    def lmatvec_with_rank(self, x: jnp.ndarray, rank: int) -> jnp.ndarray:
        ## TODO: copy the current CUR to a new object ( so that the original CUR is not modified), truncate W to rank, re-compute the U_core, then apply the truncated CUR.
        # Copy the current CUR to a new object
        new_cur = self.truncate_to_rank(rank)
        return new_cur.lmatvec(x)

    def truncate_to_rank(self, rank: int):
        """
        Truncate the current CUR to rank.
        
        Default behavior: if auto_convert_to_svd=True and the original CUR used
        QR or inverse updates, the truncated version will be converted to SVD
        for better accuracy.
        """
        new_cur = CURIncremental(self.A, self.n_row, debug=self.debug, pivot_sampling=self.pivot_sampling, random_seed=0, store_preconditioner_matrices=self.store_preconditioner_matrices, profile=self.profile, use_jit=self.use_jit, update_U_option=self.original_update_U_option, M_inv_matvec=self.M_inv_matvec, auto_convert_to_svd=self.auto_convert_to_svd)
        new_cur.k = rank
        new_cur.array_capacity = rank
        new_cur.I_array = self.I_array[:rank]
        new_cur.J_array = self.J_array[:rank]
        if self.store_preconditioner_matrices:
            new_cur.W_squared_core = self.W_squared_core[:rank, :rank]
        new_cur.W_core = self.W_core[:rank, :rank]
        if self.update_U_option == 'svd' or self.update_U_option == 'qr':
            new_cur.prepare_U_core_solver()
        else:  # 'inverse'
            new_cur.U_core = jnp.linalg.pinv(self.W_core[:rank, :rank])
        
        # Default behavior: if using QR or inverse updates, convert to SVD for better accuracy
        if new_cur.auto_convert_to_svd and (new_cur.update_U_option == 'qr' or new_cur.update_U_option == 'inverse'):
            new_cur._convert_to_svd()
        
        return new_cur
    
    def woodbury_solve(self, b: jnp.ndarray, M_inv_matvec: callable = None) -> jnp.ndarray:
        """
        Solve (D + C U R) x = b using Woodbury formula.
        
        Woodbury: (D + C U R)^{-1} = D^{-1} - D^{-1} C (U^{-1} + R D^{-1} C)^{-1} R D^{-1}
        
        where:
        - C = A[:, J] (selected columns)
        - R = A[I, :] (selected rows)
        - U is the core matrix factor
        - D is a diagonal matrix (applied via M_inv_matvec)
        
        Note: Requires store_preconditioner_matrices=True during initialization
        and that build() has been called.
        
        Parameters
        ----------
        b : jnp.ndarray, shape (n,)
            Right-hand side vector
        M_inv_matvec : callable, optional
            Function to apply D^{-1}. If None, uses self.M_inv_matvec.
            
        Returns
        -------
        jnp.ndarray, shape (n,)
            Solution vector x
        """
        solver = self.get_woodbury_solver(M_inv_matvec)
        return solver.solve(b)
    
    def get_woodbury_solver(self, M_inv_matvec: callable = None):
        """
        Get a Woodbury solver object for use with GMRES or other iterative solvers.
        
        Note: Requires store_preconditioner_matrices=True during initialization
        and that build() has been called.
        
        Parameters
        ----------
        M_inv_matvec : callable, optional
            Function to apply D^{-1}. If None, uses self.M_inv_matvec.
            
        Returns
        -------
        CUR_woodbury_solver
            Solver object with solve(b) method
        """
        if not self.store_preconditioner_matrices or self.W_squared_core is None:
            raise ValueError(
                "Woodbury solver requires store_preconditioner_matrices=True "
                "during initialization and that build() has been called."
            )
        
        M_inv = M_inv_matvec if M_inv_matvec is not None else self.M_inv_matvec
        return CUR_woodbury_solver(self, M_inv_matvec=M_inv)
    
    def to_unified(self):
        """
        Convert to CUR wrapper for consistent interface.
        
        Returns
        -------
        CUR
            CUR wrapper
        """
        from unified_cur import CUR
        return CUR(self, implementation_type='incremental')

def prepare_U_core_qr_solver(W_core: jnp.ndarray, rcond: float | None = None, sigma1_estimate: float | None = None):
    """
    Solves \|W_core x - b\|_2^2 + lambda_1^2 \|x\|_2^2 by QR factorization.
    
    If USE_CPQR is True, uses column-pivoted QR (CPQR) for better numerical stability.
    Otherwise, uses regular QR with augmented matrix [W_core; lambda_1 * I] to match SVD regularization.
    lambda_1 = sigma1_estimate * rcond matches SVD's cutoff threshold:
    SVD uses tol = rcond * max(s), and this lambda_1 matches that scale.
    
    Returns:
        If USE_CPQR: (Q, R_plus, piv, bad_cols) - CPQR factors
        Otherwise: (Q, R) - regular QR factors
    """
    if USE_CPQR:
        # Use column-pivoted QR
        Q, R_plus, piv, bad_cols = prepare_U_core_cpqr_solver(W_core, rcond=rcond)
        return (Q, R_plus, piv, bad_cols)
    else:
        # Regular QR with regularization
        # Extract only the k×k submatrix for QR computation (k is current rank)
        if rcond is None:
            eps = jnp.finfo(W_core.dtype).eps
            rcond = 10.0 * eps * max(W_core.shape)
        lambda_1 = sigma1_estimate * rcond if sigma1_estimate is not None else 0.0
        block_matrix = jnp.concatenate([W_core, lambda_1 * jnp.eye(W_core.shape[1])], axis=0)
        Q, R = jnp.linalg.qr(block_matrix, mode='reduced')
        # Return Q and R (Q will be (m+n, n) and R will be (n, n) in reduced mode)
        # We need to extract the relevant parts
        n = W_core.shape[0]
        return (Q[:n, :n], R[:n, :n])




def prepare_U_core_svd_solver(W_core: jnp.ndarray, rcond: float | None = None):
    """
    Precompute an SVD-based pseudoinverse handle for the current core W = A[I, J].
    Requires store_preconditioner_matrices=True so W_core is available.
    """

    # To avoid recompiling, we padd
    # TO avoid 
    U, s, Vh = jnp.linalg.svd(W_core, full_matrices=True)
    if rcond is None:
        eps = jnp.finfo(W_core.dtype).eps
        rcond = 10.0 * eps * max(W_core.shape)
    tol = rcond * jnp.max(s)
    inv_s = jnp.where(s > tol, 1.0 / s, 0.0)
    return (U, inv_s, Vh)


# Not JIT-compiled separately - called from within step_jit_full which is already JIT-compiled
# JIT decorator removed to avoid nested JIT compilation overhead
def U_core_matvec_jit(U, b: jnp.ndarray, update_U_option: str) -> jnp.ndarray:
    """
    Apply W^{-1} to b using linear solves.
    - If update_U_option == 'qr': QR mode - uses triangular solve
        * If USE_CPQR: (Q, R_plus, piv, bad_cols) - CPQR with pivoting
        * Otherwise: (Q, R) - regular QR
    - If update_U_option == 'svd': SVD mode (U, inv_s, Vh)
    - If update_U_option == 'inverse': inverse mode (U is a matrix)
    
    Note: When k=0, U might still be a matrix even for 'svd'/'qr' modes.
    """
    if update_U_option == 'qr':
        # For QR mode, U should always be a tuple (initialized as tuple in __init__)
        if not isinstance(U, tuple):
            # This should never happen - indicates a bug in initialization/update
            raise ValueError(f"U_core is not a tuple for QR mode. Got type: {type(U)}")
        # Use USE_CPQR flag to determine format
        if USE_CPQR:
            # CPQR mode: (Q, R_plus, piv, bad_cols) - 4 elements
            Q, R_plus, piv, bad_cols = U
            return solve_U_core_cpqr(Q, R_plus, piv, bad_cols, b)
        else:
            # Regular QR mode: (Q, R) - 2 elements
            Q, R = U
            return jax.scipy.linalg.solve_triangular(R, Q.conj().T @ b, lower=False)
    elif update_U_option == 'svd':
        # SVD mode: W = U @ diag(s) @ Vh
        # W^{-1} = Vh.conj().T @ diag(1/s) @ U.conj().T
        UU, inv_s, Vh = U
        return Vh.conj().T @ (inv_s * (UU.conj().T @ b))
    else:  # 'inverse'
        # Inverse mode: U is the inverse matrix directly
        return U @ b
        

from jax.scipy.linalg import qr as cpqr, solve_triangular



def prepare_U_core_cpqr_solver(W_core: jnp.ndarray,
                               rcond: float | None = None):
    """
    (Not actually CPQR) - just regular QR that ignores small diagonals. The reason is that CPQR does not run on GPU in JAX without a lot of effort.

    W_core: shape (m, n), with m >= n preferred.

    Returns:
        Q       : (m, n)  - orthonormal columns
        piv     : (n,)    - column permutation indices (A[:, piv] = Q @ R)
        R_plus  : (n, n)  - 'pseudo-inverse' of R in CPQR basis, with rows
                            corresponding to small diag(R) zeroed out
        rcond   : float   - rcond actually used
        tol     : float   - threshold used on |diag(R)|

    So the solve will be:
        c = Q.T @ b
        y = R_plus @ c
        x[piv] = y
    """

    m, n = W_core.shape

    # Column-pivoted QR: W_core[:, piv] = Q @ R
    # Explicitly pass use_magma to use MAGMA when available (only if JAX_GPU_MAGMA_PATH is set)
    # Use full_matrices=False for better compatibility (equivalent for square matrices)
    # Q, R, piv = jax.lax.linalg.qr(W_core, full_matrices=False, pivoting=False)#, use_magma=_MAGMA_AVAILABLE)
    Q, R = jax.lax.linalg.qr(W_core, full_matrices=False, pivoting=False)#, use_magma=_MAGMA_AVAILABLE)
    piv = jnp.arange(n)
    # rcond as in your SVD path
    eps = jnp.finfo(W_core.dtype).eps
    if rcond is None:
        rcond = 100.0 * eps * max(m, n)

    diag_R = jnp.abs(jnp.diag(R))
    sigma1_est = jnp.max(diag_R)
    tol = rcond * sigma1_est

    # Mark "small" diagonals (bad directions to be zeroed out); shape (n,)
    bad_cols = diag_R < tol
    
    # For rank-deficient case: zero out columns corresponding to small diagonals
    # Create a mask for columns to zero out
    R_plus = R.copy()
    # Zero out columns where diagonal is too small
    # Use jnp.where to handle boolean indexing properly
    col_mask = bad_cols[:, None]  # (n, 1) for broadcasting
    R_plus = jnp.where(col_mask, 0.0, R_plus)
    
    # For the diagonal entries of bad columns, we want to avoid division by zero
    # in the solve step, so we set them to 1 (which makes them effectively ignored)
    row_indices = jnp.arange(n)
    diag_mask = (row_indices[:, None] == row_indices[None, :]) & bad_cols[:, None]
    R_plus = jnp.where(diag_mask, 1.0, R_plus)

    return Q, R_plus, piv, bad_cols


# Not JIT-compiled - will be inlined into calling JIT function
def solve_U_core_cpqr(Q: jnp.ndarray,
                      R_plus: jnp.ndarray,
                      piv: jnp.ndarray,
                      bad_cols: jnp.ndarray,
                      b: jnp.ndarray) -> jnp.ndarray:
    """
    Solve min_x ||W_core x - b||_2 using the CPQR factors prepared above.

    Inputs:
        Q      : (m, n)  - orthonormal columns from QR
        R_plus : (n, n)  - upper triangular R with bad columns zeroed
        piv    : (n,)    - column permutation indices
        bad_cols: (n,)   - boolean array marking rank-deficient columns
        b      : (m,) or (m, nrhs) - right-hand side

    Returns:
        x      : (n,) or (n, nrhs) - solution (same ndim as input)
    """
    m, n = Q.shape

    # Project RHS into CPQR coordinates (handles both 1D and 2D automatically)
    c = Q.conj().T @ b           # (n,) or (n, nrhs)

    # Zero out components corresponding to bad columns (rank-deficient directions)
    # bad_cols is boolean array, so we can use it directly for masking
    if c.ndim == 1:
        c = jnp.where(bad_cols, 0.0, c)
    else:
        c = jnp.where(bad_cols[:, None], 0.0, c)
    
    # Solve R_plus @ y = c using triangular solve
    # R_plus has bad columns zeroed and diagonal entries set to 1 for stability
    y = solve_triangular(R_plus, c, lower=False)

    # Undo column pivoting
    # The factorization is W_core[:, piv] = Q @ R
    # To solve W_core @ x = b, we solve in permuted space: (Q @ R) @ x_perm = b
    # where x_perm[i] = x[piv[i]]
    # So: R @ x_perm = Q.T @ b = c, giving x_perm = y
    # Then: x[piv[i]] = x_perm[i] = y[i]
    # To get x from x_perm, we use: x = x_perm[inverse_piv] where inverse_piv[piv[i]] = i
    inverse_piv = jnp.zeros(n, dtype=jnp.int32)
    inverse_piv = inverse_piv.at[piv].set(jnp.arange(n, dtype=jnp.int32))
    x = y[inverse_piv]

    return x


# Not JIT-compiled - will be inlined into calling JIT function
def solve_U_core_cpqr_transpose(Q: jnp.ndarray,
                                 R_plus: jnp.ndarray,
                                 piv: jnp.ndarray,
                                 bad_cols: jnp.ndarray,
                                 b: jnp.ndarray) -> jnp.ndarray:
    """
    Solve A^T @ x = b using the CPQR factors prepared above.
    This computes x = (A^T)^{-1} @ b = (A^{-1})^T @ b.
    JIT-compiled for performance.
    
    Inputs:
        Q      : (m, n)  - orthonormal columns from QR
        R_plus : (n, n)  - upper triangular R with bad columns zeroed
        piv    : (n,)    - column permutation indices
        bad_cols: (n,)   - boolean array marking rank-deficient columns
        b      : (n,) or (n, nrhs) - right-hand side (note: shape is (n,) for A^T)
    
    Returns:
        x      : (m,) or (m, nrhs) - solution (same ndim as input)
    """
    m, n = Q.shape
    
    # Permute b according to piv
    b_perm = b[piv]  # Permute b according to piv
    
    # Zero out components corresponding to bad columns
    if b_perm.ndim == 1:
        c_perm = jnp.where(bad_cols, 0.0, b_perm)
    else:
        c_perm = jnp.where(bad_cols[:, None], 0.0, b_perm)
    
    y = solve_triangular(R_plus, c_perm, lower=False, trans=1)
    x = Q @ y
    
    return x



# Not JIT-compiled separately - called from within step_jit_full which is already JIT-compiled
# JIT decorator removed to avoid nested JIT compilation overhead
def U_core_lmatvec_jit(U, b: jnp.ndarray, update_U_option: str) -> jnp.ndarray:
    """
    Left-apply row vector to W^{-1}: returns b @ W^{-1} = (W^{-1})^H @ b^H.
    - If update_U_option == 'qr': QR mode
        * If USE_CPQR: (Q, R_plus, piv, bad_cols) - CPQR with pivoting
        * Otherwise: (Q, R) - regular QR
    - If update_U_option == 'svd': SVD mode (U, inv_s, Vh)
    - If update_U_option == 'inverse': inverse mode (U is a matrix)
    
    Note: When k=0, U might still be a matrix even for 'svd'/'qr' modes.
    """
    if update_U_option == 'qr':
        # For QR mode, U should always be a tuple
        if not isinstance(U, tuple):
            raise ValueError(f"U_core is not a tuple for QR mode in lmatvec. Got type: {type(U)}")
        # Use USE_CPQR flag to determine format
        if USE_CPQR:
            # CPQR mode: (Q, R_plus, piv, bad_cols) - 4 elements
            Q, R_plus, piv, bad_cols = U
            return solve_U_core_cpqr_transpose(Q, R_plus, piv, bad_cols, b)
        else:
            # Regular QR mode: (Q, R) - 2 elements
            Q, R = U
            return Q @ jax.scipy.linalg.solve_triangular(R, b, trans=1)
    elif update_U_option == 'svd':
        # SVD mode: W = U @ diag(s) @ Vh
        # W^{-1} = Vh.conj().T @ diag(1/s) @ U.conj().T
        # (W^{-1})^H = U @ diag(1/s) @ Vh
        # Check if U is a tuple (SVD mode) or matrix (not yet initialized)
        UU, inv_s, Vh = U
        if b.ndim == 1:
            return UU @ (inv_s * (Vh @ b.conj()))
        elif b.ndim == 2:
            return UU @ (inv_s[:, None] * (Vh @ b.conj()))
        else:
            raise ValueError("b must be 1D or 2D")
    else:  # 'inverse'
        # Inverse mode: U is the inverse matrix directly
        # (W^{-1})^H = U^H
        return U.conj().T @ b



