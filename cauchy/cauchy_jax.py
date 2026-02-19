"""
Highly optimized JAX implementation of Cauchy matrix operations and LU decomposition.
Uses fixed-size arrays to avoid recompilation, similar to cur.py.
Stores indices only, not L/U columns.
"""

import os
import re
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# Enable nsight profiling if requested
if os.environ.get('ENABLE_NSIGHT', '0') == '1':
    os.environ['NSIGHT_PROF'] = '1'
    # Set CUDA profiling flags
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Synchronous launches for profiling
    print("NSIGHT profiling enabled via ENABLE_NSIGHT=1")

# Enable JAX profiling if requested
if os.environ.get('ENABLE_JAX_PROFILE', '0') == '1':
    xla_flags = os.environ.get('XLA_FLAGS', '')
    if '--xla_gpu_enable_triton_gemm=false' not in xla_flags:
        os.environ['XLA_FLAGS'] = xla_flags + ' --xla_gpu_enable_triton_gemm=false'
    print("JAX profiling enabled via ENABLE_JAX_PROFILE=1")

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import time

# Module-level constant PRNG key to ensure it's truly static and avoids recompilation
_STATIC_RNG_KEY = jax.random.PRNGKey(0)

# Upper-bound (Rakau) numeric stabilization: generator column reparametrizations that
# preserve the represented matrix (g @ b.T) but improve conditioning.
#
# Configure with `CAUCHY_UB_BALANCE_GENERATORS`:
# - "0"/"off"/"false": disable
# - "1"/"balance": enable scaling/balancing (default)
# - "ortho": enable (cheap) 2-column orthogonalization
# - "both": enable balancing + orthogonalization
_UB_STAB_SPEC = os.environ.get("CAUCHY_UB_BALANCE_GENERATORS", "1").strip().lower()
if _UB_STAB_SPEC in {"0", "false", "no", "off"}:
    _UB_STAB_BALANCE = False
    _UB_STAB_ORTHO = False
else:
    if _UB_STAB_SPEC in {"1", "true", "yes", "on"}:
        _UB_STAB_SPEC = "balance"
    tokens = [t for t in re.split(r"[,+\s]+", _UB_STAB_SPEC) if t]
    if any(t in {"both", "all"} for t in tokens):
        _UB_STAB_BALANCE = True
        _UB_STAB_ORTHO = True
    else:
        _UB_STAB_BALANCE = any(t in {"balance", "bal", "scale", "scaling"} for t in tokens)
        _UB_STAB_ORTHO = any(t in {"ortho", "orth", "orthogonalize", "qr"} for t in tokens)
        # Unknown truthy value: keep backward-compatible behavior.
        if not _UB_STAB_BALANCE and not _UB_STAB_ORTHO:
            _UB_STAB_BALANCE = True
            _UB_STAB_ORTHO = False
_ENABLE_UB_GENERATOR_STABILIZATION = _UB_STAB_BALANCE or _UB_STAB_ORTHO


@jax.jit
def _balance_generator_columns(g: jnp.ndarray, b: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Rebalance generator columns without changing the represented matrix.

    For diagonal scaling S: g <- g S, b <- b S^{-1}, we have (g b^T) unchanged.
    This helps avoid overflow/underflow in the Rakau upper-bound and LU updates.
    """
    g_abs = jnp.abs(g)
    b_abs = jnp.abs(b)
    g_max = jnp.max(g_abs, axis=0)
    b_max = jnp.max(b_abs, axis=0)

    ones_g = jnp.ones_like(g_max)
    ones_b = jnp.ones_like(b_max)
    g_max_safe = jnp.where(g_max > 0, g_max, ones_g)
    b_max_safe = jnp.where(b_max > 0, b_max, ones_b)

    g_scaled = g / g_max_safe
    b_scaled = b / b_max_safe
    g_norm = g_max_safe * jnp.linalg.norm(g_scaled, axis=0)
    b_norm = b_max_safe * jnp.linalg.norm(b_scaled, axis=0)

    finite = (g_norm > 0) & (b_norm > 0) & jnp.isfinite(g_norm) & jnp.isfinite(b_norm)
    ratio = jnp.where(finite, b_norm / g_norm, jnp.ones_like(g_norm))
    scale = jnp.sqrt(ratio)

    # Avoid pathological scales that can create inf/0 even if the ratio is finite.
    finfo = jnp.finfo(scale.dtype)
    scale = jnp.clip(scale, jnp.sqrt(finfo.tiny), jnp.sqrt(finfo.max))

    g_bal = g * scale[jnp.newaxis, :]
    b_bal = b / scale[jnp.newaxis, :]
    return g_bal, b_bal

@jax.jit
def _orthogonalize_generator_columns_r2(g: jnp.ndarray, b: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    For r=2 only: orthonormalize the columns of b (QR / modified Gram-Schmidt)
    and update g so that (g @ b.T) is preserved exactly in exact arithmetic.
    """
    b0 = b[:, 0]
    b1 = b[:, 1]

    r00 = jnp.linalg.norm(b0)
    r00_safe = jnp.where(r00 > 0, r00, jnp.ones_like(r00))
    q0 = b0 / r00_safe

    r01 = jnp.vdot(q0, b1)
    u1 = b1 - q0 * r01
    r11 = jnp.linalg.norm(u1)
    r11_safe = jnp.where(r11 > 0, r11, jnp.ones_like(r11))
    q1 = u1 / r11_safe

    # Form Q and R so that b = Q R and g <- g R^T keeps g b^T invariant.
    b_new = jnp.stack([q0, q1], axis=1)
    zero = jnp.array(0.0, dtype=b.dtype)
    R = jnp.array([[r00, r01], [zero, r11]], dtype=b.dtype)
    g_new = g @ R.T
    return g_new, b_new

def _stabilize_generators(g: jnp.ndarray, b: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    if _UB_STAB_BALANCE:
        g, b = _balance_generator_columns(g, b)
    if _UB_STAB_ORTHO and g.shape[1] == 2:
        g, b = _orthogonalize_generator_columns_r2(g, b)
    # If both are enabled, re-balance after orthogonalization to keep columns comparable.
    if _UB_STAB_BALANCE and _UB_STAB_ORTHO:
        g, b = _balance_generator_columns(g, b)
    return g, b

@jax.jit
def frob_norm_from_uv(U: jnp.ndarray, V: jnp.ndarray):
    """
    Frobenius norm of A = U @ V^* where U:(m,2), V:(n,2) (rank<=2).
    Works for real or complex.
    Returns scalar (same real dtype as U/V).
    """
    return jnp.sqrt(frob_norm2_from_uv(U, V))

@jax.jit
def frob_norm2_from_uv(U: jnp.ndarray, V: jnp.ndarray):
    Gu = U.conj().T @ U
    Gv = V.conj().T @ V
    return jnp.real(jnp.sum(Gv * Gu.T))



@jax.jit
def _column_j(c, q, g, b, j):
    """Compute j-th column for rank-1 or rank-r generators."""
    # Optimized: avoid .at[].get() by using direct indexing (faster)
    q_j = q[j]
    b_j = b[j]  # (r,)
    numerator = jnp.sum(g * b_j, axis=-1)
    return numerator / (c - q_j)

@jax.jit
def _row_i(c, q, g, b, i):
    """Compute i-th row for rank-1 or rank-r generators."""
    # Optimized: avoid .at[].get() by using direct indexing (faster)
    c_i = c[i]
    g_i = g[i]  # (r,)
    numerator = jnp.sum(b * g_i, axis=-1)
    return numerator / (c_i - q)

@jax.jit
def _row_block(c, q, g, b, i_indices):
    """
    Compute block of rows efficiently without vmap.
    
    Args:
        i_indices: (k,) array of row indices
    
    Returns:
        rows: (k, m) array where rows[k] = g[i_indices[k]] * b / (c[i_indices[k]] - q)
    """
    # Extract row parameters
    c_I = c[i_indices]  # (k,)
    g_I = g[i_indices]  # (k, r)
    
    # Compute numerators via matrix multiplication: (k, m)
    numerators = g_I @ b.T
    
    # Vectorize: c_I[:, None] is (k, 1), q[None, :] is (1, m)
    c_I_col = c_I[:, None]  # (k, 1)
    q_row = q[None, :]  # (1, m)
    
    rows = numerators / (c_I_col - q_row)  # (k, m)
    return rows

@jax.jit
def _make_C(c, q, g, b):
    """Compute full Cauchy matrix: C = outer(g,b) / outer(c,ones) - outer(ones,q)."""
    numerator = g @ b.T  # (n, m)
    return numerator / (c[:, None] - q[None, :])


# Barnes-Hut matvec (optimized)
def barnes_hut_matvec(c, q, g, b, theta=1, eps2=1e-14):
    """
    Compute row norms (|C|^2 @ 1) using Barnes-Hut algorithm (O(N log N)).
    Supports rank-1 and higher-rank Cauchy-like generators.
    
    JAX-optimized version that automatically uses GPU (CUDA) or CPU implementation
    based on JAX backend availability.
    
    NOTE: NOT JIT-compiled because barnes_hut_2d uses block_until_ready().
    Removing @jax.jit eliminates JIT exit/re-entry overhead (major source of 90% overhead).
    
    Optimizations:
    - Reduced operations
    - Pre-computed repeated calculations
    - Minimal memory allocations
    - Combined operations where possible
    """
    if g.shape[1] != b.shape[1]:
        raise ValueError("g and b must have the same generator dimension")
    generator_rank = g.shape[1]
    
    n_q = q.size
    n_c = c.size
    
    # Backend selection: auto (default) = Rakau on CPU or GPU.
    _bh_backend = os.environ.get("CAUCHY_BH_BACKEND", "auto").lower()
    if _bh_backend == "auto":
        _bh_backend = "rakau"
    if _bh_backend != "rakau":
        raise ValueError(
            f"Unsupported CAUCHY_BH_BACKEND='{_bh_backend}'. "
            "Only 'rakau' (or 'auto') is supported."
        )

    from .Barnes_hut import barnes_hut_rakau as _bh_rakau

    # Keep everything in float64 to avoid extra casts in Rakau wrapper.
    all_x = jnp.concatenate([q.real, c.real]).astype(jnp.float64)
    all_y = jnp.concatenate([q.imag, c.imag]).astype(jnp.float64)
    zeros_c = jnp.zeros(n_c, dtype=jnp.float64)
    use_gpu = jax.default_backend() == "gpu"

    def _bh_from_mass(mass_q):
        all_charges = jnp.concatenate([jnp.asarray(mass_q, dtype=jnp.float64), zeros_c])
        potentials = _bh_rakau.barnes_hut_2d(
            all_x,
            all_y,
            all_charges,
            eps2=eps2,
            theta=theta,
            use_gpu=use_gpu,
            backend="rakau",
        )
        return potentials[n_q:].astype(jnp.float64)

    # Fast path for rank-1 generators (preserves existing performance)
    if generator_rank == 1:
        potentials_c = _bh_from_mass((jnp.abs(b[:, 0]) ** 2))
        g_squared = jnp.abs(g[:, 0]) ** 2
        return potentials_c * g_squared

    # Higher-rank case: exploit Hermitian symmetry to cut barnes-hut calls roughly in half.
    row_squared_norms = jnp.zeros(n_c, dtype=jnp.float64)
    # Diagonal terms (purely real masses)
    for a in range(generator_rank):
        mass_q = b[:, a] * jnp.conj(b[:, a])
        phi_real = _bh_from_mass(jnp.real(mass_q))
        cross_g = g[:, a] * jnp.conj(g[:, a])
        row_squared_norms = row_squared_norms + cross_g.real * phi_real
    # Off-diagonal pairs; (a,b) and (b,a) contribute the same real value
    for a in range(generator_rank):
        for b_idx in range(a + 1, generator_rank):
            mass_q = b[:, a] * jnp.conj(b[:, b_idx])
            phi_real = _bh_from_mass(jnp.real(mass_q))
            phi_imag = _bh_from_mass(jnp.imag(mass_q))
            cross_g = g[:, a] * jnp.conj(g[:, b_idx])
            contrib = cross_g.real * phi_real - cross_g.imag * phi_imag
            row_squared_norms = row_squared_norms + 2.0 * contrib

    return row_squared_norms


class CauchyCUR:
    """
    Highly optimized JAX-based LU decomposition for Cauchy matrices.
    Uses fixed-size arrays to avoid recompilation.
    Stores indices only, not L/U columns.
    """
    def __init__(self, c, q, g, b, use_jit=True, profile=False, block_size=1, barnes_hut_theta=1, barnes_hut_eps2=1e-14, time_steps=False, use_exact_norm=False):
        """
        Initialize LU decomposition.
        
        Args:
            c, q, g, b: Cauchy matrix parameters
            use_jit: Use JIT compilation (default: True)
            profile: Enable profiling (disables JIT)
            block_size: Number of pivots to sample before recomputing row norms (default: 1)
            time_steps: Enable detailed timing of each step (default: False)
            use_exact_norm: Use exact norm computation instead of Barnes-Hut (default: False)
        """
        self.c = jnp.asarray(c)
        self.q = jnp.asarray(q)
        g = jnp.asarray(g)
        b = jnp.asarray(b)
        
        # Convert 1D to 2D if needed
        if g.ndim == 1:
            g = g[:, None]
        if b.ndim == 1:
            b = b[:, None]
        
        self.g = g
        self.b = b

        if self.g.shape[0] != self.c.shape[0]:
            raise ValueError("g must have the same length as c")
        if self.b.shape[0] != self.q.shape[0]:
            raise ValueError("b must have the same length as q")
        if self.g.shape[1] != self.b.shape[1]:
            raise ValueError("g and b must share the same generator rank")
        self.generator_rank = self.g.shape[1]
        
        self.n = c.size
        self.m = q.size
        
        self.use_jit = use_jit and not profile
        self.profile = profile
        self.time_steps = time_steps
        self.block_size = block_size
        self.barnes_hut_theta = barnes_hut_theta
        self.barnes_hut_eps2 = barnes_hut_eps2
        self.use_exact_norm = use_exact_norm
        
        # Timing data (only populated if time_steps=True)
        self.timing_data = None
        
        # Fixed-size arrays (padded with sentinel values)
        self.column_chosen = jnp.zeros(self.m, dtype=bool)
        self.row_chosen = jnp.zeros(self.n, dtype=bool)
        
        # Store row_squared_norms and the rank at which they were computed
        self.row_squared_norms = None
        self.frobenius_norms = []
        self.frobenius_norms_rank = []
        self.g_b_norms = []
        self.max_row_norms = []
    def build(self, rank: int, use_generator_norm_sampling=False, frob_norm_relative_tolerance=1e-15,
              column_sampling='random', row_sampling='random', frob_tolerance=None,
              residual_norm_kind='frobenius'):
        """
        Build the LU decomposition up to the specified rank.
        
        Optimized: 
        - Index updates are done without blocking until the end.
        - Reuses pre-computed row_squared_norms for pivot selection to avoid redundant computation.
        
        Args:
            rank: Target rank
            use_generator_norm_sampling: If True, use row norms of g @ b^H for sampling (no BH needed)
            gb_tolerance: float or None - stop early when g_b norm falls below this tolerance 
                         (if None, no early stopping; tolerance is compared to norm, not squared)
        """
        # Set default tolerance if not provided (consistent with other methods)
        if residual_norm_kind not in ('frobenius', 'max_row'):
            raise ValueError("residual_norm_kind must be 'frobenius' or 'max_row'")

        if frob_norm_relative_tolerance is None:
            frob_norm_relative_tolerance_val = -1.0  # Negative means no early stopping
        elif residual_norm_kind == 'frobenius':
            frob_norm_relative_tolerance_val = frob_norm_relative_tolerance * frob_norm_from_uv(self.g, self.b)
        else:
            frob_norm_relative_tolerance_val = None
        
        i_indices = jnp.zeros(rank, dtype=jnp.int32)
        j_indices = jnp.zeros(rank, dtype=jnp.int32)
        
        actual_rank = rank
        frob_tol_sq = None
        if frob_tolerance is not None and frob_tolerance > 0:
            frob_tol_sq = float(frob_tolerance) ** 2

        for iter in range(rank):
            if use_generator_norm_sampling:
                # Use row norms of g @ b^H matrix (computed efficiently without forming full matrix)
                row_weights = _compute_row_norms_gb_matrix(self.g, self.b)
                # Frobenius norm of g @ b^H is sqrt(sum of all row norms squared)
                frobenius_norm_sq = jnp.sum(row_weights)
                self.frobenius_norms_squared.append(frobenius_norm_sq)
                self.frobenius_norms_squared_rank.append(None) # Just don't compute in this case
                self.g_b_norms.append(frob_norm_from_uv(self.g, self.b))
                # Store row_weights as row_squared_norms for compatibility
                self.row_squared_norms = row_weights
                max_row_norm = jnp.sqrt(jnp.max(row_weights))
                self.max_row_norms.append(max_row_norm)
                if residual_norm_kind == 'max_row' and frob_norm_relative_tolerance_val is None:
                    frob_norm_relative_tolerance_val = float(max_row_norm) * frob_norm_relative_tolerance
                # Pick pivot using generator norms
                i, j, row, col = _pick_pivot_with_generator_norms(
                    self.c, self.q, self.g, self.b, self.column_chosen, self.row_chosen,
                    column_sampling=column_sampling, row_sampling=row_sampling, rng_key=None
                )
            else:
                # Compute row_squared_norms before picking pivot (needed for pivot selection)
                if self.use_exact_norm:
                    # Use exact norm computation from full matrix
                    C_full = _make_C(self.c, self.q, self.g, self.b)
                    # For row_squared_norms, compute row norms from full matrix
                    row_squared_norms = jnp.sum(jnp.abs(C_full)**2, axis=1)
                    frobenius_norm_sq = jnp.linalg.norm(C_full, 'fro')**2
                else:
                    # Use Barnes-Hut approximation
                    row_squared_norms = barnes_hut_matvec(self.c, self.q, self.g, self.b,
                                                 theta=self.barnes_hut_theta, eps2=self.barnes_hut_eps2)
                    frobenius_norm_sq = jnp.sum(row_squared_norms)
                self.frobenius_norms.append(jnp.sqrt(frobenius_norm_sq))
                max_row_norm = jnp.sqrt(jnp.max(row_squared_norms))
                self.max_row_norms.append(max_row_norm)
                if residual_norm_kind == 'max_row' and frob_norm_relative_tolerance_val is None:
                    frob_norm_relative_tolerance_val = float(max_row_norm) * frob_norm_relative_tolerance

                # Store row_squared_norms for pivot selection
                self.row_squared_norms = row_squared_norms
                self.g_b_norms.append(frob_norm_from_uv(self.g, self.b))
                if frob_tol_sq is not None and float(frobenius_norm_sq) <= frob_tol_sq:
                    actual_rank = iter
                    break
                # Use pre-computed row_squared_norms to avoid redundant computation
                # (This fixes a performance bug where _pick_pivot() was recomputing row norms)
                # Use None to use default static key - this avoids recompilation
                # Randomness comes from changing probability distributions, not key state
                i, j, row, col = _pick_pivot_with_row_squared_norms(
                    self.c, self.q, self.g, self.b, self.column_chosen, self.row_chosen,
                    row_squared_norms,
                    column_sampling=column_sampling, row_sampling=row_sampling,
                    rng_key=None)  # Use default static key to avoid recompilation
            self.g, self.b = step(self.c, self.q, self.g, self.b, i, j, row, col)
            
            # Compute and store Frobenius norm AFTER the step (to see how it decreases)
            # Optimized: batch the updates without blocking (JAX will optimize these)
            self.column_chosen = self.column_chosen.at[j].set(True)
            self.row_chosen = self.row_chosen.at[i].set(True)
            # Zero out chosen rows/columns in g and b (broadcast row_chosen/column_chosen to match g/b shapes)
            # self.g has shape (n, generator_rank), self.row_chosen has shape (n,)
            # self.b has shape (m, generator_rank), self.column_chosen has shape (m,)
            row_chosen_expanded = jnp.expand_dims(self.row_chosen, axis=1)  # (n, 1)
            column_chosen_expanded = jnp.expand_dims(self.column_chosen, axis=1)  # (m, 1)
            self.g = jnp.where(row_chosen_expanded, 0.0, self.g)
            self.b = jnp.where(column_chosen_expanded, 0.0, self.b)
            # Optimized: update indices without blocking - only block at the end
            i_indices = i_indices.at[iter].set(i)
            j_indices = j_indices.at[iter].set(j)
            
            # Check early stopping condition (compare norm, not squared)
            norms_for_stop = self.frobenius_norms if residual_norm_kind == 'frobenius' else self.max_row_norms
            if frob_norm_relative_tolerance_val is not None:
                if frob_norm_relative_tolerance_val > 0 and len(norms_for_stop) > 0:
                    current_norm = float(norms_for_stop[-1])
                    if current_norm < frob_norm_relative_tolerance_val:
                        actual_rank = iter + 1
                        break
                elif frob_norm_relative_tolerance_val < 0 and len(norms_for_stop) > 0:
                    actual_rank = iter + 1
                    break
        
        # Block only once at the end to ensure all updates are complete
        i_indices.block_until_ready()
        j_indices.block_until_ready()
        
        # Truncate indices if we stopped early
        if actual_rank < rank:
            i_indices = i_indices[:actual_rank]
            j_indices = j_indices[:actual_rank]
        
        return i_indices, j_indices
    
    def _is_cpu_backend(self):
        """Check if JAX is using CPU backend."""
        try:
            devices = jax.devices()
            return all(d.device_kind == 'cpu' for d in devices)
        except:
            return True  # Default to CPU if detection fails
    
    def build_in_block(self, rank: int, oversample_factor=10):
        """
        Build the LU decomposition using block pivot selection and block step updates.
        
        More efficient than build() for large problems as it:
        1. Selects pivots sequentially within blocks
        2. Applies steps sequentially within blocks
        3. Recomputes row_squared_norms only every block_size iterations
        
        Args:
            rank: Total rank to build
            oversample_factor: Oversampling factor for pivot selection (default: 10)
        
        Returns:
            i_indices, j_indices: (rank,) arrays of selected pivot indices
        """
        return self.build_in_block_sequential(rank, oversample_factor)

    def build_in_block_sequential(self, rank: int, oversample_factor=10):
        """
        Alternative implementation: compute row_squared_norms once, then select pivots sequentially.
        
        Algorithm:
        1. Compute row_squared_norms once per block
        2. For block_size iterations:
           - Select single pivot using pre-computed row_squared_norms
           - Apply step update
           - Update indices
        3. Recompute row_squared_norms for next block
        
        This is different from build_in_block which selects all block_size pivots at once.
        
        Args:
            rank: Total rank to build
            oversample_factor: Oversampling factor (not used in this implementation, kept for API compatibility)
        
        Returns:
            i_indices, j_indices: (rank,) arrays of selected pivot indices
        """
        import time
        
        # Initialize timing data if requested
        if self.time_steps:
            self.timing_data = {
                'row_squared_norms': [],
                'pick_pivot': [],
                'step': [],
                'update_indices': [],
                'total_per_block': [],
                'num_blocks': 0
            }
        
        i_indices_all = jnp.zeros(rank, dtype=jnp.int32)
        j_indices_all = jnp.zeros(rank, dtype=jnp.int32)
        current_rank = jnp.array(0, dtype=jnp.int32)
        
        # Store initial g_b_norm at rank 0 (before any processing)
        self.g_b_norms.append(float(frob_norm_from_uv(self.g, self.b)))
        
        # Pre-compute number of blocks needed
        num_blocks = (rank + self.block_size - 1) // self.block_size
        
        for block_idx in range(num_blocks):
            block_start_time = time.perf_counter() if self.time_steps else None
            current_rank_int = int(current_rank)
            remaining = rank - current_rank_int
            block_size_this = min(int(self.block_size), remaining)
            
            # Compute row_squared_norms once for this block
            t0 = time.perf_counter() if self.time_steps else None
            if self.use_exact_norm:
                # Use exact norm computation from full matrix
                C_full = _make_C(self.c, self.q, self.g, self.b)
                row_squared_norms = jnp.sum(jnp.abs(C_full)**2, axis=1)
                frobenius_norm_sq = jnp.linalg.norm(C_full, 'fro')**2
            else:
                # Use Barnes-Hut approximation
                row_squared_norms = barnes_hut_matvec(self.c, self.q, self.g, self.b,
                                             theta=self.barnes_hut_theta, eps2=self.barnes_hut_eps2)
                frobenius_norm_sq = jnp.sum(row_squared_norms)
            row_squared_norms.block_until_ready()
            # Store row_squared_norms and the rank at which they were computed
            self.frobenius_norms_squared.append(frobenius_norm_sq)
            self.frobenius_norms_squared_rank.append(int(current_rank))
            if self.time_steps:
                self.timing_data['row_squared_norms'].append(time.perf_counter() - t0)
            
            # Process block using JIT-compiled sequential loop
            # But compute g_b_norms after each iteration for tracking
            t_block = time.perf_counter() if self.time_steps else None
            
            # Process each iteration separately to track g_b_norms per iteration
            for iter_in_block in range(block_size_this):
                # Process single iteration
                single_iter_jax = jnp.array(1, dtype=jnp.int32)
                (self.g, self.b, self.column_chosen, self.row_chosen, 
                 i_indices_all, j_indices_all, current_rank) = _process_sequential_block(
                    self.c, self.q, self.g, self.b,
                    self.column_chosen, self.row_chosen, row_squared_norms,
                    i_indices_all, j_indices_all, current_rank,
                    single_iter_jax,
                    column_sampling='random', row_sampling='random',
                    rng_key=None
                )
                # Store g_b_norms after each iteration (like build() does)
                self.g_b_norms.append(float(frob_norm_from_uv(self.g, self.b)))
            if self.time_steps:
                self.g.block_until_ready()
                self.b.block_until_ready()
                i_indices_all.block_until_ready()
                j_indices_all.block_until_ready()
                block_time = time.perf_counter() - t_block
                # Approximate timing per operation (for compatibility with old timing structure)
                self.timing_data['pick_pivot'].extend([block_time / (3 * block_size_this)] * block_size_this)
                self.timing_data['step'].extend([block_time / (3 * block_size_this)] * block_size_this)
                self.timing_data['update_indices'].extend([block_time / (3 * block_size_this)] * block_size_this)
            
            if self.time_steps:
                block_end_time = time.perf_counter()
                self.timing_data['total_per_block'].append(block_end_time - block_start_time)
                self.timing_data['num_blocks'] += 1
            
            # Break if we've reached the rank
            if int(current_rank) >= rank:
                break
        
        # Block to ensure all updates are complete
        i_indices_all.block_until_ready()
        j_indices_all.block_until_ready()
        return i_indices_all, j_indices_all

# Helper functions for sampling (JIT-compiled for efficiency)
@partial(jax.jit, static_argnames=('size', 'sampling',))
def _sample_index(size, weights, chosen_mask, sampling, rng_key):
    """
    Sample an index (row or column) using the specified strategy.
    
    Args:
        size: Size of the index space
        weights: Weights for each index (squared norms or weights from generator)
        chosen_mask: Boolean array indicating which indices are already chosen
        sampling: 'uniform', 'greedy', or 'random'
        rng_key: JAX RNG key
    
    Returns:
        Sampled index
    """
    if sampling == 'uniform':
        not_chosen = (~chosen_mask).astype(jnp.float32)
        sum_not_chosen = jnp.sum(not_chosen)
        # Avoid division by zero - if all items are chosen, return first available (shouldn't happen)
        probs = jnp.where(sum_not_chosen > 0, not_chosen / sum_not_chosen, not_chosen)
        return jax.random.choice(rng_key, jnp.arange(size, dtype=jnp.int32), p=probs)
    elif sampling == 'greedy':
        weights_masked = jnp.where(chosen_mask, -jnp.inf, weights)
        return jnp.argmax(weights_masked).astype(jnp.int32)
    else:  # random
        weights_masked = jnp.where(chosen_mask, 0.0, weights)
        weights_masked = jnp.maximum(weights_masked, 0.0)
        sum_weights = jnp.sum(weights_masked)
        # Avoid division by zero - if all weights are zero, fall back to uniform
        probs = jnp.where(sum_weights > 0, weights_masked / sum_weights, 
                         (~chosen_mask).astype(jnp.float32) / jnp.maximum(jnp.sum(~chosen_mask), 1.0))
        return jax.random.choice(rng_key, jnp.arange(size, dtype=jnp.int32), p=probs)

@partial(jax.jit, static_argnames=('size', 'sampling', 'max_candidates'))
def _sample_index_candidates(size, weights, chosen_mask, sampling, max_candidates, rng_key):
    """
    Sample multiple candidate indices (for block selection).
    
    Args:
        size: Size of the index space
        weights: Weights for each index (squared norms or weights from generator)
        chosen_mask: Boolean array indicating which indices are already chosen
        sampling: 'uniform', 'greedy', or 'random'
        max_candidates: Maximum number of candidates to sample
        rng_key: JAX RNG key
    
    Returns:
        Array of sampled candidate indices
    """
    if sampling == 'uniform':
        available = ~chosen_mask
        n_available = jnp.sum(available)
        available_indices = jnp.arange(size, dtype=jnp.int32)
        available_padded = jnp.concatenate([
            available_indices,
            jnp.zeros(max(0, max_candidates - size), dtype=jnp.int32)
        ])[:max_candidates]
        return jnp.where(
            n_available > 0,
            jax.random.choice(rng_key, available_padded, shape=(max_candidates,), replace=True),
            jnp.full(max_candidates, -1, dtype=jnp.int32)
        )
    elif sampling == 'greedy':
        weights_masked = jnp.where(chosen_mask, -jnp.inf, weights)
        sorted_indices = jnp.argsort(weights_masked).astype(jnp.int32)
        return sorted_indices[-max_candidates:][::-1]
    else:  # random
        weights_masked = jnp.where(chosen_mask, 0.0, weights)
        weights_masked = jnp.maximum(weights_masked, 0.0)
        probs = weights_masked / (jnp.sum(weights_masked) )
        return jax.random.choice(rng_key, jnp.arange(size, dtype=jnp.int32), 
                                shape=(max_candidates,), p=probs, replace=True)


@jax.jit
def step(c, q, g, b, i, j, row, col):
    """
    Compute one step of the LU decomposition for rank-1 or rank-r generators.
    """
    d = row[j]
    g_i = g[i]  # (r,)
    b_j = b[j]  # (r,)
    col_factor = (col / d)[:, None]  # (n, 1)
    row_factor = (row / d)[:, None]  # (m, 1)
    g_new = g - col_factor * g_i
    b_new = b - row_factor * b_j
    return g_new, b_new

@partial(jax.jit, static_argnames=('column_sampling', 'row_sampling'))
def _process_sequential_block(c, q, g, b, column_chosen, row_chosen, row_squared_norms, 
                              i_indices_all, j_indices_all, current_rank, 
                              block_size_this, column_sampling='random', row_sampling='random', rng_key=None):
    """
    Process a block of pivots sequentially using JIT-compiled fori_loop.
    
    This function is JIT-compiled and processes block_size_this pivots in a single loop.
    Uses fori_loop with dynamic upper bound. Note: will recompile for different block_size_this values,
    but this is acceptable since typically only 1-2 different values occur (regular block size and final smaller block).
    """
    if rng_key is None:
        rng_key = _STATIC_RNG_KEY
    
    def loop_body(idx, carry):
        g, b, column_chosen, row_chosen, i_indices_all, j_indices_all, current_rank = carry
        
        # Pick pivot using pre-computed row_squared_norms
        i, j, row, col = _pick_pivot_with_row_squared_norms(
            c, q, g, b, column_chosen, row_chosen, row_squared_norms,
            column_sampling, row_sampling, rng_key
        )
        
        # Apply step update
        g_new, b_new = step(c, q, g, b, i, j, row, col)
        
        # Update chosen arrays
        column_chosen_new = column_chosen.at[j].set(True)
        row_chosen_new = row_chosen.at[i].set(True)
        
        # Zero out chosen rows/columns in g and b (matching build() behavior)
        row_chosen_expanded = jnp.expand_dims(row_chosen_new, axis=1)  # (n, 1)
        column_chosen_expanded = jnp.expand_dims(column_chosen_new, axis=1)  # (m, 1)
        g_new = jnp.where(row_chosen_expanded, 0.0, g_new)
        b_new = jnp.where(column_chosen_expanded, 0.0, b_new)
        
        # Update output indices
        i_indices_all_new = i_indices_all.at[current_rank].set(i)
        j_indices_all_new = j_indices_all.at[current_rank].set(j)
        
        # Update current rank
        current_rank_new = current_rank + 1
        
        return (g_new, b_new, column_chosen_new, row_chosen_new, 
                i_indices_all_new, j_indices_all_new, current_rank_new)
    
    initial_carry = (g, b, column_chosen, row_chosen, i_indices_all, j_indices_all, current_rank)
    # fori_loop handles block_size_this=0 correctly (no iterations)
    final_carry = jax.lax.fori_loop(0, block_size_this, loop_body, initial_carry)
    
    return final_carry

@partial(jax.jit, static_argnames=('n_candidates', 'column_sampling', 'row_sampling'))
def _update_with_upper_bound_and_rejection(c, q, g, b, column_chosen, row_chosen, row_squared_norms_ub, col_pivots, row_pivots, current_rank,
                              n_candidates, column_sampling='random', row_sampling='random', rng_key=None):
    rng_key, key_rows, key_acc, key_cols = jax.random.split(rng_key, 4)
    if row_sampling == 'greedy':
        i = jnp.argmax(row_squared_norms_ub)
        found = True
        row = _row_i(c, q, g, b, i)
    else:
        row_candidates = _sample_index_candidates(
            c.shape[0], row_squared_norms_ub, row_chosen, row_sampling, n_candidates, key_rows
        )
        row_block = _row_block(c, q, g, b, row_candidates)
        row_sq = jnp.sum(jnp.abs(row_block) ** 2, axis=1)
        ub = row_squared_norms_ub[row_candidates]
        ratios = jnp.minimum(1.0, jnp.where(ub > 0, row_sq / ub, 0.0))

        accept_mask = jax.random.uniform(key_acc, (n_candidates,)) < ratios

        first_idx = jnp.argmax(accept_mask)
        found = jnp.any(accept_mask)
        i = row_candidates[first_idx]
        row = row_block[first_idx]
    j = _sample_index(q.size, jnp.abs(row) ** 2, column_chosen,
                      column_sampling, key_cols)

    def _update(_):
        col = _column_j(c, q, g, b, j)
        d = row[j]
        g_i = g[i]
        b_j = b[j]
        g_new = g - (col / d)[:, None] * g_i
        b_new = b - (row / d)[:, None] * b_j
        return (g_new, b_new,
                column_chosen.at[j].set(True),
                row_chosen.at[i].set(True),
                current_rank + 1)
    g_new, b_new, column_chosen_new, row_chosen_new, current_rank_new = jax.lax.cond(
        found, _update,
        lambda _: (g, b, column_chosen, row_chosen, current_rank),
        operand=None,
    )
    col_pivots_new = col_pivots.at[current_rank].set(j)
    row_pivots_new = row_pivots.at[current_rank].set(i)
    return g_new, b_new, column_chosen_new, row_chosen_new, col_pivots_new, row_pivots_new, current_rank_new, found

@partial(jax.jit, static_argnames=('column_sampling', 'row_sampling'))
def _pick_pivot_with_row_squared_norms(c, q, g, b, chosen_columns, chosen_rows, row_squared_norms, column_sampling='random', row_sampling='random', rng_key=None):
    """
    Pick a pivot using pre-computed row_squared_norms.
    
    This version avoids recomputing row_squared_norms, useful when row_squared_norms
    are computed once per block and reused for multiple pivots.
    """
    if rng_key is None:
        rng_key = _STATIC_RNG_KEY
    
    # Sample row using helper function
    i = _sample_index(c.size, row_squared_norms, chosen_rows, row_sampling, rng_key)
    
    # Compute row and sample column
    row = _row_i(c, q, g, b, i)
    row_squared = jnp.abs(row)**2
    j = _sample_index(q.size, row_squared, chosen_columns, column_sampling, rng_key)
    
    col = _column_j(c, q, g, b, j)
    return i, j, row, col


@jax.jit
def _compute_row_norms_gb_matrix(g, b):
    """
    Compute row norms of g @ b^H efficiently without forming the full (n, m) matrix.

    For each row i: ||(g @ b^H)[i,:]||^2 = sum_j |sum_k g[i,k] * conj(b[j,k])|^2

    Efficient computation:
    ||row_i||^2 = sum_k sum_l g[i,k] * conj(g[i,l]) * (b^H @ b)[k,l]
                 = real(sum((g @ (b^H @ b)) * conj(g), axis=1))

    Args:
        g: (n, r) array
        b: (m, r) array
    
    Returns:
        row_norms_sq: (n,) array with ||(g @ b^H)[i,:]||^2 for each row i
    """
    # Compute b^H @ b once: (r, m) @ (m, r) = (r, r)
    b_H_b = jnp.conj(b).T @ b  # (r, r)
    
    # Row norms: ||row_i||^2 = real(sum((g @ b_H_b) * conj(g), axis=1))
    g_conj = jnp.conj(g)  # (n, r)
    row_norms_sq = jnp.real(jnp.sum((g @ b_H_b) * g_conj, axis=1))  # (n,)
    
    return row_norms_sq



@partial(jax.jit, static_argnames=('column_sampling', 'row_sampling'))
def _pick_pivot_with_generator_norms(c, q, g, b, chosen_columns, chosen_rows, 
                                     column_sampling='random', row_sampling='random', rng_key=None):
    """
    Pick pivot using row norms of g @ b^H matrix for row sampling, 
    but actual column norms (from selected row) for column sampling.
    
    This avoids Barnes-Hut for row norms, but uses the standard method for columns.
    """
    if rng_key is None:
        rng_key = _STATIC_RNG_KEY
    # Sample row using generator norms (no BH needed)
    row_weights = _compute_row_norms_gb_matrix(g, b)
    i = _sample_index(c.shape[0], row_weights, chosen_rows, row_sampling, rng_key)
    
    # Compute actual row and use its squared norm for column sampling (same as standard method)
    row = _row_i(c, q, g, b, i)
    row_squared = jnp.abs(row)**2
    j = _sample_index(q.shape[0], row_squared, chosen_columns, column_sampling, rng_key)
    
    col = _column_j(c, q, g, b, j)
    return i, j, row, col

@partial(jax.jit, static_argnames=('max_rank', 'column_sampling', 'row_sampling', 'allow_extra_after_tol'))
def _build_generator_norm_jitted_inner(c, q, g_init, b_init, max_rank, rng_keys,
                                       column_sampling='random', row_sampling='random',
                                       gb_tolerance_relative=None, allow_extra_after_tol=True):
    """
    Fully JIT-compiled inner loop for generator norm sampling with early stopping.
    
    Uses while_loop to allow early stopping when g_b norm crosses tolerance.
    
    Args:
        c: (n,) complex array
        q: (m,) complex array
        g_init: (n, r) complex array - initial generator matrix
        b_init: (m, r) complex array - initial generator matrix
        max_rank: int - maximum number of iterations
        rng_keys: (max_rank, 2) PRNG key array
        column_sampling: str - sampling method for columns
        row_sampling: str - sampling method for rows
        gb_tolerance: float or None - stop when g_b norm falls below this (if None or negative, run to max_rank; tolerance is compared to norm, not squared)
    
    Returns:
        i_indices: (max_rank,) int32 array
        j_indices: (max_rank,) int32 array
        g_b_norms: (max_rank+1,) float64 array (includes initial norm)
        g_final: (n, r) complex array
        b_final: (m, r) complex array
        actual_rank: int32 - actual number of iterations completed
    """
    n = c.shape[0]
    m = q.shape[0]
    
    # Initialize state
    g = g_init
    b = b_init
    column_chosen = jnp.zeros(m, dtype=jnp.bool_)
    row_chosen = jnp.zeros(n, dtype=jnp.bool_)
    i_indices = jnp.full(max_rank, -1, dtype=jnp.int32)
    j_indices = jnp.full(max_rank, -1, dtype=jnp.int32)
    g_b_norms = jnp.full(max_rank + 1, -1.0, dtype=jnp.float64)
    
    # Compute initial g_b norm
    initial_norm = frob_norm_from_uv(g, b)
    g_b_norms = g_b_norms.at[0].set(initial_norm)
    gb_tolerance = initial_norm * gb_tolerance_relative
    
    def cond_fun(carry):
        iter_count, gb_norm_current, extra_steps_left, _ = carry
        if allow_extra_after_tol:
            in_extra = extra_steps_left >= 0
            continue_tolerance = jnp.where(in_extra, extra_steps_left > 0, gb_norm_current > gb_tolerance)
        else:
            continue_tolerance = gb_norm_current > gb_tolerance
        continue_iter = iter_count < max_rank
        return jnp.logical_and(continue_iter, continue_tolerance)
    
    def body_fun(carry):
        iter_count, _, extra_steps_left, (g, b, column_chosen, row_chosen, i_indices, j_indices, g_b_norms) = carry
        
        # Compute row norms using generator norms
        
        # Pick pivot using generator norms
        i, j, row, col = _pick_pivot_with_generator_norms(
            c, q, g, b, column_chosen, row_chosen,
            column_sampling=column_sampling, row_sampling=row_sampling,
            rng_key=rng_keys[iter_count]
        )
        
        # Apply step update
        g_new, b_new = step(c, q, g, b, i, j, row, col)
        
        # Update chosen arrays
        column_chosen_new = column_chosen.at[j].set(True)
        row_chosen_new = row_chosen.at[i].set(True)
        
        # Zero out chosen rows/columns
        row_chosen_expanded = jnp.expand_dims(row_chosen_new, axis=1)
        column_chosen_expanded = jnp.expand_dims(column_chosen_new, axis=1)
        g_new = jnp.where(row_chosen_expanded, 0.0, g_new)
        b_new = jnp.where(column_chosen_expanded, 0.0, b_new)
        
        # Update indices
        i_indices_new = i_indices.at[iter_count].set(i)
        j_indices_new = j_indices.at[iter_count].set(j)
        
        # Store g_b norm after this iteration
        g_b_norm_current = frob_norm_from_uv(g_new, b_new)
        g_b_norms_new = g_b_norms.at[iter_count + 1].set(g_b_norm_current)

        # Update extra step tracker: once tolerance is hit, allow exactly one more iteration if enabled
        if allow_extra_after_tol:
            hit_tol = jnp.logical_and(gb_tolerance_relative >= 0, g_b_norm_current <= gb_tolerance)
            extra_steps_left_new = jnp.where(extra_steps_left >= 0,
                                             extra_steps_left - 1,
                                             jnp.where(hit_tol, jnp.int32(1), jnp.int32(-1)))
        else:
            extra_steps_left_new = extra_steps_left
        
        state = (g_new, b_new, column_chosen_new, row_chosen_new, 
                 i_indices_new, j_indices_new, g_b_norms_new)
        
        return (iter_count + 1, g_b_norm_current, extra_steps_left_new, state)
    
    initial_state = (g, b, column_chosen, row_chosen, i_indices, j_indices, g_b_norms)
    initial_carry = (0, initial_norm, jnp.int32(-1), initial_state)
    
    final_iter_count, _, _, final_state = jax.lax.while_loop(cond_fun, body_fun, initial_carry)
    g_final, b_final, _, _, i_indices_final, j_indices_final, g_b_norms_final = final_state
    
    return i_indices_final, j_indices_final, g_b_norms_final, g_final, b_final, final_iter_count

def build_with_generator_norm_sampling(c, q, g, b, rank, block_size, 
                                       rng_seed=42, time_blocks=False, use_jitted_inner=False,
                                       relative_gb_tolerance=None, allow_extra_after_tol=True,
                                       column_sampling='random', row_sampling='random'):
    """
    Build CUR decomposition using generator norm sampling (no rejection sampling, no Barnes-Hut).
    
    Simple sequential iteration - no blocks needed since generator norm computation is fast.
    Uses row norms of g @ b^H for pivot selection (computed efficiently without forming full matrix).
    
    Args:
        c: (n,) complex array - first generator vector
        q: (m,) complex array - second generator vector  
        g: (n, r) complex array - generator matrix for rows
        b: (m, r) complex array - generator matrix for columns
        rank: Target rank of the decomposition (maximum)
        block_size: Ignored (kept for API compatibility)
        rng_seed: Random seed for reproducibility
        time_blocks: If True, time different blocks of computation and return timing data
        use_jitted_inner: If True, use fully JIT-compiled inner loop (faster but no detailed timing)
        gb_tolerance: float or None - stop early when g_b norm falls below this tolerance 
                     (if None, no early stopping; tolerance is compared to norm, not squared)
    
    Returns:
        i_indices: (rank,) array of row indices (padded with -1 if stopped early)
        j_indices: (rank,) array of column indices (padded with -1 if stopped early)
        g_b_norms: Array of g_b norms at each step
        ranks_after_step: Array of ranks after each step
        frobenius_norms_squared: Array of Frobenius norms squared at each step (NaN-filled)
        timing_data: (Optional) Dict with timing breakdown if time_blocks=True
    """
    
    # Use fully JIT-compiled version if requested
    if use_jitted_inner:
        # Set default tolerance if not provided (consistent with build_with_rejection_sampling)
        if relative_gb_tolerance is None:
            relative_gb_tolerance = -1.0  # Negative means no early stopping
        
        rng_key = jax.random.PRNGKey(rng_seed)
        rng_keys = jax.random.split(rng_key, rank)
        i_indices_all, j_indices_all, g_b_norms, _, _, actual_rank = _build_generator_norm_jitted_inner(
            c, q, g, b, rank, rng_keys, column_sampling, row_sampling, relative_gb_tolerance,
            allow_extra_after_tol=allow_extra_after_tol
        )
        
        # Create ranks array based on actual iterations
        actual_rank_int = int(actual_rank)
        ranks_after_step = jnp.arange(actual_rank_int + 1, dtype=jnp.int32)
        # Pad g_b_norms to match expected size (only keep actual_rank+1 values)
        if actual_rank_int < rank:
            g_b_norms = g_b_norms[:actual_rank_int + 1]
            i_indices_all = i_indices_all[:actual_rank_int]
            j_indices_all = j_indices_all[:actual_rank_int]
        
        frobenius_norms_squared = jnp.full(len(g_b_norms), jnp.nan, dtype=jnp.float64)
        
        if time_blocks:
            # Return empty timing data for JIT version
            timing_data = {
                'row_norm_computation': [],
                'pivot_selection': [],
                'step_update': [],
                'state_updates': [],
                'total_per_iteration': []
            }
            return (i_indices_all, j_indices_all, g_b_norms, ranks_after_step, 
                    frobenius_norms_squared, timing_data)
        else:
            return (i_indices_all, j_indices_all, g_b_norms, ranks_after_step, 
                    frobenius_norms_squared)
    
    # Original version with detailed timing
    n = c.shape[0]
    m = q.shape[0]
    
    # Initialize timing data structures if timing is enabled
    if time_blocks:
        timing_data = {
            'row_norm_computation': [],
            'pivot_selection': [],
            'step_update': [],
            'state_updates': [],
            'total_per_iteration': []
        }
    
    g_current = g.copy()
    b_current = b.copy()
    column_chosen = jnp.zeros(m, dtype=jnp.bool_)
    row_chosen = jnp.zeros(n, dtype=jnp.bool_)
    
    i_indices_all = jnp.zeros(rank, dtype=jnp.int32)
    j_indices_all = jnp.zeros(rank, dtype=jnp.int32)
    
    # Preallocate g_b_norms array (rank+1 to include initial norm)
    g_b_norms = jnp.zeros(rank + 1, dtype=jnp.float64)
    
    # Compute initial g_b norm at rank 0
    initial_g_b_norm = float(frob_norm_from_uv(g, b))
    g_b_norms = g_b_norms.at[0].set(initial_g_b_norm)
    
    # Set default tolerance if not provided (consistent with build_with_rejection_sampling)
    if relative_gb_tolerance is None:
        relative_gb_tolerance = -1.0  # Negative means no early stopping
    
    # Initialize PRNG key and split for all iterations at once
    rng_key = jax.random.PRNGKey(rng_seed)
    rng_keys = jax.random.split(rng_key, rank)
    
    gb_tolerance_val = initial_g_b_norm * relative_gb_tolerance
    extra_steps_left = -1 if allow_extra_after_tol else 0  # countdown once tolerance is hit
    # Simple sequential loop - no blocks needed
    actual_rank = rank
    for iter in range(rank):
        iter_start_time = time.time() if time_blocks else None
        
        # Compute row norms using generator norms (very fast - O(r^2))
        if time_blocks:
            t0 = time.time()
        
        row_weights = _compute_row_norms_gb_matrix(g_current, b_current)
        
        if time_blocks:
            row_weights.block_until_ready()
            timing_data['row_norm_computation'].append(time.time() - t0)
        
        # Pick pivot using generator norms (no rejection sampling)
        if time_blocks:
            t0 = time.time()
        
        i, j, row, col = _pick_pivot_with_generator_norms(
            c, q, g_current, b_current, column_chosen, row_chosen,
            column_sampling=column_sampling, row_sampling=row_sampling, rng_key=rng_keys[iter]
        )
        
        if time_blocks:
            timing_data['pivot_selection'].append(time.time() - t0)
        
        # Apply step update
        if time_blocks:
            t0 = time.time()
        
        g_current, b_current = step(c, q, g_current, b_current, i, j, row, col)
        
        if time_blocks:
            timing_data['step_update'].append(time.time() - t0)
        
        # Update state
        if time_blocks:
            t0 = time.time()
        
        column_chosen = column_chosen.at[j].set(True)
        row_chosen = row_chosen.at[i].set(True)
        
        # Zero out chosen rows/columns
        row_chosen_expanded = jnp.expand_dims(row_chosen, axis=1)
        column_chosen_expanded = jnp.expand_dims(column_chosen, axis=1)
        g_current = jnp.where(row_chosen_expanded, 0.0, g_current)
        b_current = jnp.where(column_chosen_expanded, 0.0, b_current)
        
        # Update indices
        i_indices_all = i_indices_all.at[iter].set(i)
        j_indices_all = j_indices_all.at[iter].set(j)
        
        # Store g_b norm after this iteration
        g_b_norm_current = frob_norm_from_uv(g_current, b_current)
        g_b_norm_current_val = float(g_b_norm_current)
        g_b_norms = g_b_norms.at[iter + 1].set(g_b_norm_current_val)
        
        if time_blocks:
            g_b_norm_current.block_until_ready()
            i_indices_all.block_until_ready()
            j_indices_all.block_until_ready()
            timing_data['state_updates'].append(time.time() - t0)
            if iter_start_time is not None:
                timing_data['total_per_iteration'].append(time.time() - iter_start_time)
        
        # Check early stopping condition with optional one extra step after first hit
        if gb_tolerance_val > 0 and g_b_norm_current_val < gb_tolerance_val:
            if allow_extra_after_tol:
                if extra_steps_left < 0:
                    extra_steps_left = 1  # allow one more iteration
            else:
                actual_rank = iter + 1
                break

        if allow_extra_after_tol:
            if extra_steps_left >= 0:
                extra_steps_left -= 1
                if extra_steps_left == 0:
                    actual_rank = iter + 1
                    break
    
    # Block to ensure all updates are complete
    g_b_norms.block_until_ready()
    i_indices_all.block_until_ready()
    j_indices_all.block_until_ready()
    
    # Only keep norms for actual iterations
    if actual_rank < rank:
        g_b_norms = g_b_norms[:actual_rank + 1]
        i_indices_all = i_indices_all[:actual_rank]
        j_indices_all = j_indices_all[:actual_rank]
    
    ranks_after_step = jnp.arange(actual_rank + 1, dtype=jnp.int32)
    
    # Return NaN array for frobenius_norms_squared (not computed, but consistent with API)
    frobenius_norms_squared = jnp.full(actual_rank + 1, jnp.nan, dtype=jnp.float64)
    
    if time_blocks:
        return (i_indices_all, j_indices_all, g_b_norms, ranks_after_step, 
                frobenius_norms_squared, timing_data)
    else:
        return (i_indices_all, j_indices_all, g_b_norms, ranks_after_step, 
                frobenius_norms_squared)



def build_with_upper_bound(c, q, g, b, rank, n_candidates, compute_row_norms_squared,
	                                 rng_seed=42, time_blocks=False, gb_tolerance=None,
	                                 allow_extra_after_tol=True, column_sampling='random',
	                                 row_sampling='random',
	                                 frob_norm_relative_tolerance=None,
	                                 return_true_norms: bool = False,
	                                 residual_norm_kind: str = 'frobenius',
	                                 return_max_row_norms: bool = False):
    if residual_norm_kind not in ('frobenius', 'max_row'):
        raise ValueError("residual_norm_kind must be 'frobenius' or 'max_row'")
    if time_blocks:
        timing_data = {
            'row_norm_computation': [],
            'frobenius_norm_computation': [],
            'rejection_sampling_block': [],
            'index_extraction': [],
            'state_updates': [],
            'total_per_iteration': []
        }

    n = c.shape[0]
    m = q.shape[0]

    g_current = g.copy()
    b_current = b.copy()
    if _ENABLE_UB_GENERATOR_STABILIZATION:
        g_current, b_current = _stabilize_generators(g_current, b_current)
    column_chosen = jnp.zeros(m, dtype=jnp.bool_)
    row_chosen = jnp.zeros(n, dtype=jnp.bool_)
    col_pivots = jnp.zeros(rank, dtype=jnp.int32)
    row_pivots = jnp.zeros(rank, dtype=jnp.int32)

    max_iterations = rank * 5
    iteration_count = 0
    current_rank = 0
    # Store ||F||_F for ranks 0..current_rank (inclusive).
    frobenius_norms = jnp.zeros(rank + 1, dtype=jnp.float64)
    max_row_norms = jnp.zeros(rank + 1, dtype=jnp.float64)
    accepted = []
    end_next_step = False

    # Initialize JAX random key (proper JAX way - split for each iteration)
    rng_key = jax.random.PRNGKey(rng_seed)

    @jax.jit
    def update_frobenius_norm(row_squared_norms, current_rank, frobenius_norms):
        frobenius_norms = frobenius_norms.at[current_rank].set(jnp.sqrt(jnp.sum(row_squared_norms)))
        return frobenius_norms
    @jax.jit
    def update_max_row_norm(row_squared_norms, current_rank, max_row_norms):
        max_row_norms = max_row_norms.at[current_rank].set(jnp.sqrt(jnp.max(row_squared_norms)))
        return max_row_norms

    row_squared_norms_init = compute_row_norms_squared(g_current, b_current)
    frobenius_norms = update_frobenius_norm(row_squared_norms_init, current_rank, frobenius_norms)
    max_row_norms = update_max_row_norm(row_squared_norms_init, current_rank, max_row_norms)
    # Handle None tolerance (disable early stopping)
    if frob_norm_relative_tolerance is None:
        norm_tolerance = -1
    else:
        norm_base = max_row_norms[0] if residual_norm_kind == 'max_row' else frobenius_norms[0]
        norm_tolerance = float(norm_base) * frob_norm_relative_tolerance


    while current_rank < rank and iteration_count < max_iterations:
        iter_start_time = time.time() if time_blocks else None


        # scale = np.sqrt(0.5 * (np.mean(np.abs(g_current)) + np.mean(np.abs(b_current))))
        # eps = np.finfo(float).tiny*0

        # mc = np.mean(np.abs(c)) + eps
        # mq = np.mean(np.abs(q)) + eps
        # mg = np.mean(np.abs(g_current)) + eps
        # mb = np.mean(np.abs(b_current)) + eps

        # scale = np.sqrt((mc * mq) / (mg * mb))
        # c = c / scale**2
        # q = q / scale**2
        # g_current = g_current / scale
        # b_current = b_current / scale
        # plan = GbUpperBoundPlan(np.asarray(c), np.asarray(q), 5, max_leaf=64)
        # def compute_row_norms_squared(g_current, b_current):
        #     ub_np = plan.compute((g_current), (b_current))
        #     return ub_np
        # if current_rank % 10 == 0:
        #     print('mc: ', mc, 'mq: ', mq, 'mg: ', mg, 'mb: ', mb, 'scale: ', scale)
    

        if time_blocks:
            t0 = time.time()
        if _ENABLE_UB_GENERATOR_STABILIZATION:
            g_current, b_current = _stabilize_generators(g_current, b_current)
        row_squared_norms = compute_row_norms_squared(g_current, b_current)
        row_squared_norms = jnp.where(row_chosen, 0.0, row_squared_norms)
        frobenius_norms = update_frobenius_norm(row_squared_norms, current_rank, frobenius_norms)
        max_row_norms = update_max_row_norm(row_squared_norms, current_rank, max_row_norms)

        if return_true_norms:
            C_full = _make_C(jnp.asarray(c), jnp.asarray(q), jnp.asarray(g_current), jnp.asarray(b_current))
            frobenius_norm_sq_true = jnp.linalg.norm(C_full, 'fro')**2
            frobenius_norms = update_frobenius_norm(frobenius_norm_sq_true, current_rank, frobenius_norms)

        if time_blocks:
            row_squared_norms.block_until_ready()
            timing_data['row_norm_computation'].append(time.time() - t0)
            t0 = time.time()
            
        # if float(frobenius_norms[current_rank]) <= frobenius_norm_tolerance:
        #     print('mc: ', mc, 'mq: ', mq, 'mg: ', mg, 'mb: ', mb, 'scale: ', scale)

        current_norms = max_row_norms if residual_norm_kind == 'max_row' else frobenius_norms
        if end_next_step:
            break
        elif float(current_norms[current_rank]) <= norm_tolerance and allow_extra_after_tol:
            end_next_step = True
        elif float(current_norms[current_rank]) <= norm_tolerance and not allow_extra_after_tol:
            break


        # Split the key for this iteration (proper JAX way)
        rng_key, rng_key_iter = jax.random.split(rng_key)
        g_current, b_current, column_chosen, row_chosen, col_pivots, row_pivots, current_rank, accepted_new = _update_with_upper_bound_and_rejection(
            c, q, g_current, b_current, column_chosen, row_chosen, row_squared_norms, col_pivots, row_pivots, current_rank,
            n_candidates, column_sampling, row_sampling, rng_key_iter
        )
        accepted.append(accepted_new)
        iteration_count += 1
        if time_blocks:
            g_current[0].block_until_ready()
            timing_data['rejection_sampling_block'].append(time.time() - t0)

    # Ensure the final residual norm (after `current_rank` accepted pivots) is recorded.
    row_squared_norms_final = compute_row_norms_squared(g_current, b_current)
    row_squared_norms_final = jnp.where(row_chosen, 0.0, row_squared_norms_final)
    frobenius_norms = update_frobenius_norm(row_squared_norms_final, current_rank, frobenius_norms)
    max_row_norms = update_max_row_norm(row_squared_norms_final, current_rank, max_row_norms)

    col_pivots = col_pivots[:current_rank]
    row_pivots = row_pivots[:current_rank]
    frobenius_norms = frobenius_norms[:current_rank + 1]
    max_row_norms = max_row_norms[:current_rank + 1]

    accepted = jnp.array(accepted)

    if time_blocks:
        timing_data['index_extraction'] = [0.0]

    result = (row_pivots,
              col_pivots,
              accepted,
              frobenius_norms)
    if return_max_row_norms:
        result = result + (max_row_norms,)

    if time_blocks:
        return result + (timing_data,)
    return result


def build_with_upper_bound_rakau(c, q, g, b, rank, block_size=3,
                                 C=5.0, max_leaf=64, rng_seed=42,
                                 time_blocks=False, gb_tolerance=None, allow_extra_after_tol=True,
                                 column_sampling='random', row_sampling='random',
                                 frob_norm_relative_tolerance=None, return_true_norms=False,
                                 residual_norm_kind: str = 'frobenius',
                                 return_max_row_norms: bool = False):
    """
    Build CUR decomposition using Rakau factor-C upper bounds and rejection sampling.

    """
    from .Barnes_hut.gb_upper_bound_rakau import GbUpperBoundPlan

    c = jnp.asarray(c)
    q = jnp.asarray(q)
    g = jnp.asarray(g)
    b = jnp.asarray(b)

    if g.ndim == 1:
        g = g[:, None]
    if b.ndim == 1:
        b = b[:, None]
    if g.shape[0] != c.shape[0] or b.shape[0] != q.shape[0]:
        raise ValueError("g/b must match lengths of c/q respectively")
    if g.shape[1] != b.shape[1]:
        raise ValueError("g and b must share generator dimension")
    if g.shape[1] != 2:
        raise ValueError("Rakau upper bound only supports r=2")
    if C < 1.0:
        raise ValueError("C must be >= 1.0")
    if block_size <= 0:
        raise ValueError("block_size must be >= 1")

    plan = GbUpperBoundPlan(np.asarray(c), np.asarray(q), C, max_leaf=max_leaf)

    def _row_norms_squared(g_current, b_current):
        ub_np = plan.compute((g_current), (b_current))
        return ub_np

    output =  build_with_upper_bound(
        c, q, g, b, rank, block_size, _row_norms_squared,
        rng_seed=rng_seed, time_blocks=time_blocks, gb_tolerance=gb_tolerance,
        allow_extra_after_tol=allow_extra_after_tol, column_sampling=column_sampling,
        row_sampling=row_sampling,
        frob_norm_relative_tolerance=frob_norm_relative_tolerance,
        return_true_norms=return_true_norms,
        residual_norm_kind=residual_norm_kind,
        return_max_row_norms=return_max_row_norms,
    )
    plan.close()
    return output
