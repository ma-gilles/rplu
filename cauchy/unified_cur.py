"""
Unified API for CUR decomposition with different sampling methods.

This module provides a single interface to switch between different CUR sampling methods:
- 'rejection': Rejection sampling (with optional Barnes-Hut or exact norms)
- 'generator_norm': Generator norm sampling (fast, no Barnes-Hut needed)
- 'upper_bound': Upper-bound rejection sampling (Rakau factor-C or JAX BH-style)
- 'iterative': Iterative method using CauchyCUR class

All methods return a consistent format for easy swapping.
"""

import numpy as np
import jax.numpy as jnp
from typing import Tuple, Optional, Dict, Any
from cauchy.cauchy_jax import (
    build_with_generator_norm_sampling,
    build_with_upper_bound_rakau,
    CauchyCUR,
    frob_norm_from_uv
)

MAX_LEAF_DEFAULT = 16
C_DEFAULT = 5

def build_cur_unified(c, q, g, b, rank, sampling_method='upper_bound',
                     block_size=None, use_exact_norm=False, barnes_hut_theta=10.0,
                     rng_seed=42, time_blocks=False,
                     allow_extra_after_tol=False, upper_bound_backend='auto',
                     frob_norm_relative_tolerance=1e-13,
                     residual_norm_kind='frobenius',
                     **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Unified API for building CUR decomposition with different sampling methods.
    
    Parameters
    ----------
    c : array
        (n,) complex array - first generator vector
    q : array
        (m,) complex array - second generator vector
    g : array
        (n, r) complex array - generator matrix for rows
    b : array
        (m, r) complex array - generator matrix for columns
    rank : int
        Target rank of the decomposition
    sampling_method : str
        Sampling method to use:
        - 'rejection': Rejection sampling (supports Barnes-Hut or exact norms)
        - 'generator_norm': Generator norm sampling (fast, no Barnes-Hut)
        - 'upper_bound': Upper-bound rejection sampling (Rakau or JAX backend)
        - 'iterative': Iterative method using CauchyCUR class
    block_size : int
        Block size for CUR (used by rejection and iterative methods)
    use_exact_norm : bool
        Whether to use exact norm computation (only for rejection method)
    barnes_hut_theta : float
        Barnes-Hut theta parameter (only for rejection method with Barnes-Hut)
    rng_seed : int
        Random seed for reproducibility
    relative_gb_tolerance : float
        Relative tolerance factor for ||gb|| stopping criterion. 
        Stops when ||gb|| < relative_gb_tolerance * initial ||gb||.
        Default: 1e-15
    time_blocks : bool
        If True, return detailed timing information
    allow_extra_after_tol : bool
        If True, perform exactly one extra pivot after crossing the tolerance (default: False).
    upper_bound_backend : str
        Backend for 'upper_bound': 'auto' (prefer Rakau when r=2), 'rakau', or 'jax'.
    **kwargs
        Additional arguments passed to specific methods
        For 'upper_bound': C (float), max_leaf (int), upper_bound_backend ('auto'|'rakau'|'jax')
    
    Returns
    -------
    i_indices : array
        (rank,) array of row indices (padded with -1 if stopped early)
    j_indices : array
        (rank,) array of column indices (padded with -1 if stopped early)
    g_b_norms : array
        Array of ||gb|| norms at each step (normalized to actual norms, not squared)
    ranks_after_step : array
        Array of ranks after each step
    metadata : dict
        Dictionary with additional information:
        - 'method': The sampling method used
        - 'frobenius_norms_squared': Array of Frobenius norms squared (if available)
        - 'accepted_fractions': Array of acceptance fractions (for rejection sampling)
        - 'timing_data': Timing breakdown (if time_blocks=True)
        - 'initial_gb_norm': Initial ||gb|| norm
        - 'final_gb_norm': Final ||gb|| norm
        - 'max_row_norms': Array of max row norms at each step (if available)
        - 'residual_norm_kind': Which norm is used for stopping/summary ('frobenius' or 'max_row')
        - 'residual_norms': Array of residual norms used for stopping (if available)
    """
    if residual_norm_kind not in ('frobenius', 'max_row'):
        raise ValueError("residual_norm_kind must be 'frobenius' or 'max_row'")
    c = jnp.asarray(c)
    q = jnp.asarray(q)
    g = jnp.asarray(g)
    b = jnp.asarray(b)
    
    # Get initial ||gb|| norm
    initial_gb_norm = float(frob_norm_from_uv(g, b))
    
    column_sampling = kwargs.get('column_sampling', 'random')
    row_sampling = kwargs.get('row_sampling', 'random')
    upper_bound_backend = kwargs.get('upper_bound_backend', upper_bound_backend)


    if sampling_method == 'upper_bound' and row_sampling == 'greedy':
        block_size = 1
    if block_size is None and sampling_method == 'upper_bound':
        if kwargs.get('C', C_DEFAULT) is not None and kwargs.get('C', C_DEFAULT) > 0:
            block_size = np.ceil(kwargs.get('C', C_DEFAULT)/2).astype(int)
        else:
            block_size = None
        
    # Store hyperparameters in metadata for printing
    metadata = {
        'method': sampling_method,
        'initial_gb_norm': initial_gb_norm,
        'frob_norm_relative_tolerance': frob_norm_relative_tolerance,
        'block_size': block_size,
        'row_sampling': row_sampling,
        'column_sampling': column_sampling,
        'residual_norm_kind': residual_norm_kind,
    }

    if sampling_method == 'upper_bound':
        if upper_bound_backend == 'auto':
            upper_bound_backend = 'rakau' if g.shape[1] == 2 else 'jax'
        
        if upper_bound_backend == 'rakau':
            # Use rakau backend
            return_max_row_norms = residual_norm_kind == 'max_row'
            result = build_with_upper_bound_rakau(
                c, q, g, b, rank, block_size=block_size,
                C=kwargs.get('C', C_DEFAULT),
                max_leaf=kwargs.get('max_leaf', MAX_LEAF_DEFAULT),
                rng_seed=rng_seed,
                time_blocks=time_blocks,
                allow_extra_after_tol=allow_extra_after_tol,
                column_sampling=column_sampling,
                row_sampling=row_sampling,
                frob_norm_relative_tolerance=frob_norm_relative_tolerance,
                return_true_norms=kwargs.get('return_true_norms', False),
                residual_norm_kind=residual_norm_kind,
                return_max_row_norms=return_max_row_norms,
            )
            
            # build_with_upper_bound_rakau returns: (row_pivots, col_pivots, accepted, frobenius_norms[, max_row_norms]) or with timing_data
            if time_blocks:
                if return_max_row_norms:
                    row_pivots, col_pivots, accepted, frobenius_norms, max_row_norms, timing_data = result
                else:
                    row_pivots, col_pivots, accepted, frobenius_norms, timing_data = result
                    max_row_norms = None
            else:
                if return_max_row_norms:
                    row_pivots, col_pivots, accepted, frobenius_norms, max_row_norms = result
                else:
                    row_pivots, col_pivots, accepted, frobenius_norms = result
                    max_row_norms = None
                timing_data = None
            
            # Convert to numpy arrays
            row_pivots = np.asarray(row_pivots)
            col_pivots = np.asarray(col_pivots)
            frobenius_norms = np.asarray(frobenius_norms)
            if max_row_norms is not None:
                max_row_norms = np.asarray(max_row_norms)
            accepted = np.asarray(accepted) if len(accepted) > 0 else np.array([])
            
            # Filter out -1 padding if present (indicates unused entries)
            valid_mask = row_pivots >= 0
            if np.any(valid_mask):
                i_indices = row_pivots[valid_mask]
                j_indices = col_pivots[valid_mask]
            else:
                i_indices = row_pivots
                j_indices = col_pivots
            
            # Convert frobenius_norms (which are actual norms, not squared) to squared
            # Create ranks array (frobenius_norms has length current_rank + 1, with initial norm at index 0)
            g_b_norms = np.full(len(frobenius_norms), np.nan)
            accepted_fractions = accepted
        else:
            # JAX backend not yet implemented in new API
            raise NotImplementedError(f"JAX backend for 'upper_bound' method not yet implemented. Use 'rakau' backend.")
        
        metadata['frobenius_norms'] = frobenius_norms
        metadata['max_row_norms'] = max_row_norms
        metadata['residual_norms'] = max_row_norms if residual_norm_kind == 'max_row' else frobenius_norms
        metadata['accepted_fractions'] = accepted_fractions
        metadata['upper_bound_backend'] = upper_bound_backend
        if time_blocks and timing_data is not None:
            metadata['timing_data'] = timing_data
        # Store upper_bound specific hyperparameters
        if upper_bound_backend == 'rakau':
            metadata['C'] = kwargs.get('C', C_DEFAULT)
            metadata['max_leaf'] = kwargs.get('max_leaf', MAX_LEAF_DEFAULT)

    elif sampling_method in ('iterative', 'greedy'):
        # Greedy/iterative using CauchyCUR.build (no blocks)
        use_generator_norm = kwargs.get('use_generator_norm_sampling', False)
        cur_obj = CauchyCUR(c, q, g, b, block_size=block_size,
                            use_exact_norm=use_exact_norm,
                            barnes_hut_theta=barnes_hut_theta)
        i_indices, j_indices = cur_obj.build(
            rank, use_generator_norm_sampling=use_generator_norm,
            frob_norm_relative_tolerance=frob_norm_relative_tolerance,
            column_sampling=column_sampling,
            row_sampling=row_sampling,
            residual_norm_kind=residual_norm_kind,
        )
        if hasattr(cur_obj, 'g_b_norms') and len(cur_obj.g_b_norms) > 0:
            g_b_norms = np.asarray([float(n) for n in cur_obj.g_b_norms])
        else:
            g_b_norms = np.array([initial_gb_norm])
        # CauchyCUR stores frobenius_norms (actual norms) when use_generator_norm_sampling=False
        # or frobenius_norms_squared when use_generator_norm_sampling=True
        if hasattr(cur_obj, 'frobenius_norms') and len(cur_obj.frobenius_norms) > 0:
            # Convert actual norms to squared for consistency with metadata format
            frobenius_norms_actual = np.asarray([float(n) for n in cur_obj.frobenius_norms])
            metadata['frobenius_norms'] = frobenius_norms_actual 
        else:
            metadata['frobenius_norms'] = np.full(len(g_b_norms), np.nan)
        if hasattr(cur_obj, 'max_row_norms') and len(cur_obj.max_row_norms) > 0:
            max_row_norms = np.asarray([float(n) for n in cur_obj.max_row_norms])
        else:
            max_row_norms = np.full(len(g_b_norms), np.nan)
        metadata['max_row_norms'] = max_row_norms
        metadata['residual_norms'] = max_row_norms if residual_norm_kind == 'max_row' else metadata['frobenius_norms']
        # Store iterative/greedy specific hyperparameters
        metadata['barnes_hut_theta'] = barnes_hut_theta
        metadata['use_exact_norm'] = use_exact_norm
        metadata['use_generator_norm_sampling'] = use_generator_norm

    elif sampling_method == 'rejection':
        # Rejection sampling via iterative CauchyCUR.build (Barnes–Hut or exact norms).
        # Note: a dedicated block-rejection routine may be added later; this path fixes the
        # unified API so that 'rejection' works as documented.
        cur_obj = CauchyCUR(c, q, g, b, block_size=block_size,
                            use_exact_norm=use_exact_norm,
                            barnes_hut_theta=barnes_hut_theta)
        i_indices, j_indices = cur_obj.build(
            rank,
            use_generator_norm_sampling=False,
            frob_norm_relative_tolerance=frob_norm_relative_tolerance,
            column_sampling=column_sampling,
            row_sampling=row_sampling,
            residual_norm_kind=residual_norm_kind,
        )
        if hasattr(cur_obj, 'g_b_norms') and len(cur_obj.g_b_norms) > 0:
            g_b_norms = np.asarray([float(n) for n in cur_obj.g_b_norms])
        else:
            g_b_norms = np.array([initial_gb_norm])

        if hasattr(cur_obj, 'frobenius_norms') and len(cur_obj.frobenius_norms) > 0:
            frobenius_norms_actual = np.asarray([float(n) for n in cur_obj.frobenius_norms])
            metadata['frobenius_norms'] = frobenius_norms_actual
        else:
            metadata['frobenius_norms'] = np.full(len(g_b_norms), np.nan)
        if hasattr(cur_obj, 'max_row_norms') and len(cur_obj.max_row_norms) > 0:
            max_row_norms = np.asarray([float(n) for n in cur_obj.max_row_norms])
        else:
            max_row_norms = np.full(len(g_b_norms), np.nan)
        metadata['max_row_norms'] = max_row_norms
        metadata['residual_norms'] = max_row_norms if residual_norm_kind == 'max_row' else metadata['frobenius_norms']

        metadata['barnes_hut_theta'] = barnes_hut_theta
        metadata['use_exact_norm'] = use_exact_norm
        metadata['use_generator_norm_sampling'] = False

    # Convert to numpy arrays
    i_indices = np.asarray(i_indices)
    j_indices = np.asarray(j_indices)
    
    # Create ranks_after_step array (ranks from 0 to actual_rank)
    # g_b_norms has length actual_rank + 1 (includes initial norm at index 0)
    # So ranks_after_step should be [0, 1, 2, ..., actual_rank]
    actual_rank = len(i_indices)
    ranks_after_step = np.arange(len(g_b_norms), dtype=np.int32)

    # Get final ||gb|| norm
    if len(g_b_norms) > 0:
        final_gb_norm = float(g_b_norms[-1])
    else:
        final_gb_norm = initial_gb_norm
    
    metadata['final_gb_norm'] = final_gb_norm
    
    # Get Frobenius norm information
    initial_residual_norm = None
    final_residual_norm = None
    residual_norms = metadata.get('residual_norms', None)
    if residual_norms is not None:
        initial_residual_norm = float(residual_norms[0])
        final_residual_norm = float(residual_norms[-1])
        # if len(frob_norms) > 0 and not np.isnan(frob_norms[0]):
        #     initial_frob_norm = float(frob_norms[0])
        # if len(frob_norms) > 0 and not np.isnan(frob_norms[-1]):
        #     final_frob_norm = float(frob_norms[-1])
    
    # Get matrix dimensions and final rank
    n, m = c.shape[0], q.shape[0]
    final_rank = len(i_indices)
    
    # Build hyperparameter string based on method
    hyperparams = []
    if sampling_method == 'upper_bound':
        backend = metadata.get('upper_bound_backend', 'auto')
        hyperparams.append(f"backend={backend}")
        if backend == 'rakau':
            C_val = metadata.get('C', C_DEFAULT)
            max_leaf_val = metadata.get('max_leaf', MAX_LEAF_DEFAULT)
            hyperparams.append(f"C={C_val:.1f}")
            hyperparams.append(f"max_leaf={max_leaf_val}")
        hyperparams.append(f"block_size={block_size}")
        hyperparams.append(f"sampling={row_sampling}")
    elif sampling_method == 'generator_norm':
        hyperparams.append(f"block_size={block_size}")
        hyperparams.append(f"sampling={row_sampling}")
        use_jit = metadata.get('use_jitted_inner', True)
        if not use_jit:
            hyperparams.append("no_jit")
    elif sampling_method in ('iterative', 'greedy', 'rejection'):
        hyperparams.append(f"block_size={block_size}")
        bh_theta = metadata.get('barnes_hut_theta', 10.0)
        use_exact = metadata.get('use_exact_norm', False)
        use_gen_norm = metadata.get('use_generator_norm_sampling', False)
        if use_exact:
            hyperparams.append("exact_norm")
        else:
            hyperparams.append(f"BH_theta={bh_theta:.1f}")
        if use_gen_norm:
            hyperparams.append("gen_norm")
        hyperparams.append(f"sampling={row_sampling}")
    
    hyperparam_str = ", ".join(hyperparams) if hyperparams else ""
    
    # Print summary in a single line (focus on Frobenius norm)
    if initial_residual_norm is not None and final_residual_norm is not None:
        initial_norm_val = initial_residual_norm
        final_norm_val = final_residual_norm
        rel_norm = final_residual_norm / initial_residual_norm if initial_residual_norm > 0 else np.nan
        norm_label = "||F||" if residual_norm_kind == 'frobenius' else "||row||_max"
        tol_label = "frob_tol" if residual_norm_kind == 'frobenius' else "row_tol"
        
        if frob_norm_relative_tolerance is None:
            print(f"CUR[{sampling_method}] | matrix: {n}×{m} | rank: {final_rank} | "
                  f"{norm_label}: {final_norm_val:.2e} (init: {initial_norm_val:.2e}, "
                  f"rel: {rel_norm:.2e}, {tol_label}: disabled) | {hyperparam_str}")
        else:
            print(f"CUR[{sampling_method}] | matrix: {n}×{m} | rank: {final_rank} | "
                  f"{norm_label}: {final_norm_val:.2e} (init: {initial_norm_val:.2e}, "
                  f"rel: {rel_norm:.2e}, {tol_label}: {frob_norm_relative_tolerance:.2e}) | {hyperparam_str}")

    return i_indices, j_indices, g_b_norms, ranks_after_step, metadata
