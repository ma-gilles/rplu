"""
Efficiently compute PivotedQRCUR for multiple ranks with incremental timing.

This module provides functions to compute PivotedQRCUR for many different ranks
by building once at the maximum rank and truncating for lower ranks, with
detailed timing information for each component.
"""

import time
import jax
import jax.numpy as jnp
import jax.scipy.linalg
from typing import List, Dict, Optional, Tuple
from pivoted_qr import pivoted_qr, PivotedQR
from pivoted_qr_cur import PivotedQRCUR, AOperatorTranspose
from matrix_classes import AOperator


def compute_pivoted_qr_cur_multi_rank(
    Aop: AOperator,
    col_norms_squared: jnp.ndarray,
    row_norms_squared: jnp.ndarray,
    ranks: List[int],
    tol: Optional[float] = None,
    debug: bool = False,
    use_jit: bool = True,
    random_seed: Optional[int] = None,
    column_sampling: str = 'greedy',
    use_svd_for_T: bool = True,
    compute_T_via_pinv: bool = False,
) -> Dict[int, Dict]:
    """
    Compute PivotedQRCUR for multiple ranks efficiently.
    
    This function:
    1. Builds pivoted QR for columns and rows once at max_rank
    2. For each rank, computes T using truncated R matrices
    3. Returns results for all ranks with detailed timing information
    
    Parameters
    ----------
    Aop : AOperator
        Matrix operator
    col_norms_squared : jnp.ndarray
        Squared column norms
    row_norms_squared : jnp.ndarray
        Squared row norms
    ranks : List[int]
        List of ranks to compute (must be sorted, max_rank will be used)
    tol : float, optional
        Tolerance for early termination
    debug : bool
        Enable debug output
    use_jit : bool
        Use JIT-compiled functions
    random_seed : int, optional
        Random seed
    column_sampling : str
        Column sampling mode ('greedy', 'random', 'uniform')
    use_svd_for_T : bool
        Use SVD for computing T
    compute_T_via_pinv : bool
        Compute T via pinv (alternative method)
        
    Returns
    -------
    Dict[int, Dict]
        Dictionary mapping rank -> result dict with keys:
        - 'I': row indices (k,)
        - 'J': column indices (k,)
        - 'T_u', 'T_s', 'T_vt': SVD factors of T (if use_svd_for_T)
        - 'T_core': core matrix T (if not use_svd_for_T)
        - 'k': actual rank
        - 'timings': dict with timing breakdown:
            - 'qr_cols': cumulative time for QR on columns up to rank k
            - 'qr_rows': cumulative time for QR on rows up to rank k
            - 'compute_T': time to compute T at rank k
            - 'total': total time for rank k
    """
    if not ranks:
        raise ValueError("ranks list cannot be empty")
    
    max_rank = max(ranks)
    ranks_sorted = sorted(set(ranks))  # Remove duplicates and sort
    
    print(f"Building PivotedQRCUR for ranks {ranks_sorted} (max_rank={max_rank})...")
    
    # Step 1: Compute pivoted QR on A to select columns (at max_rank)
    print(f"\nStep 1: Computing pivoted QR on A (rank={max_rank})...")
    t0_cols = time.time()
    qr_result_cols = pivoted_qr(
        Aop,
        col_norms_squared,
        rank=max_rank,
        tol=tol,
        debug=debug,
        use_jit=use_jit,
        return_timings=True,
        random_seed=random_seed,
        column_sampling=column_sampling
    )
    time_qr_cols_total = time.time() - t0_cols
    
    I_cols = qr_result_cols['I']  # Column indices
    R_cols_full = qr_result_cols['R']  # max_rank x max_rank
    timings_qr_cols = qr_result_cols.get('timings', [])  # Cumulative timings
    k_actual_cols = qr_result_cols['k']
    
    print(f"  Selected {k_actual_cols} columns")
    print(f"  Total time: {time_qr_cols_total:.3f}s")
    
    # Step 2: Compute pivoted QR on A^T to select rows (at max_rank)
    print(f"\nStep 2: Computing pivoted QR on A^T (rank={max_rank})...")
    A_T = AOperatorTranspose(Aop)
    t0_rows = time.time()
    qr_result_rows = pivoted_qr(
        A_T,
        row_norms_squared,
        rank=max_rank,
        tol=tol,
        debug=debug,
        use_jit=use_jit,
        return_timings=True,
        random_seed=random_seed,
        column_sampling=column_sampling
    )
    time_qr_rows_total = time.time() - t0_rows
    
    I_rows = qr_result_rows['I']  # Row indices
    R_rows_full = qr_result_rows['R']  # max_rank x max_rank
    timings_qr_rows = qr_result_rows.get('timings', [])  # Cumulative timings
    k_actual_rows = qr_result_rows['k']
    
    print(f"  Selected {k_actual_rows} rows")
    print(f"  Total time: {time_qr_rows_total:.3f}s")
    
    # Determine actual max rank achievable
    k_max = min(k_actual_cols, k_actual_rows, max_rank)
    ranks_to_compute = [k for k in ranks_sorted if k <= k_max]
    
    print(f"\nStep 3: Computing T for ranks {ranks_to_compute}...")
    
    results = {}
    
    # For each rank, create a PivotedQRCUR object with the computed results
    for k in ranks_to_compute:
        print(f"\n  Computing T for rank {k}...")
        t0_T = time.time()
        
        # Truncate R matrices
        R_cols = R_cols_full[:k, :k]
        R_rows = R_rows_full[:k, :k]
        J_k = I_cols[:k]
        I_k = I_rows[:k]
        
        # Create a PivotedQRCUR object and populate it with the computed results
        # This ensures compatibility with estimate_approximation_error
        cur_obj = PivotedQRCUR(
            Aop,
            col_norms_squared,
            row_norms_squared=row_norms_squared,
            max_rank=k,
            tol=tol,
            debug=debug,
            use_jit=use_jit,
            random_seed=random_seed,
            use_svd_for_T=use_svd_for_T,
            compute_T_via_pinv=compute_T_via_pinv,
            column_sampling=column_sampling,
        )
        
        # Set the internal state directly (bypassing build to avoid recomputation)
        cur_obj.I_array = I_k
        cur_obj.J_array = J_k
        cur_obj.k = k
        cur_obj.qr_result = {
            'I': I_cols,
            'R': R_cols_full,
            'k': k_actual_cols,
            'timings': timings_qr_cols,
        }
        cur_obj.qr_result_T = {
            'I': I_rows,
            'R': R_rows_full,
            'k': k_actual_rows,
            'timings': timings_qr_rows,
        }
        
        # Compute YtXZt for rank k
        # Y^T X Z^T = (A[:, J])^T @ A @ (A[I, :])^T
        YtXZt = jnp.zeros((k, k), dtype=Aop.dtype)
        
        for i, i_idx in enumerate(I_k):
            row_i = Aop.get_row(int(i_idx))  # (p,)
            A_row_i = Aop.matvec(row_i)  # (n,)
            AA_row_i = Aop.lmatvec_with_C(A_row_i, J_k)  # (k,)
            YtXZt = YtXZt.at[:, i].set(AA_row_i)
        
        # Compute T from R_cols, R_rows, and YtXZt (same logic as PivotedQRCUR.build)
        if compute_T_via_pinv:
            # Method 1: Compute T = pinv(A[I, J]) directly
            W = jnp.zeros((k, k), dtype=Aop.dtype)
            for i, i_idx in enumerate(I_k):
                row_i = Aop.get_row(int(i_idx))
                W = W.at[i, :].set(row_i[J_k])
            
            U, s, Vh = jnp.linalg.svd(W, full_matrices=False)
            eps = jnp.finfo(W.dtype).eps
            rcond = 10.0 * eps * max(W.shape)
            tol_svd = rcond * jnp.max(s)
            sT_thresholded = jnp.where(s > tol_svd, 1.0 / s, 0.0)
            
            # Store in cur_obj
            cur_obj.T_u = cur_obj.T_u.at[:k, :k].set(Vh.conj().T)
            cur_obj.T_s = cur_obj.T_s.at[:k].set(sT_thresholded)
            cur_obj.T_vt = cur_obj.T_vt.at[:k, :k].set(U.conj().T)
            cur_obj.T_use_svd = True
        else:
            # Method 2: Compute T via normal equations
            if use_svd_for_T:
                # Step 1: Solve R_cols^T R_cols @ temp = Y^T X Z^T using SVD
                U_cols, s_cols, Vt_cols = jnp.linalg.svd(R_cols, full_matrices=False)
                eps = jnp.finfo(R_cols.dtype).eps
                rcond_cols = jnp.sqrt(10.0 * eps) * max(R_cols.shape)
                tol_cols = rcond_cols * jnp.max(s_cols)
                inv_s_cols_sq = jnp.where(s_cols > tol_cols, 1.0 / (s_cols ** 2), 0.0)
                temp = Vt_cols.T @ (inv_s_cols_sq[:, None] * (Vt_cols @ YtXZt))
                
                # Step 2: Solve R_rows^T R_rows @ T = temp^T using SVD
                U_rows, s_rows, Vt_rows = jnp.linalg.svd(R_rows, full_matrices=False)
                rcond_rows = jnp.sqrt(10.0 * eps) * max(R_rows.shape)
                tol_rows = rcond_rows * jnp.max(s_rows)
                inv_s_rows_sq = jnp.where(s_rows > tol_rows, 1.0 / (s_rows ** 2), 0.0)
                T = Vt_rows.T @ (inv_s_rows_sq[:, None] * (Vt_rows @ temp.T))
                T = T.T
                
                # SVD threshold T
                uT, sT, vT = jnp.linalg.svd(T, full_matrices=False)
                eps = jnp.finfo(T.dtype).eps
                rcond = 10.0 * eps * max(T.shape)
                tol = rcond * jnp.max(sT)
                sT_thresholded = jnp.where(sT > tol, sT, 0.0)
                
                # Store in cur_obj
                cur_obj.T_u = cur_obj.T_u.at[:k, :k].set(uT)
                cur_obj.T_s = cur_obj.T_s.at[:k].set(sT_thresholded)
                cur_obj.T_vt = cur_obj.T_vt.at[:k, :k].set(vT)
                cur_obj.T_use_svd = True
            else:
                # Use triangular solves (faster but less stable)
                temp1 = jax.scipy.linalg.solve_triangular(R_cols, YtXZt, lower=False, trans=1)
                temp = jax.scipy.linalg.solve_triangular(R_cols, temp1, lower=False)
                temp2 = jax.scipy.linalg.solve_triangular(R_rows, temp.T, lower=False, trans=1)
                T = jax.scipy.linalg.solve_triangular(R_rows, temp2, lower=False).T
                
                # Store in cur_obj
                cur_obj.T_core = cur_obj.T_core.at[:k, :k].set(T)
                cur_obj.T_use_svd = False
        
        time_T = time.time() - t0_T
        
        # Get cumulative timings for QR steps up to rank k
        # timings_qr_cols[i] is time for step i+1, so cumulative up to k is sum of first k
        time_qr_cols_k = float(jnp.sum(timings_qr_cols[:k])) if len(timings_qr_cols) >= k else time_qr_cols_total
        time_qr_rows_k = float(jnp.sum(timings_qr_rows[:k])) if len(timings_qr_rows) >= k else time_qr_rows_total
        
        total_time_k = time_qr_cols_k + time_qr_rows_k + time_T
        
        print(f"    QR cols (cumulative): {time_qr_cols_k:.3f}s")
        print(f"    QR rows (cumulative): {time_qr_rows_k:.3f}s")
        print(f"    Compute T: {time_T:.3f}s")
        print(f"    Total: {total_time_k:.3f}s")
        
        # Store result with PivotedQRCUR object and timings
        results[k] = {
            'cur_obj': cur_obj,  # The actual PivotedQRCUR object (compatible with estimate_approximation_error)
            'timings': {
                'qr_cols': time_qr_cols_k,
                'qr_rows': time_qr_rows_k,
                'compute_T': time_T,
                'total': total_time_k,
            }
        }
    
    print(f"\n✓ Completed computation for {len(results)} ranks")
    
    return results

