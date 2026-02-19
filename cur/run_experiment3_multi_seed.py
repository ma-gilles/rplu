#!/usr/bin/env python3
"""
Run experiment 3 (solver comparison) for a single method.
Supports 4 methods: unpreconditioned, svd_precon, cur_precon, pivoted_qr_cur
Each method runs once with a single seed.
"""

import os
import sys
import json
import argparse
import numpy as np

# Add project root to path
_this_dir = os.path.dirname(os.path.abspath(__file__))
_proj_root = os.path.dirname(_this_dir)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)
# Also add cur directory to path for relative imports
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)


def run_single_experiment(
    method,  # 'unpreconditioned', 'svd_precon', 'cur_greedy', 'cur_random', 'pivoted_qr_cur_greedy', 'pivoted_qr_cur_random'
    problem_size=320,
    svd_rank=16,
    cur_rank=512,
    tau=1e1,
    restart=20,
    maxit=1000,
    tol=1e-16,
    output_dir=None,
    gpu_id=0,
    seed=42,
):
    """
    Run a single experiment 3 method.
    
    Parameters
    ----------
    method : str
        Method to run: 'unpreconditioned', 'svd_precon', 'cur_greedy', 'cur_random', 
                      'pivoted_qr_cur_greedy', 'pivoted_qr_cur_random'
    problem_size : int
        Problem size (dim x dim x dim) for Toeplitz
    svd_rank : int
        Rank for SVD preconditioner
    cur_rank : int
        Rank for CUR/RPLU preconditioner
    tau : float
        Parameter for problem
    restart : int
        GMRES restart parameter
    maxit : int
        Maximum GMRES iterations
    tol : float
        Tolerance for GMRES
    output_dir : str
        Directory to save results
    gpu_id : int
        GPU ID to use
    seed : int
        Random seed
    """
    print(f"\n{'='*60}")
    print(f"Running {method} on experiment 3 (size={problem_size})")
    print(f"{'='*60}\n")
    
    # Create output file name
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"experiment3_{method}.json")
    
    # Import and run directly instead of via subprocess for better control
    import jax
    import jax.numpy as jnp
    import time
    import gc
    
    # Import from benchmark_preconditioners (relative import since we're in cur/)
    from benchmark_preconditioners import (
        _build_operator_and_rhs,
        solve_with_gmres, benchmark_unpreconditioned, benchmark_svd_precon,
        benchmark_cur_precon, benchmark_pivoted_qr_cur_precon
    )
    from randomized_svd import WoodburyPreconditioner
    from pivoted_qr_cur import PivotedQRCUR
    
    # Import helper functions used by README-driven experiment scripts
    from readme_helpers import configure_gpu, determine_restart_budget, TARGET_REL_ERROR
    
    # Configure GPU - set CUDA_VISIBLE_DEVICES BEFORE importing JAX
    if gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        configure_gpu(gpu_id)
    
    # Import JAX and verify GPU is available BEFORE building operator
    # This must happen before any JAX operations to ensure GPU is detected
    import jax
    if gpu_id >= 0:
        # Force JAX to use GPU backend
        try:
            devices = jax.devices()
            backend = jax.default_backend()
            if backend != 'gpu' or not any('cuda' in str(d).lower() or 'gpu' in str(d).lower() for d in devices):
                raise RuntimeError(
                    f"GPU requested (gpu_id={gpu_id}) but JAX is using {backend} backend. "
                    f"Devices: {devices}. CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}. "
                    f"This usually means CUDA libraries aren't accessible to jaxlib."
                )
            print(f"GPU verified: JAX devices={devices}, backend={backend}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize GPU: {e}")
    
    # Build operator and RHS (this will call _configure_platform internally)
    # But _configure_platform uses setdefault, so our CUDA_VISIBLE_DEVICES should stick
    Cop, A_mv, b, M_inv_matvec, x_true = _build_operator_and_rhs(
        problem_size, tau, use_laplace=True, rng_seed=seed
    )
    
    result_data = None
    
    try:
        if method == 'unpreconditioned':
            print("  Running with callback (for convergence plots)...")
            res_with_callback = benchmark_unpreconditioned(
                problem_size=problem_size, tau=tau, use_laplace=True,
                tol=tol, restart=restart, maxit=maxit,
                use_gpu=(gpu_id >= 0), use_jax=True, verbose=False
            )
            
            error_history = res_with_callback.get("error_history", [])
            budget, reached, cycle_idx = determine_restart_budget(
                error_history, TARGET_REL_ERROR, maxit
            )
            
            print("  Running without callback (for accurate timing)...")
            start_time = time.time()
            x, info, iters, gmres_t = solve_with_gmres(
                A_mv, b, M_matvec=None, tol=tol, restart=restart, maxit=budget,
                use_jax=True, verbose=False, callback=None, x_true=None
            )
            timing_gmres_t = time.time() - start_time
            
            rel = float(jnp.linalg.norm(A_mv(x) - b).block_until_ready() / jnp.linalg.norm(b).block_until_ready())
            err = float(jnp.linalg.norm(x - x_true).block_until_ready() / jnp.linalg.norm(x_true).block_until_ready())
            jax.clear_caches()
            
            result_data = {
                "time": timing_gmres_t,
                "gmres_time": timing_gmres_t,
                "precon_time": 0.0,
                "total_time": timing_gmres_t,
                "iters": budget * restart,
                "info": int(info) if isinstance(info, (jnp.ndarray, np.ndarray)) else (int(info) if info != "N/A" else 0),
                "rel_res": rel,
                "error": err,
                "error_history": error_history,
                "time_history": res_with_callback.get("time_history", []),
                "restart_cycles_budget": budget,
                "restart_cycles_target": int(cycle_idx) if reached and cycle_idx is not None else None,
                "target_error": TARGET_REL_ERROR,
                "target_error_reached": reached,
                "final_error_meets_target": bool(err <= TARGET_REL_ERROR)
            }
            
        elif method == 'svd_precon':
            print("  Running with callback (for convergence plots)...")
            res_with_callback = benchmark_svd_precon(
                problem_size=problem_size, svd_rank=svd_rank, svd_ranks=[svd_rank],
                tau=tau, use_laplace=True, tol=tol, restart=restart, maxit=maxit,
                use_gpu=(gpu_id >= 0), use_jax=True, verbose=False, show_trace=False
            )
            
            error_history = res_with_callback.get("error_history", [])
            budget, reached, cycle_idx = determine_restart_budget(
                error_history, TARGET_REL_ERROR, maxit
            )
            
            print("  Running without callback (for accurate timing)...")
            precon_start = time.time()
            # Get detailed timings from randomized_svd to see where time is spent
            from cur.randomized_svd import randomized_svd
            U, s, Vt, timings = randomized_svd(
                Cop, n_components=svd_rank, n_oversamples=0,
                random_state=seed, batch_matvec=False, return_timings=True
            )
            U_lr = U * jnp.sqrt(s)
            V_lr = Vt.T * jnp.sqrt(s)
            Svals = s
            if timings:
                print(f"  RSVD timing breakdown:")
                for key, val in timings.items():
                    print(f"    {key}: {val:.2f}s")
            M_svd = WoodburyPreconditioner(M_inv_matvec, U_lr, V_lr)
            precon_time = time.time() - precon_start
            
            start_time = time.time()
            x, info, iters, gmres_t = solve_with_gmres(
                A_mv, b, M_matvec=M_svd, tol=tol, restart=restart, maxit=budget,
                use_jax=True, verbose=False, callback=None, x_true=None
            )
            timing_gmres_t = time.time() - start_time
            
            rel = float(jnp.linalg.norm(A_mv(x) - b).block_until_ready() / jnp.linalg.norm(b).block_until_ready())
            err = float(jnp.linalg.norm(x - x_true).block_until_ready() / jnp.linalg.norm(x_true).block_until_ready())
            total_t = precon_time + timing_gmres_t
            jax.clear_caches()
            
            result_data = {
                "time": timing_gmres_t,
                "gmres_time": timing_gmres_t,
                "precon_time": precon_time,
                "total_time": total_t,
                "iters": budget * restart,
                "info": int(info) if isinstance(info, (jnp.ndarray, np.ndarray)) else (int(info) if info != "N/A" else 0),
                "rel_res": rel,
                "error": err,
                "sigma1": float(Svals[0]),
                "sigmar": float(Svals[min(len(Svals) - 1, svd_rank - 1)]),
                "rank": svd_rank,
                "error_history": error_history,
                "time_history": res_with_callback.get("time_history", []),
                "restart_cycles_budget": budget,
                "restart_cycles_target": int(cycle_idx) if reached and cycle_idx is not None else None,
                "target_error": TARGET_REL_ERROR,
                "target_error_reached": reached,
                "final_error_meets_target": bool(err <= TARGET_REL_ERROR)
            }
            
        elif method == 'cur_greedy':
            print("  Running with callback (for convergence plots)...")
            res_with_callback = benchmark_cur_precon(
                problem_size=problem_size, cur_rank=cur_rank, tau=tau, use_laplace=True,
                tol=tol, restart=restart, maxit=maxit,
                use_gpu=(gpu_id >= 0), use_jax=True, verbose=False, sampling='greedy'
            )
            
            error_history = res_with_callback.get("error_history", [])
            budget, reached, cycle_idx = determine_restart_budget(
                error_history, TARGET_REL_ERROR, maxit
            )
            
            print("  Running without callback (for accurate timing)...")
            # Rebuild operator
            Cop, A_mv, b, M_inv_matvec, x_true = _build_operator_and_rhs(
                problem_size, tau, use_laplace=True, rng_seed=seed
            )
            
            precon_start = time.time()
            row_norms = Cop.compute_all_row_norms_squared()
            from cur import CURIncremental, CUR_woodbury_solver
            cur = CURIncremental(
                Cop, row_norms, debug=False, pivot_sampling='greedy',
                store_preconditioner_matrices=True, profile=False, use_jit=True,
                M_inv_matvec=M_inv_matvec, random_seed=seed
            )
            build_result = cur.build(int(cur_rank), timing=True)
            if build_result is None or 'I' not in build_result:
                raise ValueError(f"CUR build failed for rank {cur_rank}")
            I_cur = build_result['I'].block_until_ready()
            if cur.k == 0:
                raise ValueError(f"CUR build completed but k=0 (no columns selected)")
            # Ensure W_squared_core is computed
            if not hasattr(cur, 'W_squared_core') or cur.W_squared_core is None:
                cur._compute_W_squared_core()
            M_cur = CUR_woodbury_solver(cur, M_inv_matvec=M_inv_matvec)
            precon_time = time.time() - precon_start
            
            start_time = time.time()
            x, info, iters, gmres_t = solve_with_gmres(
                A_mv, b, M_matvec=M_cur.solve, tol=tol, restart=restart, maxit=budget,
                use_jax=True, verbose=False, callback=None, x_true=None
            )
            timing_gmres_t = time.time() - start_time
            
            rel = float(jnp.linalg.norm(A_mv(x) - b).block_until_ready() / jnp.linalg.norm(b).block_until_ready())
            err = float(jnp.linalg.norm(x - x_true).block_until_ready() / jnp.linalg.norm(x_true).block_until_ready())
            total_t = precon_time + timing_gmres_t
            jax.clear_caches()
            
            result_data = {
                "time": timing_gmres_t,
                "gmres_time": timing_gmres_t,
                "precon_time": precon_time,
                "total_time": total_t,
                "iters": budget * restart,
                "info": int(info) if isinstance(info, (jnp.ndarray, np.ndarray)) else (int(info) if info != "N/A" else 0),
                "rel_res": rel,
                "error": err,
                "rank": int(cur_rank),
                "error_history": error_history,
                "time_history": res_with_callback.get("time_history", []),
                "restart_cycles_budget": budget,
                "restart_cycles_target": int(cycle_idx) if reached and cycle_idx is not None else None,
                "target_error": TARGET_REL_ERROR,
                "target_error_reached": reached,
                "final_error_meets_target": bool(err <= TARGET_REL_ERROR)
            }
            
        elif method == 'cur_random':
            print("  Running with callback (for convergence plots)...")
            res_with_callback = benchmark_cur_precon(
                problem_size=problem_size, cur_rank=cur_rank, tau=tau, use_laplace=True,
                tol=tol, restart=restart, maxit=maxit,
                use_gpu=(gpu_id >= 0), use_jax=True, verbose=False, sampling='random'
            )
            
            error_history = res_with_callback.get("error_history", [])
            budget, reached, cycle_idx = determine_restart_budget(
                error_history, TARGET_REL_ERROR, maxit
            )
            
            print("  Running without callback (for accurate timing)...")
            # Rebuild operator
            Cop, A_mv, b, M_inv_matvec, x_true = _build_operator_and_rhs(
                problem_size, tau, use_laplace=True, rng_seed=seed
            )
            
            precon_start = time.time()
            row_norms = Cop.compute_all_row_norms_squared()
            from cur import CURIncremental, CUR_woodbury_solver
            cur = CURIncremental(
                Cop, row_norms, debug=False, pivot_sampling='random',
                store_preconditioner_matrices=True, profile=False, use_jit=True,
                M_inv_matvec=M_inv_matvec, random_seed=seed
            )
            build_result = cur.build(int(cur_rank), timing=True)
            if build_result is None or 'I' not in build_result:
                raise ValueError(f"CUR build failed for rank {cur_rank}")
            I_cur = build_result['I'].block_until_ready()
            if cur.k == 0:
                raise ValueError(f"CUR build completed but k=0 (no columns selected)")
            # Ensure W_squared_core is computed
            if not hasattr(cur, 'W_squared_core') or cur.W_squared_core is None:
                cur._compute_W_squared_core()
            M_cur = CUR_woodbury_solver(cur, M_inv_matvec=M_inv_matvec)
            precon_time = time.time() - precon_start
            
            start_time = time.time()
            x, info, iters, gmres_t = solve_with_gmres(
                A_mv, b, M_matvec=M_cur.solve, tol=tol, restart=restart, maxit=budget,
                use_jax=True, verbose=False, callback=None, x_true=None
            )
            timing_gmres_t = time.time() - start_time
            
            rel = float(jnp.linalg.norm(A_mv(x) - b).block_until_ready() / jnp.linalg.norm(b).block_until_ready())
            err = float(jnp.linalg.norm(x - x_true).block_until_ready() / jnp.linalg.norm(x_true).block_until_ready())
            total_t = precon_time + timing_gmres_t
            jax.clear_caches()
            
            result_data = {
                "time": timing_gmres_t,
                "gmres_time": timing_gmres_t,
                "precon_time": precon_time,
                "total_time": total_t,
                "iters": budget * restart,
                "info": int(info) if isinstance(info, (jnp.ndarray, np.ndarray)) else (int(info) if info != "N/A" else 0),
                "rel_res": rel,
                "error": err,
                "rank": int(cur_rank),
                "error_history": error_history,
                "time_history": res_with_callback.get("time_history", []),
                "restart_cycles_budget": budget,
                "restart_cycles_target": int(cycle_idx) if reached and cycle_idx is not None else None,
                "target_error": TARGET_REL_ERROR,
                "target_error_reached": reached,
                "final_error_meets_target": bool(err <= TARGET_REL_ERROR)
            }
            
        elif method == 'pivoted_qr_cur_greedy':
            print("  Running with callback (for convergence plots)...")
            res_with_callback = benchmark_pivoted_qr_cur_precon(
                problem_size=problem_size, cur_rank=cur_rank, tau=tau, use_laplace=True,
                tol=tol, restart=restart, maxit=maxit,
                use_gpu=(gpu_id >= 0), use_jax=True, verbose=False,
                sampling='greedy', random_seed=seed, use_adaptive_rank=True  # Stop when column norms become negative
            )
            
            error_history = res_with_callback.get("error_history", [])
            budget, reached, cycle_idx = determine_restart_budget(
                error_history, TARGET_REL_ERROR, maxit
            )
            
            print("  Running without callback (for accurate timing)...")
            # Rebuild operator
            Cop, A_mv, b, M_inv_matvec, x_true = _build_operator_and_rhs(
                problem_size, tau, use_laplace=True, rng_seed=seed
            )
            
            precon_start = time.time()
            row_norms_sq = Cop.compute_all_row_norms_squared()
            A_T = Cop.T
            col_norms_sq = A_T.compute_all_row_norms_squared()
            
            pqr_cur = PivotedQRCUR(
                Cop, col_norms_sq,
                row_norms_squared=row_norms_sq,
                max_rank=cur_rank,
                debug=False,
                column_sampling='greedy',
                random_seed=seed,
                store_preconditioner_matrices=True,
                M_inv_matvec=M_inv_matvec,
                compute_T_via_pinv=True  # Use T_use_svd method
            )
            pqr_cur.build(rank=cur_rank, timing=True)
            
            # Check if column norms became negative and adapt rank
            actual_rank = pqr_cur.k
            iter_when_negative = -1
            if pqr_cur.qr_result is not None:
                iter_when_negative = pqr_cur.qr_result.get('iter_when_all_norms_are_negative', -1)
            if pqr_cur.qr_result_T is not None:
                iter_T_neg = pqr_cur.qr_result_T.get('iter_when_all_norms_are_negative', -1)
                if iter_T_neg >= 0:
                    if iter_when_negative < 0 or iter_T_neg < iter_when_negative:
                        iter_when_negative = iter_T_neg
            
            if iter_when_negative >= 0:
                print(f"  Column norms became negative at iteration {iter_when_negative}, using rank {iter_when_negative}")
                actual_rank = iter_when_negative
                # Rebuild with adaptive rank
                pqr_cur = PivotedQRCUR(
                    Cop, col_norms_sq,
                    row_norms_squared=row_norms_sq,
                    max_rank=actual_rank,
                    debug=False,
                    column_sampling='greedy',
                    random_seed=seed,
                    store_preconditioner_matrices=True,
                    M_inv_matvec=M_inv_matvec,
                    compute_T_via_pinv=True  # Use T_use_svd method
                )
                pqr_cur.build(rank=actual_rank)
            
            M_pqr = pqr_cur.get_woodbury_solver()
            precon_time = time.time() - precon_start
            
            start_time = time.time()
            x, info, iters, gmres_t = solve_with_gmres(
                A_mv, b, M_matvec=M_pqr.solve, tol=tol, restart=restart, maxit=budget,
                use_jax=True, verbose=False, callback=None, x_true=None
            )
            timing_gmres_t = time.time() - start_time
            
            rel = float(jnp.linalg.norm(A_mv(x) - b).block_until_ready() / jnp.linalg.norm(b).block_until_ready())
            err = float(jnp.linalg.norm(x - x_true).block_until_ready() / jnp.linalg.norm(x_true).block_until_ready())
            total_t = precon_time + timing_gmres_t
            jax.clear_caches()
            
            result_data = {
                "time": timing_gmres_t,
                "gmres_time": timing_gmres_t,
                "precon_time": precon_time,
                "total_time": total_t,
                "iters": budget * restart,
                "info": int(info) if isinstance(info, (jnp.ndarray, np.ndarray)) else (int(info) if info != "N/A" else 0),
                "rel_res": rel,
                "error": err,
                "rank": int(actual_rank),  # Actual rank used (may be less than requested if column norms became negative)
                "requested_rank": int(cur_rank),  # Rank that was requested
                "iter_when_negative": int(iter_when_negative) if iter_when_negative >= 0 else None,
                "error_history": error_history,
                "time_history": res_with_callback.get("time_history", []),
                "restart_cycles_budget": budget,
                "restart_cycles_target": int(cycle_idx) if reached and cycle_idx is not None else None,
                "target_error": TARGET_REL_ERROR,
                "target_error_reached": reached,
                "final_error_meets_target": bool(err <= TARGET_REL_ERROR)
            }
            
        elif method == 'pivoted_qr_cur_random':
            print("  Running with callback (for convergence plots)...")
            res_with_callback = benchmark_pivoted_qr_cur_precon(
                problem_size=problem_size, cur_rank=cur_rank, tau=tau, use_laplace=True,
                tol=tol, restart=restart, maxit=maxit,
                use_gpu=(gpu_id >= 0), use_jax=True, verbose=False,
                sampling='random', random_seed=seed, use_adaptive_rank=True  # Stop when column norms become negative
            )
            
            error_history = res_with_callback.get("error_history", [])
            budget, reached, cycle_idx = determine_restart_budget(
                error_history, TARGET_REL_ERROR, maxit
            )
            
            print("  Running without callback (for accurate timing)...")
            # Rebuild operator
            Cop, A_mv, b, M_inv_matvec, x_true = _build_operator_and_rhs(
                problem_size, tau, use_laplace=True, rng_seed=seed
            )
            
            precon_start = time.time()
            row_norms_sq = Cop.compute_all_row_norms_squared()
            A_T = Cop.T
            col_norms_sq = A_T.compute_all_row_norms_squared()
            
            pqr_cur = PivotedQRCUR(
                Cop, col_norms_sq,
                row_norms_squared=row_norms_sq,
                max_rank=cur_rank,
                debug=False,
                column_sampling='random',
                random_seed=seed,
                store_preconditioner_matrices=True,
                M_inv_matvec=M_inv_matvec,
                compute_T_via_pinv=True  # Use T_use_svd method
            )
            pqr_cur.build(rank=cur_rank, timing=True)
            
            # Check if column norms became negative and adapt rank
            actual_rank = pqr_cur.k
            iter_when_negative = -1
            if pqr_cur.qr_result is not None:
                iter_when_negative = pqr_cur.qr_result.get('iter_when_all_norms_are_negative', -1)
            if pqr_cur.qr_result_T is not None:
                iter_T_neg = pqr_cur.qr_result_T.get('iter_when_all_norms_are_negative', -1)
                if iter_T_neg >= 0:
                    if iter_when_negative < 0 or iter_T_neg < iter_when_negative:
                        iter_when_negative = iter_T_neg
            
            if iter_when_negative >= 0:
                print(f"  Column norms became negative at iteration {iter_when_negative}, using rank {iter_when_negative}")
                actual_rank = iter_when_negative
                # Rebuild with adaptive rank
                pqr_cur = PivotedQRCUR(
                    Cop, col_norms_sq,
                    row_norms_squared=row_norms_sq,
                    max_rank=actual_rank,
                    debug=False,
                    column_sampling='random',
                    random_seed=seed,
                    store_preconditioner_matrices=True,
                    M_inv_matvec=M_inv_matvec,
                    compute_T_via_pinv=True  # Use T_use_svd method
                )
                pqr_cur.build(rank=actual_rank)
            
            M_pqr = pqr_cur.get_woodbury_solver()
            precon_time = time.time() - precon_start
            
            start_time = time.time()
            x, info, iters, gmres_t = solve_with_gmres(
                A_mv, b, M_matvec=M_pqr.solve, tol=tol, restart=restart, maxit=budget,
                use_jax=True, verbose=False, callback=None, x_true=None
            )
            timing_gmres_t = time.time() - start_time
            
            rel = float(jnp.linalg.norm(A_mv(x) - b).block_until_ready() / jnp.linalg.norm(b).block_until_ready())
            err = float(jnp.linalg.norm(x - x_true).block_until_ready() / jnp.linalg.norm(x_true).block_until_ready())
            total_t = precon_time + timing_gmres_t
            jax.clear_caches()
            
            result_data = {
                "time": timing_gmres_t,
                "gmres_time": timing_gmres_t,
                "precon_time": precon_time,
                "total_time": total_t,
                "iters": budget * restart,
                "info": int(info) if isinstance(info, (jnp.ndarray, np.ndarray)) else (int(info) if info != "N/A" else 0),
                "rel_res": rel,
                "error": err,
                "rank": int(actual_rank),  # Actual rank used (may be less than requested if column norms became negative)
                "requested_rank": int(cur_rank),  # Rank that was requested
                "iter_when_negative": int(iter_when_negative) if iter_when_negative >= 0 else None,
                "error_history": error_history,
                "time_history": res_with_callback.get("time_history", []),
                "restart_cycles_budget": budget,
                "restart_cycles_target": int(cycle_idx) if reached and cycle_idx is not None else None,
                "target_error": TARGET_REL_ERROR,
                "target_error_reached": reached,
                "final_error_meets_target": bool(err <= TARGET_REL_ERROR)
            }
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Save result
        method_result = {
            'method': method,
            'problem_size': problem_size,
            'svd_rank': svd_rank,
            'cur_rank': cur_rank,
            'tau': tau,
            'restart': restart,
            'maxit': maxit,
            'tol': tol,
            'seed': seed,
            'result': result_data
        }
        
        method_output_file = os.path.join(output_dir, f"experiment3_{method}_result.json")
        with open(method_output_file, 'w') as f:
            json.dump(method_result, f, indent=2)
        
        print(f"Successfully completed: {method_output_file}")
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        # Write error to output file so we can see what went wrong
        try:
            with open(method_output_file.replace('.json', '_error.txt'), 'w') as f:
                f.write(f"Error: {e}\n\n")
                traceback.print_exc(file=f)
        except:
            pass
        return False
    finally:
        jax.clear_caches()
        gc.collect()


def main():
    parser = argparse.ArgumentParser(
        description='Run experiment 3 for a single method'
    )
    parser.add_argument('--method', type=str, required=True,
                       choices=['unpreconditioned', 'svd_precon', 'cur_greedy', 'cur_random', 
                               'pivoted_qr_cur_greedy', 'pivoted_qr_cur_random'],
                       help='Method to run')
    parser.add_argument('--problem-size', type=int, default=320,
                       help='Problem size (dim x dim x dim) for Toeplitz')
    parser.add_argument('--svd-rank', type=int, default=16,
                       help='Rank for SVD preconditioner')
    parser.add_argument('--cur-rank', type=int, default=512,
                       help='Rank for CUR/RPLU preconditioner')
    parser.add_argument('--tau', type=float, default=np.nan,
                       help='Parameter for problem (Unused)')
    parser.add_argument('--restart', type=int, default=20,
                       help='GMRES restart parameter')
    parser.add_argument('--maxit', type=int, default=1000,
                       help='Maximum GMRES iterations')
    parser.add_argument('--tol', type=float, default=1e-16,
                       help='Tolerance for GMRES')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--gpu-id', type=int, default=0,
                       help='GPU ID to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    success = run_single_experiment(
        method=args.method,
        problem_size=args.problem_size,
        svd_rank=args.svd_rank,
        cur_rank=args.cur_rank,
        tau=args.tau,
        restart=args.restart,
        maxit=args.maxit,
        tol=args.tol,
        output_dir=args.output_dir,
        gpu_id=args.gpu_id,
        seed=args.seed,
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
