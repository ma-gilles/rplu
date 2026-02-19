import os
import sys

import time
import numpy as np
import traceback as _traceback
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
# Ensure project root (parent of this 'cur' package) is on sys.path
_this_dir = os.path.dirname(os.path.abspath(__file__))
_proj_root = os.path.dirname(_this_dir)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

# Local imports (root-level modules)
from kernels import drifted_aniso_gaussian_kernel3d
from conv_nd_operator import ConvNDOperator
from randomized_svd import randomized_svd, WoodburyPreconditioner
# Imports from this package
from cur import CURIncremental, CUR_woodbury_solver
from laplace import InvLaplacianDirichlet3D, laplacian3d_dirichlet


def solve_with_gmres(A_matvec, b, M_matvec=None, tol=1e-16, restart=50, maxit=1, use_jax=True, verbose=False, callback=None, x_true=None):
    """Minimal GMRES wrapper (JAX version by default).
    
    Args:
        callback: Optional function(x, restart_num, error) called after each restart with current solution x
        x_true: Optional exact solution for error computation in callback
    """
    from scipy.sparse.linalg import gmres as scipy_gmres
    atol_value = float(tol * jnp.linalg.norm(b))
    print(f"ATOL: {atol_value}")
    print(f"TOL: {tol}")
    start = time.time()
    if use_jax:
        if callback is not None:
            # Use custom JAX GMRES with callback support
            print("Using custom JAX GMRES with callback support")
            x, info = gmres_with_callback(A_matvec, b, M_matvec, restart=restart, maxiter=maxit,
                                         atol=atol_value, callback=callback, x_true=x_true)
        else:
            print("Using JAX GMRES")
            x, info = jax.scipy.sparse.linalg.gmres(A_matvec, b, restart=restart, maxiter=maxit,
                                                    atol=atol_value, tol=0.0,
                                                    M=M_matvec if M_matvec is not None else None)
        x.block_until_ready()
        elapsed = time.time() - start
        print(f"GMRES time: {elapsed:.3f}s")
        return x, info, "N/A", elapsed
    # SciPy fallback
    b_np = np.array(b)
    def A_mv_np(x_np):
        return np.array(A_matvec(jnp.asarray(x_np)))
    from scipy.sparse.linalg import LinearOperator
    A_op = LinearOperator((b_np.size, b_np.size), matvec=A_mv_np, dtype=np.float64)
    if M_matvec is not None:
        def M_mv_np(x_np):
            return np.array(M_matvec(jnp.asarray(x_np)))
        M_op = LinearOperator((b_np.size, b_np.size), matvec=M_mv_np, dtype=np.float64)
    else:
        M_op = None
    iters = [0]
    def cb(_):
        iters[0] += 1
    print("Using SciPy GMRES with callback")
    x_np, info = scipy_gmres(A_op, b_np, restart=restart, maxiter=maxit,
                             atol=atol_value, rtol=0.0, callback=cb, callback_type='pr_norm', M=M_op)
    elapsed = time.time() - start
    return jnp.asarray(x_np), info, iters[0], elapsed


def gmres_with_callback(A_matvec, b, M_matvec=None, restart=50, maxiter=1, atol=1e-8, callback=None, x_true=None):
    """
    JAX GMRES implementation with callback support at each restart.
    
    This implements restarted GMRES where callbacks are invoked after each restart cycle.
    For each restart cycle, we solve for an update to the current solution.
    
    Args:
        A_matvec: Function computing A @ x
        b: Right-hand side vector
        M_matvec: Optional preconditioner function M^{-1} @ x (right preconditioning)
        restart: Number of iterations per restart cycle (Krylov subspace size)
        maxiter: Maximum number of restart cycles
        atol: Absolute tolerance for convergence
        callback: Optional function(x, restart_num, error) called after each restart
                  with current solution x, restart number, and relative error ||x - x_true|| / ||x_true||
        x_true: Exact solution for error computation (required if callback is provided)
    
    Returns:
        x: Solution vector
        info: Convergence info (0 = converged, >0 = maxiter reached without convergence)
    """
    b_norm = float(jnp.linalg.norm(b).block_until_ready())
    atol_value = float(atol)
    
    # Convert x_true to JAX array if provided
    if x_true is not None:
        x_true = jnp.asarray(x_true)
        x_true_norm = float(jnp.linalg.norm(x_true).block_until_ready())
    
    # Initial guess
    x = jnp.zeros_like(b)
    
    # Call callback with initial guess (use efficient computation)
    if callback is not None:
        if x_true is not None:
            # Initial guess is zeros, so error is just ||x_true|| / ||x_true|| = 1.0
            error = 1.0
        else:
            # Fallback to residual if x_true not provided
            r0 = b - A_matvec(x)
            error = float(jnp.linalg.norm(r0).block_until_ready() / max(b_norm, 1e-16))
        callback(x, 0, error)
    
    for restart_cycle in range(maxiter):
        # Compute residual r = b - A @ x
        r = b - A_matvec(x)
        
        # Check convergence (defer blocking)
        r_norm_sq = jnp.sum(r * r)  # Squared norm, cheaper than full norm
        r_norm = float(jnp.sqrt(r_norm_sq).block_until_ready())
        
        if r_norm <= atol_value:
            return x, 0
        
        # Build operator for this restart cycle
        # We solve: find dx such that A @ dx ≈ r
        # Use JAX GMRES with built-in preconditioner support (M parameter) to avoid transpose issues
        inner_tol = max(atol_value / max(b_norm, 1.0), 1e-12)
        dx, info_inner = jax.scipy.sparse.linalg.gmres(
            A_matvec, r, restart=restart, maxiter=1,
            atol=inner_tol, tol=0.0,
            M=M_matvec  # Right preconditioning - JAX handles this correctly without transpose issues
        )
        
        # Update solution
        x = x + dx
        
        # Compute error for callback (only if needed, and batch the computation)
        if callback is not None:
            if x_true is not None:
                # Compute error: reuse the residual we'll compute next, or compute separately
                # But minimize blocking - compute error and residual together if possible
                error_diff = x - x_true
                error_norm_sq = jnp.sum(error_diff * error_diff)
                error = float(jnp.sqrt(error_norm_sq).block_until_ready() / max(x_true_norm, 1e-16))
            else:
                # Fallback to residual if x_true not provided
                r_new = b - A_matvec(x)
                r_new_norm = float(jnp.linalg.norm(r_new).block_until_ready())
                error = r_new_norm / max(b_norm, 1e-16)
                # Use this for convergence check too
                if r_new_norm <= atol_value:
                    callback(x, restart_cycle + 1, error)
                    return x, 0
            callback(x, restart_cycle + 1, error)
        else:
            # Only check convergence if no callback (callback path handles it above)
            r_new = b - A_matvec(x)
            r_new_norm = float(jnp.linalg.norm(r_new).block_until_ready())
            if r_new_norm <= atol_value:
                return x, 0
    
    # Final convergence check
    r_final = b - A_matvec(x)
    r_final_norm = float(jnp.linalg.norm(r_final).block_until_ready())
    info = 0 if r_final_norm <= atol_value else maxiter
    
    return x, info


def randomized_lowrank_factor(Cop, rank, oversamp=0, rng=None, batch_matvec=False):
    """Return U_lr, V_lr, s such that U_lr @ V_lr.T ≈ C (symmetric factorization)."""
    if rng is None:
        rng = np.random.default_rng(0)
    U, s, Vt = randomized_svd(Cop, n_components=int(rank), n_oversamples=int(oversamp),
                              random_state=rng.integers(0, 2**31), batch_matvec=batch_matvec)
    U_lr = U * jnp.sqrt(s)
    V_lr = Vt.T * jnp.sqrt(s)
    return U_lr, V_lr, s

def _configure_platform(use_gpu: bool, gpu_id: int = 2):
    if use_gpu:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(gpu_id))
    else:
        os.environ['JAX_PLATFORMS'] = 'cpu'
    jax.config.update("jax_enable_x64", True)


def _build_operator_and_rhs(problem_size: int, tau: float, use_laplace: bool = True, rng_seed: int = 0):
    rng = np.random.default_rng(rng_seed)
    d = int(problem_size)
    shape = (d, d, d)
    kernel_width = 2 * problem_size + 1
    scale = problem_size / 300 # Scale the kernel to the problem size
    kernel = drifted_aniso_gaussian_kernel3d(
        (kernel_width, kernel_width, kernel_width),
        sigmas=np.array([1.0, 1.0, 1.0]) * 80 *scale, 
        alpha=1.0,
        beta=0.000,
        delta=50 * scale
    )

    Cop = ConvNDOperator(jnp.asarray(kernel, dtype=jnp.float64), shape, dtype=jnp.float64)
    N = Cop.N
    
    # Define M_inv_matvec based on use_laplace flag
    if use_laplace:
        invL = InvLaplacianDirichlet3D(d, d, d)
        def M_inv_matvec(x):
            return invL.apply(x.reshape(d, d, d)).reshape(d*d*d)
        def M_matvec(x):
            return laplacian3d_dirichlet(x.reshape(d, d, d)).reshape(d*d*d)
    else:
        def M_inv_matvec(x):
            return x / tau
        def M_matvec(x):
            return x * tau
    
    def A_mv(x):
        return M_matvec(x) + Cop.matvec(x)
    
    x_true = jnp.asarray(rng.standard_normal(N))
    b = A_mv(x_true)
    return Cop, A_mv, b, M_inv_matvec, x_true


def benchmark_unpreconditioned(problem_size=64, tau=1e2, use_laplace=True, tol=1e-8, restart=50, maxit=1, use_gpu=True, use_jax=True, verbose=True):
    _configure_platform(use_gpu)
    Cop, A_mv, b, M_inv_matvec, x_true = _build_operator_and_rhs(problem_size, tau, use_laplace)
    if verbose:
        print("\n[Unpreconditioned]")
    
    # Track error and timing at each restart
    error_history = []
    time_history = []
    gmres_start_time = None
    
    def error_callback(x, restart_num, error):
        nonlocal gmres_start_time
        if gmres_start_time is None:
            gmres_start_time = time.time()
        elapsed = time.time() - gmres_start_time
        error_history.append(float(error))
        time_history.append(float(elapsed))
        if verbose:
            print(f"  Restart {restart_num}: error={error:.2e}, time={elapsed:.3f}s")
    
    x, info, iters, gmres_t = solve_with_gmres(
        A_mv, b, M_matvec=None, tol=tol, restart=restart, maxit=maxit, 
        use_jax=use_jax, verbose=verbose, callback=error_callback, x_true=x_true
    )
    rel = float(jnp.linalg.norm(A_mv(x) - b).block_until_ready() / jnp.linalg.norm(b).block_until_ready())
    err = float(jnp.linalg.norm(x - x_true).block_until_ready() / jnp.linalg.norm(x_true).block_until_ready())
    jax.clear_caches()
    return dict(time=gmres_t, gmres_time=gmres_t, precon_time=0.0, total_time=gmres_t, iters=iters, info=info, rel_res=rel, error=err,
                error_history=error_history, time_history=time_history)


def benchmark_svd_precon(problem_size=64, svd_rank=32, svd_ranks=None, tau=1e2, use_laplace=True, tol=1e-8, restart=50, maxit=1, use_gpu=True, use_jax=True, verbose=True, show_trace=False):
    _configure_platform(use_gpu)
    Cop, A_mv, b, M_inv_matvec, x_true = _build_operator_and_rhs(problem_size, tau, use_laplace)
    # Build candidate rank list: explicit list or fallback halving
    if svd_ranks is not None and len(svd_ranks) > 0:
        candidates = [int(r) for r in svd_ranks if int(r) > 0]
    else:
        r = int(svd_rank)
        candidates = []
        while r >= 1:
            candidates.append(r)
            r //= 2
    last_err = None
    for r in candidates:
        try:
            if verbose:
                print(f"\n[SVD Woodbury] Trying rank={r}")
            try:
                precon_start = time.time()
                U_lr, V_lr, Svals = randomized_lowrank_factor(Cop, rank=r, oversamp=0, rng=np.random.default_rng(0), batch_matvec=False)
            except Exception as e_fac:
                last_err = e_fac
                print(f"  Factorization failed at rank={r}: {e_fac}")
                if show_trace:
                    print(_traceback.format_exc())
                continue

            M_svd = WoodburyPreconditioner(M_inv_matvec, U_lr, V_lr)
            precon_time = time.time() - precon_start
            
            # Track error and timing at each restart
            error_history = []
            time_history = []
            gmres_start_time = None
            
            def error_callback(x, restart_num, error):
                nonlocal gmres_start_time
                if gmres_start_time is None:
                    gmres_start_time = time.time()
                elapsed = time.time() - gmres_start_time
                error_history.append(float(error))
                time_history.append(float(elapsed))
                if verbose:
                    print(f"    Restart {restart_num}: error={error:.2e}, time={elapsed:.3f}s")
            
            try:
                x, info, iters, gmres_t = solve_with_gmres(
                    A_mv, b, M_matvec=M_svd, tol=tol, restart=restart, maxit=maxit, 
                    use_jax=use_jax, verbose=verbose, callback=error_callback, x_true=x_true
                )
            except Exception as e_solve:
                last_err = e_solve
                print(f"  GMRES solve failed at rank={r}: {e_solve}")
                if show_trace:
                    print(_traceback.format_exc())
                continue

            rel = float(jnp.linalg.norm(A_mv(x) - b).block_until_ready() / jnp.linalg.norm(b).block_until_ready())
            err = float(jnp.linalg.norm(x - x_true).block_until_ready() / jnp.linalg.norm(x_true).block_until_ready())
            total_t = precon_time + gmres_t
            jax.clear_caches()
            return dict(time=gmres_t, gmres_time=gmres_t, precon_time=precon_time, total_time=total_t, iters=iters, info=info, rel_res=rel, error=err,
                        sigma1=float(Svals[0]), sigmar=float(Svals[min(len(Svals) - 1, r - 1)]), rank=r,
                        error_history=error_history, time_history=time_history)
        except Exception as e:
            last_err = e
            jax.clear_caches()
            print(f"  Rank {r} unexpected failure: {e}")
            if show_trace:
                print(_traceback.format_exc())
            continue

        finally:
            try:
                jax.clear_caches()
            except Exception:
                pass
    raise RuntimeError(f"All SVD ranks failed: {candidates}. Last error: {last_err}")


def benchmark_cur_precon(problem_size=64, cur_rank=128, tau=1e2, use_laplace=True, tol=1e-8, restart=50, maxit=1, use_gpu=True, use_jax=True, verbose=True, sampling='random'):
    """
    Benchmark CURIncremental preconditioner.
    
    Parameters
    ----------
    sampling : str
        Sampling strategy: 'random' or 'greedy'
    """
    _configure_platform(use_gpu)
    Cop, A_mv, b, M_inv_matvec, x_true = _build_operator_and_rhs(problem_size, tau, use_laplace)
    if verbose:
        print(f"\n[CUR Woodbury] (sampling={sampling})")
    precon_start = time.time()
    row_norms = Cop.compute_all_row_norms_squared()
    
    # Convert sampling to pivot_sampling format expected by CURIncremental
    pivot_sampling = sampling if sampling in ['random', 'greedy', 'uniform'] else 'random'
    
    cur = CURIncremental(Cop, row_norms, debug=False, pivot_sampling=pivot_sampling,
                         store_preconditioner_matrices=True, profile=False, use_jit=True, M_inv_matvec=M_inv_matvec,
                         random_seed=42)
    build_result = cur.build(int(cur_rank), timing=True)
    if build_result is None or 'I' not in build_result:
        raise ValueError(f"CUR build failed for rank {cur_rank}, problem_size={problem_size}")
    I_cur = build_result['I'].block_until_ready()
    
    if cur.k == 0:
        raise ValueError(f"CUR build completed but k=0 (no columns selected) for rank {cur_rank}, problem_size={problem_size}")
    
    # Track when norms went negative
    iter_when_negative = build_result.get('iter_when_all_norms_are_negative', -1)
    
    # Ensure W_squared_core is computed
    if not hasattr(cur, 'W_squared_core') or cur.W_squared_core is None:
        if verbose:
            print(f"  Computing W_squared_core...")
        cur._compute_W_squared_core()
    
    M_cur = CUR_woodbury_solver(cur, M_inv_matvec=M_inv_matvec)
    precon_time = time.time() - precon_start
    
    # Track error and timing at each restart
    error_history = []
    time_history = []
    gmres_start_time = None
    
    def error_callback(x, restart_num, error):
        nonlocal gmres_start_time
        if gmres_start_time is None:
            gmres_start_time = time.time()
        elapsed = time.time() - gmres_start_time
        error_history.append(float(error))
        time_history.append(float(elapsed))
        if verbose:
            print(f"  Restart {restart_num}: error={error:.2e}, time={elapsed:.3f}s")
    
    x, info, iters, gmres_t = solve_with_gmres(
        A_mv, b, M_matvec=M_cur.solve, tol=tol, restart=restart, maxit=maxit, 
        use_jax=use_jax, verbose=verbose, callback=error_callback, x_true=x_true
    )
    rel = float(jnp.linalg.norm(A_mv(x) - b).block_until_ready() / jnp.linalg.norm(b))
    err = float(jnp.linalg.norm(x - x_true).block_until_ready() / jnp.linalg.norm(x_true))
    total_t = precon_time + gmres_t
    jax.clear_caches()
    return dict(time=gmres_t, gmres_time=gmres_t, precon_time=precon_time, total_time=total_t, iters=iters, info=info, rel_res=rel, error=err, rank=int(cur_rank),
                error_history=error_history, time_history=time_history, sampling=sampling,
                iter_when_all_norms_are_negative=int(iter_when_negative))


def benchmark_pivoted_qr_cur_precon(problem_size=64, cur_rank=128, tau=1e2, use_laplace=True, 
                                    tol=1e-8, restart=50, maxit=1, use_gpu=True, use_jax=True, 
                                    verbose=True, sampling='random', random_seed=42, 
                                    use_adaptive_rank=True):
    """
    Benchmark PivotedQRCUR preconditioner with Woodbury solver.
    
    Parameters
    ----------
    sampling : str
        Sampling strategy: 'random', 'greedy', or 'uniform'
    random_seed : int
        Random seed for reproducibility
    use_adaptive_rank : bool
        If True, reduce rank to iteration when all norms became negative
    """
    _configure_platform(use_gpu)
    Cop, A_mv, b, M_inv_matvec, x_true = _build_operator_and_rhs(problem_size, tau, use_laplace)
    if verbose:
        print(f"\n[PivotedQRCUR Woodbury] (sampling={sampling})")
    precon_start = time.time()
    
    # Compute norms
    row_norms_sq = Cop.compute_all_row_norms_squared()
    A_T = Cop.T
    col_norms_sq = A_T.compute_all_row_norms_squared()
    
    # Build PivotedQRCUR with preconditioner matrices precomputed
    from pivoted_qr_cur import PivotedQRCUR
    cur_obj = PivotedQRCUR(
        Cop, col_norms_sq,
        row_norms_squared=row_norms_sq,
        max_rank=cur_rank,
        debug=False,
        column_sampling=sampling,
        random_seed=random_seed,
        store_preconditioner_matrices=True,
        M_inv_matvec=M_inv_matvec,
        compute_T_via_pinv=True  # Use T_use_svd method
    )
    result = cur_obj.build(rank=cur_rank, timing=True)
    
    # Check if norms went negative and adapt rank if requested
    actual_rank = cur_obj.k
    iter_when_negative = -1
    if cur_obj.qr_result is not None:
        iter_when_negative = cur_obj.qr_result.get('iter_when_all_norms_are_negative', -1)
    if cur_obj.qr_result_T is not None:
        iter_T_neg = cur_obj.qr_result_T.get('iter_when_all_norms_are_negative', -1)
        if iter_T_neg >= 0:
            if iter_when_negative < 0 or iter_T_neg < iter_when_negative:
                iter_when_negative = iter_T_neg
    
    if use_adaptive_rank and iter_when_negative >= 0:
        # Use the rank at which norms became negative
        actual_rank = iter_when_negative
        if verbose:
            print(f"  Norms became negative at iteration {iter_when_negative}, using rank {actual_rank}")
        # Rebuild with adaptive rank (preconditioner matrices will be recomputed)
        cur_obj = PivotedQRCUR(
            Cop, col_norms_sq,
            row_norms_squared=row_norms_sq,
            max_rank=actual_rank,
            debug=False,
            column_sampling=sampling,
            random_seed=random_seed,
            store_preconditioner_matrices=True,
            M_inv_matvec=M_inv_matvec,
            compute_T_via_pinv=True  # Use T_use_svd method
        )
        cur_obj.build(rank=actual_rank)
    
    # Get Woodbury solver (uses precomputed matrices from build)
    M_cur = cur_obj.get_woodbury_solver()
    precon_time = time.time() - precon_start
    
    # Track error and timing at each restart
    error_history = []
    time_history = []
    gmres_start_time = None
    
    def error_callback(x, restart_num, error):
        nonlocal gmres_start_time
        if gmres_start_time is None:
            gmres_start_time = time.time()
        elapsed = time.time() - gmres_start_time
        error_history.append(float(error))
        time_history.append(float(elapsed))
        if verbose:
            print(f"  Restart {restart_num}: error={error:.2e}, time={elapsed:.3f}s")
    
    x, info, iters, gmres_t = solve_with_gmres(
        A_mv, b, M_matvec=M_cur.solve, tol=tol, restart=restart, maxit=maxit, 
        use_jax=use_jax, verbose=verbose, callback=error_callback, x_true=x_true
    )
    rel = float(jnp.linalg.norm(A_mv(x) - b).block_until_ready() / jnp.linalg.norm(b))
    err = float(jnp.linalg.norm(x - x_true).block_until_ready() / jnp.linalg.norm(x_true))
    total_t = precon_time + gmres_t
    jax.clear_caches()
    
    return dict(time=gmres_t, gmres_time=gmres_t, precon_time=precon_time, total_time=total_t, 
                iters=iters, info=info, rel_res=rel, error=err, rank=int(actual_rank),
                requested_rank=int(cur_rank),
                iter_when_all_norms_are_negative=int(iter_when_negative),
                error_history=error_history, time_history=time_history,
                sampling=sampling, random_seed=random_seed,
                use_adaptive_rank=use_adaptive_rank)
