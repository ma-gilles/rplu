#!/usr/bin/env python3

"""
Test RPLU vs Complete LU on the non-symmetric matrix function
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import time
import os
import sys

# Add the current directory and repo root to the path for local imports
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.dirname(_here))

from rplu import lu as rplu_lu

fig_dir = 'plots/cauchy'
os.makedirs(fig_dir, exist_ok=True)

TICK_LABELSIZE = 32
LEGEND_FONTSIZE = 28
ANNOTATION_FONTSIZE = 28
POINT_ALPHA = 0.35
PIVOT_ALPHA = 0.9
PIVOT_SIZE = 400
SMILE_PIVOT_SIZE = 700


def _match_xy_tick_spacing(ax, nbins=6):
    locator = MaxNLocator(nbins=nbins)
    ax.xaxis.set_major_locator(locator)
    ax.yaxis.set_major_locator(locator)


def _lupp_row_pivots(A, k, eps=1e-12):
    """
    Row pivots from LU with partial pivoting (LUPP).

    Returns indices into the original row order.
    """
    A_work = A.copy()
    m, n = A_work.shape
    perm = np.arange(m)
    pivots = []
    max_steps = min(k, m, n)

    for j in range(max_steps):
        pivot_offset = np.argmax(np.abs(A_work[j:, j]))
        pivot_row = j + pivot_offset
        if np.abs(A_work[pivot_row, j]) < eps:
            break
        if pivot_row != j:
            A_work[[j, pivot_row], :] = A_work[[pivot_row, j], :]
            perm[[j, pivot_row]] = perm[[pivot_row, j]]
        pivots.append(perm[j])

        # Eliminate below the pivot
        if j + 1 < m:
            A_work[j + 1:, j] /= A_work[j, j]
            A_work[j + 1:, j + 1:] -= np.outer(A_work[j + 1:, j], A_work[j, j + 1:])

    return np.array(pivots, dtype=int)


def _iterative_cur_lupp(A, block_size=50, eps=1e-6, max_rank=None, seed=0):
    """
    IterativeCUR (Algorithm 3.1) using LUPP for column/row selection.
    """
    rng = np.random.default_rng(seed)
    m, n = A.shape
    b = int(block_size)
    if b <= 0:
        raise ValueError("block_size must be positive")
    c = max(int(np.floor(1.1 * b)), b)

    # Sketch matrix and initial residual sketch S_col^0 = G A
    G = rng.standard_normal((c, m))
    GA = G @ A
    S_col = GA.copy()
    GA_norm = np.linalg.norm(GA, ord='fro')
    if GA_norm == 0:
        raise ValueError("Sketch of A has zero norm; cannot proceed.")

    J = []
    I = []
    C = None
    R = None
    U = None

    rho = np.linalg.norm(S_col, ord='fro') / GA_norm
    k = 0

    while rho > eps:
        if max_rank is not None and len(J) >= max_rank:
            break

        # Column selection on sketched residual S_col
        cols = _lupp_row_pivots(S_col.T, b)
        cols = cols.tolist()
        if len(cols) == 0:
            break

        # Row residual using the newly selected columns
        if C is None:
            S_row = A[:, cols]
        else:
            S_row = A[:, cols] - C @ (U @ R[:, cols])

        # Row selection on row residual
        rows = _lupp_row_pivots(S_row, b).tolist()
        if len(rows) == 0:
            break

        # Append indices
        J.extend(cols)
        I.extend(rows)
        J = list(dict.fromkeys(J))
        I = list(dict.fromkeys(I))

        # Update CUR factors
        C = A[:, J]
        R = A[I, :]
        U = np.linalg.pinv(A[np.ix_(I, J)])

        # Update sketched residual S_col = G(A - C U R)
        S_col = GA - (G @ C) @ (U @ R)
        rho = np.linalg.norm(S_col, ord='fro') / GA_norm
        k += 1

    if C is None or R is None or U is None:
        return None, [], [], None

    A_hat = C @ U @ R
    return A_hat, I, J, rho



def create_smile_two_sets_matrix(n1, n2):
    """
    Create a matrix between two sets of points from a smile:
    Set 1: mouth + one eye
    Set 2: face + other eye
    
    Returns:
    - A: (n1, n2) complex matrix between the two sets
    - X1: (n1,) complex coordinates of first set
    - X2: (n2,) complex coordinates of second set
    """
    # Create points in a smile shape
    small = int(np.ceil(min(n1, n2) ** (1.0/2)))
    eye_points = small
    mouth_points = int(np.ceil(n1/10.0))
    face_points = n2 - eye_points
    
    # Ensure we have enough points
    if mouth_points + eye_points > n1:
        mouth_points = n1 - eye_points
    if face_points + eye_points > n2:
        face_points = n2 - eye_points
    
    X1 = np.zeros(n1, dtype=complex)  # Set 1: mouth + left eye
    X2 = np.zeros(n2, dtype=complex)  # Set 2: face + right eye
    idx1 = 0
    idx2 = 0
    
    # Left eye (goes to set 1)
    for i in range(eye_points):
        while True:
            x = 2 * np.random.rand() - 1
            y = 2 * np.random.rand() - 1
            if x**2 + y**2 <= 1.0:
                X1[idx1] = (x - 4.0) + 1j * (y + 4.0)
                idx1 += 1
                break
    
    # Mouth (goes to set 1)
    for x in list(np.linspace(-5.0, 5.0, mouth_points)):
        X1[idx1] = x + 1j * (x**2 / 16.0 - 5.0)
        idx1 += 1
    
    # Right eye (goes to set 2)
    for i in range(eye_points):
        while True:
            x = 2 * np.random.rand() - 1
            y = 2 * np.random.rand() - 1
            if x**2 + y**2 <= 1.0:
                X2[idx2] = (x + 4.0) + 1j * (y + 4.0)
                idx2 += 1
                break
    
    # Face (goes to set 2)
    for theta in list(np.linspace(0, 2*np.pi, face_points)):
        X2[idx2] = 10.0 * np.cos(theta) + 1j * 10.0 * np.sin(theta)
        idx2 += 1
    
    # Create complex matrix between the two sets using 1/(x1 - x2) like spiral matrix
    X1_col = X1.reshape(-1, 1)   # Column vector
    X2_row = X2.reshape(1, -1)   # Row vector
    A = 1 / (X1_col - X2_row)
    
    return A, X1, X2

import numpy as np

#n=2000, gap=0.15, turns=10.0, decay=0.1, clustering_factor=2.0
def create_spiral_matrix_new(n,
                         gap: float = 0.0,
                         turns: float = 6,
                         decay: float = 0.075,
                         clustering_factor: float = 8.0,
                         kind: str = "log",
                         jitter: float = 0.0,
                         seed=None):
    """
    Build a Cauchy-like kernel A_ij = 1 / (x1_i - x2_j) between two spirals in C,
    with controllable point clustering toward the spiral centers.

    Parameters
    ----------
    n : int
        Number of points per spiral (A is n x n).
    gap : float
        Horizontal separation between the two mirrored spirals (in complex-plane units).
    turns : float
        Number of 2π revolutions on each spiral.
    decay : float
        Spiral tightness. For 'log': r(t)=exp(-decay*t). For 'arch': r(t)=max(1-decay*t,0).
    clustering_factor : float >= 1
        Density control along the parameter t.
        1.0 = uniform; larger -> more points near the center (small radius).
    kind : {'log','arch'}
        Spiral type: logarithmic ('log') or archimedean ('arch').
    jitter : float
        Optional small complex Gaussian noise to break perfect symmetry.
    seed : int or None
        RNG seed (used only if jitter > 0).

    Returns
    -------
    A : (n,n) complex ndarray
        Kernel with entries 1 / (x1_i - x2_j).
    x1 : (n,) complex ndarray
        Points on the first spiral.
    x2 : (n,) complex ndarray
        Points on the second (mirrored and shifted) spiral.
    """
    n = int(n)
    if n <= 1:
        raise ValueError("n must be >= 2")
    if clustering_factor < 1:
        raise ValueError("clustering_factor must be >= 1")

    # Parameterize spiral by t in [0, T]; center corresponds to large t (small radius)
    T = 2 * np.pi * float(turns)
    u = (np.arange(n) + 0.5) / n  # midpoints avoid endpoints
    p = float(clustering_factor)

    # Bias samples toward t ~ T (the center) when p > 1:
    # g(u) = 1 - (1-u)^p  (monotone; g(u)=u when p=1)
    t = T * (1.0 - (1.0 - u) ** p)

    # Radius profile
    if kind == "log":
        r = np.exp(-float(decay) * t)
    elif kind == "arch":
        r = np.maximum(1.0 - float(decay) * t, 0.0)
    else:
        raise ValueError("kind must be 'log' or 'arch'")

    # Complex spiral
    z = r * np.exp(1j * t)

    # Optional tiny isotropic noise
    if jitter > 0:
        rng = np.random.default_rng(seed)
        z = z + jitter * (rng.standard_normal(n) + 1j * rng.standard_normal(n))

    # Two arms: mirror across origin and separate horizontally by 'gap'
    d = float(gap) / 2.0
    x1 = z + d
    x2 = -z - d

    # Cauchy-like kernel
    #x1 -= 0.5 - 0.5 * 1j
    # x1 += -1 - 1* 1j
    x1 +=  -1.4 * (1 + 1j)

    A = 1.0 / (x1[:, None] - x2[None, :])


    return A, x1, x2


import numpy as np


def test_rplu_vs_complete_lu(n=200, max_rank=30, gap=1e-4, exp_factor=6, extent_multiplier=1):
    """Test RPLU vs Complete LU on the non-symmetric matrix"""
    
    print(f"Testing RPLU vs Complete LU on {n}x{n} matrix")
    print("=" * 60)
    print("Visualization method: Histogram")
    print("=" * 60)
    
    # Create the matrix with detailed info
    print("Creating matrix...")
    A, x1, x2 = create_spiral_matrix_new(n)#, gap=gap, exp_factor=exp_factor, extent_multiplier=extent_multiplier, clustering_factor=3.0)
    
    print(f"Matrix shape: {A.shape}")
    print(f"Matrix condition number: {np.linalg.cond(A):.2e}")
    print(f"Matrix Frobenius norm: {np.linalg.norm(A, 'fro'):.6f}")
    
    # Check if matrix is symmetric
    is_symmetric = np.allclose(A, A.T, rtol=1e-10, atol=1e-10)
    print(f"Is symmetric: {is_symmetric}")
    
    # Check if matrix is complex
    is_complex = np.iscomplexobj(A)
    print(f"Is complex: {is_complex}")
    
    # Test Complete LU (Greedy RPLU) - run once with max_rank
    print("\n" + "="*40)
    print("Testing Complete LU (Greedy RPLU) - single run with max_rank")
    print("="*40)
    
    start_time = time.time()
    L_greedy_full, U_greedy_full, pivots_greedy_full = rplu_lu(A, max_rank, pivot='greedy')
    complete_lu_time = time.time() - start_time
    
    print(f"Complete LU (Greedy) total time for max_rank {max_rank}: {complete_lu_time:.4f} seconds")
    print(f"Pivots shape: {pivots_greedy_full.shape}")
    
    # Test RPLU (Random) - run 30 times with max_rank
    print("\n" + "="*40)
    print("Testing RPLU (Random) - 30 runs with max_rank")
    print("="*40)
    
    # Run RPLU (Random) 30 times
    num_runs = 30
    rplu_random_runs = []
    rplu_random_total_time = 0
    
    for run in range(num_runs):
        start_time = time.time()
        L_random_full, U_random_full, pivots_random_full = rplu_lu(A, max_rank, pivot='random')
        run_time = time.time() - start_time
        rplu_random_total_time += run_time
        
        # Store results for this run
        run_errors = []
        for rank in range(1, max_rank + 1):
            L_random_k = L_random_full[:, :rank]
            U_random_k = U_random_full[:rank, :]
            A_approx_random = L_random_k @ U_random_k
            run_error = np.linalg.norm(A - A_approx_random, 'fro')
            run_errors.append(run_error)
        
        rplu_random_runs.append(run_errors)
        
        if (run + 1) % 10 == 0:  # Print progress every 10 runs
            print(f"  Completed {run + 1}/{num_runs} runs...")
    
    print(f"RPLU (Random) total time for {num_runs} runs: {rplu_random_total_time:.4f} seconds")
    print(f"Average time per run: {rplu_random_total_time/num_runs:.4f} seconds")
    
    # Convert to numpy array for easier computation
    rplu_random_runs = np.array(rplu_random_runs)  # Shape: (30, max_rank)
    
    # Compute mean and standard deviation for each rank
    rplu_random_errors_mean = np.mean(rplu_random_runs, axis=0)
    rplu_random_errors_std = np.std(rplu_random_runs, axis=0)
    rplu_random_errors_min = np.min(rplu_random_runs, axis=0)
    rplu_random_errors_max = np.max(rplu_random_runs, axis=0)
    
    # Use the first run for pivots (they're all the same for the same matrix)
    L_random_full, U_random_full, pivots_random_full = rplu_lu(A, max_rank, pivot='random')
    
    # Test different ranks by extracting first k columns/rows
    ranks_to_test = list(range(1, max_rank + 1))  # 1 to max_rank in increments of 1
    rplu_greedy_errors = []
    rplu_random_pivots = []
    rplu_greedy_pivots = []
    
    print(f"\n" + "="*40)
    print("Testing different ranks by extraction")
    print("="*40)
    
    for rank in ranks_to_test:
        if rank % 2 == 0 or rank == 1:  # Print every 2nd rank and rank 1 (since max_rank=10)
            print(f"\nTesting rank {rank} (extracted from max_rank)")
        
        # Extract first k columns/rows for Random RPLU
        L_random_k = L_random_full[:, :rank]
        U_random_k = U_random_full[:rank, :]
        pivots_random_k = pivots_random_full[:rank]
        
        # Extract first k columns/rows for Greedy RPLU
        L_greedy_k = L_greedy_full[:, :rank]
        U_greedy_k = U_greedy_full[:rank, :]
        pivots_greedy_k = pivots_greedy_full[:rank]
        
        # Compute low-rank approximations
        A_approx_greedy = L_greedy_k @ U_greedy_k
        
        rplu_greedy_error = np.linalg.norm(A - A_approx_greedy, 'fro')
        
        rplu_greedy_errors.append(rplu_greedy_error)
        rplu_random_pivots.append(pivots_random_k)
        rplu_greedy_pivots.append(pivots_greedy_k)
        
        if rank % 2 == 0 or rank == 1:  # Print every 2nd rank and rank 1 (since max_rank=10)
            print(f"  RPLU (Random) mean error: {rplu_random_errors_mean[rank-1]:.2e} ± {rplu_random_errors_std[rank-1]:.2e}")
            print(f"  RPLU (Greedy) error: {rplu_greedy_error:.2e}")
            print(f"  Relative errors: {rplu_random_errors_mean[rank-1]/np.linalg.norm(A, 'fro'):.2e} / {rplu_greedy_error/np.linalg.norm(A, 'fro'):.2e}")
    
    # Test SVD for comparison
    print("\n" + "="*40)
    print("Testing SVD")
    print("="*40)
    
    start_time = time.time()
    s = np.linalg.svd(A, compute_uv=False)

    svd_time = time.time() - start_time
    
    print(f"SVD time: {svd_time:.4f} seconds")
    print(f"SVD singular values shape: {s.shape}")
    print(f"First 10 singular values: {s[:10]}")
    
    # Test SVD approximations for different ranks
    svd_errors = []
    for rank in ranks_to_test:
        svd_error = np.sqrt(np.sum(np.abs(s[rank:])**2))
        svd_errors.append(svd_error)
        # if rank % 5 == 0 or rank == 1:  # Print every 5th rank and rank 1
        #     print(f"  SVD rank {rank} (effective {max_effective_rank}) error: {svd_error:.2e}")
    
    # Create separate plots for better visualization
    print("\n========================================")
    print("Creating separate plots...")
    print("========================================")
    
    # Create 2D points: handle complex numbers
    points_x = x1
    points_y = x2
    
    # Plot 1: RPLU (Random) pivot selection order visualization
    fig1, ax1 = plt.subplots(1, 1, figsize=(12, 10))
    
    # Check if points are complex and plot individual coordinates
    if np.iscomplexobj(points_x) or np.iscomplexobj(points_y):
        # Handle complex numbers - plot individual coordinates
        points_x_real = np.real(points_x)
        points_x_imag = np.imag(points_x)
        points_y_real = np.real(points_y)
        points_y_imag = np.imag(points_y)
        
        # Plot individual x and y coordinates with enhanced styling
        ax1.scatter(points_x_real, points_x_imag, c='#1f77b4', s=110, alpha=POINT_ALPHA, 
                    label='X coordinates', marker='o', edgecolors='none', linewidth=0, zorder=2)
        ax1.scatter(points_y_real, points_y_imag, c='#2ca02c', s=110, alpha=POINT_ALPHA, 
                    label='Y coordinates', marker='s', edgecolors='none', linewidth=0, zorder=2)
        
        # Show RPLU (Random) pivot selection order for first 10 pivots
        if len(rplu_random_pivots) > 9:  # rank 10 is index 9
            pivots_random_10 = rplu_random_pivots[9][:10]  # First 10 pivots
            
            if len(pivots_random_10) > 0:
                # Plot pivot selection order with numbers
                for i, pivot in enumerate(pivots_random_10):
                    row_idx, col_idx = pivot
                    # Plot row and column pivots
                    ax1.scatter(points_x_real[row_idx], points_x_imag[row_idx], 
                               c='#d62728', s=PIVOT_SIZE, marker='^', alpha=PIVOT_ALPHA, 
                               edgecolors='none', linewidth=0, zorder=5)
                    ax1.scatter(points_y_real[col_idx], points_y_imag[col_idx], 
                               c='#ff7f0e', s=PIVOT_SIZE, marker='D', alpha=PIVOT_ALPHA, 
                               edgecolors='none', linewidth=0, zorder=5)
                    
        
        # Enhanced legend and styling
        ax1.legend(['X coordinates', 'Y coordinates', 'X pivots', 'Y pivots'], 
                   loc='upper right', fontsize=14, framealpha=0.9, fancybox=True, shadow=True)
        ax1.grid(True, alpha=0.3, zorder=1)
        ax1.tick_params(axis='both', which='major', labelsize=TICK_LABELSIZE)
        _match_xy_tick_spacing(ax1)
        ax1.set_xlabel('Real Part', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Imaginary Part', fontsize=16, fontweight='bold')
        ax1.set_title(f'RPLU (Random) Pivot Selection Order (n={n})', 
                      fontsize=18, fontweight='bold', pad=20)
        
        # Adjust layout
        plt.tight_layout()
    else:
        # Original code for real numbers
        # Create dual y-axes for better visibility
        ax1_twin = ax1.twinx()
        
        # Plot histograms on the left y-axis
        n_x, bins_x, _ = ax1.hist(points_x, bins=50, color='green', alpha=0.6, label='x coordinates', density=True, edgecolor='black', linewidth=0.5)
        n_y, bins_y, _ = ax1.hist(points_y, bins=50, color='orange', alpha=0.6, label='y coordinates', density=True, edgecolor='black', linewidth=0.5)
        
        # Set y-axis limits for histograms
        max_density = max(np.max(n_x), np.max(n_y))
        ax1.set_ylim(0, max_density * 1.1)
        ax1.set_ylabel('Density (Histogram)', color='black', fontsize=14)
        ax1.set_xlabel('Coordinate Value', fontsize=14)
        
        # Show RPLU (Random) pivot selection order for first 20 pivots on the right y-axis
        if len(rplu_random_pivots) > 19:  # rank 20 is index 19
            pivots_random_20 = rplu_random_pivots[19][:20]  # First 20 pivots
            
            if len(pivots_random_20) > 0:
                # Plot pivot selection order with numbers on the right y-axis
                for i, pivot in enumerate(pivots_random_20):
                    row_idx, col_idx = pivot
                    # Plot row and column pivots on the line
                    ax1_twin.scatter(points_x[row_idx], i+1, 
                                   c='green', s=90, marker='^', alpha=PIVOT_ALPHA,
                                   edgecolors='none', linewidth=0, zorder=5)
                    ax1_twin.scatter(points_y[col_idx], i+1, 
                                   c='orange', s=90, marker='D', alpha=PIVOT_ALPHA,
                                   edgecolors='none', linewidth=0, zorder=5)
                    
        
        # Set y-axis limits for pivot points
        ax1_twin.set_ylim(0, 21)
        ax1_twin.set_ylabel('Pivot Order', color='red', fontsize=14)
        ax1_twin.tick_params(axis='y', labelcolor='red', labelsize=TICK_LABELSIZE)
        
        ax1.legend(['x coordinates', 'y coordinates'], loc='upper right', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', which='major', labelsize=TICK_LABELSIZE)
    
    # Save first plot
    filename1 = f'{fig_dir}/rplu_random_pivots_n{n}.png'
    plt.savefig(filename1, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: RPLU (Greedy) pivot selection order visualization
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 10))
    
    # Check if points are complex and plot individual coordinates
    if np.iscomplexobj(points_x) or np.iscomplexobj(points_y):
        # Handle complex numbers - plot individual coordinates
        points_x_real = np.real(points_x)
        points_x_imag = np.imag(points_x)
        points_y_real = np.real(points_y)
        points_y_imag = np.imag(points_y)
        
        # Plot individual x and y coordinates with enhanced styling
        ax2.scatter(points_x_real, points_x_imag, c='#1f77b4', s=110, alpha=POINT_ALPHA, 
                    label='X coordinates', marker='o', edgecolors='none', linewidth=0, zorder=2)
        ax2.scatter(points_y_real, points_y_imag, c='#2ca02c', s=110, alpha=POINT_ALPHA, 
                    label='Y coordinates', marker='s', edgecolors='none', linewidth=0, zorder=2)
        
        # Show RPLU (Greedy) pivot selection order for first 10 pivots
        if len(rplu_greedy_pivots) > 9:  # rank 10 is index 9
            pivots_greedy_10 = rplu_greedy_pivots[9][:10]  # First 10 pivots
            
            if len(pivots_greedy_10) > 0:
                # Plot pivot selection order with numbers
                for i, pivot in enumerate(pivots_greedy_10):
                    row_idx, col_idx = pivot
                    # Plot row and column pivots
                    ax2.scatter(points_x_real[row_idx], points_x_imag[row_idx], 
                               c='#d62728', s=PIVOT_SIZE, marker='^', alpha=PIVOT_ALPHA, 
                               edgecolors='none', linewidth=0, zorder=5)
                    ax2.scatter(points_y_real[col_idx], points_y_imag[col_idx], 
                               c='#ff7f0e', s=PIVOT_SIZE, marker='D', alpha=PIVOT_ALPHA, 
                               edgecolors='none', linewidth=0, zorder=5)
        
        # Enhanced legend and styling
        ax2.legend(['X coordinates', 'Y coordinates', 'X pivots', 'Y pivots'], 
                   loc='upper right', fontsize=14, framealpha=0.9, fancybox=True, shadow=True)
        ax2.grid(True, alpha=0.3, zorder=1)
        ax2.tick_params(axis='both', which='major', labelsize=TICK_LABELSIZE)
        _match_xy_tick_spacing(ax2)
        # Make plot square by setting equal aspect ratio
        ax2.set_aspect('equal')
        # ax2.set_xlabel('Real Part', fontsize=16, fontweight='bold')
        # ax2.set_ylabel('Imaginary Part', fontsize=16, fontweight='bold')
        # ax2.set_title(f'RPLU (Greedy) Pivot Selection Order (n={n})', 
        #               fontsize=18, fontweight='bold', pad=20)
        
        # Adjust layout
        plt.tight_layout()
    else:
        # Original code for real numbers
        # Create dual y-axes for better visibility
        ax2_twin = ax2.twinx()
        
        # Plot histograms on the left y-axis
        n_x2, bins_x2, _ = ax2.hist(points_x, bins=50, color='green', alpha=0.6, label='x coordinates', density=True, edgecolor='black', linewidth=0.5)
        n_y2, bins_y2, _ = ax2.hist(points_y, bins=50, color='orange', alpha=0.6, label='y coordinates', density=True, edgecolor='black', linewidth=0.5)
        
        # Set y-axis limits for histograms
        max_density2 = max(np.max(n_x2), np.max(n_y2))
        ax2.set_ylim(0, max_density2 * 1.1)
        ax2.set_ylabel('Density (Histogram)', color='black', fontsize=14)
        ax2.set_xlabel('Coordinate Value', fontsize=14)
        
        # Show RPLU (Greedy) pivot selection order for first 20 pivots on the right y-axis
        if len(rplu_greedy_pivots) > 19:  # rank 20 is index 19
            pivots_greedy_20 = rplu_greedy_pivots[19][:20]  # First 20 pivots
            
            if len(pivots_greedy_20) > 0:
                # Plot pivot selection order with numbers on the right y-axis
                for i, pivot in enumerate(pivots_greedy_20):
                    row_idx, col_idx = pivot
                    # Plot row and column pivots on the line
                    ax2_twin.scatter(points_x[row_idx], i+1, 
                                   c='purple', s=90, marker='^', alpha=PIVOT_ALPHA,
                                   edgecolors='none', linewidth=0, zorder=5)
                    ax2_twin.scatter(points_y[col_idx], i+1, 
                                   c='brown', s=90, marker='D', alpha=PIVOT_ALPHA,
                                   edgecolors='none', linewidth=0, zorder=5)
                    
        
        # Set y-axis limits for pivot points
        ax2_twin.set_ylim(0, 21)
        ax2_twin.set_ylabel('Pivot Order', color='red', fontsize=14)
        ax2_twin.tick_params(axis='y', labelcolor='red', labelsize=TICK_LABELSIZE)
        
        ax2.legend(['x coordinates', 'y coordinates'], loc='upper right', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', which='major', labelsize=TICK_LABELSIZE)
    
    # Save second plot
    filename2 = f'{fig_dir}/rplu_greedy_pivots_n{n}.png'
    plt.savefig(filename2, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Error comparison across ranks with pivot visualization
    fig3, ax3 = plt.subplots(1, 1, figsize=(12, 8))
    ranks = list(range(1, max_rank + 1))
    
    # Plot RPLU (Random) with error bars
    ax3.semilogy(ranks, rplu_random_errors_mean, 'b-o', linewidth=2.5, markersize=8, 
                 label='RPLU', zorder=3)
    ax3.fill_between(ranks, 
                     rplu_random_errors_min, 
                     rplu_random_errors_max, 
                     alpha=0.2, color='blue', zorder=1)
    
    # Plot Greedy LU and SVD
    ax3.semilogy(ranks, rplu_greedy_errors, 'r-s', linewidth=2.5, markersize=8, 
                 label='Greedy LU', zorder=3)
    ax3.semilogy(ranks, svd_errors, 'k--', linewidth=2.5, markersize=8, 
                 label='SVD', zorder=3)
    
    # Set x-axis ticks to only show integer ranks
    ax3.set_xticks(ranks)
    ax3.set_xticklabels([str(r) for r in ranks])
    
    # Add pivot visualization on the right y-axis
    ax3_twin = ax3.twinx()
    
    # Use the first run's pivots for visualization
    if len(rplu_random_pivots) > 0:
        # Get pivot indices for the first run at max_rank
        first_run_pivots = rplu_random_pivots[-1]  # Last element has max_rank pivots
        
        # Plot pivot order on the right y-axis
        pivot_ranks = list(range(1, len(first_run_pivots) + 1))
        pivot_values = [i for i in range(1, len(first_run_pivots) + 1)]
        
        # Create a scatter plot for pivots with a different style
        ax3_twin.scatter(pivot_ranks, pivot_values, c='purple', s=60, marker='o', 
                         alpha=0.6, label='Pivot Order', zorder=5, edgecolors='none', linewidth=0)
        
    
    # Set y-axis limits and labels
    ax3.set_ylabel('Frobenius Norm Error', fontsize=16, fontweight='bold')
    ax3.set_xlabel('Rank', fontsize=16, fontweight='bold')
    ax3_twin.set_ylabel('Pivot Order', fontsize=16, fontweight='bold', color='purple')
    
    # Set y-axis limits for pivot visualization
    ax3_twin.set_ylim(0, max_rank + 1)
    ax3_twin.tick_params(axis='y', labelcolor='purple', labelsize=TICK_LABELSIZE)
    
    # Customize the plot appearance
    ax3.grid(True, alpha=0.3, zorder=1)
    ax3.tick_params(axis='both', which='major', labelsize=TICK_LABELSIZE)
    ax3.legend(loc='upper right', fontsize=14, framealpha=0.9, fancybox=True, shadow=True)
    
    # Add title
    ax3.set_title(f'RPLU vs Greedy LU vs SVD Error Comparison (n={n})', 
                  fontsize=18, fontweight='bold', pad=20)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save third plot
    filename3 = f'{fig_dir}/rplu_error_comparison_n{n}.png'
    plt.savefig(filename3, dpi=300, bbox_inches='tight')
    plt.close()
    

    
    print(f"Plots saved as:")
    print(f"  {filename1}")
    print(f"  {filename2}")
    print(f"  {filename3}")
    
    # Print detailed summary
    print("\n" + "="*60)
    print("DETAILED SUMMARY")
    print("="*60)
    print(f"Matrix size: {n}x{n}")
    print(f"RPLU (Greedy) total time: {complete_lu_time:.4f} seconds")
    print(f"RPLU (Random) total time: {rplu_random_total_time:.4f} seconds")
    print(f"SVD time: {svd_time:.4f} seconds")
    
    # Find best results
    best_rplu_random_rank_idx = np.argmin(rplu_random_errors_mean)
    best_rplu_random_rank = ranks_to_test[best_rplu_random_rank_idx]
    best_rplu_random_error = rplu_random_errors_mean[best_rplu_random_rank_idx]
    
    best_rplu_greedy_rank_idx = np.argmin(rplu_greedy_errors)
    best_rplu_greedy_rank = ranks_to_test[best_rplu_greedy_rank_idx]
    best_rplu_greedy_error = rplu_greedy_errors[best_rplu_greedy_rank_idx]
    
    best_svd_rank_idx = np.argmin(svd_errors)
    best_svd_rank = ranks_to_test[best_svd_rank_idx]
    best_svd_error = svd_errors[best_svd_rank_idx]
    
    print(f"\nBest Results:")
    print(f"  Best RPLU (Random): Rank {best_rplu_random_rank}, Error {best_rplu_random_error:.2e}")
    print(f"  Best RPLU (Greedy): Rank {best_rplu_greedy_rank}, Error {best_rplu_greedy_error:.2e}")
    print(f"  Best SVD:  Rank {best_svd_rank}, Error {best_svd_error:.2e}")
    
    # Verify results are correct
    print(f"\nVerification:")
    print(f"  RPLU Random best error vs SVD: {best_rplu_random_error/best_svd_error:.2f}x")
    print(f"  RPLU Greedy best error vs SVD: {best_rplu_greedy_error/best_svd_error:.2f}x")
    print(f"  RPLU Random vs Greedy: {best_rplu_random_error/best_rplu_greedy_error:.2f}x")
    
    # Check if SVD is indeed optimal (should be <= all other methods)
    print(f"\nOptimality Check:")
    svd_optimal_count = 0
    total_ranks = len(ranks_to_test)
    for i, rank in enumerate(ranks_to_test):
        if svd_errors[i] <= rplu_random_errors_mean[i] and svd_errors[i] <= rplu_greedy_errors[i]:
            svd_optimal_count += 1
        else:
            if rank % 2 == 0:  # Only print warnings for every 2nd rank to avoid spam
                print(f"  WARNING: SVD not optimal at rank {rank}!")
                print(f"    SVD: {svd_errors[i]:.2e}, Random: {rplu_random_errors_mean[i]:.2e}, Greedy: {rplu_greedy_errors[i]:.2e}")
    
    print(f"  SVD optimal for {svd_optimal_count}/{total_ranks} ranks ({100*svd_optimal_count/total_ranks:.1f}%)")
    
    return {
        'rplu_ranks': ranks_to_test,
        'rplu_random_errors': rplu_random_errors_mean,
        'rplu_random_errors_std': rplu_random_errors_std,
        'rplu_greedy_errors': rplu_greedy_errors,
        'rplu_random_pivots': rplu_random_pivots,
        'rplu_greedy_pivots': rplu_greedy_pivots,
        'best_rplu_random_rank': best_rplu_random_rank,
        'best_rplu_random_error': best_rplu_random_error,
        'best_rplu_greedy_rank': best_rplu_greedy_rank,
        'best_rplu_greedy_error': best_rplu_greedy_error,
        'best_svd_rank': best_svd_rank,
        'best_svd_error': best_svd_error,
        'svd_errors': svd_errors,
        'x1': x1,
        'x2': x2
    }


def unified_matrix_analysis(A, points_x, points_y, matrix_name, max_rank, matrix_type, block_size=None, eps=1e-6, seed=0):
    """
    Unified analysis function that applies the same RPLU analysis and plotting
    to any matrix, ensuring consistency between different matrix types.
    
    Args:
        A: Input matrix
        points_x, points_y: Point coordinates for visualization
        matrix_name: Name/identifier for the matrix (for file naming)
        max_rank: Maximum rank to test
        matrix_type: Type of matrix ("spiral" or "smile") for file naming
    """
    print(f"Testing RPLU vs Complete LU on {A.shape[0]}x{A.shape[1]} matrix")
    t0 = time.time()
    
    # Create figure directory
    fig_dir = 'plots/cauchy'
    os.makedirs(fig_dir, exist_ok=True)
    
    # Test parameters
    ranks_to_test = list(range(1, max_rank + 1))
    pivot_size = SMILE_PIVOT_SIZE if matrix_type == "smile" else PIVOT_SIZE
    if block_size is None:
        block_size = min(20, max_rank)
    
    # Test Complete LU (Greedy RPLU) - single run with max_rank
    print("=" * 40)
    print("Testing Complete LU (Greedy RPLU) - single run with max_rank")
    print("=" * 40)
    
    start_time = time.time()
    L_greedy_full, U_greedy_full, rplu_greedy_pivots = rplu_lu(A, max_rank, pivot='greedy')
    complete_lu_time = time.time() - start_time
    
    print(f"Complete LU (Greedy) total time for max_rank {max_rank}: {complete_lu_time:.4f} seconds")
    print(f"Pivots shape: {rplu_greedy_pivots.shape}")

    # Test C2PLU (row-norm-greedy) - single run with max_rank
    print("\n" + "=" * 40)
    print("Testing C2PLU (Row-Norm Greedy) - single run with max_rank")
    print("=" * 40)

    start_time = time.time()
    L_c2plu_full, U_c2plu_full, c2plu_pivots = rplu_lu(A, max_rank, pivot='row_norm_greedy')
    c2plu_time = time.time() - start_time

    print(f"C2PLU total time for max_rank {max_rank}: {c2plu_time:.4f} seconds")
    print(f"Pivots shape: {c2plu_pivots.shape}")
    
    # Test RPLU - 30 runs with max_rank
    print("\n" + "=" * 40)
    print("Testing RPLU - 30 runs with max_rank")
    print("=" * 40)
    
    rplu_random_errors = []
    rplu_random_pivots = []
    rplu_random_total_time = 0
    
    for run in range(30):
        start_time = time.time()
        L_random_full, U_random_full, pivots_random = rplu_lu(A, max_rank, pivot='random')
        run_time = time.time() - start_time
        rplu_random_total_time += run_time
        
        # Compute errors for each rank
        run_errors = []
        for rank in ranks_to_test:
            L_random_k = L_random_full[:, :rank]
            U_random_k = U_random_full[:rank, :]
            A_random_k = L_random_k @ U_random_k
            error = np.linalg.norm(A - A_random_k, 'fro')
            run_errors.append(error)
        
        rplu_random_errors.append(run_errors)
        rplu_random_pivots.append(pivots_random)
        
        if (run + 1) % 10 == 0:
            print(f"Completed {run + 1}/30 runs")
    
    # Compute mean and std for RPLU
    rplu_random_errors = np.array(rplu_random_errors)
    rplu_random_errors_mean = np.mean(rplu_random_errors, axis=0)
    rplu_random_errors_std = np.std(rplu_random_errors, axis=0)
    rplu_random_errors_min = np.min(rplu_random_errors, axis=0)
    rplu_random_errors_max = np.max(rplu_random_errors, axis=0)
    
    print(f"RPLU total time: {rplu_random_total_time:.4f} seconds")
    
    # Compute CPLU errors for each rank
    rplu_greedy_errors = []
    for rank in ranks_to_test:
        L_greedy_k = L_greedy_full[:, :rank]
        U_greedy_k = U_greedy_full[:rank, :]
        A_greedy_k = L_greedy_k @ U_greedy_k
        error = np.linalg.norm(A - A_greedy_k, 'fro')
        rplu_greedy_errors.append(error)

    # Compute C2PLU errors for each rank
    c2plu_errors = []
    for rank in ranks_to_test:
        L_c2plu_k = L_c2plu_full[:, :rank]
        U_c2plu_k = U_c2plu_full[:rank, :]
        A_c2plu_k = L_c2plu_k @ U_c2plu_k
        error = np.linalg.norm(A - A_c2plu_k, 'fro')
        c2plu_errors.append(error)

    # LUPP-CUR (IterativeCUR) - 30 runs (min/max shaded)
    print("\n" + "=" * 40)
    print("Testing IterativeCUR - 30 runs with max_rank")
    print("=" * 40)

    num_runs = 30
    lupp_runs = []
    lupp_total_time = 0.0
    lupp_rhos = []
    lupp_ranks = []

    for run in range(num_runs):
        start_time = time.time()
        A_hat, I, J, rho = _iterative_cur_lupp(
            A,
            block_size=block_size,
            eps=eps,
            max_rank=max_rank,
            seed=seed + run,
        )
        run_time = time.time() - start_time
        lupp_total_time += run_time
        lupp_rhos.append(rho)

        if A_hat is None:
            run_errors = [float("inf")] * len(ranks_to_test)
            lupp_ranks.append(0)
        else:
            lupp_rank = min(len(I), len(J))
            lupp_ranks.append(lupp_rank)
            run_errors = []
            for rank in ranks_to_test:
                if rank <= lupp_rank:
                    Jk = J[:rank]
                    Ik = I[:rank]
                    Ck = A[:, Jk]
                    Rk = A[Ik, :]
                    Uk = np.linalg.pinv(A[np.ix_(Ik, Jk)])
                    A_k = Ck @ Uk @ Rk
                    run_errors.append(np.linalg.norm(A - A_k, "fro"))
                else:
                    run_errors.append(np.linalg.norm(A - A_hat, "fro"))

        lupp_runs.append(run_errors)
        if (run + 1) % 10 == 0:
            print(f"Completed {run + 1}/{num_runs} runs")

    lupp_runs = np.array(lupp_runs)
    lupp_errors_mean = np.mean(lupp_runs, axis=0)
    lupp_errors_std = np.std(lupp_runs, axis=0)
    lupp_errors_min = np.min(lupp_runs, axis=0)
    lupp_errors_max = np.max(lupp_runs, axis=0)

    # Keep downstream variable name the same as older plots/prints
    lupp_errors = lupp_errors_mean

    rho_finite = [r for r in lupp_rhos if np.isfinite(r)]
    rho_min = min(rho_finite) if rho_finite else float("nan")
    rho_max = max(rho_finite) if rho_finite else float("nan")
    rank_min = min(lupp_ranks) if lupp_ranks else 0
    rank_max = max(lupp_ranks) if lupp_ranks else 0
    print(
        f"IterativeCUR total time: {lupp_total_time:.4f} seconds "
        f"(avg {lupp_total_time/num_runs:.4f}/run); "
        f"rho in [{rho_min:.2e}, {rho_max:.2e}]; "
        f"rank in [{rank_min}, {rank_max}]"
    )
    
    # Compute SVD errors for each rank
    print("\nComputing SVD for comparison...")
    start_time = time.time()
    s = np.linalg.svd(A, compute_uv=False)
    svd_time = time.time() - start_time
    
    svd_errors = []
    for rank in ranks_to_test:
        error = np.sqrt(np.sum(np.abs(s[rank:])**2))
        svd_errors.append(error)
    
    print(f"SVD computation time: {svd_time:.4f} seconds")
    
    # Print summary for each rank
    print("\n" + "=" * 60)
    print("ERROR COMPARISON BY RANK")
    print("=" * 60)
    print(f"{'Rank':<6} {'RPLU':<15} {'CPLU':<15} {'C2PLU':<15} {'LUPP':<15} {'SVD':<15} {'R/CPLU':<10} {'R/C2':<10}")
    print("-" * 60)
    
    for i, rank in enumerate(ranks_to_test):
        rplu_random_error = rplu_random_errors_mean[i]
        rplu_greedy_error = rplu_greedy_errors[i]
        c2_error = c2plu_errors[i]
        lupp_error = lupp_errors[i]
        svd_error = svd_errors[i]
        ratio_cplu = rplu_random_error / rplu_greedy_error
        ratio_c2 = rplu_random_error / c2_error
        
        print(f"{rank:<6} {rplu_random_error:<15.2e} {rplu_greedy_error:<15.2e} {c2_error:<15.2e} {lupp_error:<15.2e} {svd_error:<15.2e} {ratio_cplu:<10.2f} {ratio_c2:<10.2f}")
    
    t_plot_start = time.time()
    # Plot 1: RPLU pivot selection order visualization
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 10))
    
    # Check if points are complex and plot individual coordinates
    if np.iscomplexobj(points_x) or np.iscomplexobj(points_y):
        # Handle complex numbers - plot individual coordinates
        points_x_real = np.real(points_x)
        points_x_imag = np.imag(points_x)
        points_y_real = np.real(points_y)
        points_y_imag = np.imag(points_y)
        
        # Plot individual x and y coordinates with enhanced styling
        ax1.scatter(points_x_real, points_x_imag, c='#1f77b4', s=110, alpha=POINT_ALPHA, 
                    label='X coordinates', marker='o', edgecolors='none', linewidth=0, zorder=2)
        ax1.scatter(points_y_real, points_y_imag, c='#2ca02c', s=110, alpha=POINT_ALPHA, 
                    label='Y coordinates', marker='s', edgecolors='none', linewidth=0, zorder=2)
        
        # Show RPLU pivot selection order for first 10 pivots
        if len(rplu_random_pivots) > 9:  # rank 10 is index 9
            pivots_random_10 = rplu_random_pivots[-1][:10]  # First 10 pivots
            
            if len(pivots_random_10) > 0:
                # Plot pivot selection order with numbers
                for i, pivot in enumerate(pivots_random_10):
                    row_idx, col_idx = pivot
                    # Plot row and column pivots
                    ax1.scatter(points_x_real[row_idx], points_x_imag[row_idx], 
                               c='#d62728', s=pivot_size, marker='^', alpha=PIVOT_ALPHA, 
                               edgecolors='none', linewidth=0, zorder=5)
                    ax1.scatter(points_y_real[col_idx], points_y_imag[col_idx], 
                               c='#ff7f0e', s=pivot_size, marker='D', alpha=PIVOT_ALPHA, 
                               edgecolors='none', linewidth=0, zorder=5)
                    
        
        # Make plot square by setting equal aspect ratio
        ax1.set_aspect('equal')
        # Remove axis labels and title for subfigure use
        # ax1.set_xlabel('Real Part', fontsize=16, fontweight='bold')
        # ax1.set_ylabel('Imaginary Part', fontsize=16, fontweight='bold')
        # ax1.set_title(f'RPLU (Random) Pivot Selection Order - {matrix_type.capitalize()} Matrix', 
        #               fontsize=18, fontweight='bold', pad=20)
        
        # Remove legend for clean subfigure use
        # ax1.legend(['X', 'Y', 'X pivots', 'Y pivots'], 
        #            loc='upper right', fontsize=18, framealpha=0.9, fancybox=True, shadow=True)
        ax1.grid(True, alpha=0.3, zorder=1)
        ax1.tick_params(axis='both', which='major', labelsize=TICK_LABELSIZE)
        _match_xy_tick_spacing(ax1)
        
        # Adjust layout
        plt.tight_layout()
    
    # Save first plot
    filename1 = f'{fig_dir}/{matrix_type}_random_pivots_{matrix_name}.png'
    plt.savefig(filename1, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: CPLU (Greedy) pivot selection order visualization
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 10))
    
    # Check if points are complex and plot individual coordinates
    if np.iscomplexobj(points_x) or np.iscomplexobj(points_y):
        # Handle complex numbers - plot individual coordinates
        points_x_real = np.real(points_x)
        points_x_imag = np.imag(points_x)
        points_y_real = np.real(points_y)
        points_y_imag = np.imag(points_y)
        
        # Plot individual x and y coordinates with enhanced styling
        ax2.scatter(points_x_real, points_x_imag, c='#1f77b4', s=110, alpha=POINT_ALPHA, 
                    label='X coordinates', marker='o', edgecolors='none', linewidth=0, zorder=2)
        ax2.scatter(points_y_real, points_y_imag, c='#2ca02c', s=110, alpha=POINT_ALPHA, 
                    label='Y coordinates', marker='s', edgecolors='none', linewidth=0, zorder=2)
        
        # Show RPLU (Greedy) pivot selection order for first 10 pivots
        if len(rplu_greedy_pivots) > 9:  # rank 10 is index 9
            pivots_greedy_10 = rplu_greedy_pivots[:10]  # First 10 pivots
            
            if len(pivots_greedy_10) > 0:
                # Plot pivot selection order with numbers
                for i, pivot in enumerate(pivots_greedy_10):
                    row_idx, col_idx = pivot
                    # Plot row and column pivots
                    ax2.scatter(points_x_real[row_idx], points_x_imag[row_idx], 
                               c='#d62728', s=pivot_size, marker='^', alpha=PIVOT_ALPHA, 
                               edgecolors='none', linewidth=0, zorder=5)
                    ax2.scatter(points_y_real[col_idx], points_y_imag[col_idx], 
                               c='#ff7f0e', s=pivot_size, marker='D', alpha=PIVOT_ALPHA, 
                               edgecolors='none', linewidth=0, zorder=5)
        
        # Make plot square by setting equal aspect ratio
        ax2.set_aspect('equal')
        # Remove axis labels and title for subfigure use
        # ax2.set_xlabel('Real Part', fontsize=16, fontweight='bold')
        # ax2.set_ylabel('Imaginary Part', fontsize=16, fontweight='bold')
        # ax2.set_title(f'RPLU (Greedy) Pivot Selection Order - {matrix_type.capitalize()} Matrix', 
        #               fontsize=18, fontweight='bold', pad=20)
        
        # Remove legend for clean subfigure use
        # ax2.legend(['X coordinates', 'Y coordinates', 'X pivots', 'Y pivots'], 
        #            loc='upper right', fontsize=18, framealpha=0.9, fancybox=True, shadow=True)
        ax2.grid(True, alpha=0.3, zorder=1)
        ax2.tick_params(axis='both', which='major', labelsize=TICK_LABELSIZE)
        _match_xy_tick_spacing(ax2)
        
        # Adjust layout
        plt.tight_layout()
    
    # Save second plot
    filename2 = f'{fig_dir}/{matrix_type}_cplu_pivots_{matrix_name}.png'
    plt.savefig(filename2, dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 3: C2PLU pivot selection order visualization
    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 10))

    if np.iscomplexobj(points_x) or np.iscomplexobj(points_y):
        points_x_real = np.real(points_x)
        points_x_imag = np.imag(points_x)
        points_y_real = np.real(points_y)
        points_y_imag = np.imag(points_y)

        ax3.scatter(points_x_real, points_x_imag, c='#1f77b4', s=110, alpha=POINT_ALPHA, 
                    label='X coordinates', marker='o', edgecolors='none', linewidth=0, zorder=2)
        ax3.scatter(points_y_real, points_y_imag, c='#2ca02c', s=110, alpha=POINT_ALPHA, 
                    label='Y coordinates', marker='s', edgecolors='none', linewidth=0, zorder=2)

        pivots_c2plu_10 = c2plu_pivots[:10]
        if len(pivots_c2plu_10) > 0:
            for i, pivot in enumerate(pivots_c2plu_10):
                row_idx, col_idx = pivot
                ax3.scatter(points_x_real[row_idx], points_x_imag[row_idx], 
                            c='#d62728', s=pivot_size, marker='^', alpha=PIVOT_ALPHA, 
                            edgecolors='none', linewidth=0, zorder=5)
                ax3.scatter(points_y_real[col_idx], points_y_imag[col_idx], 
                            c='#ff7f0e', s=pivot_size, marker='D', alpha=PIVOT_ALPHA, 
                            edgecolors='none', linewidth=0, zorder=5)

        ax3.set_aspect('equal')
        ax3.grid(True, alpha=0.3, zorder=1)
        ax3.tick_params(axis='both', which='major', labelsize=TICK_LABELSIZE)
        _match_xy_tick_spacing(ax3)
        plt.tight_layout()

    filename3 = f'{fig_dir}/{matrix_type}_c2plu_pivots_{matrix_name}.png'
    plt.savefig(filename3, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Error comparison across ranks
    fig4, ax4 = plt.subplots(1, 1, figsize=(10, 10))
    ranks = list(range(1, max_rank + 1))
    
    # Plot RPLU with error bars
    ax4.semilogy(ranks, rplu_random_errors_mean, 'b-o', linewidth=3, markersize=10, 
                 label='RPLU', zorder=3)
    ax4.fill_between(ranks, 
                     rplu_random_errors_min, 
                     rplu_random_errors_max, 
                     alpha=0.2, color='blue', zorder=1)
    
    # Plot CPLU, C2PLU, LUPP-CUR, and SVD
    ax4.semilogy(ranks, rplu_greedy_errors, 'r-s', linewidth=3, markersize=10, 
                 label='CPLU', zorder=3)
    ax4.semilogy(ranks, c2plu_errors, 'g-^', linewidth=3, markersize=10, 
                 label='C2PLU', zorder=3)
    ax4.semilogy(ranks, lupp_errors, 'm-d', linewidth=3, markersize=10, 
                 label='IterativeCUR', zorder=3)
    ax4.fill_between(
        ranks,
        lupp_errors_min,
        lupp_errors_max,
        alpha=0.2,
        color='magenta',
        zorder=1,
    )
    ax4.semilogy(ranks, svd_errors, 'k--', linewidth=3, markersize=10, 
                 label='SVD', zorder=3)
    
    # Set x-axis ticks to only show integer ranks
    ax4.set_xticks(ranks)
    ax4.set_xticklabels([str(r) for r in ranks])
    
    # For line plots, ensure square figure without distorting data
    # Set y-axis limits to maintain proper proportions
    y_candidates = (
        rplu_random_errors_mean
        + rplu_greedy_errors
        + c2plu_errors
        + lupp_errors
        + svd_errors
    )
    y_candidates = [v for v in y_candidates if np.isfinite(v)]
    y_min = min(y_candidates) if y_candidates else 1e-16
    y_max = max(y_candidates) if y_candidates else 1.0
    
    # Remove axis labels and title for subfigure use
    # ax3.set_ylabel('Frobenius Norm Error', fontsize=16, fontweight='bold')
    # ax3.set_xlabel('Rank', fontsize=16, fontweight='bold')
    # ax3.set_title(f'RPLU vs Greedy LU vs SVD Error Comparison - {matrix_type.capitalize()} Matrix', 
    #               fontsize=18, fontweight='bold', pad=20)
    
    # Customize the plot appearance
    ax4.grid(True, alpha=0.3, zorder=1)
    ax4.tick_params(axis='both', which='major', labelsize=TICK_LABELSIZE)
    # Remove legend for clean subfigure use
    # ax3.legend(loc='upper right', fontsize=18, framealpha=0.9, fancybox=True, shadow=True)
    
    # Adjust layout to maintain square proportions
    plt.tight_layout()
    
    # Save fourth plot
    filename4 = f'{fig_dir}/{matrix_type}_error_comparison_{matrix_name}.png'
    plt.savefig(filename4, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 5: Legend for pivot selection plots
    fig5, ax5 = plt.subplots(1, 1, figsize=(7, 6))
    
    ax5.scatter([], [], c='#1f77b4', s=150, marker='o', label='x points')
    ax5.scatter([], [], c='#2ca02c', s=150, marker='s', label='y points')
    ax5.scatter([], [], c='#d62728', s=pivot_size, marker='^', label='x pivots')
    ax5.scatter([], [], c='#ff7f0e', s=pivot_size, marker='D', label='y pivots')
    
    # Set plot limits to be invisible
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
    ax5.set_aspect('equal')
    
    # Remove axis elements
    ax5.set_xticks([])
    ax5.set_yticks([])
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    ax5.spines['bottom'].set_visible(False)
    ax5.spines['left'].set_visible(False)
    
    ax5.legend(
        loc='center',
        fontsize=LEGEND_FONTSIZE,
        framealpha=1.0,
        facecolor='white',
        fancybox=True,
        shadow=True,
        ncol=1,
        borderpad=0.4,
        labelspacing=0.3,
        handletextpad=0.5,
        handlelength=1.2,
    )
    
    # Adjust layout
    plt.tight_layout(pad=0.05)
    
    filename5 = f'{fig_dir}/{matrix_type}_pivot_legend_{matrix_name}.png'
    plt.savefig(filename5, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 6: Legend for algorithm names (white background)
    fig6, ax6 = plt.subplots(1, 1, figsize=(10, 10))
    
    ax6.plot([], [], 'b-o', linewidth=3, markersize=10, label='RPLU')
    ax6.plot([], [], 'r-s', linewidth=3, markersize=10, label='CPLU')
    ax6.plot([], [], 'g-^', linewidth=3, markersize=10, label='C2PLU')
    ax6.plot([], [], 'm-d', linewidth=3, markersize=10, label='IterativeCUR')
    ax6.plot([], [], 'k--', linewidth=3, markersize=10, label='Optimal')
    
    # Set plot limits to be invisible
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.set_aspect('equal')
    
    # Remove axis elements
    ax6.set_xticks([])
    ax6.set_yticks([])
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)
    ax6.spines['bottom'].set_visible(False)
    ax6.spines['left'].set_visible(False)
    
    ax6.legend(
        loc='center',
        fontsize=LEGEND_FONTSIZE,
        framealpha=1.0,
        facecolor='white',
        fancybox=True,
        shadow=True,
        ncol=1,
        columnspacing=2,
    )
    
    # Adjust layout
    plt.tight_layout()
    
    filename6 = f'{fig_dir}/{matrix_type}_alg_legend_{matrix_name}.png'
    plt.savefig(filename6, dpi=300, bbox_inches='tight')
    plt.close()
    plot_time = time.time() - t_plot_start
    
    print(f"Plots saved as:")
    print(f"  {filename1}")
    print(f"  {filename2}")
    print(f"  {filename3}")
    print(f"  {filename4}")
    print(f"  {filename5}")
    print(f"  {filename6}")
    
    # Print detailed summary
    print("\n" + "="*60)
    print("DETAILED SUMMARY")
    print("="*60)
    print(f"Matrix size: {matrix_name}")
    print(f"CPLU total time: {complete_lu_time:.4f} seconds")
    print(f"C2PLU total time: {c2plu_time:.4f} seconds")
    print(f"RPLU total time: {rplu_random_total_time:.4f} seconds")
    print(f"IterativeCUR total time: {lupp_total_time:.4f} seconds")
    print(f"SVD time: {svd_time:.4f} seconds")
    print(f"Plotting time: {plot_time:.4f} seconds")
    print(f"Total analysis time: {time.time() - t0:.4f} seconds")
    

    return {
        'rplu_random_errors': rplu_random_errors,
        'rplu_greedy_errors': rplu_greedy_errors,
        'c2plu_errors': c2plu_errors,
        'lupp_errors': lupp_errors,
        'lupp_errors_std': lupp_errors_std,
        'lupp_errors_min': lupp_errors_min,
        'lupp_errors_max': lupp_errors_max,
        'svd_errors': svd_errors,
        'rplu_random_pivots': rplu_random_pivots,
        'rplu_greedy_pivots': rplu_greedy_pivots,
        'c2plu_pivots': c2plu_pivots,
    }

# def test_singular_value_decay(n=200):
    # """Test singular value decay to understand low-rank structure"""
    
    # print(f"\nAnalyzing singular value decay for {n}x{n} matrix")
    # print("=" * 50)
    
    # A, times_col, times_row = create_line_matrix_nonsymmetric_detailed(n)
    
    # # Compute SVD
    # U, s, Vt = np.linalg.svd(A)
    
    # # Plot singular values
    # plt.figure(figsize=(10, 6))
    # plt.semilogy(range(1, len(s)+1), s, 'o-', markersize=4, linewidth=1)
    # plt.xlabel('Index')
    # plt.ylabel('Singular Value')
    # plt.title(f'Singular Value Decay (n={n})')
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.savefig(f'{fig_dir}/singular_value_decay.png', dpi=300, bbox_inches='tight')
    # plt.show()
    
    # # Print some statistics
    # print(f"First 10 singular values: {s[:10]}")
    # print(f"Condition number: {s[0]/s[-1]:.2e}")
    # print(f"Sum of first 10 singular values: {np.sum(s[:10]):.6f}")
    # print(f"Sum of all singular values: {np.sum(s):.6f}")
    # print(f"Ratio (first 10 / total): {np.sum(s[:10])/np.sum(s):.4f}")

def test_unified_matrices():
    """
    Unified test function that tests both spiral matrix and smile matrix
    using the same plotting script for consistency
    """
    print("="*80)
    print("UNIFIED MATRIX TEST - SPIRAL + SMILE")
    print("="*80)
    
    # Test 1: Spiral Matrix
    print("\n" + "="*60)
    print("TESTING SPIRAL MATRIX")
    print("="*60)
    
    # Fixed parameters for spiral matrix
    n = 4000
    max_rank = 10
    
    print(f"Matrix size: n={n}, max_rank={max_rank}")
    print("-" * 50)
    
    # Create spiral matrix
    A_spiral, points_x, points_y = create_spiral_matrix_new(n)#, gap, exp_factor, extent_multiplier)
    
    # Test spiral matrix using unified analysis function
    results_spiral = unified_matrix_analysis(A_spiral, points_x, points_y, n, max_rank, "spiral")
    
    print(f"\nSpiral matrix test completed successfully!")
    print(f"Plots saved as 'spiral_*.png' files")
    
    # Test 2: Smile Matrix
    print("\n" + "="*60)
    print("TESTING SMILE MATRIX")
    print("=" * 60)
    
    # Fixed parameters for smile test
    n1, n2 = 1000, 1500
    print(f"Testing two-sets smile matrix with n1={n1}, n2={n2}")
    
    # Create smile matrix
    A_smile, points_x, points_y = create_smile_two_sets_matrix(n1, n2)
    
    # Test smile matrix using unified analysis function
    results_smile = unified_matrix_analysis(A_smile, points_x, points_y, f"{n1}x{n2}", max_rank, "smile")
    
    print(f"\nTwo-sets smile matrix test completed successfully!")
    print(f"Results saved as 'smile_*.png' files")
    
    print("\n" + "="*80)
    print("UNIFIED TEST COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    return {
        'spiral_results': results_spiral,
        'smile_results': results_smile
    }


def test_mtest_both_sets(n=2000, max_rank=20, block_sizes=None, eps=1e-6, seed=0):
    """
    Run unified analysis on both spiral and two-set smile matrices at size n.
    """
    if block_sizes is None:
        block_sizes = [10, 20, 40]
    print("="*80)
    print(f"MTEST BOTH SETS (n={n})")
    print("="*80)

    t0 = time.time()
    print("\n" + "="*60)
    print("TESTING SPIRAL MATRIX")
    print("="*60)
    t_build = time.time()
    A_spiral, points_x, points_y = create_spiral_matrix_new(n)
    print(f"Spiral matrix build time: {time.time() - t_build:.4f} seconds")
    results_spiral = {}
    for block_size in block_sizes:
        print(f"\nUsing IterativeCUR block_size={block_size}")
        label = f"{n}_b{block_size}"
        results_spiral[block_size] = unified_matrix_analysis(
            A_spiral,
            points_x,
            points_y,
            label,
            max_rank,
            "spiral",
            block_size=block_size,
            eps=eps,
            seed=seed,
        )

    print("\n" + "="*60)
    print("TESTING SMILE MATRIX (two sets)")
    print("=" * 60)
    t_build = time.time()
    A_smile, points_x, points_y = create_smile_two_sets_matrix(n, n)
    print(f"Smile matrix build time: {time.time() - t_build:.4f} seconds")
    results_smile = {}
    for block_size in block_sizes:
        print(f"\nUsing IterativeCUR block_size={block_size}")
        label = f"{n}x{n}_b{block_size}"
        results_smile[block_size] = unified_matrix_analysis(
            A_smile,
            points_x,
            points_y,
            label,
            max_rank,
            "smile",
            block_size=block_size,
            eps=eps,
            seed=seed,
        )

    print("\n" + "="*80)
    print("MTEST BOTH SETS COMPLETED")
    print("="*80)
    print(f"Total mtest time: {time.time() - t0:.4f} seconds")
    return {
        'spiral_results': results_spiral,
        'smile_results': results_smile,
    }


if __name__ == "__main__":
    import sys
    
    test_mtest_both_sets(n=4000, max_rank=10, block_sizes=[5,])
    sys.exit(0)

    
    # Run unified test with both matrix types
    results = test_unified_matrices()
