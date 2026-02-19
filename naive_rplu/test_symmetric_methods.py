#!/usr/bin/env python3
"""Test symmetric methods (symRPLU, RPCholesky, RPLU, SVD) on symmetric matrices - Trace only"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'Randomly-Pivoted-Cholesky'))
from symmetric_rplu import rank12_symmetric
from rplu import lu
from rpcholesky import rpcholesky
from rpqr import rpqr

def create_spiral_matrix(n, bandwidth=2):
    """Create spiral-shaped kernel matrix from Randomly-Pivoted-Cholesky"""
    times = np.linspace(0, 2, n)
    times = times ** 5
    times = times[::-1]
    x = np.exp(.2 * times) * np.cos(times)
    y = np.exp(.2 * times) * np.sin(times)
    X = np.column_stack((x, y))
    
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist = np.linalg.norm(X[i] - X[j])
            A[i, j] = np.exp(-dist / bandwidth)
    
    return A, X

def create_smile_matrix(n, bandwidth=2.0):
    """Create smile-shaped kernel matrix from Randomly-Pivoted-Cholesky"""
    small = int(np.ceil(n ** (1.0/2)))
    eye_points = small
    mouth_points = int(np.ceil(n/10.0))
    face_points = n - 2 * eye_points - mouth_points
    
    X = np.zeros((n, 2))
    idx = 0
    
    # Eyes
    for x_shift in [-4.0, 4.0]:
        for i in range(eye_points):
            while True:
                x = 2 * np.random.rand() - 1
                y = 2 * np.random.rand() - 1
                if x**2 + y**2 <= 1.0:
                    X[idx, 0] = x + x_shift
                    X[idx, 1] = y + 4.0
                    idx += 1
                    break
    
    # Mouth
    for x in list(np.linspace(-5.0, 5.0, mouth_points)):
        X[idx, 0] = x
        X[idx, 1] = x**2 / 16.0 - 5.0
        idx += 1
    
    # Face
    for theta in list(np.linspace(0, 2*np.pi, face_points)):
        X[idx, 0] = 10.0 * np.cos(theta)
        X[idx, 1] = 10.0 * np.sin(theta)
        idx += 1
    
    # Create kernel matrix
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist = np.linalg.norm(X[i] - X[j])
            A[i, j] = np.exp(-dist / bandwidth)
    
    return A, X

def test_symmetric_methods(n=200, max_rank=30, include_rpqr=False, plot_rpqr=False):
    """Test symRPLU, RPCholesky, RPLU, RPQR, SVD on symmetric matrices - Trace only"""
    print(f"Testing symmetric methods on {n}x{n} matrices (Trace only)")
    if include_rpqr:
        print(f"RPQR included: {'Yes' if plot_rpqr else 'No (computed but not plotted)'}")
    
    # Create figure directory
    fig_dir = 'plots/symmetric_methods'
    os.makedirs(fig_dir, exist_ok=True)
    
    # Test matrices
    matrices = [
        ("Spiral", create_spiral_matrix(n, bandwidth=2)[0]),
        ("Smiley", create_smile_matrix(n, bandwidth=2.0)[0])
    ]

    def error_function(A, reconstruction):
        return np.sum(np.linalg.svd(A - reconstruction, compute_uv=False))
    
    def error_function_spd(A, reconstruction):
        return np.trace(A - reconstruction)

    for name, A in matrices:
        print(f"\nTesting {name} matrix")
        is_psd = True
        
        # Test methods
        ranks = list(range(1, max_rank + 1))
        
        # symRPLU (30 runs)
        print("Running symRPLU...")
        sym_errors = []
        for run in range(30):
            R, _ = rank12_symmetric(A, n_iter=max_rank, pivot='random', diag_ok=True)
            run_errors = []
            for rank in ranks:
                if rank <= R.shape[1]:
                    R_k = R[:, :rank]
                    reconstruction = R_k @ R_k.T
                    # Use trace since everything should be SPD
                    error = error_function_spd(A, reconstruction)
                else:
                    error = float('inf')
                run_errors.append(error)
            sym_errors.append(run_errors)
        
        sym_errors = np.array(sym_errors)
        sym_mean = np.mean(sym_errors, axis=0)
        sym_std = np.std(sym_errors, axis=0)
        
        # CPLU (single run)
        print("Running CPLU...")
        R_greedy, _ = rank12_symmetric(A, n_iter=max_rank, pivot='greedy', diag_ok=True)
        cplu_errors = []
        for rank in ranks:
            if rank <= R_greedy.shape[1]:
                R_k = R_greedy[:, :rank]
                reconstruction = R_k @ R_k.T
                error = error_function(A, reconstruction)
            else:
                error = float('inf')
            cplu_errors.append(error)
        
        # RPLU (30 runs)
        print("Running RPLU...")
        rplu_errors = []
        for run in range(30):
            L, U, _ = lu(A, n_iter=max_rank, pivot='random')
            run_errors = []
            for rank in ranks:
                if rank <= L.shape[1]:
                    L_k = L[:, :rank]
                    U_k = U[:rank, :]
                    reconstruction = L_k @ U_k
                    error = error_function(A, reconstruction)
                else:
                    error = float('inf')
                run_errors.append(error)
            rplu_errors.append(run_errors)
        
        rplu_errors = np.array(rplu_errors)
        rplu_mean = np.mean(rplu_errors, axis=0)
        rplu_std = np.std(rplu_errors, axis=0)
        
        # RPQR (30 runs, optional)
        if include_rpqr:
            print("Running RPQR...")
            rpqr_errors = []
            for run in range(30):
                Q, F, _ = rpqr(A, k=max_rank, accelerated=False)
                run_errors = []
                for rank in ranks:
                    if rank <= Q.shape[1]:
                        Q_k = Q[:, :rank]
                        F_k = F[:, :rank].T
                        reconstruction = Q_k @ F_k
                        error = np.sum(np.linalg.svd(A - reconstruction, compute_uv=False))
                    else:
                        error = float('inf')
                    run_errors.append(error)
                rpqr_errors.append(run_errors)
            
            rpqr_errors = np.array(rpqr_errors)
            rpqr_mean = np.mean(rpqr_errors, axis=0)
            rpqr_std = np.std(rpqr_errors, axis=0)
        
        # RPCholesky (30 runs, only for PSD)
        if is_psd:
            print("Running RPCholesky...")
            rpchol_errors = []
            for run in range(30):
                result = rpcholesky(A, k=max_rank, accelerated=False)
                run_errors = []
                for rank in ranks:
                    if rank <= result.G.shape[1]:
                        G_k = result.G[:rank]
                        reconstruction = G_k.T @ G_k
                        error = error_function_spd(A, reconstruction)
                    else:
                        error = float('inf')
                    run_errors.append(error)
                rpchol_errors.append(run_errors)
            
            rpchol_errors = np.array(rpchol_errors)
            rpchol_mean = np.mean(rpchol_errors, axis=0)
            rpchol_std = np.std(rpchol_errors, axis=0)
        
        # SVD
        print("Computing SVD...")
        s = np.linalg.svd(A, compute_uv=False)
        svd_errors = [np.sum(s[rank:]) for rank in ranks]
        
        # Plot error comparison
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        ax.semilogy(ranks, sym_mean, 'b-o', linewidth=3, markersize=10, label='symRPLU')
        ax.fill_between(ranks, sym_mean - sym_std, sym_mean + sym_std, alpha=0.2, color='blue')
        
        ax.semilogy(ranks, cplu_errors, 'r-s', linewidth=3, markersize=10, label='CPLU')
        ax.semilogy(ranks, rplu_mean, 'g-^', linewidth=3, markersize=10, label='RPLU')
        ax.fill_between(ranks, rplu_mean - rplu_std, rplu_mean + rplu_std, alpha=0.2, color='green')
        
        if include_rpqr and plot_rpqr:
            ax.semilogy(ranks, rpqr_mean, 'm-d', linewidth=3, markersize=10, label='RPQR')
            ax.fill_between(ranks, rpqr_mean - rpqr_std, rpqr_mean + rpqr_std, alpha=0.2, color='magenta')
        
        if is_psd:
            ax.semilogy(ranks, rpchol_mean, 'c-x', linewidth=3, markersize=10, label='RPCholesky')
            ax.fill_between(ranks, rpchol_mean - rpchol_std, rpchol_mean + rpchol_std, alpha=0.2, color='cyan')
        
        ax.semilogy(ranks, svd_errors, 'k--', linewidth=3, markersize=10, label='SVD')
        
        # Remove axis labels and title for clean individual plots (like test_rplu_vs_complete_lu.py)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('')
        
        # Professional tick styling - match test_rplu_vs_complete_lu.py style
        ax.tick_params(axis='both', which='major', labelsize=32)
        
        # Enhanced grid and background
        ax.grid(True, alpha=0.6, linestyle='-', linewidth=1.2, color='#666666')
        ax.set_facecolor('white')
        
        # Set consistent axis limits
        # ax.set_xlim(0, 30)
        # ax.set_ylim(1e-8, 1e2)
        
        # Enhanced spine styling
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('#404040')
        
        # Fix y-axis tick formatting for better readability
        ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
        
        # Set y-axis ticks at every factor of 2 (50, 100, 200, etc.)
        ax.yaxis.set_major_locator(plt.FixedLocator([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 400, 800]))
        
        # Force the tick locator to use our custom ticks
        ax.yaxis.set_major_locator(plt.FixedLocator([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 400, 800]))
        ax.yaxis.set_minor_locator(plt.NullLocator())
        
        # Ensure ticks are properly formatted
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}' if x >= 1 else f'{x:.1f}'))
        
        plt.tight_layout()
        plt.savefig(f'{fig_dir}/{name.lower()}_trace_n{n}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create separate legend figure
        fig_legend, ax_legend = plt.subplots(1, 1, figsize=(8, 2))
        
        # Create legend handles
        legend_handles = []
        legend_labels = []
        
        # Add all methods to legend
        legend_handles.append(plt.Line2D([0], [0], color='blue', marker='o', linewidth=3, markersize=10))
        legend_labels.append('symRPLU')
        
        legend_handles.append(plt.Line2D([0], [0], color='red', marker='s', linewidth=3, markersize=10))
        legend_labels.append('CPLU')
        
        legend_handles.append(plt.Line2D([0], [0], color='green', marker='^', linewidth=3, markersize=10))
        legend_labels.append('RPLU')
        
        if include_rpqr and plot_rpqr:
            legend_handles.append(plt.Line2D([0], [0], color='magenta', marker='d', linewidth=3, markersize=10))
            legend_labels.append('RPQR')
        
        if is_psd:
            legend_handles.append(plt.Line2D([0], [0], color='cyan', marker='x', linewidth=3, markersize=10))
            legend_labels.append('RPCholesky')
        
        legend_handles.append(plt.Line2D([0], [0], color='black', marker='', linewidth=3, linestyle='--'))
        legend_labels.append('SVD')
        
        # Create legend
        ax_legend.legend(legend_handles, legend_labels, loc='center', ncol=len(legend_handles), 
                        fontsize=16, frameon=False)
        ax_legend.set_xlim(0, 1)
        ax_legend.set_ylim(0, 1)
        ax_legend.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{fig_dir}/{name.lower()}_legend_n{n}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plot and legend saved: {name}")
        
        # Create additional plot showing the points (like test_rplu_vs_complete_lu.py)
        fig_points, ax_points = plt.subplots(1, 1, figsize=(10, 10))
        
        # Get the points from the matrix creation
        if name == "Spiral":
            _, X = create_spiral_matrix(n, bandwidth=2)
            points_x = X[:, 0]
            points_y = X[:, 1]
        elif name == "Smiley":
            _, X = create_smile_matrix(n, bandwidth=2.0)
            points_x = X[:, 0]
            points_y = X[:, 1]
        
        # Plot the points with enhanced styling
        ax_points.scatter(points_x, points_y, c='#1f77b4', s=150, alpha=0.7, 
                         marker='o', edgecolors='black', linewidth=0.8, zorder=2)
        
        # Make plot square by setting equal aspect ratio
        ax_points.set_aspect('equal')
        
        # Remove axis labels and title for clean individual plots
        ax_points.set_xlabel('')
        ax_points.set_ylabel('')
        ax_points.set_title('')
        
        # Professional tick styling
        ax_points.tick_params(axis='both', which='major', labelsize=32)
        
        # Enhanced grid and background
        ax_points.grid(True, alpha=0.6, linestyle='-', linewidth=1.2, color='#666666')
        ax_points.set_facecolor('white')
        
        # Enhanced spine styling
        for spine in ax_points.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('#404040')
        
        plt.tight_layout()
        plt.savefig(f'{fig_dir}/{name.lower()}_points_n{n}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Points plot saved: {name}")
    
    print(f"\nAll tests completed! Plots saved in {fig_dir}")

if __name__ == "__main__":
    # Easy way to control RPQR: set these flags
    test_symmetric_methods(n=400, max_rank=30, include_rpqr=False, plot_rpqr=False)
