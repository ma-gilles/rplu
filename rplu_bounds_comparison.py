"""
Theoretical bounds comparison for RPLU (Randomly Pivoted LU).

Implements bounds from:
- Theorem 3.3 (Doubling): E[||A - Â^(k)||_F^2] ≤ 4^k * ||A - A_k||_F^2
  Taking sqrt: sqrt(E[...]) ≤ 2^k * ||A - A_k||_F

Note: Unlike RPCholesky, there is no oversampled bound for RPLU.
Plots show Frobenius norm (not squared) for consistency with RPCholesky trace plots.

Reference: "Randomly Pivoted LU" paper, Theorem 3.3
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def create_matrix_with_geometric_singular_values(n, m, rho):
    """
    Create a general (non-symmetric) matrix with geometrically decaying singular values.
    
    σ_i = ρ^(i-1) for i = 1, ..., min(n,m)
    In array indexing: singular_values[i] = ρ^i for i = 0, ..., min(n,m)-1
    
    Parameters:
        n: number of rows
        m: number of columns
        rho: decay rate (0 < rho < 1)
    
    Returns:
        A: n x m matrix with prescribed singular values
    """
    k = min(n, m)
    # Random orthogonal matrices
    U, _ = np.linalg.qr(np.random.randn(n, n))
    V, _ = np.linalg.qr(np.random.randn(m, m))
    
    # Singular values: [1, ρ, ρ^2, ..., ρ^(k-1)]
    singular_values = np.array([rho**i for i in range(k)])
    
    # Construct A = U @ Σ @ V^T
    Sigma = np.zeros((n, m))
    np.fill_diagonal(Sigma, singular_values)
    A = U @ Sigma @ V.T
    
    return A


def compute_doubling_bound(singular_values, k):
    """
    Compute the doubling bound for RPLU at rank k (non-squared).
    
    Theorem 3.3: E[||A - Â^(k)||_F^2] ≤ 4^k * ||A - A_k||_F^2
    Taking sqrt: bound ≤ 2^k * ||A - A_k||_F
    
    where ||A - A_k||_F = sqrt(sum(singular_values[k:]^2))
    
    Parameters:
        singular_values: array of singular values sorted descending [σ_1, σ_2, ..., σ_n]
        k: rank (number of steps)
    
    Returns:
        The bound: 2^k * ||A - A_k||_F
    """
    if k >= len(singular_values):
        return 0.0
    tail_norm = np.sqrt(np.sum(singular_values[k:] ** 2))  # ||A - A_k||_F
    return (2 ** k) * tail_norm


def compute_optimal_rank_k_error(singular_values, k):
    """
    Compute the optimal rank-k approximation error in Frobenius norm.
    
    ||A - A_k||_F = sqrt(sum(singular_values[k:]^2))
    
    Parameters:
        singular_values: array of singular values sorted descending
        k: rank of approximation
    
    Returns:
        ||A - A_k||_F
    """
    if k >= len(singular_values):
        return 0.0
    return np.sqrt(np.sum(singular_values[k:] ** 2))


def run_rplu_single_trial(A, max_rank):
    """
    Run one RPLU trajectory up to max_rank and return ||residual||_F after each step.
    
    RPLU samples pivot (i,j) with probability proportional to |A^(k-1)_{i,j}|^2.
    
    Parameters:
        A: input matrix (n x m)
        max_rank: maximum number of steps
    
    Returns:
        List of ||residual||_F values after steps 1, 2, ..., max_rank
    """
    n, m = A.shape
    residual = A.copy().astype(np.float64)
    errors = []

    for step in range(max_rank):
        # Compute squared entries
        entries_sq = residual ** 2
        total = np.sum(entries_sq)
        
        if total <= 1e-30:
            # Fill remaining with current error
            remaining = max_rank - step
            errors.extend([np.linalg.norm(residual, 'fro')] * remaining)
            break

        # Sample pivot (i, j) proportional to |A_{i,j}|^2
        probs = entries_sq.flatten() / total
        idx = np.random.choice(n * m, p=probs)
        i, j = idx // m, idx % m
        
        pivot_val = residual[i, j]
        if abs(pivot_val) <= 1e-14:
            remaining = max_rank - step
            errors.extend([np.linalg.norm(residual, 'fro')] * remaining)
            break

        # Schur complement update: A^(k) = A^(k-1) - col_j * row_i / pivot
        col = residual[:, j].copy()
        row = residual[i, :].copy()
        residual = residual - np.outer(col, row) / pivot_val
        
        # Store Frobenius norm of residual
        errors.append(np.linalg.norm(residual, 'fro'))

    return errors


def create_legend_figure(plot_dir, colors):
    """Create a separate legend figure for use in publications."""
    fig, ax = plt.subplots(figsize=(8, 2))
    
    # Create dummy plots for legend
    ax.plot([], [], 'o-', color=colors['observed'], markersize=8, 
            label='RPLU (observed)')
    ax.plot([], [], '^-', color=colors['doubling'], markersize=6,
            label='Doubling: $2^k \\cdot \\|A - A_k\\|_F$')
    ax.plot([], [], 'k-', linewidth=3,
            label='Optimal: $\\|A - A_k\\|_F$')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    ax.legend(loc='center', ncol=3, fontsize=14, framealpha=0.95)
    
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/rplu_bounds_legend.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_presentation_plots(rho_values, n=200, m=200, max_rank=50, n_runs=30):
    """
    Create publication-quality plots for top journals.
    No labels - those will be added later.
    """
    plot_dir = 'plots/theoretical_bounds'
    os.makedirs(plot_dir, exist_ok=True)
    
    # Publication style - clean, high quality
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times', 'Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'cm',
        'font.size': 18,
        'axes.titlesize': 20,
        'axes.labelsize': 18,
        'xtick.labelsize': 22,
        'ytick.labelsize': 22,
        'xtick.major.size': 14,
        'ytick.major.size': 14,
        'xtick.minor.size': 7,
        'ytick.minor.size': 7,
        'xtick.major.width': 2.0,
        'ytick.major.width': 2.0,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'lines.linewidth': 2.5,
        'lines.markersize': 8,
        'figure.dpi': 300,
        'savefig.dpi': 600,
        'savefig.format': 'pdf',
        'axes.linewidth': 1.5,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.8,
        'grid.linestyle': '-',
    })
    
    # Professional color palette
    colors = {
        'observed': '#2E7D32',      # Dark green
        'doubling': '#C62828',      # Dark red
        'optimal': '#212121',       # Near black
    }
    
    for rho in rho_values:
        print(f"\nCreating publication plot for RPLU with ρ = {rho}")
        
        k = min(n, m)
        singular_values = np.array([rho**i for i in range(k)])
        
        # Run simulations
        rplu_runs = []
        for _ in range(n_runs):
            A = create_matrix_with_geometric_singular_values(n, m, rho)
            errors = run_rplu_single_trial(A, max_rank)
            rplu_runs.append(errors)
        
        rplu_runs = np.array(rplu_runs)
        rplu_mean = np.mean(rplu_runs, axis=0)
        rplu_min = np.min(rplu_runs, axis=0)
        rplu_max = np.max(rplu_runs, axis=0)
        
        # Add k=0 point (initial error before any steps)
        initial_error = np.sqrt(np.sum(singular_values ** 2))  # ||A||_F
        rplu_mean = np.concatenate([[initial_error], rplu_mean])
        rplu_min = np.concatenate([[initial_error], rplu_min])
        rplu_max = np.concatenate([[initial_error], rplu_max])
        
        ranks = np.arange(0, max_rank + 1)  # Start from 0
        
        doubling_bounds = [compute_doubling_bound(singular_values, k) for k in ranks]
        optimal_errors = [compute_optimal_rank_k_error(singular_values, k) for k in ranks]
        
        # Publication figure (standard column width ~3.5in, or double ~7in)
        fig, ax = plt.subplots(figsize=(7, 5.5))
        
        # Plot with clean lines
        ax.semilogy(ranks, rplu_mean, 'o-', color=colors['observed'], 
                    markersize=6, markeredgecolor='white', markeredgewidth=0.8,
                    linewidth=2, zorder=5)
        ax.fill_between(ranks, rplu_min, rplu_max, 
                       color=colors['observed'], alpha=0.2, zorder=1)
        
        ax.semilogy(ranks, doubling_bounds, '--', color=colors['doubling'], 
                   linewidth=2.5, zorder=3)
        
        valid_opt = np.array(optimal_errors) > 1e-16
        if np.any(valid_opt):
            ax.semilogy(ranks[valid_opt], np.array(optimal_errors)[valid_opt], 
                       '-', color=colors['optimal'], linewidth=2.5, zorder=2)
        
        # No labels (will be added later)
        ax.set_xlim(0, max_rank)
        
        # Y-limits (cap at 10^2)
        # Y-limits: fixed range 1e-10 to 1e2 for geometric decay
        ax.set_ylim(1e-10, 1e2)
        
        # Clean spines
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color('#333333')
        
        ax.tick_params(width=1.2, length=6, direction='in', top=True, right=True)
        ax.tick_params(which='minor', width=0.8, length=3, direction='in', top=True, right=True)
        
        plt.tight_layout(pad=0.5)
        
        # Save as PNG
        filename_png = f'{plot_dir}/rplu_rho{rho}_n{n}_k{max_rank}.png'
        plt.savefig(filename_png, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Saved: {filename_png}")
        plt.close()
    
    # Separate legend
    fig_leg, ax_leg = plt.subplots(figsize=(8, 1.2))
    
    ax_leg.plot([], [], 'o-', color=colors['observed'], markersize=8, 
               linewidth=2, label='RPLU (observed)')
    ax_leg.plot([], [], '--', color=colors['doubling'], linewidth=2.5,
               label='Doubling Bound')
    ax_leg.plot([], [], '-', color=colors['optimal'], linewidth=2.5,
               label='Optimal')
    
    ax_leg.axis('off')
    ax_leg.legend(loc='center', ncol=3, fontsize=12, frameon=False)
    
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/rplu_legend.png', dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Legend saved: {plot_dir}/rplu_legend.png")


def create_polynomial_decay_plots(decay_types, n=200, m=200, max_rank=50, n_runs=30):
    """
    Create publication-quality plots for polynomial decay cases.
    decay_types is a list of tuples: (name, singular_values)
    """
    plot_dir = 'plots/theoretical_bounds'
    os.makedirs(plot_dir, exist_ok=True)
    
    # Publication style
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times', 'Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'cm',
        'font.size': 18,
        'axes.titlesize': 20,
        'axes.labelsize': 18,
        'xtick.labelsize': 22,
        'ytick.labelsize': 22,
        'xtick.major.size': 14,
        'ytick.major.size': 14,
        'xtick.minor.size': 7,
        'ytick.minor.size': 7,
        'xtick.major.width': 2.0,
        'ytick.major.width': 2.0,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'lines.linewidth': 2.5,
        'lines.markersize': 8,
        'figure.dpi': 300,
        'savefig.dpi': 600,
        'axes.linewidth': 1.5,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.8,
    })
    
    colors = {
        'observed': '#2E7D32',
        'doubling': '#C62828',
        'optimal': '#212121',
    }
    
    for name, singular_values in decay_types:
        print(f"\nCreating publication plot for RPLU with {name} decay")
        
        # Run simulations
        rplu_runs = []
        for _ in range(n_runs):
            A = create_matrix_from_singular_values(n, m, singular_values)
            errors = run_rplu_single_trial(A, max_rank)
            rplu_runs.append(errors)
        
        rplu_runs = np.array(rplu_runs)
        rplu_mean = np.mean(rplu_runs, axis=0)
        rplu_min = np.min(rplu_runs, axis=0)
        rplu_max = np.max(rplu_runs, axis=0)
        
        # Add k=0 point (initial error before any steps)
        initial_error = np.sqrt(np.sum(singular_values ** 2))  # ||A||_F
        rplu_mean = np.concatenate([[initial_error], rplu_mean])
        rplu_min = np.concatenate([[initial_error], rplu_min])
        rplu_max = np.concatenate([[initial_error], rplu_max])
        
        ranks = np.arange(0, max_rank + 1)  # Start from 0
        
        doubling_bounds = [compute_doubling_bound(singular_values, k) for k in ranks]
        optimal_errors = [compute_optimal_rank_k_error(singular_values, k) for k in ranks]
        
        fig, ax = plt.subplots(figsize=(7, 5.5))
        
        ax.semilogy(ranks, rplu_mean, 'o-', color=colors['observed'], 
                    markersize=6, markeredgecolor='white', markeredgewidth=0.8,
                    linewidth=2, zorder=5)
        ax.fill_between(ranks, rplu_min, rplu_max, 
                       color=colors['observed'], alpha=0.2, zorder=1)
        
        # Plot doubling bound up to and including first point that exceeds y-limit
        doubling_bounds = np.array(doubling_bounds)
        # Find first index where bound exceeds 1e2 (y-axis limit)
        exceed_idx = np.where(doubling_bounds > 1e2)[0]
        if len(exceed_idx) > 0:
            last_valid = exceed_idx[0] + 1  # Include the first point that exceeds
        else:
            last_valid = len(doubling_bounds)
        
        if last_valid > 0:
            ax.semilogy(ranks[:last_valid], doubling_bounds[:last_valid], 
                       '--', color=colors['doubling'], linewidth=2.5, zorder=3)
        
        valid_opt = np.array(optimal_errors) > 1e-16
        if np.any(valid_opt):
            ax.semilogy(ranks[valid_opt], np.array(optimal_errors)[valid_opt], 
                       '-', color=colors['optimal'], linewidth=2.5, zorder=2)
        
        ax.set_xlim(0, max_rank)
        
        # Y-limits
        all_vals = np.concatenate([rplu_mean, np.array(optimal_errors)])
        all_vals = all_vals[all_vals > 1e-16]
        if len(all_vals) > 0:
            ax.set_ylim(min(all_vals) / 5, 1e2)
        
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color('#333333')
        
        ax.tick_params(width=1.2, length=6, direction='in', top=True, right=True)
        ax.tick_params(which='minor', width=0.8, length=3, direction='in', top=True, right=True)
        
        plt.tight_layout(pad=0.5)
        
        safe_name = name.replace('/', '_').replace('^', '').replace(' ', '_')
        filename_png = f'{plot_dir}/rplu_{safe_name}_n{n}_k{max_rank}.png'
        plt.savefig(filename_png, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Saved: {filename_png}")
        plt.close()


def create_matrix_from_singular_values(n, m, singular_values):
    """Create a matrix with given singular values."""
    k = min(n, m, len(singular_values))
    U, _ = np.linalg.qr(np.random.randn(n, n))
    V, _ = np.linalg.qr(np.random.randn(m, m))
    Sigma = np.zeros((n, m))
    np.fill_diagonal(Sigma, singular_values[:k])
    return U @ Sigma @ V.T


if __name__ == "__main__":
    # Parameters
    n = 200
    m = 200
    max_rank = 50
    n_runs = 30
    rho_values = [0.75, 0.40, 0.25]  # Geometric decay rates
    
    print("="*70)
    print("RPLU Theoretical Bounds Analysis")
    print("="*70)
    print(f"Matrix size: {n}×{m}")
    print(f"Max rank: {max_rank}")
    print(f"Number of trials: {n_runs}")
    print(f"Singular value decay rates (ρ): {rho_values}")
    print()
    print("Bound being compared:")
    print("  Doubling (Theorem 3.3): sqrt(E[||A - Â^(k)||_F^2]) ≤ 2^k · ||A - A_k||_F")
    print()
    print("Key differences from RPCholesky:")
    print("  - Uses 2^k factor (same as RPCholesky for non-squared)")
    print("  - Uses Frobenius norm (not trace)")
    print("  - Works on general matrices (not just PSD)")
    print("  - No oversampled bound exists")
    print("="*70)
    
    # Run analysis and create plots
    create_presentation_plots(rho_values, n, m, max_rank, n_runs)
    
    # Also create polynomial decay plots
    print("\n" + "="*70)
    print("Creating polynomial decay plots...")
    print("="*70)
    
    k_max = min(n, m)
    poly_decays = [
        ("1_k2", np.array([1.0 / (k + 1)**2 for k in range(k_max)])),  # 1/k^2 decay
        ("1_k", np.array([1.0 / (k + 1) for k in range(k_max)])),      # 1/k decay
    ]
    create_polynomial_decay_plots(poly_decays, n, m, max_rank, n_runs)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("• Doubling bound gives 2^k factor for RPLU (same as RPCholesky)")
    print("• Geometric decay ρ < 1/2 needed for bound to guarantee convergence")
    print("• RPLU improves on CPLU which requires ρ < 1/4")
    print("• Observed error is typically much better than the bound")
    print("="*70)
