"""
Theoretical bounds comparison for RPCholesky.

Implements bounds from:
- Lemma 5.5 (Error Doubling): E[tr(A - Â^(k))] ≤ 2^k * tr(A - A_k)
- Theorem 2.3 (Main Theorem): E[tr(A - Â^(k))] ≤ (1+ε) * tr(A - A_r)
  when k ≥ r/ε + r*log(1/(ε*η)), where η = tr(A - A_r)/tr(A)

Reference: "Randomly Pivoted Cholesky" by Chen et al. (2023)
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def create_symmetric_geometric_decay_matrix(n, rho):
    """
    Create a symmetric PSD matrix with geometrically decaying eigenvalues: λ_i = ρ^(i-1).
    
    Note: eigenvalues are sorted in DESCENDING order: λ_1 = 1, λ_2 = ρ, λ_3 = ρ^2, ...
    In array indexing: eigenvalues[0] = 1, eigenvalues[1] = ρ, eigenvalues[i] = ρ^i
    """
    U, _ = np.linalg.qr(np.random.randn(n, n))
    # eigenvalues[i] = ρ^i, so eigenvalues = [1, ρ, ρ^2, ..., ρ^(n-1)]
    eigenvalues = np.array([rho**i for i in range(n)])
    Lambda = np.diag(eigenvalues)
    A = U @ Lambda @ U.T
    return A


def compute_doubling_lemma_bound(eigenvalues, k):
    """
    Compute the doubling lemma bound for RPCholesky at rank k.
    
    Lemma 5.5: E[tr(A - Â^(k))] ≤ 2^k * tr(A - A_k)
    
    where tr(A - A_k) = sum_{i=k+1}^n λ_i = sum(eigenvalues[k:]) in 0-indexed arrays.
    
    Parameters:
        eigenvalues: array of eigenvalues sorted descending [λ_1, λ_2, ..., λ_n]
        k: rank (number of steps)
    
    Returns:
        The bound: 2^k * tr(A - A_k)
    """
    if k >= len(eigenvalues):
        return 0.0
    tail_sum = np.sum(eigenvalues[k:])  # tr(A - A_k)
    return (2 ** k) * tail_sum


def compute_main_theorem_bound(eigenvalues, k, epsilon):
    """
    Compute the main theorem bound for RPCholesky at step k with parameter epsilon.
    
    Theorem 2.3: E[tr(A - Â^(k))] ≤ (1+ε) * tr(A - A_r)
    when k ≥ r/ε + r*log(1/(ε*η)), where η = tr(A - A_r)/tr(A)
    
    For a given k and ε, we find the LARGEST r such that the condition holds,
    since larger r gives smaller tr(A - A_r).
    
    Parameters:
        eigenvalues: array of eigenvalues sorted descending
        k: number of steps taken
        epsilon: oversampling parameter
    
    Returns:
        (bound, r): the bound value and the optimal rank r used
    """
    n = len(eigenvalues)
    total_trace = np.sum(eigenvalues)
    
    if total_trace < 1e-14:
        return 0.0, 0
    
    if k <= 0:
        # No steps taken, no bound from this theorem
        return float('inf'), 0
    
    # For each r, compute the required k and check if our k satisfies it
    best_r = 0
    
    for r in range(1, n):  # Search all possible r from 1 to n-1
        # tr(A - A_r) = sum of tail eigenvalues
        tail_trace = np.sum(eigenvalues[r:])
        
        eta = tail_trace / total_trace
        
        # Required k from theorem: r/ε + r*log+(1/(ε*η))
        # If ε*η >= 1, the log+ term is zero.
        eps_eta = epsilon * eta
        if eps_eta <= 0:
            continue
        log_plus_term = max(np.log(1.0 / eps_eta), 0.0)
        required_k = r / epsilon + r * log_plus_term
        
        if k >= required_k:
            best_r = r
    
    # Compute bound using best_r
    if best_r == 0:
        return float('inf'), 0
    
    tail_trace = np.sum(eigenvalues[best_r:])
    bound = (1 + epsilon) * tail_trace
    
    return bound, best_r


def compute_main_theorem_bound_optimized(eigenvalues, k):
    """
    Find the optimal epsilon that gives the tightest main theorem bound.
    
    Searches over a range of epsilon values and returns the best bound.
    
    Parameters:
        eigenvalues: array of eigenvalues sorted descending
        k: number of steps taken
    
    Returns:
        (bound, optimal_epsilon, optimal_r): best bound and the parameters used
    """
    # Search over log-spaced epsilon values
    epsilon_values = np.logspace(-3, 3, 200)
    
    best_bound = float('inf')
    best_epsilon = 1.0
    best_r = 0
    
    for eps in epsilon_values:
        bound, r = compute_main_theorem_bound(eigenvalues, k, eps)
        if bound < best_bound:
            best_bound = bound
            best_epsilon = eps
            best_r = r
    
    return best_bound, best_epsilon, best_r


def compute_optimal_rank_k_error(eigenvalues, k):
    """
    Compute the optimal rank-k approximation error in trace.
    
    tr(A - A_k) = sum_{i=k+1}^n λ_i = sum(eigenvalues[k:])
    
    Parameters:
        eigenvalues: array of eigenvalues sorted descending
        k: rank of approximation
    
    Returns:
        tr(A - A_k)
    """
    if k >= len(eigenvalues):
        return 0.0
    return np.sum(eigenvalues[k:])


def run_rpchol_single_trial(A, max_rank):
    """
    Run one RPCholesky trajectory up to max_rank and return tr(residual) after each step.
    
    RPCholesky samples pivot i with probability proportional to A^(k-1)_{ii}.
    
    Parameters:
        A: PSD matrix
        max_rank: maximum number of steps
    
    Returns:
        List of trace(residual) values after steps 1, 2, ..., max_rank
    """
    n = A.shape[0]
    residual = A.copy()
    errors = []

    for step in range(max_rank):
        # Get diagonal elements (should be non-negative for PSD)
        diag_elements = np.diag(residual)
        diag_elements = np.clip(diag_elements, a_min=0, a_max=None)

        total = np.sum(diag_elements)
        if total <= 1e-30:
            # Fill remaining with current trace
            remaining = max_rank - step
            errors.extend([np.trace(residual)] * remaining)
            break

        # Sample pivot proportional to diagonal
        probs = diag_elements / total
        pivot_idx = np.random.choice(n, p=probs)
        pivot_val = residual[pivot_idx, pivot_idx]
        
        if abs(pivot_val) <= 1e-14:
            remaining = max_rank - step
            errors.extend([np.trace(residual)] * remaining)
            break

        # Schur complement update: A^(k) = A^(k-1) - col * row / pivot
        col = residual[:, pivot_idx]
        residual = residual - np.outer(col, col) / pivot_val
        
        # Store trace of residual after this step
        errors.append(np.trace(residual))

    return errors


def create_legend_figure(plot_dir, colors):
    """Create a separate legend figure for use in publications."""
    fig, ax = plt.subplots(figsize=(8, 2))
    
    # Create dummy plots for legend
    ax.plot([], [], 'o-', color=colors['observed'], markersize=8, 
            label='RPCholesky (observed)')
    ax.plot([], [], 's--', color=colors['main_theorem'], markersize=6,
            label='Main Theorem (opt. ε)')
    ax.plot([], [], '^-', color=colors['doubling'], markersize=6,
            label='Doubling: $2^k \\cdot \\mathrm{tr}(A - A_k)$')
    ax.plot([], [], 'k-', linewidth=3,
            label='Optimal: $\\mathrm{tr}(A - A_k)$')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    ax.legend(loc='center', ncol=2, fontsize=14, framealpha=0.95)
    
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/legend.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_presentation_plots(rho_values, n=200, max_rank=50, n_runs=30):
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
        'main_theorem': '#1565C0',  # Dark blue  
        'doubling': '#C62828',      # Dark red
        'optimal': '#212121',       # Near black
    }
    
    for rho in rho_values:
        print(f"\nCreating publication plot for ρ = {rho}")
        
        eigenvalues = np.array([rho**i for i in range(n)])
        
        # Run simulations
        rpchol_runs = []
        for _ in range(n_runs):
            A = create_symmetric_geometric_decay_matrix(n, rho)
            errors = run_rpchol_single_trial(A, max_rank)
            rpchol_runs.append(errors)
        
        rpchol_runs = np.array(rpchol_runs)
        rpchol_mean = np.mean(rpchol_runs, axis=0)
        rpchol_min = np.min(rpchol_runs, axis=0)
        rpchol_max = np.max(rpchol_runs, axis=0)
        
        # Add k=0 point (initial error before any steps)
        initial_error = np.sum(eigenvalues)  # tr(A)
        rpchol_mean = np.concatenate([[initial_error], rpchol_mean])
        rpchol_min = np.concatenate([[initial_error], rpchol_min])
        rpchol_max = np.concatenate([[initial_error], rpchol_max])
        
        ranks = np.arange(0, max_rank + 1)  # Start from 0
        
        doubling_bounds = [compute_doubling_lemma_bound(eigenvalues, k) for k in ranks]
        main_bounds = [compute_main_theorem_bound_optimized(eigenvalues, k)[0] for k in ranks]
        optimal_errors = [compute_optimal_rank_k_error(eigenvalues, k) for k in ranks]
        
        # Publication figure (standard column width ~3.5in, or double ~7in)
        fig, ax = plt.subplots(figsize=(7, 5.5))
        
        # Plot with clean lines
        ax.semilogy(ranks, rpchol_mean, 'o-', color=colors['observed'], 
                    markersize=6, markeredgecolor='white', markeredgewidth=0.8,
                    linewidth=2, zorder=5)
        ax.fill_between(ranks, rpchol_min, rpchol_max, 
                       color=colors['observed'], alpha=0.2, zorder=1)
        
        valid_main = np.array(main_bounds) < 1e15
        if np.any(valid_main):
            ax.semilogy(ranks[valid_main], np.array(main_bounds)[valid_main], 
                       's-', color=colors['main_theorem'], markersize=5,
                       markeredgecolor='white', markeredgewidth=0.8,
                       linewidth=2, zorder=4)
        
        ax.semilogy(ranks, doubling_bounds, '--', color=colors['doubling'], 
                   linewidth=2.5, zorder=3)
        
        valid_opt = np.array(optimal_errors) > 1e-16
        if np.any(valid_opt):
            ax.semilogy(ranks[valid_opt], np.array(optimal_errors)[valid_opt], 
                       '-', color=colors['optimal'], linewidth=2.5, zorder=2)
        
        # No labels (will be added later)
        ax.set_xlim(0, max_rank)
        
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
        filename_png = f'{plot_dir}/rpchol_rho{rho}_n{n}_k{max_rank}.png'
        plt.savefig(filename_png, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Saved: {filename_png}")
        plt.close()
    
    # Separate legend
    fig_leg, ax_leg = plt.subplots(figsize=(8, 1.2))
    
    ax_leg.plot([], [], 'o-', color=colors['observed'], markersize=8, 
               linewidth=2, label='RPCholesky (observed)')
    ax_leg.plot([], [], 's-', color=colors['main_theorem'], markersize=6,
               linewidth=2, label='Main Theorem')
    ax_leg.plot([], [], '--', color=colors['doubling'], linewidth=2.5,
               label='Doubling Lemma')
    ax_leg.plot([], [], '-', color=colors['optimal'], linewidth=2.5,
               label='Optimal')
    
    ax_leg.axis('off')
    ax_leg.legend(loc='center', ncol=4, fontsize=12, frameon=False)
    
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/rpchol_legend.png', dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Legend saved: {plot_dir}/rpchol_legend.png")


def create_polynomial_decay_plots(decay_types, n=200, max_rank=50, n_runs=30):
    """
    Create publication-quality plots for polynomial decay cases.
    decay_types is a list of tuples: (name, decay_function)
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
        'main_theorem': '#1565C0',
        'doubling': '#C62828',
        'optimal': '#212121',
    }
    
    for name, eigenvalues in decay_types:
        print(f"\nCreating publication plot for {name} decay")
        
        # Run simulations
        rpchol_runs = []
        for _ in range(n_runs):
            A = create_matrix_from_eigenvalues(n, eigenvalues)
            errors = run_rpchol_single_trial(A, max_rank)
            rpchol_runs.append(errors)
        
        rpchol_runs = np.array(rpchol_runs)
        rpchol_mean = np.mean(rpchol_runs, axis=0)
        rpchol_min = np.min(rpchol_runs, axis=0)
        rpchol_max = np.max(rpchol_runs, axis=0)
        
        # Add k=0 point (initial error before any steps)
        initial_error = np.sum(eigenvalues)  # tr(A)
        rpchol_mean = np.concatenate([[initial_error], rpchol_mean])
        rpchol_min = np.concatenate([[initial_error], rpchol_min])
        rpchol_max = np.concatenate([[initial_error], rpchol_max])
        
        ranks = np.arange(0, max_rank + 1)  # Start from 0
        
        doubling_bounds = [compute_doubling_lemma_bound(eigenvalues, k) for k in ranks]
        main_bounds = [compute_main_theorem_bound_optimized(eigenvalues, k)[0] for k in ranks]
        optimal_errors = [compute_optimal_rank_k_error(eigenvalues, k) for k in ranks]
        
        fig, ax = plt.subplots(figsize=(7, 5.5))
        
        ax.semilogy(ranks, rpchol_mean, 'o-', color=colors['observed'], 
                    markersize=6, markeredgecolor='white', markeredgewidth=0.8,
                    linewidth=2, zorder=5)
        ax.fill_between(ranks, rpchol_min, rpchol_max, 
                       color=colors['observed'], alpha=0.2, zorder=1)
        
        valid_main = np.array(main_bounds) < 1e15
        if np.any(valid_main):
            ax.semilogy(ranks[valid_main], np.array(main_bounds)[valid_main], 
                       's-', color=colors['main_theorem'], markersize=5,
                       markeredgecolor='white', markeredgewidth=0.8,
                       linewidth=2, zorder=4)
        
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
        all_vals = np.concatenate([rpchol_mean, np.array(optimal_errors)])
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
        filename_png = f'{plot_dir}/rpchol_{safe_name}_n{n}_k{max_rank}.png'
        plt.savefig(filename_png, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Saved: {filename_png}")
        plt.close()


def create_matrix_from_eigenvalues(n, eigenvalues):
    """Create a symmetric PSD matrix with given eigenvalues."""
    U, _ = np.linalg.qr(np.random.randn(n, n))
    Lambda = np.diag(eigenvalues[:n])
    return U @ Lambda @ U.T


if __name__ == "__main__":
    # Parameters
    n = 200
    max_rank = 50
    n_runs = 30
    rho_values = [0.75, 0.40, 0.25]  # Geometric decay rates
    
    print("="*70)
    print("RPCholesky Theoretical Bounds Analysis")
    print("="*70)
    print(f"Matrix size: {n}×{n}")
    print(f"Max rank: {max_rank}")
    print(f"Number of trials: {n_runs}")
    print(f"Eigenvalue decay rates (ρ): {rho_values}")
    print()
    print("Bounds being compared:")
    print("  1. Doubling Lemma (Lemma 5.5): E[tr(A - Â^(k))] ≤ 2^k · tr(A - A_k)")
    print("  2. Main Theorem (Theorem 2.3): E[tr(A - Â^(k))] ≤ (1+ε) · tr(A - A_r)")
    print("     when k ≥ r/ε + r·log(1/(ε·η)), η = tr(A - A_r)/tr(A)")
    print("  3. Optimal rank-k error: tr(A - A_k)")
    print("="*70)
    
    # Run analysis and create plots
    create_presentation_plots(rho_values, n, max_rank, n_runs)
    
    # Also create polynomial decay plots
    print("\n" + "="*70)
    print("Creating polynomial decay plots...")
    print("="*70)
    
    # Polynomial decay: σ_k = 1/(k+1)^p
    poly_decays = [
        ("1_k2", np.array([1.0 / (k + 1)**2 for k in range(n)])),  # 1/k^2 decay
        ("1_k", np.array([1.0 / (k + 1) for k in range(n)])),      # 1/k decay
    ]
    create_polynomial_decay_plots(poly_decays, n, max_rank, n_runs)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("• Doubling lemma gives 2^k factor (tight for some matrices)")
    print("• Main theorem with optimal ε can be tighter for oversampling (k > r)")
    print("• Geometric decay ρ < 1/2 needed for doubling lemma to guarantee convergence")
    print("• Plots show observed error is typically much better than bounds")
    print("="*70)
