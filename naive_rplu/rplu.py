import numpy as np


def lu(A, n_iter, pivot='random', eps=1e-32):
    """
    Perform LU decomposition with specified pivoting strategy
    
    Parameters:
    -----------
    A : ndarray
        Input matrix
    n_iter : int
        Number of iterations (rank of approximation)
    pivot : str
        Pivoting strategy: 'random' or 'greedy'
    eps : float
        Small threshold for early stopping
    
    Returns:
    --------
    L : ndarray
        Lower triangular matrix
    U : ndarray
        Upper triangular matrix  
    pivots : ndarray
        Pivot indices
    """
    n, m = A.shape
    L = np.zeros((n, n_iter), dtype=A.dtype)
    U = np.zeros((n_iter, m), dtype=A.dtype)
    
    # Keep track of remaining indices
    remaining_rows = np.arange(n)
    remaining_cols = np.arange(m)
    pivots = np.zeros((n_iter, 2), dtype=int)
    
    # Make a copy to avoid modifying original
    A_work = A.copy()
    
    for k in range(n_iter):
        if A_work.size == 0:
            print(f'Matrix exhausted at iteration {k}')
            break
            
        # Choose pivot based on strategy
        if pivot == 'random':
            # Random pivoting: sample with probability proportional to |A_ij|^2
            C = np.abs(A_work)**2
            total_norm = np.sum(C)
            if total_norm < eps:
                print(f'Matrix norm too small at iteration {k}')
                break
            C = C / total_norm
            flat_idx = np.random.choice(C.size, 1, p=C.reshape(-1))[0]
            i_local, j_local = np.unravel_index(flat_idx, A_work.shape)
        elif pivot == 'greedy':
            # Greedy pivoting: choose largest element
            i_local, j_local = np.unravel_index(np.argmax(np.abs(A_work)), A_work.shape)
            if np.abs(A_work[i_local, j_local]) < eps:
                print(f'Largest element too small at iteration {k}')
                break
        elif pivot == 'norm_greedy':
            # Norm-greedy pivoting: choose column with highest norm, then largest element in that column
            col_norms = np.linalg.norm(A_work, axis=0)
            if np.max(col_norms) < eps:
                print(f'Column norms too small at iteration {k}')
                break
            j_local = np.argmax(col_norms)
            i_local = np.argmax(np.abs(A_work[:, j_local]))
            if np.abs(A_work[i_local, j_local]) < eps:
                print(f'Largest element in chosen column too small at iteration {k}')
                break
        elif pivot == 'row_norm_greedy':
            # Row-norm-greedy pivoting: choose row with highest norm, then largest element in that row
            row_norms = np.linalg.norm(A_work, axis=1)
            if np.max(row_norms) < eps:
                print(f'Row norms too small at iteration {k}')
                break
            i_local = np.argmax(row_norms)
            j_local = np.argmax(np.abs(A_work[i_local, :]))
            if np.abs(A_work[i_local, j_local]) < eps:
                print(f'Largest element in chosen row too small at iteration {k}')
                break
        else:
            raise ValueError(f"Unknown pivot strategy: {pivot}")
        
        # Get global indices
        i_global = remaining_rows[i_local]
        j_global = remaining_cols[j_local]
        
        # Store pivot
        pivots[k, 0] = i_global
        pivots[k, 1] = j_global
        
        # Extract pivot element
        pivot_val = A_work[i_local, j_local]
        
        # Extract column and row
        l_col = A_work[:, j_local] / pivot_val
        u_row = A_work[i_local, :]
        
        # Store in L and U matrices
        L[remaining_rows, k] = l_col
        U[k, remaining_cols] = u_row
        
        # Update remaining matrix (Schur complement)
        A_work = A_work - np.outer(l_col, u_row)
        
        # Remove the pivot row and column
        mask_rows = np.arange(A_work.shape[0]) != i_local
        mask_cols = np.arange(A_work.shape[1]) != j_local
        
        A_work = A_work[mask_rows][:, mask_cols]
        remaining_rows = remaining_rows[mask_rows]
        remaining_cols = remaining_cols[mask_cols]
    
    return L, U, pivots
