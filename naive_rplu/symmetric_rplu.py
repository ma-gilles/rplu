import numpy as np
from numpy.linalg import qr, eigh, pinv


def _sqrt_psd(A):
    """
    Compute the square root of a positive semi-definite matrix A.
    Returns A^(1/2) such that A^(1/2) @ A^(1/2) = A.
    """
    eigenvals, eigenvecs = eigh(A)
    # Ensure eigenvalues are non-negative (clip to 0)
    eigenvals = np.maximum(eigenvals, 0)
    return eigenvecs @ np.diag(np.sqrt(eigenvals)) @ eigenvecs.T



def rank12_symmetric(A, n_iter, *, pivot='random', diag_ok=True,
                     eps=1e-32, rng=None):
    """
    Symmetric deflation that also returns a low-rank factor R (A ≈ R Rᵀ).

    Parameters
    ----------
    A : (n,n) ndarray, symmetric
    n_iter : int
        Max number of pivot events.
    pivot : {'random', 'greedy'}
    diag_ok : bool
        If False, forbid diagonal pivots (rank-2 only).
    eps : float
        Tiny threshold for early stop.
    rng : np.random.Generator or None

    Returns
    -------
    residual : ndarray  (n,n)
        Remaining matrix after ≤ n_iter steps (in full space).
    R : ndarray  (n, r)
        Low-rank factor accumulated so far (r ≤ 2 n_iter).
    pivots : ndarray  (k, 2)
        Global pivot indices (k ≤ n_iter).
    """
    if rng is None:
        rng = np.random.default_rng()

    n = A.shape[0]
    A_work = A.copy().astype(float)
    remaining = np.arange(n)          # global labels
    pivots = []
    R_cols = []                       # store columns as we generate them

    for _ in range(n_iter):
        m = A_work.shape[0]
        if m == 0:
            break

        # -------- 1. choose entry (i,j) ----------
        if pivot == 'random':
            W = A_work**2
            if not diag_ok:
                np.fill_diagonal(W, 0.0)
            tot = W.sum()
            if tot < eps:
                break
            idx = rng.choice(W.size, p=W.ravel() / tot)
            i_loc, j_loc = divmod(idx, m)
        elif pivot == 'greedy':
            W = np.abs(A_work)
            if not diag_ok:
                np.fill_diagonal(W, 0.0)
            i_loc, j_loc = divmod(W.argmax(), m)
            if W[i_loc, j_loc] < eps:
                break
        else:
            raise ValueError("pivot must be 'random' or 'greedy'")

        i_glob, j_glob = remaining[i_loc], remaining[j_loc]
        pivots.append((i_glob, j_glob))

        # -------- 2. build Δ = B Bᵀ & collect B into R_cols ----------
        if i_loc == j_loc:           # rank-1  (diagonal)
            u = A_work[:, [i_loc]]
            aii = A_work[i_loc, i_loc]
            b = u / np.sqrt(aii)     # column vector
            
            # Create full-size column for global matrix
            b_full = np.zeros((n, 1))
            b_full[remaining, 0] = b.flatten()
            R_cols.append(b_full)

            Delta = b @ b.T
            rows_drop = [i_loc]

        else:                        # rank-2  (off-diag)
            u = A_work[:, [i_loc]]
            v = A_work[:, [j_loc]]
            C = np.array([[A_work[i_loc, i_loc], A_work[i_loc, j_loc]],
                          [A_work[i_loc, j_loc], A_work[j_loc, j_loc]]],
                         float)
            C_pinv = pinv(C)         # always safe
            C_half = _sqrt_psd(C_pinv)
            
            # two rank-1 columns
            B = np.hstack([u, v]) @ C_half    # m×2
            
            # Create full-size columns for global matrix
            b1_full = np.zeros((n, 1))
            b2_full = np.zeros((n, 1))
            b1_full[remaining, 0] = B[:, 0]
            b2_full[remaining, 0] = B[:, 1]
            R_cols.extend([b1_full, b2_full])

            Delta = B @ B.T
            rows_drop = [i_loc, j_loc]

        # update residual
        A_work -= Delta

        # deflate rows/cols
        mask = np.ones(m, bool)
        mask[rows_drop] = False
        A_work = A_work[mask][:, mask]
        remaining = remaining[mask]

    # stack stored columns into a single matrix
    R = np.hstack(R_cols) if R_cols else np.empty((n, 0))

    # Reconstruct residual matrix in full space
    residual_full = np.zeros((n, n))
    if len(remaining) > 0:
        residual_full[np.ix_(remaining, remaining)] = A_work

    return  R, np.array(pivots, int)


# -----------------------------------------------------------
# demo
# -----------------------------------------------------------
if __name__ == "__main__":
    n = 40
    Q, _ = qr(np.random.randn(n, n))
    lam = 0.6 ** np.arange(n) + 1.0
    A = Q @ np.diag(lam) @ Q.T          # SPD test matrix

    res, R, piv = rank12_symmetric(A, n_iter=5, pivot='random', diag_ok=True)
    print("pivots:", piv)
    print("rank used:", R.shape[1])
    
    # Check the complete reconstruction: A ≈ res + R Rᵀ
    reconstruction = res + R @ R.T
    print("‖A - (res + R Rᵀ)‖₂  =", np.linalg.norm(A - reconstruction))
    print("‖A‖₂ =", np.linalg.norm(A))
    print("relative error =", np.linalg.norm(A - reconstruction) / np.linalg.norm(A))
