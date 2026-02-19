"""
FMM-based matvecs for Cauchy-like matrices using pyfmmlib2d.
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import numpy as np


def _default_pyfmmlib2d_path() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "vendor", "pyfmmlib2d")


def _import_pyfmmlib2d(vendor_path: Optional[str] = None):
    if vendor_path:
        sys.path.insert(0, vendor_path)
    try:
        import pyfmmlib2d  # type: ignore
    except ImportError as exc:
        msg = (
            "pyfmmlib2d is required for FMM matvecs. "
            "Install it or set PYFMMLIB2D_PATH to the vendor directory."
        )
        raise ImportError(msg) from exc
    return pyfmmlib2d


def _points_from_complex(z: np.ndarray) -> np.ndarray:
    pts = np.vstack([np.real(z), np.imag(z)])
    return np.asfortranarray(pts, dtype=np.float64)


class CauchyFMMOperator:
    """
    Cauchy-like linear operator using pyfmmlib2d ZFMM.

    Matrix entries:
        C_ij = sum_r g[i, r] * b[j, r] / (c[i] - q[j])
    """

    def __init__(
        self,
        c: np.ndarray,
        q: np.ndarray,
        g: np.ndarray,
        b: np.ndarray,
        precision: int = 4,
        vendor_path: Optional[str] = None,
    ) -> None:
        pyfmmlib2d = _import_pyfmmlib2d(
            vendor_path or os.environ.get("PYFMMLIB2D_PATH") or _default_pyfmmlib2d_path()
        )
        self._zfmm = pyfmmlib2d.ZFMM
        self.precision = precision

        self.c = np.asarray(c, dtype=np.complex128)
        self.q = np.asarray(q, dtype=np.complex128)
        g = np.asarray(g, dtype=np.complex128)
        b = np.asarray(b, dtype=np.complex128)
        if g.ndim == 1:
            g = g[:, None]
        if b.ndim == 1:
            b = b[:, None]
        if g.shape[1] != b.shape[1]:
            raise ValueError("g and b must have the same generator rank")
        if g.shape[0] != self.c.shape[0]:
            raise ValueError("g must have the same length as c")
        if b.shape[0] != self.q.shape[0]:
            raise ValueError("b must have the same length as q")

        self.g = g
        self.b = b
        self.shape = (self.c.shape[0], self.q.shape[0])
        self._c_pts = _points_from_complex(self.c)
        self._q_pts = _points_from_complex(self.q)
        self._c_pts_conj = _points_from_complex(np.conj(self.c))
        self._q_pts_conj = _points_from_complex(np.conj(self.q))

    def _zfmm_potential(self, source: np.ndarray, target: np.ndarray, dipstr: np.ndarray) -> np.ndarray:
        out = self._zfmm(
            source=source,
            target=target,
            dipstr=np.asfortranarray(dipstr),
            compute_target_potential=True,
            precision=self.precision,
        )
        return out["target"]["u"]

    def matvec(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the Cauchy-like operator.

        Accepts (m,) or (m, k) inputs and returns (n,) or (n, k) outputs.
        """
        x = np.asarray(x, dtype=np.complex128)
        if x.ndim == 1:
            return self._matvec_vector(x)
        if x.ndim == 2:
            if x.shape[0] != self.shape[1]:
                raise ValueError("matvec expects shape (m,) or (m, k)")
            return np.column_stack([self._matvec_vector(x[:, i]) for i in range(x.shape[1])])
        raise ValueError("matvec expects a 1D or 2D array")

    def rmatvec(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the adjoint of the Cauchy-like operator.

        Accepts (n,) or (n, k) inputs and returns (m,) or (m, k) outputs.
        """
        x = np.asarray(x, dtype=np.complex128)
        if x.ndim == 1:
            return self._rmatvec_vector(x)
        if x.ndim == 2:
            if x.shape[0] != self.shape[0]:
                raise ValueError("rmatvec expects shape (n,) or (n, k)")
            return np.column_stack([self._rmatvec_vector(x[:, i]) for i in range(x.shape[1])])
        raise ValueError("rmatvec expects a 1D or 2D array")

    def matmat(self, x: np.ndarray) -> np.ndarray:
        return self.matvec(x)

    def rmatmat(self, x: np.ndarray) -> np.ndarray:
        return self.rmatvec(x)

    def _matvec_vector(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0] != self.shape[1]:
            raise ValueError("matvec expects shape (m,) or (m, k)")
        y = np.zeros(self.shape[0], dtype=np.complex128)
        for r in range(self.g.shape[1]):
            charges = self.b[:, r] * x
            pot = self._zfmm_potential(self._q_pts, self._c_pts, charges)
            y += self.g[:, r] * pot
        return y

    def _rmatvec_vector(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0] != self.shape[0]:
            raise ValueError("rmatvec expects shape (n,) or (n, k)")
        y = np.zeros(self.shape[1], dtype=np.complex128)
        for r in range(self.g.shape[1]):
            charges = np.conj(self.g[:, r]) * x
            pot = self._zfmm_potential(self._c_pts_conj, self._q_pts_conj, charges)
            y += -np.conj(self.b[:, r]) * pot
        return y


def randomized_svd_operator(
    matvec,
    rmatvec,
    shape: Tuple[int, int],
    rank: int,
    oversample: int = 10,
    n_iter: int = 1,
    seed: int = 0,
    
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Randomized SVD for a general matvec/rmatvec operator.
    """
    if n_iter > 0:
        print(f"Running {n_iter} power iterations for randomized SVD")

    n, m = shape
    rank = min(rank, n, m)
    l = min(rank + oversample, n, m)
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((m, l)) + 1j * rng.standard_normal((m, l))

    def apply_matvec(X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            return matvec(X)
        return np.column_stack([matvec(X[:, i]) for i in range(X.shape[1])])

    def apply_rmatvec(X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            return rmatvec(X)
        return np.column_stack([rmatvec(X[:, i]) for i in range(X.shape[1])])

    Y = apply_matvec(omega)
    for _ in range(n_iter):
        Y, _ = np.linalg.qr(Y, mode="reduced")
        Y = apply_matvec(apply_rmatvec(Y))
    Q, _ = np.linalg.qr(Y, mode="reduced")
    
    B = apply_rmatvec(Q).conj().T
    U_hat, s, Vh = np.linalg.svd(B, full_matrices=False)
    U = Q @ U_hat
    return U[:, :rank], s[:rank], Vh[:rank, :]
