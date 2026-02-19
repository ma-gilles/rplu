"""
Rakau-backed certified factor-C upper bound for Cauchy-like row norms.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

_DEFAULT_LIB = Path(__file__).resolve().parents[2] / "rakau" / "librakau_gb_ub.so"
_LIB_PATH = Path(os.environ.get("RAKAU_GB_UB_LIB", _DEFAULT_LIB))
try:
    _LIB = ctypes.CDLL(str(_LIB_PATH))
except OSError as e:
    raise OSError(
        f"Failed to load Rakau upper-bound library at {_LIB_PATH}. "
        "Build `rakau/librakau_gb_ub.so` (CPU-only) or set `RAKAU_GB_UB_LIB` to its path."
    ) from e

_LIB.rakau_gb_upper_bound_2d_r2.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # cx
    ctypes.POINTER(ctypes.c_double),  # cy
    ctypes.c_size_t,                  # n
    ctypes.POINTER(ctypes.c_double),  # qx
    ctypes.POINTER(ctypes.c_double),  # qy
    ctypes.c_size_t,                  # m
    ctypes.POINTER(ctypes.c_double),  # g_re
    ctypes.POINTER(ctypes.c_double),  # g_im
    ctypes.POINTER(ctypes.c_double),  # b_re
    ctypes.POINTER(ctypes.c_double),  # b_im
    ctypes.c_size_t,                  # r
    ctypes.c_double,                  # C
    ctypes.c_size_t,                  # max_leaf
    ctypes.POINTER(ctypes.c_double),  # out
]
_LIB.rakau_gb_upper_bound_2d_r2.restype = ctypes.c_int

_LIB.rakau_gb_ub_plan_create_2d_r2.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # cx
    ctypes.POINTER(ctypes.c_double),  # cy
    ctypes.c_size_t,                  # n
    ctypes.POINTER(ctypes.c_double),  # qx
    ctypes.POINTER(ctypes.c_double),  # qy
    ctypes.c_size_t,                  # m
    ctypes.c_double,                  # C
    ctypes.c_size_t,                  # max_leaf
    ctypes.POINTER(ctypes.c_int),     # err
]
_LIB.rakau_gb_ub_plan_create_2d_r2.restype = ctypes.c_void_p

_LIB.rakau_gb_ub_plan_compute_2d_r2.argtypes = [
    ctypes.c_void_p,                  # plan
    ctypes.POINTER(ctypes.c_double),  # g_re
    ctypes.POINTER(ctypes.c_double),  # g_im
    ctypes.POINTER(ctypes.c_double),  # b_re
    ctypes.POINTER(ctypes.c_double),  # b_im
    ctypes.POINTER(ctypes.c_double),  # out
]
_LIB.rakau_gb_ub_plan_compute_2d_r2.restype = ctypes.c_int

_LIB.rakau_gb_ub_plan_compute_weights_2d_r2.argtypes = [
    ctypes.c_void_p,                  # plan
    ctypes.POINTER(ctypes.c_double),  # weights
    ctypes.POINTER(ctypes.c_double),  # out
]
_LIB.rakau_gb_ub_plan_compute_weights_2d_r2.restype = ctypes.c_int

_LIB.rakau_gb_ub_plan_destroy.argtypes = [ctypes.c_void_p]
_LIB.rakau_gb_ub_plan_destroy.restype = None


class GbUpperBoundPlan:
    """
    Precomputed geometry plan for repeated factor-C upper bounds (r=2 only).

    This precomputes the interaction lists for fixed c/q and C. Use compute()
    with different g/b weights to get very fast online bounds.
    """

    def __init__(self, c: np.ndarray, q: np.ndarray, C: float, *, max_leaf: int = 64) -> None:
        if C < 1.0:
            raise ValueError("C must be >= 1.0")

        c = np.asarray(c, dtype=np.complex128).ravel()
        q = np.asarray(q, dtype=np.complex128).ravel()

        self._n = c.size
        self._m = q.size
        self._closed = False

        cx = np.ascontiguousarray(c.real, dtype=np.float64)
        cy = np.ascontiguousarray(c.imag, dtype=np.float64)
        qx = np.ascontiguousarray(q.real, dtype=np.float64)
        qy = np.ascontiguousarray(q.imag, dtype=np.float64)

        err = ctypes.c_int(0)
        ptr = _LIB.rakau_gb_ub_plan_create_2d_r2(
            cx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            cy.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_size_t(self._n),
            qx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            qy.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_size_t(self._m),
            ctypes.c_double(C),
            ctypes.c_size_t(max_leaf),
            ctypes.byref(err),
        )
        if not ptr:
            raise RuntimeError(f"rakau_gb_ub_plan_create_2d_r2 failed with code {err.value}")
        self._ptr = ptr

    def close(self) -> None:
        if not self._closed and self._ptr:
            _LIB.rakau_gb_ub_plan_destroy(self._ptr)
            self._ptr = None
            self._closed = True

    def __enter__(self) -> "GbUpperBoundPlan":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def compute(self, g: np.ndarray, b: np.ndarray) -> np.ndarray:
        if self._closed or not self._ptr:
            raise RuntimeError("GbUpperBoundPlan is closed")

        g = np.asarray(g, dtype=np.complex128)
        b = np.asarray(b, dtype=np.complex128)

        if g.ndim != 2 or b.ndim != 2:
            raise ValueError("g and b must be 2D arrays")
        if g.shape[1] != b.shape[1]:
            raise ValueError("g and b must share generator dimension")
        if g.shape[0] != self._n or b.shape[0] != self._m:
            raise ValueError("g/b must match c/q lengths")
        if g.shape[1] != 2:
            raise ValueError("GbUpperBoundPlan only supports r=2")

        g_re = np.ascontiguousarray(g.real, dtype=np.float64)
        g_im = np.ascontiguousarray(g.imag, dtype=np.float64)
        b_re = np.ascontiguousarray(b.real, dtype=np.float64)
        b_im = np.ascontiguousarray(b.imag, dtype=np.float64)

        out = np.empty(self._n, dtype=np.float64)
        rc = _LIB.rakau_gb_ub_plan_compute_2d_r2(
            self._ptr,
            g_re.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            g_im.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            b_re.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            b_im.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if rc != 0:
            raise RuntimeError(f"rakau_gb_ub_plan_compute_2d_r2 failed with code {rc}")
        return out

    def compute_weights(self, weights: np.ndarray) -> np.ndarray:
        if self._closed or not self._ptr:
            raise RuntimeError("GbUpperBoundPlan is closed")

        weights = np.asarray(weights, dtype=np.float64).ravel()
        if weights.size != self._m:
            raise ValueError("weights must match q length")

        out = np.empty(self._n, dtype=np.float64)
        rc = _LIB.rakau_gb_ub_plan_compute_weights_2d_r2(
            self._ptr,
            weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if rc != 0:
            raise RuntimeError(f"rakau_gb_ub_plan_compute_weights_2d_r2 failed with code {rc}")
        return out
