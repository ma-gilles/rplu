"""
Loewner matrix and sampling utilities.

Provides functions for building Loewner matrices and sampling points.
"""

import numpy as np
from typing import Tuple


def build_loewner_cauchy_matrix_generator(x: np.ndarray, y: np.ndarray, fx: np.ndarray, fy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert Loewner matrix to rank-2 Cauchy-like matrix form.
    
    The Loewner matrix L[i,j] = (f(x_i) - f(y_j)) / (x_i - y_j) can be written as:
    L[i,j] = f(x_i) / (x_i - y_j) - f(y_j) / (x_i - y_j)
    
    This is a rank-2 Cauchy-like matrix: C[i,j] = sum_k g[i,k] * b[j,k] / (c[i] - q[j])
    where:
        c = x (row points)
        q = y (column points)
        g = [f(x), 1] (n x 2 matrix)
        b = [1, -f(y)] (m x 2 matrix)
    
    Parameters
    ----------
    x : array
        Row points (complex)
    y : array
        Column points (complex)
    fx : array
        Function values at x points (pre-computed)
    fy : array
        Function values at y points (pre-computed)
    
    Returns
    -------
    c : array
        Row points
    q : array
        Column points
    g : array
        Row generator matrix (n x 2)
    b : array
        Column generator matrix (m x 2)
    """
    c = x
    q = y
    alpha = np.sqrt(np.max([np.max(np.abs(fx)), np.max(np.abs(fy))]))
    g = np.column_stack([fx / alpha, np.ones(len(x), dtype=fx.dtype) * alpha])
    b = np.column_stack([np.ones(len(y), dtype=fy.dtype) * alpha, -fy / alpha])
    
    return c, q, g, b


def build_loewner_cauchy_matrix(x: np.ndarray, y: np.ndarray, f: callable) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience wrapper that evaluates f at x/y and returns the Cauchy-like factors.
    """
    fx = f(x)
    fy = f(y)
    return build_loewner_cauchy_matrix_generator(x, y, fx, fy)


def sample_unit_disk(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample n points uniformly in the unit disk.
    
    Uses polar coordinates with uniform angle and square-root radius
    to achieve uniform distribution.
    
    Parameters
    ----------
    n : int
        Number of points to sample
    rng : numpy.random.Generator
        Random number generator
    
    Returns
    -------
    points : array
        Complex points in the unit disk
    """
    r = np.sqrt(rng.random(n))
    theta = rng.uniform(0.0, 2 * np.pi, size=n)
    return r * np.exp(1j * theta)
