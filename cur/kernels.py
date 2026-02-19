#!/usr/bin/env python3
"""
Kernel functions for convolution operators
"""

import numpy as np

def drifted_aniso_gaussian_kernel3d(shape, sigmas=(4.0, 2.5, 2.5),
                                    alpha=1.0, beta=0.15, delta=1.0):
    """
    Build a **non-even**, **spatially decaying** 3D kernel sampled from the Green's function of
    `-nu Δ + a·∇ + mu` (constant coefficients). The drift `a` breaks even symmetry.
    
    Args:
        shape: (nz, ny, nx) tuple for kernel dimensions
        sigmas: (sx, sy, sz) tuple for standard deviations in each direction
        alpha: Scaling factordrifted_aniso_gaussian_kernel3d
        beta: Drift parameter (breaks evenness)
        delta: Shift parameter in x direction
    
    Returns:
        Kernel array of shape (nz, ny, nx) as float64
    """
    nz, ny, nx = shape
    cz, cy, cx = ( (n-1)//2 for n in (nz, ny, nx) )
    z = np.arange(nz) - cz
    y = np.arange(ny) - cy
    x = np.arange(nx) - cx
    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
    sx, sy, sz = sigmas  # note: sx for x, sy for y, sz for z
    # drift (breaks evenness) and shift in x:
    core = np.exp(beta * X) * np.exp(-0.5*((X - delta)**2/sx**2 + Y**2/sy**2 + Z**2/sz**2))
    s = alpha * core
    return s.astype(np.float64)
