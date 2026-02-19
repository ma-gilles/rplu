# JAX 3D Laplacian with homogeneous Dirichlet BCs (no Neumann, no +mu I term)
# and a fast inverse of (-nu * Δ_h) via separable 3D DST-I.
#
# Conventions:
#   - laplacian3d_dirichlet(u) returns  (-Δ_h) u   (SPD under Dirichlet).
#   - InvLaplacianDirichlet3D.apply(z) returns y solving (-nu Δ_h) y = z.
#
# Notes:
#   - Grid spacings hx,hy,hz allowed.
#   - DST-I implemented via an FFT-based odd extension (norm=None).
#   - Inverse uses the standard scaling 2/(n+1) per axis.

import jax
import jax.numpy as jnp
from functools import partial

jax.config.update("jax_enable_x64", True)

# ---------- 3D Laplacian (Dirichlet BCs) ----------
@partial(jax.jit, static_argnames=("hx","hy","hz"))
def laplacian3d_dirichlet(u, hx: float = 1.0, hy: float = 1.0, hz: float = 1.0):
    """
    u: array [nz, ny, nx], assumed zero on the boundary (Dirichlet).
    Returns (-Δ_h) u with 2nd-order FD and homogeneous Dirichlet BCs.
    """
    # Pad with zeros (Dirichlet)
    uzp = jnp.pad(u, ((1,1),(0,0),(0,0)), mode="constant")
    uyp = jnp.pad(u, ((0,0),(1,1),(0,0)), mode="constant")
    uxp = jnp.pad(u, ((0,0),(0,0),(1,1)), mode="constant")

    dzz = (uzp[2:, :, :] - 2*u + uzp[:-2, :, :]) / (hz*hz)
    dyy = (uyp[:, 2:, :] - 2*u + uyp[:, :-2, :]) / (hy*hy)
    dxx = (uxp[:, :, 2:] - 2*u + uxp[:, :, :-2]) / (hx*hx)

    return -(dxx + dyy + dzz)

# ---------- 1D DST-I (norm=None) via FFT odd extension ----------
def _dst1(x, axis=-1):
    """
    DST-I (type-1), norm=None. For length n along axis:
      Y_k = sum_{j=1}^n x_j sin(pi*j*k/(n+1)), k=1..n
    Implementation via FFT on an odd extension of length 2*(n+1).
    """
    x = jnp.swapaxes(x, axis, -1)
    n = x.shape[-1]
    m = n + 1  # extension size base
    # y length = 2m, with y[0]=0, y[1:n+1]=x, y[n+1]=0, y[n+2:2m]= -x[::-1]
    zeros_head = jnp.zeros_like(x[..., :1])
    zeros_mid  = jnp.zeros_like(x[..., :1])
    y = jnp.concatenate([zeros_head, x, zeros_mid, -x[..., ::-1]], axis=-1)  # [..., 2m]
    Y = jnp.fft.fft(y, axis=-1)  # complex
    # DST-I coefficients are -Im(FFT)[1:n+1], take k=1..n
    coeff = -jnp.imag(Y[..., 1:m])  # [..., n]
    coeff = jnp.swapaxes(coeff, -1, axis)
    return coeff.astype(x.dtype)

def _idst1(X, axis=-1):
    """
    Inverse of DST-I with norm=None:
      x_j = (2/(n+1)) * sum_{k=1}^n X_k sin(pi*j*k/(n+1))
    """
    X = jnp.swapaxes(X, axis, -1)
    n = X.shape[-1]
    m = n + 1
    # Build complex spectrum for odd extension such that ifft reconstructs y whose
    # entries 1..n equal x. This mirrors the forward construction.
    dtype = jnp.complex64 if X.dtype in (jnp.float32, jnp.complex64) else jnp.complex128
    Yhat = jnp.zeros(X.shape[:-1] + (2*m,), dtype=dtype)
    Yhat = Yhat.at[..., 1:m].set(-1j * X)
    Yhat = Yhat.at[..., (2*m-1):(m):-1].set(1j * X)
    y = jnp.fft.ifft(Yhat, axis=-1).real
    x_rec = y[..., 1:m]
    x_rec = jnp.swapaxes(x_rec, -1, axis)
    return x_rec.astype(X.dtype)

# ---------- Separable 3D DST-I and inverse ----------
def _dstn1_3d(u):
    u = _dst1(u, axis=0)
    u = _dst1(u, axis=1)
    u = _dst1(u, axis=2)
    return u

def _idstn1_3d(U):
    U = _idst1(U, axis=2)
    U = _idst1(U, axis=1)
    U = _idst1(U, axis=0)
    return U

# ---------- Fast inverse of (-nu * Δ_h) under Dirichlet BCs ----------
class InvLaplacianDirichlet3D:
    """
    Solve y from (-nu Δ_h) y = z with homogeneous Dirichlet BCs.
    Eigenvalues: lam = nu * [ (2-2cos(pi*kx/(nx+1)))/hx^2 + ... ]  for kx=1..nx, etc.
    """
    def __init__(self, nz:int, ny:int, nx:int, nu:float = 1.0,
                 hx:float = 1.0, hy:float = 1.0, hz:float = 1.0):
        self.nz, self.ny, self.nx = nz, ny, nx
        self.nu = float(nu)
        self.hx, self.hy, self.hz = float(hx), float(hy), float(hz)

        # kx = jnp.arange(1, nx+1, dtype=jnp.float64)
        # ky = jnp.arange(1, ny+1, dtype=jnp.float64)
        # kz = jnp.arange(1, nz+1, dtype=jnp.float64)

        # lamx = (2.0 - 2.0*jnp.cos(jnp.pi * kx / (nx + 1))) / (self.hx*self.hx)
        # lamy = (2.0 - 2.0*jnp.cos(jnp.pi * ky / (ny + 1))) / (self.hy*self.hy)
        # lamz = (2.0 - 2.0*jnp.cos(jnp.pi * kz / (nz + 1))) / (self.hz*self.hz)

        # self.LAM = self.nu * (lamz[:,None,None] + lamy[None,:,None] + lamx[None,None,:])  # [nz,ny,nx]

    
    @partial(jax.jit, static_argnums=(0,))
    def apply(self, z):
        """
        z: array [nz, ny, nx]
        returns y = (-nu Δ_h)^{-1} z  under Dirichlet BCs
        """
        kx = jnp.arange(1, self.nx+1, dtype=jnp.float64)
        ky = jnp.arange(1, self.ny+1, dtype=jnp.float64)
        kz = jnp.arange(1, self.nz+1, dtype=jnp.float64)

        lamx = (2.0 - 2.0*jnp.cos(jnp.pi * kx / (self.nx + 1))) / (self.hx*self.hx)
        lamy = (2.0 - 2.0*jnp.cos(jnp.pi * ky / (self.ny + 1))) / (self.hy*self.hy)
        lamz = (2.0 - 2.0*jnp.cos(jnp.pi * kz / (self.nz + 1))) / (self.hz*self.hz)

        LAM =  self.nu * (lamz[:,None,None] + lamy[None,:,None] + lamx[None,None,:])  # [nz,ny,nx]

        Zhat = _dstn1_3d(z)         # DST-I along each axis
        Yhat = Zhat / LAM      # divide by eigenvalues
        y = _idstn1_3d(Yhat)        # inverse DST-I (with scaling)
        return y

# ---------------- Quick self-test (remove in production) ----------------
if __name__ == "__main__":
    nz, ny, nx = 32, 33, 34
    hx = hy = hz = 1.0
    nu = 1.0

    invL = InvLaplacianDirichlet3D(nz, ny, nx, nu, hx, hy, hz)

    # Random RHS with zero boundary (not required for solve)
    key = jax.random.PRNGKey(0)
    z = jax.random.normal(key, (nz, ny, nx))
    y = invL.apply(z)

    # Check: apply (-nu Δ_h) to y and compare with z
    Dy = nu * laplacian3d_dirichlet(y, hx, hy, hz)
    rel = jnp.linalg.norm((Dy - z).ravel()) / jnp.linalg.norm(z.ravel())
    print("Dirichlet inverse check, rel. residual ~", float(rel))
