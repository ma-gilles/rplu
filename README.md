# RPLU

This repository contains code to reproduce the result of "Low-Rank Approximation by Randomly Pivoted LU".


## 0) Installation

These are necessary for Cauchy-like experiments.

```bash
conda create -n cauchycur -c conda-forge -y \
  python=3.11 pip \
  numpy scipy=1.16.3 matplotlib pytest \
  cmake ninja cxx-compiler \
  boost-cpp tbb tbb-devel xsimd requests baryrat
conda activate cauchycur

# Avoid picking up ~/.local site-packages (common on shared/HPC systems).
export PYTHONNOUSERSITE=1
```

### CUDA vs non-CUDA (JAX)

- Non-CUDA (CPU, including macOS):

  ```bash
  python -m pip install -U jax jaxlib
  ```

  Use this on CPU-only systems. Do not use `jax[cuda12]` unless you are on an NVIDIA CUDA setup.

- CUDA (GPU): install JAX per the official guide for your CUDA version:
  https://jax.readthedocs.io/en/latest/installation.html

  Example for CUDA 12.x (adjust for your CUDA major version):

  ```bash
  python -m pip install -U "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
  ```

- For the baryrat experiments:

  ```bash
  python -m pip install -U baryrat
  ```

Verify your install:

```bash
python -c "import jax; print(jax.devices())"
python -c "import scipy; print(scipy.__version__)"
```

## 1) Clone / update

Rakau is included directly in this repository under `rakau/` (no submodule, no separate clone).

```bash
git clone <REPO_URL>
cd cauchy
```

If you already cloned:

```bash
git pull
```

## 2) Build Rakau (Barnes-Hut) wrapper (required)

The Cauchy-like upper-bound backend uses `cauchy/Barnes_hut/gb_upper_bound_rakau.py`, which loads the in-repo Rakau build output:
- default: `rakau/librakau_gb_ub.so`
- override env var: `RAKAU_GB_UB_LIB`

### Build prerequisites

You need:
- C++17 compiler
- Boost headers
- TBB headers + library (`tbb-devel` + `tbb`)
- xsimd
- CMake

If dependencies are installed via conda:

```bash
export CMAKE_PREFIX_PATH="$CONDA_PREFIX"
```

### CPU-only build (no CUDA)

```bash
cmake -S rakau -B rakau/build-cpu \
  -DRAKAU_WITH_CUDA=OFF -DRAKAU_WITH_ROCM=OFF \
  -DRAKAU_BUILD_TESTS=OFF -DRAKAU_BUILD_BENCHMARKS=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build rakau/build-cpu -j
```

This produces `rakau/librakau_gb_ub.so` and does not require CUDA at runtime.

## 3) Smoke test

Important: `cauchy/cauchy_jax.py` sets a default `CUDA_VISIBLE_DEVICES` if you do not set it.

```bash
export JAX_ENABLE_X64=1

# GPU (set a valid id):
export CUDA_VISIBLE_DEVICES=0

# OR CPU:
# export JAX_PLATFORMS=cpu

python -c "import jax; print(jax.devices()); from cauchy.cauchy_jax import CauchyCUR; print('import ok')"
```

Tiny CauchyCUR build (works on CPU too; exact norms path):

```bash
python - <<'PY'
import os
os.environ["JAX_ENABLE_X64"] = "1"

import numpy as np
import jax.numpy as jnp

from cauchy.cauchy_jax import CauchyCUR

rng = np.random.default_rng(0)
n = 200
m = 200
r = 2
c = rng.standard_normal(n) + 1j * rng.standard_normal(n)
q = rng.standard_normal(m) + 1j * rng.standard_normal(m)
g = rng.standard_normal((n, r))
b = rng.standard_normal((m, r))

cur = CauchyCUR(jnp.asarray(c), jnp.asarray(q), jnp.asarray(g), jnp.asarray(b), block_size=1, use_exact_norm=True)
i_idx, j_idx = cur.build(rank=10)
print("ok", i_idx.shape, j_idx.shape)
PY
```

Optional: validate the Rakau upper-bound wrapper directly (no JAX required):

```bash
python -c "import numpy as np; from cauchy.Barnes_hut.gb_upper_bound_rakau import GbUpperBoundPlan; rng=np.random.default_rng(0); c=rng.standard_normal(100)+1j*rng.standard_normal(100); q=rng.standard_normal(120)+1j*rng.standard_normal(120); plan=GbUpperBoundPlan(c,q,2.0); g=rng.standard_normal((100,2))+1j*rng.standard_normal((100,2)); b=rng.standard_normal((120,2))+1j*rng.standard_normal((120,2)); out=plan.compute(g,b); print('ok', out.shape, float(out.min()), float(out.max()))"
```

## 4) Reproduce results in manuscript

### 4.1) Toy (Figures 1-3)

#### Figure 1

```bash
python naive_rplu/test_rplu_vs_complete_lu.py
```

#### Figure 2 (symmetric)

Requires RPCholesky:

```bash
cd <../<THIS_DIR>>
git clone https://github.com/eepperly/Randomly-Pivoted-Cholesky.git
cd cauchy
```

(Or ensure `Randomly-Pivoted-Cholesky` is on `PYTHONPATH`.)

```bash
python naive_rplu/test_symmetric_methods.py
```

#### Figure 3 (bound comparison)

```bash
python rpchol_bounds_comparison.py
python rplu_bounds_comparison.py
```

### 4.2) GPU implementation (Figures 4-6)

These require CUDA to run in reasonable time, and JAX installed with CUDA as above.

```bash
python -m pip install -U jax jaxlib baryrat
```

#### Figure 4 example (row 1)

```bash
python cur/test_gpu_cur_rsvd.py --problem-type toeplitz --dim 128
```

If SVD is too memory-heavy:

```bash
python cur/test_gpu_cur_rsvd.py --problem-type toeplitz --dim 128 --skip-svd
```

#### Figure 5

```bash
python cur/run_experiment3_multi_seed.py --method=cur_greedy --problem-size=320 --output-dir=results/
```

Run separately for each method:
- `unpreconditioned`
- `svd_precon`
- `cur_greedy`
- `cur_random`
- `pivoted_qr_cur_greedy`
- `pivoted_qr_cur_random`

The full `problem_size=320` sweep can take a long time.

#### Figure 6 graph example

```bash
python cur/test_gpu_cur_rsvd.py --problem-type graph --graph-name circuit5M_dc --compute-frobenius --skip-svd
```

Note: this may download from SparseSuite first. If already downloaded, pass a local directory to `--graphs-dir`.

### 4.3) Cauchy-like implementation (Figures 7-9)

#### Figure 7: Cauchy-like benchmark

To run the benchmark vs RSVD + FMM, you need to install https://github.com/dbstein/pyfmmlib2d, which can be painful.
If you do it successfully, you can do this:
```bash
python test_loewner_cauchy_paper.py compare_methods
```

If not, you can skip by:
```bash
python test_loewner_cauchy_paper.py compare_methods --no-include-rsvd-fmm
```

To get the timing plot, similarly:
```bash
python test_loewner_cauchy_paper.py timing_fixed_d --no-include-rsvd-fmm
```

#### Figures 8 and 9: CUR-AAA vs AAA

```bash
python test_baryrat_cur_half_paper.py
```
