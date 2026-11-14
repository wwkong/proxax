# GPU & Accelerator Usage

Proxax runs natively on NVIDIA GPUs and TPUs via JAX without custom C++/CUDA builds.

## Installation

Install Proxax alongside CUDA-enabled JAX:

```bash
pip install proxax "jax[cuda12]"
```

Verify that JAX detects your GPU:

```python
import jax
print(jax.devices())  # [CudaDevice(id=0)]
```

## Memory Preallocation

By default, XLA preallocates ~75% of total GPU VRAM at startup. On shared or desktop GPUs where part of VRAM is already in use, disable preallocation or cap the fraction before importing JAX:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# or: export XLA_PYTHON_CLIENT_MEM_FRACTION=0.35
```

## When to Use GPU vs. CPU

Based on our [GPU benchmarks](https://proxax.readthedocs.io/projects/experiments/en/latest/gpu_benchmarks.html) on an NVIDIA RTX 3060 (`float64`):

- **Prefer GPU (`backend="vmap"`) when:**
  - Solving **large batches** of independent problems with `px.batch_solve(..., backend="vmap")` or `px.batch_implicit_solve` (`B >= 256`), where `vmap` batches all solves into parallel GPU kernels and scales sublinearly with `B`.
  - Solving **dense** problems (`n >= 256`), such as nonconvex-concave min-max (`PDHG`, `ExtraGradient`).
  - Embedding `px.layers.QPLayer` or `px.implicit_solve` inside a GPU neural network / `Optax` training loop to avoid host-device transfers.
- **Prefer CPU (`JAX_PLATFORMS=cpu` or `backend="threads"`) when:**
  - Solving **single small or medium problems** where GPU kernel-launch overhead exceeds compute time.
  - Running **small sequential `SpectralProblem` solves** (`<= 48x48`), where per-iteration `cusolver` SVD launches are slower than CPU LAPACK.
  - Running **`float64` `BCOO` sparse matvecs** on consumer GeForce GPUs (which have `1/64` FP64 throughput compared to FP32; datacenter GPUs such as A100/H100 provide full-rate FP64).

## Choosing `float32` vs. `float64` on GPU

- **`float32` (default):** Full tensor/FP32 throughput on all GPUs; well-suited for `1e-3`–`1e-4` tolerances, `QPLayer`, and ML training loops.
- **`float64` (`JAX_ENABLE_X64=1`):** Needed for tight tolerances (`<= 1e-6`) or ill-conditioned QPs, at the cost of reduced throughput on consumer GeForce GPUs.
