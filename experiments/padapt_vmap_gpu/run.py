"""Benchmark comparing CPU sequential, CPU vmap, and GPU vmap for PAdaptADMM.

Evaluates `PAdaptADMM(adaptive_tau=False)` in `float32` on batches of
nonconvex Box-constrained Distributed QP (DQP) instances across:
1. CPU Sequential (jitted single-instance solve in a Python loop)
2. CPU `jax.vmap` (jitted batched solve across K instances)
3. GPU `jax.vmap` (if a CUDA/GPU device is available)

Usage:
  uv run python experiments/padapt_vmap_gpu/run.py
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from proxax import px  # noqa: E402


def _sample_dqp_batch(
    batch_size: int,
    num_blocks: int,
    m_constr: int,
    n_var: int,
    seed: int = 0,
) -> tuple[
    tuple[jax.Array, ...],
    tuple[jax.Array, ...],
    tuple[jax.Array, ...],
    jax.Array,
    tuple[jax.Array, ...],
]:
  """Generates a `float32` batch of `batch_size` nonconvex DQP instances."""
  key = jax.random.key(seed)
  p_blocks: list[jax.Array] = []
  r_blocks: list[jax.Array] = []
  a_blocks: list[jax.Array] = []
  x0_blocks: list[jax.Array] = []
  rhs_total = jnp.zeros((batch_size, m_constr), dtype=jnp.float32)

  for _ in range(num_blocks):
    key, k_a, k_q, k_d, k_r, k_xh, k_x0 = jax.random.split(key, 7)
    a_i = jax.random.normal(
        k_a, (batch_size, m_constr, n_var), dtype=jnp.float32
    ) / jnp.sqrt(float(n_var))

    q_raw = jax.random.normal(
        k_q, (batch_size, n_var, n_var), dtype=jnp.float32
    )
    q_mat, _ = jnp.linalg.qr(q_raw)

    d_vec = jax.random.uniform(
        k_d, (batch_size, n_var), minval=0.5, maxval=2.0, dtype=jnp.float32
    )
    d_vec = d_vec.at[:, 0].set(-0.05)
    p_i = jnp.einsum("bij,bj,bkj->bik", q_mat, d_vec, q_mat)
    p_i = 0.5 * (p_i + jnp.swapaxes(p_i, -1, -2))

    r_i = jax.random.normal(k_r, (batch_size, n_var), dtype=jnp.float32)
    x_hat = (
        jax.random.normal(k_xh, (batch_size, n_var), dtype=jnp.float32) * 0.2
    )
    x_0 = jax.random.normal(k_x0, (batch_size, n_var), dtype=jnp.float32) * 0.2
    rhs_total = rhs_total + jnp.einsum("bmn,bn->bm", a_i, x_hat)

    p_blocks.append(p_i)
    r_blocks.append(r_i)
    a_blocks.append(a_i)
    x0_blocks.append(x_0)

  return (
      tuple(p_blocks),
      tuple(r_blocks),
      tuple(a_blocks),
      rhs_total,
      tuple(x0_blocks),
  )


def _make_solve_single(rel_tol: float = 1e-2):
  """Builds a single-instance solver function suitable for `jit` and `vmap`."""

  def solve_one(
      p_tuple: tuple[jax.Array, ...],
      r_tuple: tuple[jax.Array, ...],
      a_tuple: tuple[jax.Array, ...],
      rhs: jax.Array,
      x0_tuple: tuple[jax.Array, ...],
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    blocks = tuple(
        px.BlockTerm(
            smooth=lambda x, p=p_i, r=r_i: (
                0.5 * jnp.dot(x, p @ x) + jnp.dot(r, x)
            ),
            proximal=px.prox.Box(lower=-1.0, upper=1.0),
            linear_op=a_i,
        )
        for p_i, r_i, a_i in zip(p_tuple, r_tuple, a_tuple, strict=True)
    )
    prob = px.BlockProblem(blocks=blocks, rhs=rhs)
    g0 = jax.grad(prob.objective_value)(x0_tuple)
    g_scale = 1.0 + jnp.sqrt(sum(jnp.sum(g_i**2) for g_i in g0))
    f_scale = 1.0 + jnp.linalg.norm(
        sum(a_i @ x_i for a_i, x_i in zip(a_tuple, x0_tuple, strict=True)) - rhs
    )
    opt_tol = rel_tol * g_scale
    feas_tol = rel_tol * f_scale
    solver = px.solvers.PAdaptADMM(
        opt_tol=opt_tol,
        feas_tol=feas_tol,
        lmt_bound=1e3 * opt_tol,
        gamma_init=10.0,
        max_stages=15,
        max_stage_iters=500,
        max_inner_iters=100,
        adaptive_tau=False,
    )
    sol = px.solve(prob, x0_tuple, solver=solver, jit=True)
    return (
        sol.status == int(px.SolveStatus.CONVERGED),
        sol.stats.outer_iters,
        sol.stats.opt_residual,
    )

  return solve_one


def _bench_sequential(
    solve_jit,
    batch_data: tuple,
    batch_size: int,
    device: jax.Device,
) -> tuple[float, float, int, float]:
  """Benchmarks sequential calls to a jitted single-instance solver."""
  p_b, r_b, a_b, rhs_b, x0_b = jax.device_put(batch_data, device)

  def _run_all() -> tuple[int, float]:
    conv_count = 0
    iter_sum = 0.0
    for k in range(batch_size):
      pk = tuple(p[k] for p in p_b)
      rk = tuple(r[k] for r in r_b)
      ak = tuple(a[k] for a in a_b)
      xk = tuple(x[k] for x in x0_b)
      conv, iters, _ = solve_jit(pk, rk, ak, rhs_b[k], xk)
      jax.block_until_ready(conv)
      conv_count += int(conv)
      iter_sum += float(iters)
    return conv_count, iter_sum / batch_size

  t0 = time.perf_counter()
  _run_all()
  cold_s = time.perf_counter() - t0

  t1 = time.perf_counter()
  conv_warm, avg_iters = _run_all()
  warm_s = time.perf_counter() - t1
  return cold_s, warm_s, conv_warm, avg_iters


def _bench_vmap(
    solve_vmap_jit,
    batch_data: tuple,
    device: jax.Device,
) -> tuple[float, float, int, float]:
  """Benchmarks a single `jit(vmap(...))` call over the batch."""
  data_dev = jax.device_put(batch_data, device)

  t0 = time.perf_counter()
  conv, iters, _ = solve_vmap_jit(*data_dev)
  jax.block_until_ready(conv)
  cold_s = time.perf_counter() - t0

  t1 = time.perf_counter()
  conv, iters, _ = solve_vmap_jit(*data_dev)
  jax.block_until_ready(conv)
  warm_s = time.perf_counter() - t1
  return cold_s, warm_s, int(jnp.sum(conv)), float(jnp.mean(iters))


def main() -> None:
  """Runs CPU sequential, CPU vmap, and GPU vmap PAdaptADMM benchmarks."""
  cpu_dev = jax.devices("cpu")[0]
  try:
    gpu_devs = jax.devices("gpu")
  except RuntimeError:
    gpu_devs = []
  gpu_dev = gpu_devs[0] if gpu_devs else None

  configs = [
      (2, 16, 32),
      (2, 64, 128),
  ]
  batch_sizes = [1, 8, 32]

  print(f"CPU device: {cpu_dev}")
  print(f"GPU device: {gpu_dev if gpu_dev is not None else 'None (skipped)'}")
  header = (
      f"{'B':>2} {'(m,n)':<9} {'K':>3} | {'Variant':<14} | "
      f"{'Conv':>6} | {'Iters':>6} | {'Cold (s)':>8} | {'Warm (s)':>8} | "
      f"{'ms/inst':>8}"
  )
  print("\n" + header, flush=True)
  print("-" * len(header), flush=True)

  solve_one = _make_solve_single()
  solve_jit = jax.jit(solve_one)
  solve_vmap_jit = jax.jit(jax.vmap(solve_one))

  for b_cnt, m_c, n_v in configs:
    for k_batch in batch_sizes:
      batch_data = _sample_dqp_batch(k_batch, b_cnt, m_c, n_v, seed=0)

      rows: list[tuple[str, float, float, int, float]] = []
      cold, warm, conv, iters = _bench_sequential(
          solve_jit, batch_data, k_batch, cpu_dev
      )
      rows.append(("CPU sequential", cold, warm, conv, iters))

      cold, warm, conv, iters = _bench_vmap(
          solve_vmap_jit, batch_data, cpu_dev
      )
      rows.append(("CPU vmap", cold, warm, conv, iters))

      if gpu_dev is not None:
        cold, warm, conv, iters = _bench_vmap(
            solve_vmap_jit, batch_data, gpu_dev
        )
        rows.append(("GPU vmap", cold, warm, conv, iters))

      for label, cold_s, warm_s, conv_n, avg_it in rows:
        ms_per = 1e3 * warm_s / k_batch
        print(
            f"{b_cnt:>2} {f'({m_c},{n_v})':<9} {k_batch:>3} | "
            f"{label:<14} | {f'{conv_n}/{k_batch}':>6} | "
            f"{avg_it:>6.1f} | {cold_s:>8.3f} | {warm_s:>8.3f} | "
            f"{ms_per:>8.2f}",
            flush=True,
        )
      print("-" * len(header), flush=True)


if __name__ == "__main__":
  main()
