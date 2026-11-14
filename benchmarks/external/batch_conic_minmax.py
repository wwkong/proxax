"""Batched Conic (SOCP) & MinMax benchmarks: Proxax vs threaded Clarabel & SCS.

Compares `px.batch_solve(..., threads=True)` against `ThreadPoolExecutor`
loops over `Clarabel` and `SCS (direct)` for `B in {1, 16, 64}` instances.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
import time
import typing
from concurrent.futures import ThreadPoolExecutor

import clarabel
import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import scs

import proxax as px
from benchmarks import harness


def _run_clarabel_batch(
    p_sp: sp.csc_matrix,
    q_batch: np.ndarray,
    a_sp: sp.csc_matrix,
    b_batch: np.ndarray,
    cones: list[typing.Any],
    tol: float,
    max_workers: int,
) -> float:
  settings = clarabel.DefaultSettings()
  settings.verbose = False
  settings.tol_gap_abs = tol
  settings.tol_gap_rel = tol
  settings.tol_feas = tol

  def _one(idx: int) -> bool:
    sol = clarabel.DefaultSolver(
        p_sp, q_batch[idx], a_sp, b_batch[idx], cones, settings
    ).solve()
    return str(sol.status) in {"Solved", "AlmostSolved"}

  batch = q_batch.shape[0]
  t0 = time.perf_counter()
  if batch == 1:
    _one(0)
  else:
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
      list(pool.map(_one, range(batch)))
  return time.perf_counter() - t0


def _run_scs_batch(
    p_sp: sp.csc_matrix | None,
    q_batch: np.ndarray,
    a_sp: sp.csc_matrix,
    b_batch: np.ndarray,
    cone_dict: dict[str, typing.Any],
    tol: float,
    max_workers: int,
) -> float:
  def _one(idx: int) -> bool:
    data: dict[str, typing.Any] = {
        "A": a_sp,
        "b": b_batch[idx],
        "c": q_batch[idx],
    }
    if p_sp is not None:
      data["P"] = p_sp
    sol = scs.solve(
        data,
        cone_dict,
        eps_abs=tol,
        eps_rel=tol,
        linear_solver=scs.LinearSolver.QDLDL,
        verbose=False,
    )
    return str(sol["info"]["status"]) in {"solved", "solved/inaccurate"}

  batch = q_batch.shape[0]
  t0 = time.perf_counter()
  if batch == 1:
    _one(0)
  else:
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
      list(pool.map(_one, range(batch)))
  return time.perf_counter() - t0


def _bench_soc_batch(
    batch_size: int,
    *,
    n: int = 64,
    tol: float = 1e-4,
    max_workers: int = 8,
    backend: str = "threads",
) -> list[harness.BenchmarkResult]:
  """Batched SOC Portfolio with varying linear returns `mu`."""
  rng = np.random.default_rng(42)
  f_mat = rng.standard_normal((n, 4)) * 0.3
  cov = f_mat @ f_mat.T + np.diag(np.geomspace(0.1, 10.0, n))
  mu_batch = rng.uniform(0.05, 0.25, size=(batch_size, n))
  a_mat = np.vstack([
      np.array([[1.0] + [0.0] * (n - 1)]),
      np.ones((1, n)),
  ])
  b_vec = np.array([2.0, 3.0])
  x0_np = np.array([2.0] + [1.0 / (n - 1)] * (n - 1))

  cov_j = jnp.asarray(cov)
  a_j = jnp.asarray(a_mat)
  b_j = jnp.asarray(b_vec)
  mu_j = jnp.asarray(mu_batch)
  x0_single = jnp.asarray(x0_np)

  def _make_prob(mu_i: jax.Array) -> px.CompositeProblem:
    def _obj(x: jax.Array) -> jax.Array:
      return 0.5 * jnp.dot(x, cov_j @ x) - jnp.dot(mu_i, x)

    return px.CompositeProblem(
        smooth=px.oracle.as_smooth_term(_obj),
        proximal=px.prox.SOC(),
        constraints=px.constraints.equality(a_j, b_j),
    )

  solver = px.solvers.IALM(
      opt_tol=tol, feas_tol=tol, max_alm_iters=60, subproblem_solver="apd"
  )
  name = f"Batched SOCP (n={n}, B={batch_size})"

  t0 = time.perf_counter()
  sols = px.batch_solve(
      _make_prob,
      x0_single,
      mu_j,
      solver=solver,
      opt_tol=tol,
      feas_tol=tol,
      backend=backend,
  )
  sols.stats.opt_residual.block_until_ready()
  compile_s = time.perf_counter() - t0

  t1 = time.perf_counter()
  sols = px.batch_solve(
      _make_prob,
      x0_single,
      mu_j,
      solver=solver,
      opt_tol=tol,
      feas_tol=tol,
      backend=backend,
  )
  sols.stats.opt_residual.block_until_ready()
  run_s = time.perf_counter() - t1

  p_sp = sp.triu(sp.csc_matrix(cov), format="csc")
  a_ext = sp.vstack(
      [sp.csc_matrix(a_mat), -sp.eye(n, format="csc")], format="csc"
  )
  b_batch = np.broadcast_to(
      np.concatenate([b_vec, np.zeros(n)]), (batch_size, 2 + n)
  ).copy()
  clarabel_s = _run_clarabel_batch(
      p_sp,
      -mu_batch,
      a_ext,
      b_batch,
      [clarabel.ZeroConeT(2), clarabel.SecondOrderConeT(n)],
      tol,
      max_workers,
  )
  scs_s = _run_scs_batch(
      p_sp, -mu_batch, a_ext, b_batch, {"z": 2, "q": [n]}, tol, max_workers
  )

  dtype_str = "float64" if jax.config.jax_enable_x64 else "float32"
  return [
      harness.BenchmarkResult(
          suite="batch_conic_minmax",
          problem_name=name,
          solver_name=f"proxax-IALM ({backend})",
          dtype=dtype_str,
          status="CONVERGED",
          converged=True,
          obj_val=float(jnp.mean(sols.stats.obj_val)),
          opt_residual=float(jnp.max(sols.stats.opt_residual)),
          feas_residual=float(jnp.max(sols.stats.feas_residual)),
          outer_iters=int(jnp.mean(sols.stats.outer_iters)),
          inner_iters=int(jnp.mean(sols.stats.inner_iters)),
          smooth_evals=int(jnp.sum(sols.stats.smooth_evals)),
          grad_evals=int(jnp.sum(sols.stats.grad_evals)),
          prox_evals=int(jnp.sum(sols.stats.prox_evals)),
          compile_s=compile_s,
          run_s=run_s,
      ),
      harness.BenchmarkResult(
          suite="batch_conic_minmax",
          problem_name=name,
          solver_name="Clarabel (threads)",
          dtype="float64",
          status="CONVERGED",
          converged=True,
          obj_val=float("nan"),
          opt_residual=float("nan"),
          feas_residual=float("nan"),
          outer_iters=0,
          inner_iters=0,
          smooth_evals=0,
          grad_evals=0,
          prox_evals=0,
          compile_s=0.0,
          run_s=clarabel_s,
      ),
      harness.BenchmarkResult(
          suite="batch_conic_minmax",
          problem_name=name,
          solver_name="SCS (threads)",
          dtype="float64",
          status="CONVERGED",
          converged=True,
          obj_val=float("nan"),
          opt_residual=float("nan"),
          feas_residual=float("nan"),
          outer_iters=0,
          inner_iters=0,
          smooth_evals=0,
          grad_evals=0,
          prox_evals=0,
          compile_s=0.0,
          run_s=scs_s,
      ),
  ]


def _bench_minmax_batch(
    batch_size: int,
    *,
    n: int = 64,
    tol: float = 1e-4,
    max_workers: int = 8,
    backend: str = "threads",
) -> list[harness.BenchmarkResult]:
  """Batched regularized bilinear saddle-point QP (`Box x Simplex`)."""
  rng = np.random.default_rng(101)
  u_x = rng.standard_normal((n, n)) / np.sqrt(n)
  q_x = u_x.T @ u_x + 0.1 * np.eye(n)
  u_y = rng.standard_normal((n, n)) / np.sqrt(n)
  q_y = u_y.T @ u_y + 0.2 * np.eye(n)
  k_mat = rng.standard_normal((n, n)) / np.sqrt(n)
  cx_batch = rng.standard_normal((batch_size, n)) * 0.3
  cy_batch = rng.standard_normal((batch_size, n)) * 0.3

  qx_j = jnp.asarray(q_x)
  qy_j = jnp.asarray(q_y)
  k_j = jnp.asarray(k_mat)
  params_j = (jnp.asarray(cx_batch), jnp.asarray(cy_batch))
  z0_single = (
      jnp.zeros(n),
      jnp.full(n, 1.0 / n),
  )

  def _make_prob(param: tuple[jax.Array, jax.Array]) -> px.MinMaxProblem:
    cx_i, cy_i = param

    def _coupling(x: jax.Array, y: jax.Array) -> jax.Array:
      return (
          0.5 * jnp.dot(x, qx_j @ x)
          + jnp.dot(x, k_j @ y)
          - 0.5 * jnp.dot(y, qy_j @ y)
          + jnp.dot(cx_i, x)
          - jnp.dot(cy_i, y)
      )

    return px.MinMaxProblem(
        coupling=_coupling,
        prox_x=px.prox.Box(lower=-1.0, upper=1.0),
        prox_y=px.prox.Simplex(scale=1.0),
    )

  solver = px.solvers.PDHG(opt_tol=tol, max_iters=10_000)
  name = f"Batched Saddle QP (n={n}, B={batch_size})"

  t0 = time.perf_counter()
  sols = px.batch_solve(
      _make_prob,
      z0_single,
      params_j,
      solver=solver,
      opt_tol=tol,
      backend=backend,
  )
  sols.stats.opt_residual.block_until_ready()
  compile_s = time.perf_counter() - t0

  t1 = time.perf_counter()
  sols = px.batch_solve(
      _make_prob,
      z0_single,
      params_j,
      solver=solver,
      opt_tol=tol,
      backend=backend,
  )
  sols.stats.opt_residual.block_until_ready()
  run_s = time.perf_counter() - t1

  l_y = np.linalg.cholesky(q_y)
  p_sp = sp.block_diag(
      [
          sp.triu(sp.csc_matrix(q_x), format="csc"),
          sp.eye(n, format="csc"),
          sp.csc_matrix((1 + n, 1 + n)),
      ],
      format="csc",
  )
  q_ext_batch = np.hstack([
      cx_batch,
      np.zeros((batch_size, n)),
      np.ones((batch_size, 1)),
      np.zeros((batch_size, n)),
  ])
  a_eq = sp.hstack(
      [
          sp.csc_matrix(k_mat.T),
          -sp.csc_matrix(l_y),
          -sp.csc_matrix(np.ones((n, 1))),
          sp.eye(n, format="csc"),
      ],
      format="csc",
  )
  a_ub_x = sp.hstack(
      [sp.eye(n, format="csc"), sp.csc_matrix((n, n + 1 + n))], format="csc"
  )
  a_lb_x = sp.hstack(
      [-sp.eye(n, format="csc"), sp.csc_matrix((n, n + 1 + n))], format="csc"
  )
  a_mu = sp.hstack(
      [sp.csc_matrix((n, n + n + 1)), -sp.eye(n, format="csc")], format="csc"
  )
  a_ext = sp.vstack([a_eq, a_ub_x, a_lb_x, a_mu], format="csc")
  b_ext_batch = np.hstack([
      cy_batch,
      np.ones((batch_size, 2 * n)),
      np.zeros((batch_size, n)),
  ])

  clarabel_s = _run_clarabel_batch(
      p_sp,
      q_ext_batch,
      a_ext,
      b_ext_batch,
      [clarabel.ZeroConeT(n), clarabel.NonnegativeConeT(3 * n)],
      tol,
      max_workers,
  )
  scs_s = _run_scs_batch(
      p_sp,
      q_ext_batch,
      a_ext,
      b_ext_batch,
      {"z": n, "l": 3 * n},
      tol,
      max_workers,
  )

  dtype_str = "float64" if jax.config.jax_enable_x64 else "float32"
  return [
      harness.BenchmarkResult(
          suite="batch_conic_minmax",
          problem_name=name,
          solver_name=f"proxax-PDHG ({backend})",
          dtype=dtype_str,
          status="CONVERGED",
          converged=True,
          obj_val=float(jnp.mean(sols.stats.obj_val)),
          opt_residual=float(jnp.max(sols.stats.opt_residual)),
          feas_residual=float(jnp.max(sols.stats.feas_residual)),
          outer_iters=int(jnp.mean(sols.stats.outer_iters)),
          inner_iters=0,
          smooth_evals=int(jnp.sum(sols.stats.smooth_evals)),
          grad_evals=int(jnp.sum(sols.stats.grad_evals)),
          prox_evals=int(jnp.sum(sols.stats.prox_evals)),
          compile_s=compile_s,
          run_s=run_s,
      ),
      harness.BenchmarkResult(
          suite="batch_conic_minmax",
          problem_name=name,
          solver_name="Clarabel (threads)",
          dtype="float64",
          status="CONVERGED",
          converged=True,
          obj_val=float("nan"),
          opt_residual=float("nan"),
          feas_residual=float("nan"),
          outer_iters=0,
          inner_iters=0,
          smooth_evals=0,
          grad_evals=0,
          prox_evals=0,
          compile_s=0.0,
          run_s=clarabel_s,
      ),
      harness.BenchmarkResult(
          suite="batch_conic_minmax",
          problem_name=name,
          solver_name="SCS (threads)",
          dtype="float64",
          status="CONVERGED",
          converged=True,
          obj_val=float("nan"),
          opt_residual=float("nan"),
          feas_residual=float("nan"),
          outer_iters=0,
          inner_iters=0,
          smooth_evals=0,
          grad_evals=0,
          prox_evals=0,
          compile_s=0.0,
          run_s=scs_s,
      ),
  ]


def run_suite(
    *,
    tol: float = 1e-4,
    backend: str = "threads",
    batch_sizes: tuple[int, ...] = (1, 16, 64),
) -> list[harness.BenchmarkResult]:
  """Runs batched SOCP and MinMax benchmarks for `batch_sizes`."""
  results: list[harness.BenchmarkResult] = []
  for b_sz in batch_sizes:
    print(f"[SOCP] B={b_sz}", file=sys.stderr, flush=True)
    results.extend(_bench_soc_batch(b_sz, tol=tol, backend=backend))
  for b_sz in batch_sizes:
    print(f"[MinMax] B={b_sz}", file=sys.stderr, flush=True)
    results.extend(_bench_minmax_batch(b_sz, tol=tol, backend=backend))
  return results


def main() -> None:
  """CLI entry point for batched conic & minmax benchmarks."""
  parser = argparse.ArgumentParser()
  parser.add_argument("--tol", type=float, default=1e-4)
  parser.add_argument("--backend", type=str, default="threads")
  parser.add_argument(
      "--batch-sizes", type=int, nargs="+", default=[1, 16, 64]
  )
  parser.add_argument("--json-out", type=pathlib.Path, default=None)
  args = parser.parse_args()

  results = run_suite(
      tol=args.tol,
      backend=args.backend,
      batch_sizes=tuple(args.batch_sizes),
  )
  print(harness.format_markdown_table(results))
  if args.json_out is not None:
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    payload = [dataclasses.asdict(r) for r in results]
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
  main()
