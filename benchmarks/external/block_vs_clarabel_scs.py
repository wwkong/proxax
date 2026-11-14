"""Multi-block consensus/sharing QP: Proxax BlockProblem vs Clarabel, SCS.

Compares `DPADMM`, `PAdaptADMM`, and `ADMM` against `Clarabel` and `SCS` on
`B`-block box-constrained convex QPs with linear coupling `sum_j A_j x_j = b`.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
import time
import typing

import clarabel
import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

import proxax as px
from benchmarks import harness
from benchmarks.external import conic_helpers


@dataclasses.dataclass(frozen=True)
class BlockBenchmark:
  """A single multi-block QP benchmark instance."""

  name: str
  proxax_builder: typing.Callable[
      [], tuple[px.BlockProblem, tuple[jax.Array, ...]]
  ]
  clarabel_builder: typing.Callable[
      [float, float], harness.BenchmarkResult
  ]
  scs_builder: typing.Callable[[float, bool], harness.BenchmarkResult]


def _make_multiblock_sharing_qp(
    num_blocks: int, *, d: int = 32, m: int = 16, seed: int = 77
) -> BlockBenchmark:
  """Convex `B`-block QP with `Box(-1, 1)` constraints and `m` coupling eqs."""
  rng = np.random.default_rng(seed)
  q_list: list[np.ndarray] = []
  c_list: list[np.ndarray] = []
  a_list: list[np.ndarray] = []
  rhs_np = np.zeros(m)
  for _ in range(num_blocks):
    u_i = rng.standard_normal((d, d)) / np.sqrt(d)
    q_i = u_i.T @ u_i + 0.2 * np.eye(d)
    c_i = rng.standard_normal(d) * 0.3
    a_i = rng.standard_normal((m, d)) / np.sqrt(num_blocks * d)
    x_feas = rng.uniform(-0.5, 0.5, size=d)
    rhs_np = rhs_np + a_i @ x_feas
    q_list.append(q_i)
    c_list.append(c_i)
    a_list.append(a_i)

  n_total = num_blocks * d
  name = f"Sharing QP (B={num_blocks}, N={n_total})"

  def _proxax() -> tuple[px.BlockProblem, tuple[jax.Array, ...]]:
    blocks = []
    x0s = []
    for q_i, c_i, a_i in zip(q_list, c_list, a_list, strict=True):
      q_j = jnp.asarray(q_i)
      c_j = jnp.asarray(c_i)
      a_j = jnp.asarray(a_i)

      def _make_smooth(
          qm: jax.Array, cv: jax.Array
      ) -> typing.Callable[[jax.Array], jax.Array]:
        return lambda x: 0.5 * jnp.dot(x, qm @ x) + jnp.dot(cv, x)

      blocks.append(
          px.BlockTerm(
              smooth=_make_smooth(q_j, c_j),
              proximal=px.prox.Box(lower=-1.0, upper=1.0),
              linear_op=a_j,
          )
      )
      x0s.append(jnp.zeros(d))
    prob = px.BlockProblem(blocks=tuple(blocks), rhs=jnp.asarray(rhs_np))
    return prob, tuple(x0s)

  p_sp = sp.block_diag(
      [sp.triu(sp.csc_matrix(q_i), format="csc") for q_i in q_list],
      format="csc",
  )
  q_vec = np.concatenate(c_list)
  a_eq = sp.hstack([sp.csc_matrix(a_i) for a_i in a_list], format="csc")
  eye_n = sp.eye(n_total, format="csc")
  a_ext = sp.vstack([a_eq, eye_n, -eye_n], format="csc")
  b_ext = np.concatenate([rhs_np, np.ones(n_total), np.ones(n_total)])

  def _clarabel(opt_tol: float, feas_tol: float) -> harness.BenchmarkResult:
    cones = [
        clarabel.ZeroConeT(m),
        clarabel.NonnegativeConeT(2 * n_total),
    ]
    return conic_helpers.solve_clarabel(
        suite="block_vs_clarabel_scs",
        problem_name=name,
        p_sp=p_sp,
        q_vec=q_vec,
        a_sp=a_ext,
        b_vec=b_ext,
        cones=cones,
        opt_tol=opt_tol,
        feas_tol=feas_tol,
    )

  def _scs(tol: float, indirect: bool) -> harness.BenchmarkResult:
    return conic_helpers.solve_scs(
        suite="block_vs_clarabel_scs",
        problem_name=name,
        p_sp=p_sp,
        q_vec=q_vec,
        a_sp=a_ext,
        b_vec=b_ext,
        cone_dict={"z": m, "l": 2 * n_total},
        tol=tol,
        use_indirect=indirect,
    )

  return BlockBenchmark(
      name=name,
      proxax_builder=_proxax,
      clarabel_builder=_clarabel,
      scs_builder=_scs,
  )


def _time_block_solve(
    *,
    problem_name: str,
    solver_name: str,
    prob: px.BlockProblem,
    x0: tuple[jax.Array, ...],
    solver: typing.Any,
    opt_tol: float,
    feas_tol: float,
    time_limit: float,
) -> harness.BenchmarkResult:
  """Times a `BlockProblem` solver with warmup + timed run."""
  dtype_str = "float64" if jax.config.jax_enable_x64 else "float32"

  def _run() -> px.Solution:
    sol = px.solve(
        prob,
        x0,
        solver=solver,
        opt_tol=opt_tol,
        feas_tol=feas_tol,
    )
    sol.stats.opt_residual.block_until_ready()
    return sol

  t0 = time.perf_counter()
  sol = _run()
  compile_s = time.perf_counter() - t0

  t1 = time.perf_counter()
  sol = _run()
  run_s = time.perf_counter() - t1

  return harness.BenchmarkResult(
      suite="block_vs_clarabel_scs",
      problem_name=problem_name,
      solver_name=solver_name,
      dtype=dtype_str,
      status=sol.status_enum.name,
      converged=(sol.status_enum == px.SolveStatus.CONVERGED),
      obj_val=float(sol.stats.obj_val),
      opt_residual=float(sol.stats.opt_residual),
      feas_residual=float(sol.stats.feas_residual),
      outer_iters=int(sol.stats.outer_iters),
      inner_iters=int(sol.stats.inner_iters),
      smooth_evals=int(sol.stats.smooth_evals),
      grad_evals=int(sol.stats.grad_evals),
      prox_evals=int(sol.stats.prox_evals),
      compile_s=compile_s,
      run_s=run_s,
  )


def run_suite(
    *,
    opt_tol: float = 1e-4,
    feas_tol: float = 1e-4,
    time_limit: float = 10.0,
) -> list[harness.BenchmarkResult]:
  """Runs multi-block QP benchmarks across Proxax, Clarabel, and SCS."""
  benchmarks = [
      _make_multiblock_sharing_qp(2, d=64),
      _make_multiblock_sharing_qp(4, d=64),
      _make_multiblock_sharing_qp(8, d=64),
  ]

  proxax_solvers = [
      (
          "DPADMM",
          px.solvers.DPADMM(
              opt_tol=opt_tol, feas_tol=feas_tol, max_stages=20
          ),
      ),
      (
          "PAdaptADMM",
          px.solvers.PAdaptADMM(
              opt_tol=opt_tol,
              feas_tol=feas_tol,
              adaptive_tau=True,
              max_stages=20,
          ),
      ),
      (
          "ADMM",
          px.solvers.ADMM(
              opt_tol=opt_tol, feas_tol=feas_tol, max_iters=2000
          ),
      ),
  ]

  def _log_done(res: harness.BenchmarkResult) -> None:
    print(
        f"  -> {res.solver_name}: {res.status} "
        f"(run={1e3 * res.run_s:.2f}ms, compile={res.compile_s:.2f}s)",
        file=sys.stderr,
        flush=True,
    )

  results: list[harness.BenchmarkResult] = []
  total = len(benchmarks)
  for idx, bench in enumerate(benchmarks, start=1):
    print(f"[{idx}/{total}] {bench.name}", file=sys.stderr, flush=True)
    prob, x0 = bench.proxax_builder()
    for solver_name, solver in proxax_solvers:
      res = _time_block_solve(
          problem_name=bench.name,
          solver_name=solver_name,
          prob=prob,
          x0=x0,
          solver=solver,
          opt_tol=opt_tol,
          feas_tol=feas_tol,
          time_limit=time_limit,
      )
      _log_done(res)
      results.append(res)
    for ext_res in (
        bench.clarabel_builder(opt_tol, feas_tol),
        bench.scs_builder(min(opt_tol, feas_tol), False),
        bench.scs_builder(min(opt_tol, feas_tol), True),
    ):
      _log_done(ext_res)
      results.append(ext_res)
  return results


def main() -> None:
  """CLI entry point for multi-block QP benchmarks."""
  parser = argparse.ArgumentParser(
      description="Benchmark Proxax BlockProblem solvers vs Clarabel and SCS."
  )
  parser.add_argument("--opt-tol", type=float, default=1e-4)
  parser.add_argument("--feas-tol", type=float, default=1e-4)
  parser.add_argument("--time-limit", type=float, default=10.0)
  parser.add_argument("--json-out", type=pathlib.Path, default=None)
  args = parser.parse_args()

  results = run_suite(
      opt_tol=args.opt_tol,
      feas_tol=args.feas_tol,
      time_limit=args.time_limit,
  )
  print(harness.format_markdown_table(results))
  if args.json_out is not None:
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    payload = [dataclasses.asdict(r) for r in results]
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
  main()
