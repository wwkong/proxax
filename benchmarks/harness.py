"""Shared timing, evaluation, and reporting utilities for benchmarks."""

from __future__ import annotations

import dataclasses
import time
import typing

import jax
import jax.numpy as jnp

from proxax import px
from proxax.solvers import base


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class BenchmarkResult:
  """Recorded metrics for a single solver run on a benchmark problem."""

  suite: str
  problem_name: str
  solver_name: str
  dtype: str
  status: str
  converged: bool
  obj_val: float
  opt_residual: float
  feas_residual: float
  outer_iters: int
  inner_iters: int
  smooth_evals: int
  grad_evals: int
  prox_evals: int
  compile_s: float
  run_s: float


def time_proxax_solve(
    *,
    suite: str,
    problem_name: str,
    solver_name: str,
    prob: px.CompositeProblem | px.MinMaxProblem | px.SpectralProblem,
    x0: typing.Any,
    solver: typing.Any,
    opt_tol: float,
    feas_tol: float | None = None,
    repeats: int = 1,
) -> BenchmarkResult:
  """Times a Proxax solver on a problem, separating JIT compile and run."""
  dtype_str = "float64" if jax.config.jax_enable_x64 else "float32"

  def _run() -> px.Solution:
    kwargs: dict[str, typing.Any] = {
        "solver": solver,
        "opt_tol": opt_tol,
        "jit": True,
    }
    if feas_tol is not None and isinstance(prob, px.CompositeProblem):
      kwargs["feas_tol"] = feas_tol
    sol = px.solve(prob, x0, **kwargs)
    sol.stats.opt_residual.block_until_ready()
    return sol

  t0 = time.perf_counter()
  sol = _run()
  compile_s = time.perf_counter() - t0

  run_times: list[float] = []
  for _ in range(max(1, repeats)):
    t1 = time.perf_counter()
    sol = _run()
    run_times.append(time.perf_counter() - t1)

  best_run_s = min(run_times)
  return BenchmarkResult(
      suite=suite,
      problem_name=problem_name,
      solver_name=solver_name,
      dtype=dtype_str,
      status=sol.status_enum.name,
      converged=(sol.status_enum == base.SolveStatus.CONVERGED),
      obj_val=float(sol.stats.obj_val),
      opt_residual=float(sol.stats.opt_residual),
      feas_residual=float(sol.stats.feas_residual),
      outer_iters=int(sol.stats.outer_iters),
      inner_iters=int(sol.stats.inner_iters),
      smooth_evals=int(sol.stats.smooth_evals),
      grad_evals=int(sol.stats.grad_evals),
      prox_evals=int(sol.stats.prox_evals),
      compile_s=compile_s,
      run_s=best_run_s,
  )


def format_markdown_table(results: list[BenchmarkResult]) -> str:
  """Formats benchmark results as a GitHub-flavored Markdown table."""
  header = (
      "| Suite | Problem | Solver | Status | Obj | Opt Res | Feas Res "
      "| Grad Evals | Run (ms) |"
  )
  sep = (
      "| :--- | :--- | :--- | :--- | ---: | ---: | ---: "
      "| ---: | ---: |"
  )
  rows = [header, sep]
  for r in results:
    rows.append(
        f"| {r.suite} | {r.problem_name} | {r.solver_name} | {r.status} "
        f"| {r.obj_val:.4f} | {r.opt_residual:.2e} | {r.feas_residual:.2e} "
        f"| {r.grad_evals} | {1e3 * r.run_s:.2f} |"
    )
  return "\n".join(rows)


def dolan_more_summary(
    results: list[BenchmarkResult],
    *,
    metric: str = "grad_evals",
    taus: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0),
) -> str:
  """Computes a Dolan-More performance profile summary across solvers."""
  problems = sorted({(r.suite, r.problem_name) for r in results})
  solvers = sorted({r.solver_name for r in results})
  by_key = {(r.suite, r.problem_name, r.solver_name): r for r in results}

  best_per_prob: dict[tuple[str, str], float] = {}
  for prob_key in problems:
    vals: list[float] = []
    for s in solvers:
      res = by_key.get((prob_key[0], prob_key[1], s))
      if res is not None and res.converged:
        v = float(getattr(res, metric))
        vals.append(max(v, 1e-12))
    best_per_prob[prob_key] = min(vals) if vals else float("inf")

  tau_cols = " | ".join(f"tau<={t:g}" for t in taus)
  header = f"| Solver | Solved | {tau_cols} |"
  sep = "| :--- | ---: | " + " | ".join("---:" for _ in taus) + " |"
  lines = [header, sep]

  for s in solvers:
    ratios: list[float] = []
    for prob_key in problems:
      res = by_key.get((prob_key[0], prob_key[1], s))
      best = best_per_prob[prob_key]
      if res is not None and res.converged and jnp.isfinite(best):
        ratios.append(max(float(getattr(res, metric)), 1e-12) / best)
      else:
        ratios.append(float("inf"))
    n_probs = max(1, len(problems))
    solved = sum(1 for r in ratios if r < float("inf"))
    fracs = " | ".join(
        f"{sum(1 for r in ratios if r <= t) / n_probs:.2f}" for t in taus
    )
    lines.append(f"| {s} | {solved}/{len(problems)} | {fracs} |")

  return "\n".join(lines)
