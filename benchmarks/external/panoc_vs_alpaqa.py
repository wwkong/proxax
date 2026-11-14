"""Benchmark comparing Proxax (PANOC / IALM / IAIPAL) against alpaqa.

Covers three nonconvex problem families:
  1. Box-constrained nonconvex QPs (``proxax.PANOC`` vs ``alpaqa.panoc``).
  2. Linearly-constrained nonconvex QPs (``IALM``, ``IALM(panoc)``, ``IAIPAL``
     vs ``alpaqa.ALMSolver`` with ``alpaqa.PANOCSolver``).
  3. Nonlinear-constrained nonconvex QPs with an l2-ball constraint and linear
     inequalities (``proxax`` splitting l2-ball in ``proximal`` vs ``alpaqa``
     handling both in ``ALMSolver``).

Because ``alpaqa==0.0.1`` only provides binary wheels up to Python 3.10 while
``proxax`` requires Python >=3.11, this script supports either in-process
``alpaqa`` imports or delegating ``alpaqa`` solves to a separate Python
interpreter via ``--alpaqa-python`` (using ``--alpaqa-worker`` mode).
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import os
import pathlib
import subprocess
import sys
import typing

import numpy as np

if typing.TYPE_CHECKING:
  from benchmarks import harness as harness_mod


@dataclasses.dataclass(frozen=True)
class NonconvexQPData:
  """Problem data for a nonconvex QP with optional box, linear, and ball."""

  name: str
  suite: str
  Q: np.ndarray
  c: np.ndarray
  lb: np.ndarray | None
  ub: np.ndarray | None
  A: np.ndarray | None
  b: np.ndarray | None
  radius: float | None
  x0: np.ndarray


def _make_indefinite_qp(
    rng: np.random.Generator,
    n: int,
    m: int = 0,
    *,
    with_box: bool = True,
    radius: float | None = None,
    name: str,
    suite: str,
) -> NonconvexQPData:
  """Generates a reproducible indefinite QP with guaranteed feasible x0."""
  U, _ = np.linalg.qr(rng.standard_normal((n, n)))
  # Symmetric indefinite spectrum in [-1.0, 4.0]
  eigs = rng.uniform(-1.0, 4.0, size=n)
  eigs[: max(1, n // 5)] = rng.uniform(-1.0, -0.1, size=max(1, n // 5))
  Q = (U * eigs) @ U.T
  Q = 0.5 * (Q + Q.T)
  c = rng.standard_normal(n)

  if with_box:
    lb = np.zeros(n, dtype=np.float64)
    ub = np.ones(n, dtype=np.float64)
    x_feas = rng.uniform(0.1, 0.9, size=n)
    x0 = np.full(n, 0.5, dtype=np.float64)
  else:
    lb = None
    ub = None
    x_feas = rng.standard_normal(n)
    if radius is not None:
      norm = float(np.linalg.norm(x_feas))
      if norm > 0.5 * radius:
        x_feas = x_feas * (0.5 * radius / norm)
    x0 = np.zeros(n, dtype=np.float64)

  if m > 0:
    A = rng.standard_normal((m, n)) / np.sqrt(n)
    slack = rng.uniform(0.1, 1.0, size=m)
    b = A @ x_feas + slack
  else:
    A = None
    b = None

  return NonconvexQPData(
      name=name,
      suite=suite,
      Q=Q,
      c=c,
      lb=lb,
      ub=ub,
      A=A,
      b=b,
      radius=radius,
      x0=x0,
  )


def _compute_independent_residuals(
    prob: NonconvexQPData,
    x: np.ndarray,
    y: np.ndarray | None,
) -> tuple[float, float, float]:
  """Computes (obj_val, opt_residual, feas_residual) independently."""
  grad = prob.Q @ x + prob.c
  obj = float(0.5 * (x @ prob.Q @ x) + prob.c @ x)

  if prob.A is not None and prob.b is not None and y is not None:
    m_lin = prob.A.shape[0]
    y_pos = np.maximum(0.0, y[:m_lin])
    grad_lag = grad + prob.A.T @ y_pos
    lin_viol = np.maximum(0.0, prob.A @ x - prob.b)
  else:
    grad_lag = grad
    lin_viol = np.zeros(0, dtype=np.float64)

  if prob.lb is not None and prob.ub is not None:
    proj_x = np.clip(x, prob.lb, prob.ub)
    box_viol = np.linalg.norm(x - proj_x)
    step = np.clip(x - grad_lag, prob.lb, prob.ub)
    opt_res = float(np.linalg.norm(x - step))
  elif prob.radius is not None:
    norm_x = float(np.linalg.norm(x))
    box_viol = max(0.0, norm_x - prob.radius)
    cand = x - grad_lag
    norm_c = float(np.linalg.norm(cand))
    if norm_c > prob.radius:
      cand = cand * (prob.radius / norm_c)
    opt_res = float(np.linalg.norm(x - cand))
  else:
    box_viol = 0.0
    opt_res = float(np.linalg.norm(grad_lag))

  feas_res = float(np.hypot(np.linalg.norm(lin_viol), box_viol))
  return obj, opt_res, feas_res


def _run_alpaqa_solve(payload: dict[str, typing.Any]) -> dict[str, typing.Any]:
  """Runs alpaqa in-process on a serialized NonconvexQPData payload."""
  import time

  import alpaqa as pa

  Q = np.asarray(payload["Q"], dtype=np.float64)
  c = np.asarray(payload["c"], dtype=np.float64)
  x0 = np.asarray(payload["x0"], dtype=np.float64)
  lb = (
      np.asarray(payload["lb"], dtype=np.float64)
      if payload["lb"] is not None
      else None
  )
  ub = (
      np.asarray(payload["ub"], dtype=np.float64)
      if payload["ub"] is not None
      else None
  )
  A = (
      np.asarray(payload["A"], dtype=np.float64)
      if payload["A"] is not None
      else None
  )
  b = (
      np.asarray(payload["b"], dtype=np.float64)
      if payload["b"] is not None
      else None
  )
  radius: float | None = payload["radius"]
  tol = float(payload["tol"])
  max_iters = int(payload["max_iters"])
  time_limit = float(payload["time_limit"])
  repeats = int(payload["repeats"])
  n = Q.shape[0]

  def f(x: np.ndarray) -> float:
    return float(0.5 * (x @ (Q @ x)) + c @ x)

  def grad_f(x: np.ndarray) -> np.ndarray:
    return np.asarray(Q @ x + c, dtype=np.float64)

  panoc_params = pa.PANOCParams(
      max_iter=max_iters,
      max_time=datetime.timedelta(seconds=time_limit),
      stop_crit=pa.PANOCStopCrit.FPRNorm,
  )
  lbfgs_params = pa.LBFGSParams(memory=10)

  if A is None and radius is None:
    box = pa.Box(n)
    if lb is not None and ub is not None:
      box.lowerbound = lb
      box.upperbound = ub

    eval_counts = {"f": 0, "grad_f": 0}

    def f_cnt(x: np.ndarray) -> float:
      eval_counts["f"] += 1
      return f(x)

    def gf_cnt(x: np.ndarray) -> np.ndarray:
      eval_counts["grad_f"] += 1
      return grad_f(x)

    run_times: list[float] = []
    x_sol = x0.copy()
    stats: dict[str, typing.Any] = {}
    for _ in range(max(1, repeats)):
      eval_counts["f"] = 0
      eval_counts["grad_f"] = 0
      t0 = time.perf_counter()
      x_sol, stats = pa.panoc(
          f_cnt, gf_cnt, box, x0, tol, panoc_params, lbfgs_params
      )
      run_times.append(time.perf_counter() - t0)

    return {
        "status": str(stats["status"]).split(".")[-1],
        "converged": stats["status"] == pa.SolverStatus.Converged,
        "x": np.asarray(x_sol, dtype=np.float64).tolist(),
        "y": None,
        "outer_iters": int(stats["iterations"]),
        "inner_iters": 0,
        "smooth_evals": eval_counts["f"],
        "grad_evals": eval_counts["grad_f"],
        "prox_evals": int(stats["iterations"]),
        "run_s": min(run_times),
    }

  m_lin = 0 if A is None else A.shape[0]
  m_ball = 1 if radius is not None else 0
  m_tot = m_lin + m_ball

  prob = pa.Problem(n, m_tot)
  prob.f = f
  prob.grad_f = grad_f
  box_c = pa.Box(n)
  if lb is not None and ub is not None:
    box_c.lowerbound = lb
    box_c.upperbound = ub
  prob.C = box_c

  box_d = pa.Box(m_tot)
  box_d.lowerbound = np.full(m_tot, -np.inf, dtype=np.float64)
  box_d.upperbound = np.zeros(m_tot, dtype=np.float64)
  prob.D = box_d

  def g(x: np.ndarray) -> np.ndarray:
    parts: list[np.ndarray] = []
    if A is not None and b is not None:
      parts.append(A @ x - b)
    if radius is not None:
      parts.append(np.array([0.5 * (float(x @ x) - radius * radius)]))
    return np.concatenate(parts)

  def grad_g_prod(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = np.zeros(n, dtype=np.float64)
    if A is not None:
      out += A.T @ y[:m_lin]
    if radius is not None:
      out += y[m_lin] * x
    return out

  prob.g = g
  prob.grad_g_prod = grad_g_prod

  alm_params = pa.ALMParams(
      ε=tol,
      δ=tol,
      max_iter=100,
      max_time=datetime.timedelta(seconds=time_limit),
  )
  inner_params = pa.PANOCParams(
      max_iter=500,
      max_time=datetime.timedelta(seconds=time_limit),
      stop_crit=pa.PANOCStopCrit.FPRNorm,
  )

  run_times = []
  x_alm = x0.copy()
  y_alm = np.zeros(m_tot, dtype=np.float64)
  stats_alm: dict[str, typing.Any] = {}
  evals_f = 0
  evals_gf = 0
  for _ in range(max(1, repeats)):
    pc = pa.ProblemWithCounters(prob)
    panoc = pa.PANOCSolver(inner_params, lbfgs_params)
    alm = pa.ALMSolver(alm_params, panoc)
    t0 = time.perf_counter()
    x_alm, y_alm, stats_alm = alm(pc, x0)
    run_times.append(time.perf_counter() - t0)
    evals_f = int(pc.evaluations.f)
    evals_gf = int(pc.evaluations.grad_f)

  inner_iters = int(stats_alm.get("inner", {}).get("iterations", 0))
  return {
      "status": str(stats_alm["status"]).split(".")[-1],
      "converged": stats_alm["status"] == pa.SolverStatus.Converged,
      "x": np.asarray(x_alm, dtype=np.float64).tolist(),
      "y": np.asarray(y_alm, dtype=np.float64).tolist(),
      "outer_iters": int(stats_alm["outer_iterations"]),
      "inner_iters": inner_iters,
      "smooth_evals": evals_f,
      "grad_evals": evals_gf,
      "prox_evals": inner_iters,
      "run_s": min(run_times),
  }


def _solve_with_alpaqa(
    prob: NonconvexQPData,
    *,
    tol: float,
    max_iters: int,
    time_limit: float,
    repeats: int,
    alpaqa_python: str | None,
) -> harness_mod.BenchmarkResult:
  """Solves a NonconvexQPData instance with alpaqa (in-process or worker)."""
  from benchmarks import harness

  payload = {
      "Q": prob.Q.tolist(),
      "c": prob.c.tolist(),
      "lb": prob.lb.tolist() if prob.lb is not None else None,
      "ub": prob.ub.tolist() if prob.ub is not None else None,
      "A": prob.A.tolist() if prob.A is not None else None,
      "b": prob.b.tolist() if prob.b is not None else None,
      "radius": prob.radius,
      "x0": prob.x0.tolist(),
      "tol": tol,
      "max_iters": max_iters,
      "time_limit": time_limit,
      "repeats": repeats,
  }

  if alpaqa_python is None:
    raw = _run_alpaqa_solve(payload)
  else:
    proc = subprocess.run(
        [alpaqa_python, __file__, "--alpaqa-worker"],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=True,
    )
    raw = json.loads(proc.stdout)

  x_sol = np.asarray(raw["x"], dtype=np.float64)
  y_sol = (
      np.asarray(raw["y"], dtype=np.float64) if raw["y"] is not None else None
  )
  obj, opt_res, feas_res = _compute_independent_residuals(prob, x_sol, y_sol)
  solver_name = (
      "alpaqa-PANOC"
      if (prob.A is None and prob.radius is None)
      else "alpaqa-ALM(PANOC)"
  )
  return harness.BenchmarkResult(
      suite=prob.suite,
      problem_name=prob.name,
      solver_name=solver_name,
      dtype="float64",
      status=str(raw["status"]),
      converged=bool(raw["converged"]),
      obj_val=obj,
      opt_residual=opt_res,
      feas_residual=feas_res,
      outer_iters=int(raw["outer_iters"]),
      inner_iters=int(raw["inner_iters"]),
      smooth_evals=int(raw["smooth_evals"]),
      grad_evals=int(raw["grad_evals"]),
      prox_evals=int(raw["prox_evals"]),
      compile_s=0.0,
      run_s=float(raw["run_s"]),
  )


def _solve_with_proxax(
    prob: NonconvexQPData,
    *,
    solver_name: str,
    solver: typing.Any,
    tol: float,
    time_limit: float,
    repeats: int,
) -> harness_mod.BenchmarkResult:
  """Solves a NonconvexQPData instance with Proxax and verifies KKT."""
  import jax.numpy as jnp

  from benchmarks import harness
  from proxax import px

  Q_j = jnp.asarray(prob.Q, dtype=jnp.float64)
  c_j = jnp.asarray(prob.c, dtype=jnp.float64)

  def f_fn(x: typing.Any) -> typing.Any:
    return 0.5 * jnp.dot(x, Q_j @ x) + jnp.dot(c_j, x)

  if prob.lb is not None and prob.ub is not None:
    prox_op: typing.Any = px.prox.Box(
        lower=jnp.asarray(prob.lb, dtype=jnp.float64),
        upper=jnp.asarray(prob.ub, dtype=jnp.float64),
    )
  elif prob.radius is not None:
    prox_op = px.prox.L2Ball(radius=prob.radius)
  else:
    prox_op = px.prox.Zero()

  constraints = None
  if prob.A is not None and prob.b is not None:
    A_j = jnp.asarray(prob.A, dtype=jnp.float64)
    b_j = jnp.asarray(prob.b, dtype=jnp.float64)
    constraints = px.constraints.less_equal(A_j, b_j)

  comp = px.CompositeProblem(
      smooth=f_fn,
      proximal=prox_op,
      constraints=constraints,
  )
  x0_j = jnp.asarray(prob.x0, dtype=jnp.float64)
  feas_tol = tol if constraints is not None else None

  res = harness.time_proxax_solve(
      suite=prob.suite,
      problem_name=prob.name,
      solver_name=solver_name,
      prob=comp,
      x0=x0_j,
      solver=solver,
      opt_tol=tol,
      feas_tol=feas_tol,
      repeats=repeats,
      time_limit=time_limit,
  )

  # Independent residual verification using the same formula as alpaqa.
  if feas_tol is not None:
    sol = px.solve(
        comp,
        x0_j,
        solver=solver,
        opt_tol=tol,
        feas_tol=feas_tol,
        jit=True,
        time_limit=time_limit,
    )
  else:
    sol = px.solve(
        comp,
        x0_j,
        solver=solver,
        opt_tol=tol,
        jit=True,
        time_limit=time_limit,
    )
  x_np = np.asarray(sol.x, dtype=np.float64)
  y_np = (
      np.asarray(sol.dual, dtype=np.float64) if sol.dual is not None else None
  )
  obj, opt_res, feas_res = _compute_independent_residuals(prob, x_np, y_np)
  return dataclasses.replace(
      res,
      obj_val=obj,
      opt_residual=opt_res,
      feas_residual=feas_res,
  )


def _detect_alpaqa_python(cli_value: str | None) -> str | None:
  """Returns None if alpaqa imports natively, or path to worker Python."""
  if cli_value:
    return cli_value
  try:
    import alpaqa  # noqa: F401

    return None
  except ImportError:
    pass
  env_path = os.environ.get("ALPAQA_PYTHON")
  if env_path and pathlib.Path(env_path).exists():
    return env_path
  default_path = pathlib.Path("/tmp/alpaqa_env/bin/python")
  if default_path.exists():
    return str(default_path)
  raise RuntimeError(
      "alpaqa is not importable in the current Python environment and no "
      "--alpaqa-python worker was found. Install alpaqa or pass "
      "--alpaqa-python /path/to/python."
  )


def build_problems(seed: int = 42) -> list[NonconvexQPData]:
  """Builds box, linearly constrained, and l2-ball nonconvex QP suites."""
  rng = np.random.default_rng(seed)
  problems: list[NonconvexQPData] = []

  for n in (50, 200, 500):
    problems.append(
        _make_indefinite_qp(
            rng,
            n,
            0,
            with_box=True,
            name=f"box_qp_n{n}",
            suite="box_qp",
        )
    )

  for n, m in ((50, 25), (200, 100)):
    problems.append(
        _make_indefinite_qp(
            rng,
            n,
            m,
            with_box=True,
            name=f"lin_qp_n{n}_m{m}",
            suite="lin_qp",
        )
    )

  for n, m in ((50, 25), (200, 100)):
    problems.append(
        _make_indefinite_qp(
            rng,
            n,
            m,
            with_box=False,
            radius=2.0,
            name=f"nl_ball_qp_n{n}_m{m}",
            suite="nl_ball_qp",
        )
    )

  return problems


def run_benchmark(
    *,
    tol: float,
    time_limit: float,
    repeats: int,
    alpaqa_python: str | None,
) -> list[harness_mod.BenchmarkResult]:
  """Runs all PANOC/ALM comparisons and returns BenchmarkResults."""
  from proxax.solvers import IAIPAL
  from proxax.solvers import IALM
  from proxax.solvers import PANOC

  results: list[harness_mod.BenchmarkResult] = []
  problems = build_problems()

  for prob in problems:
    if prob.A is None and prob.radius is None:
      results.append(
          _solve_with_proxax(
              prob,
              solver_name="proxax-PANOC",
              solver=PANOC(opt_tol=tol, max_iters=2000),
              tol=tol,
              time_limit=time_limit,
              repeats=repeats,
          )
      )
      results.append(
          _solve_with_alpaqa(
              prob,
              tol=tol,
              max_iters=2000,
              time_limit=time_limit,
              repeats=repeats,
              alpaqa_python=alpaqa_python,
          )
      )
    else:
      solvers: list[tuple[str, typing.Any]] = [
          (
              "proxax-IALM(apd)",
              IALM(opt_tol=tol, feas_tol=tol, subproblem_solver="apd"),
          ),
          (
              "proxax-IALM(panoc)",
              IALM(opt_tol=tol, feas_tol=tol, subproblem_solver="panoc"),
          ),
          ("proxax-IAIPAL", IAIPAL(opt_tol=tol, feas_tol=tol)),
      ]
      for s_name, solver in solvers:
        results.append(
            _solve_with_proxax(
                prob,
                solver_name=s_name,
                solver=solver,
                tol=tol,
                time_limit=time_limit,
                repeats=repeats,
            )
        )
      results.append(
          _solve_with_alpaqa(
              prob,
              tol=tol,
              max_iters=2000,
              time_limit=time_limit,
              repeats=repeats,
              alpaqa_python=alpaqa_python,
          )
      )

  return results


def main() -> None:
  """CLI entry point for the Proxax vs alpaqa benchmark."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--alpaqa-worker",
      action="store_true",
      help="Internal mode: read JSON problem on stdin, run alpaqa, print JSON.",
  )
  parser.add_argument(
      "--alpaqa-python",
      type=str,
      default=None,
      help="Python executable with alpaqa installed (if not in current env).",
  )
  parser.add_argument(
      "--tol",
      type=float,
      default=1e-5,
      help="Optimality and feasibility tolerance.",
  )
  parser.add_argument(
      "--time-limit",
      type=float,
      default=30.0,
      help="Per-problem time limit in seconds.",
  )
  parser.add_argument(
      "--repeats",
      type=int,
      default=3,
      help="Number of steady-state timing repeats after warmup.",
  )
  args = parser.parse_args()

  if args.alpaqa_worker:
    payload = json.load(sys.stdin)
    out = _run_alpaqa_solve(payload)
    json.dump(out, sys.stdout)
    return

  import jax

  jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]
  from benchmarks import harness

  alpaqa_python = _detect_alpaqa_python(args.alpaqa_python)
  results = run_benchmark(
      tol=args.tol,
      time_limit=args.time_limit,
      repeats=args.repeats,
      alpaqa_python=alpaqa_python,
  )
  print(harness.format_markdown_table(results))
  print("\n### Dolan-More Summary (by run_s)\n")
  print(harness.dolan_more_summary(results, metric="run_s"))


if __name__ == "__main__":
  main()
