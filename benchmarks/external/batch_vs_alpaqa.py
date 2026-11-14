"""Batched nonconvex benchmark: Proxax (vmap / threads) vs. alpaqa (procs).

Covers three nonconvex QP families across batch sizes B in {1, 16, 64}:
  1. ``box_qp_batch``: min_{x in [0, 1]^n} 0.5 x^T Q x + q_i^T x (n=100).
  2. ``lin_qp_batch``: min_{x in [0, 1]^n} 0.5 x^T Q x + q_i^T x s.t. Ax <= b
     (n=50, m=25).
  3. ``nl_ball_qp_batch``: min_{||x||_2 <= 2} 0.5 x^T Q x + q_i^T x s.t. Ax <= b
     (n=50, m=25).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import datetime
import json
import multiprocessing
import os
import pathlib
import subprocess
import sys
import time
import typing

import numpy as np


@dataclasses.dataclass(frozen=True)
class BatchedAlpaqaCase:
  """Problem data for a batched nonconvex QP."""

  suite: str
  name: str
  n: int
  m: int
  batch_size: int
  Q: np.ndarray
  lb: np.ndarray | None
  ub: np.ndarray | None
  A: np.ndarray | None
  b: np.ndarray | None
  radius: float | None
  x0: np.ndarray
  q_batch: np.ndarray


@dataclasses.dataclass(frozen=True)
class BatchAlpaqaResult:
  """Recorded timing and convergence metrics."""

  suite: str
  problem_name: str
  solver_name: str
  batch_size: int
  converged: bool
  mean_obj: float
  compile_s: float
  run_s: float


def build_cases(seed: int = 42) -> list[BatchedAlpaqaCase]:
  """Builds batched box, linear, and l2-ball nonconvex QP cases."""
  rng = np.random.default_rng(seed)
  cases: list[BatchedAlpaqaCase] = []

  def _indef_q(n: int) -> np.ndarray:
    U, _ = np.linalg.qr(rng.standard_normal((n, n)))
    eigs = rng.uniform(-1.0, 4.0, size=n)
    eigs[: max(1, n // 5)] = rng.uniform(-1.0, -0.1, size=max(1, n // 5))
    Q = (U * eigs) @ U.T
    return 0.5 * (Q + Q.T)

  Q_box = _indef_q(100)
  for bs in (1, 16, 64):
    cases.append(
        BatchedAlpaqaCase(
            suite="box_qp_batch",
            name=f"box_qp_n100_B{bs}",
            n=100,
            m=0,
            batch_size=bs,
            Q=Q_box,
            lb=np.zeros(100),
            ub=np.ones(100),
            A=None,
            b=None,
            radius=None,
            x0=np.full(100, 0.5),
            q_batch=rng.standard_normal((bs, 100)),
        )
    )

  Q_lin = _indef_q(50)
  A_lin = rng.standard_normal((25, 50)) / np.sqrt(50)
  b_lin = A_lin @ rng.uniform(0.1, 0.9, size=50) + 0.5
  for bs in (1, 16, 64):
    cases.append(
        BatchedAlpaqaCase(
            suite="lin_qp_batch",
            name=f"lin_qp_n50_B{bs}",
            n=50,
            m=25,
            batch_size=bs,
            Q=Q_lin,
            lb=np.zeros(50),
            ub=np.ones(50),
            A=A_lin,
            b=b_lin,
            radius=None,
            x0=np.full(50, 0.5),
            q_batch=rng.standard_normal((bs, 50)),
        )
    )

  Q_nl = _indef_q(50)
  A_nl = rng.standard_normal((25, 50)) / np.sqrt(50)
  b_nl = A_nl @ np.zeros(50) + 0.5
  for bs in (1, 16, 64):
    cases.append(
        BatchedAlpaqaCase(
            suite="nl_ball_qp_batch",
            name=f"nl_ball_qp_n50_B{bs}",
            n=50,
            m=25,
            batch_size=bs,
            Q=Q_nl,
            lb=None,
            ub=None,
            A=A_nl,
            b=b_nl,
            radius=2.0,
            x0=np.zeros(50),
            q_batch=rng.standard_normal((bs, 50)),
        )
    )

  return cases


_ALPAQA_CTX: dict[str, typing.Any] = {}


def _alpaqa_solve_one_proc(idx: int) -> tuple[bool, float]:
  """Solves one batch element with alpaqa inside a forked worker process."""
  import alpaqa as pa

  ctx = _ALPAQA_CTX
  Q: np.ndarray = ctx["Q"]
  x0: np.ndarray = ctx["x0"]
  q_batch: np.ndarray = ctx["q_batch"]
  lb: np.ndarray | None = ctx["lb"]
  ub: np.ndarray | None = ctx["ub"]
  A: np.ndarray | None = ctx["A"]
  b: np.ndarray | None = ctx["b"]
  radius: float | None = ctx["radius"]
  tol: float = ctx["tol"]
  n: int = ctx["n"]
  c = q_batch[idx]

  panoc_params = pa.PANOCParams(
      max_iter=2000,
      max_time=datetime.timedelta(seconds=30.0),
      stop_crit=pa.PANOCStopCrit.FPRNorm,
  )
  lbfgs_params = pa.LBFGSParams(memory=10)

  def f(x: np.ndarray) -> float:
    return float(0.5 * (x @ (Q @ x)) + c @ x)

  def grad_f(x: np.ndarray) -> np.ndarray:
    return np.asarray(Q @ x + c, dtype=np.float64)

  if A is None and radius is None:
    box = pa.Box(n)
    if lb is not None and ub is not None:
      box.lowerbound = lb
      box.upperbound = ub
    x_sol, stats = pa.panoc(
        f, grad_f, box, x0, tol, panoc_params, lbfgs_params
    )
    ok = stats["status"] == pa.SolverStatus.Converged
    return ok, f(np.asarray(x_sol, dtype=np.float64))

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
      ε=tol, δ=tol, max_iter=100, max_time=datetime.timedelta(seconds=30.0)
  )
  inner_params = pa.PANOCParams(
      max_iter=500,
      max_time=datetime.timedelta(seconds=30.0),
      stop_crit=pa.PANOCStopCrit.FPRNorm,
  )
  alm = pa.ALMSolver(alm_params, pa.PANOCSolver(inner_params, lbfgs_params))
  x_alm, _, stats_alm = alm(prob, x0)
  ok = stats_alm["status"] == pa.SolverStatus.Converged
  return ok, f(np.asarray(x_alm, dtype=np.float64))


def _run_alpaqa_batch_worker(
    payload: dict[str, typing.Any],
) -> dict[str, typing.Any]:
  """Runs alpaqa in parallel across batch elements via ProcessPoolExecutor."""
  import alpaqa as _pa  # noqa: F401 - preload before fork

  Q = np.asarray(payload["Q"], dtype=np.float64)
  x0 = np.asarray(payload["x0"], dtype=np.float64)
  q_batch = np.asarray(payload["q_batch"], dtype=np.float64)
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
  repeats = int(payload["repeats"])
  n = Q.shape[0]
  bs = q_batch.shape[0]

  _ALPAQA_CTX.clear()
  _ALPAQA_CTX.update({
      "Q": Q,
      "x0": x0,
      "q_batch": q_batch,
      "lb": lb,
      "ub": ub,
      "A": A,
      "b": b,
      "radius": radius,
      "tol": tol,
      "n": n,
  })

  mp_ctx = multiprocessing.get_context("fork")
  with concurrent.futures.ProcessPoolExecutor(mp_context=mp_ctx) as pool:

    def _run_pool() -> tuple[bool, float]:
      outs = list(pool.map(_alpaqa_solve_one_proc, range(bs)))
      return all(ok for ok, _ in outs), float(np.mean([o for _, o in outs]))

    t0 = time.perf_counter()
    all_ok, mean_obj = _run_pool()
    compile_s = time.perf_counter() - t0

    run_times: list[float] = []
    for _ in range(max(1, repeats)):
      t1 = time.perf_counter()
      all_ok, mean_obj = _run_pool()
      run_times.append(time.perf_counter() - t1)

  return {
      "converged": all_ok,
      "mean_obj": mean_obj,
      "compile_s": compile_s,
      "run_s": min(run_times),
  }


def _solve_with_alpaqa_batch(
    case: BatchedAlpaqaCase,
    *,
    tol: float,
    repeats: int,
    alpaqa_python: str | None,
) -> BatchAlpaqaResult:
  """Solves a batch with alpaqa (in-process or worker)."""
  payload = {
      "Q": case.Q.tolist(),
      "lb": case.lb.tolist() if case.lb is not None else None,
      "ub": case.ub.tolist() if case.ub is not None else None,
      "A": case.A.tolist() if case.A is not None else None,
      "b": case.b.tolist() if case.b is not None else None,
      "radius": case.radius,
      "x0": case.x0.tolist(),
      "q_batch": case.q_batch.tolist(),
      "tol": tol,
      "repeats": repeats,
  }
  if alpaqa_python is None:
    raw = _run_alpaqa_batch_worker(payload)
  else:
    proc = subprocess.run(
        [alpaqa_python, __file__, "--alpaqa-worker"],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=True,
    )
    raw = json.loads(proc.stdout)

  solver_name = (
      "alpaqa-PANOC(procs)"
      if case.suite == "box_qp_batch"
      else "alpaqa-ALM(procs)"
  )
  return BatchAlpaqaResult(
      suite=case.suite,
      problem_name=case.name,
      solver_name=solver_name,
      batch_size=case.batch_size,
      converged=bool(raw["converged"]),
      mean_obj=float(raw["mean_obj"]),
      compile_s=float(raw["compile_s"]),
      run_s=float(raw["run_s"]),
  )


def _solve_with_proxax_batch(
    case: BatchedAlpaqaCase,
    *,
    solver_name: str,
    solver: typing.Any,
    backend: str,
    tol: float,
    repeats: int,
    chunk_size: int = 1,
) -> BatchAlpaqaResult:
  """Solves a nonconvex batch with px.batch_solve."""
  import jax.numpy as jnp

  from proxax import px

  Q_j = jnp.asarray(case.Q, dtype=jnp.float64)
  qb_j = jnp.asarray(case.q_batch, dtype=jnp.float64)
  x0_j = jnp.asarray(case.x0, dtype=jnp.float64)

  if case.lb is not None and case.ub is not None:
    prox_op: typing.Any = px.prox.Box(
        lower=jnp.asarray(case.lb, dtype=jnp.float64),
        upper=jnp.asarray(case.ub, dtype=jnp.float64),
    )
  elif case.radius is not None:
    prox_op = px.prox.L2Ball(radius=case.radius)
  else:
    prox_op = px.prox.Zero()

  constraints = None
  if case.A is not None and case.b is not None:
    constraints = px.constraints.less_equal(
        jnp.asarray(case.A, dtype=jnp.float64),
        jnp.asarray(case.b, dtype=jnp.float64),
    )

  def prob_fn(c_i: typing.Any) -> px.CompositeProblem:
    return px.CompositeProblem(
        smooth=lambda x: 0.5 * jnp.dot(x, Q_j @ x) + jnp.dot(c_i, x),
        proximal=prox_op,
        constraints=constraints,
    )

  def _run() -> px.Solution:
    sol = px.batch_solve(
        prob_fn,
        x0_j,
        qb_j,
        solver=solver,
        opt_tol=tol,
        feas_tol=tol,
        backend=backend,
        chunk_size=chunk_size,
    )
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

  converged = bool(
      jnp.all(sol.status == int(px.SolveStatus.CONVERGED.value))
  )
  return BatchAlpaqaResult(
      suite=case.suite,
      problem_name=case.name,
      solver_name=solver_name,
      batch_size=case.batch_size,
      converged=converged,
      mean_obj=float(jnp.mean(sol.stats.obj_val)),
      compile_s=compile_s,
      run_s=min(run_times),
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
      "alpaqa is not importable and no --alpaqa-python worker was found."
  )


def format_markdown_table(results: list[BatchAlpaqaResult]) -> str:
  """Formats results as a GitHub-flavored Markdown table."""
  header = (
      "| Suite | Problem | Solver | B | Converged | Mean Obj "
      "| Compile (s) | Run (ms) |"
  )
  sep = "| :--- | :--- | :--- | ---: | :--- | ---: | ---: | ---: |"
  rows = [header, sep]
  for r in results:
    rows.append(
        f"| {r.suite} | {r.problem_name} | {r.solver_name} | {r.batch_size} "
        f"| {r.converged} | {r.mean_obj:.4f} | {r.compile_s:.2f} "
        f"| {1e3 * r.run_s:.2f} |"
    )
  return "\n".join(rows)


def main() -> None:
  """CLI entry point for batched Proxax vs alpaqa benchmark."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--alpaqa-worker", action="store_true")
  parser.add_argument("--alpaqa-python", type=str, default=None)
  parser.add_argument("--tol", type=float, default=1e-6)
  parser.add_argument("--repeats", type=int, default=3)
  args = parser.parse_args()

  if args.alpaqa_worker:
    payload = json.load(sys.stdin)
    json.dump(_run_alpaqa_batch_worker(payload), sys.stdout)
    return

  import jax

  jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]
  from proxax.solvers import IALM
  from proxax.solvers import PANOC

  alpaqa_python = _detect_alpaqa_python(args.alpaqa_python)
  results: list[BatchAlpaqaResult] = []
  for case in build_cases():
    if case.suite == "box_qp_batch":
      solvers: list[tuple[str, typing.Any, str, int]] = [
          ("proxax-PANOC(vmap)", PANOC(opt_tol=args.tol), "vmap", 1),
          ("proxax-PANOC(threads)", PANOC(opt_tol=args.tol), "threads", 1),
      ]
      if case.batch_size > 1:
        solvers.append((
            "proxax-PANOC(threads,c=4)",
            PANOC(opt_tol=args.tol),
            "threads",
            4,
        ))
    else:
      solvers = [
          (
              "proxax-IALM(vmap)",
              IALM(opt_tol=args.tol, feas_tol=args.tol),
              "vmap",
              1,
          ),
          (
              "proxax-IALM(threads)",
              IALM(opt_tol=args.tol, feas_tol=args.tol),
              "threads",
              1,
          ),
      ]
      if case.batch_size > 1:
        solvers.append((
            "proxax-IALM(threads,c=4)",
            IALM(opt_tol=args.tol, feas_tol=args.tol),
            "threads",
            4,
        ))
    for s_name, solver, backend, chunk_size in solvers:
      results.append(
          _solve_with_proxax_batch(
              case,
              solver_name=s_name,
              solver=solver,
              backend=backend,
              tol=args.tol,
              repeats=args.repeats,
              chunk_size=chunk_size,
          )
      )
    results.append(
        _solve_with_alpaqa_batch(
            case,
            tol=args.tol,
            repeats=args.repeats,
            alpaqa_python=alpaqa_python,
        )
    )

  print(format_markdown_table(results))


if __name__ == "__main__":
  main()
