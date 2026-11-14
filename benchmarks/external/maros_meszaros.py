"""Maros-Meszaros QP benchmark runner for Proxax constrained solvers."""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import typing
import urllib.request

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks import harness
from benchmarks.external import qpsif
from proxax import px
from proxax import solvers
from proxax.solvers import base

DEFAULT_PROBLEMS = (
    "DUAL1",
    "DUAL2",
    "DUAL3",
    "DUAL4",
    "HS21",
    "HS35",
    "HS51",
    "HS52",
    "HS53",
    "HS76",
    "HS118",
    "HS268",
    "LOTSCHD",
    "QAFIRO",
    "TAME",
)

_SIF_BASE_URL = (
    "https://raw.githubusercontent.com/optimizers/"
    "maros-meszaros-mirror/master"
)


def _fetch_sif(
    name: str,
    cache_dir: pathlib.Path,
) -> pathlib.Path:
  """Returns local ``.SIF`` path, downloading into ``cache_dir`` if needed."""
  cache_dir.mkdir(parents=True, exist_ok=True)
  target = cache_dir / f"{name}.SIF"
  if not target.exists():
    url = f"{_SIF_BASE_URL}/{name}.SIF"
    urllib.request.urlretrieve(url, target)
  return target


def to_proxax_problem(
    qp: qpsif.QPSIFProblem,
) -> tuple[px.CompositeProblem, jax.Array]:
  """Converts a ``QPSIFProblem`` to a Proxax ``CompositeProblem`` and ``x0``."""
  dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
  p_mat = jnp.asarray(qp.P, dtype=dtype)
  q_vec = jnp.asarray(qp.q, dtype=dtype)

  def _smooth(x: jax.Array) -> jax.Array:
    return 0.5 * jnp.dot(x, p_mat @ x) + jnp.dot(q_vec, x) + qp.r

  lb_arr = jnp.asarray(qp.lb, dtype=dtype)
  ub_arr = jnp.asarray(qp.ub, dtype=dtype)
  if bool(np.all(np.isneginf(qp.lb)) and np.all(np.isposinf(qp.ub))):
    proximal: typing.Any = px.prox.Zero()
  else:
    proximal = px.prox.Box(lower=lb_arr, upper=ub_arr)

  constraints = None
  if qp.A.shape[0] > 0:
    a_mat = jnp.asarray(qp.A, dtype=dtype)
    if bool(np.allclose(qp.l_con, qp.u_con, rtol=0.0, atol=1e-12)):
      constraints = px.constraints.equality(
          a_mat, rhs=jnp.asarray(qp.l_con, dtype=dtype)
      )
    else:
      constraints = px.constraints.box(
          a_mat,
          lower=jnp.asarray(qp.l_con, dtype=dtype),
          upper=jnp.asarray(qp.u_con, dtype=dtype),
      )

  # Project zero onto variable box for a clean finite initial point.
  x0_np = np.clip(0.0, qp.lb, qp.ub)
  x0 = jnp.asarray(x0_np, dtype=dtype)
  return (
      px.CompositeProblem(
          smooth=_smooth,
          proximal=proximal,
          constraints=constraints,
      ),
      x0,
  )


def _qp_solvers(
    opt_tol: float,
    feas_tol: float,
) -> list[tuple[str, base.Solver]]:
  return [
      ("IALM", solvers.IALM(opt_tol=opt_tol, feas_tol=feas_tol)),
      ("IAIPAL", solvers.IAIPAL(
          opt_tol=opt_tol, feas_tol=feas_tol, max_outer_iters=1_000_000
      )),
      ("QPAPD", solvers.QPAPD(opt_tol=opt_tol, feas_tol=feas_tol)),
      ("QPAIPP", solvers.QPAIPP(opt_tol=opt_tol, feas_tol=feas_tol)),
      ("SProxALM", solvers.SProxALM(opt_tol=opt_tol, feas_tol=feas_tol)),
  ]


def run_maros_meszaros(
    *,
    problems: tuple[str, ...] = DEFAULT_PROBLEMS,
    data_dir: pathlib.Path | None = None,
    max_dim: int = 200,
    limit: int | None = None,
    opt_tol: float = 1e-4,
    feas_tol: float = 1e-4,
    solver_names: tuple[str, ...] | None = None,
    time_limit: float | None = None,
    precondition: bool = False,
) -> list[harness.BenchmarkResult]:
  """Runs Proxax constrained solvers on a curated Maros-Meszaros QP subset."""
  cache_dir = (
      data_dir
      if data_dir is not None
      else pathlib.Path.home() / ".cache" / "proxax" / "maros_meszaros"
  )
  results: list[harness.BenchmarkResult] = []
  ran = 0
  for name in problems:
    if limit is not None and ran >= limit:
      break
    sif_path = (
        cache_dir / f"{name}.SIF"
        if data_dir is not None
        else _fetch_sif(name, cache_dir)
    )
    qp = qpsif.load_qpsif(sif_path)
    n, m = qp.A.shape[1], qp.A.shape[0]
    if n > max_dim:
      continue
    ran += 1
    eff_opt_tol = opt_tol
    eff_feas_tol = feas_tol
    if precondition:
      qp, d_x, d_y, cost_scale = qpsif.ruiz_equilibrate(qp)
      if d_x.size > 0:
        eff_opt_tol = opt_tol * min(1.0, float(cost_scale * np.min(d_x)))
      if d_y.size > 0:
        eff_feas_tol = feas_tol * min(1.0, float(np.min(d_y)))
    prob, x0 = to_proxax_problem(qp)
    solvers_list = (
        [("APD", solvers.APD(opt_tol=eff_opt_tol))]
        if prob.constraints is None
        else _qp_solvers(eff_opt_tol, eff_feas_tol)
    )
    if solver_names is not None:
      allowed = set(solver_names)
      solvers_list = [(k, v) for k, v in solvers_list if k in allowed]
    for solver_name, solver in solvers_list:
      res = harness.time_proxax_solve(
          suite="maros_meszaros",
          problem_name=f"{qp.name} (n={n},m={m})",
          solver_name=solver_name,
          prob=prob,
          x0=x0,
          solver=solver,
          opt_tol=eff_opt_tol,
          feas_tol=None if prob.constraints is None else eff_feas_tol,
          time_limit=time_limit,
      )
      results.append(res)
      print(
          f"{res.problem_name:<24s} | {res.solver_name:<9s} | "
          f"{res.status:<17s} | iters={res.outer_iters:>4d} | "
          f"opt={res.opt_residual:.2e} | feas={res.feas_residual:.2e} | "
          f"{res.run_s * 1000:>7.2f}ms",
          flush=True,
      )
  return results


def run_clarabel_baseline(
    qp: qpsif.QPSIFProblem,
    *,
    opt_tol: float,
    feas_tol: float,
) -> harness.BenchmarkResult:
  """Runs Clarabel on a ``QPSIFProblem`` as an IPM baseline."""
  import clarabel  # type: ignore[import-untyped]
  import scipy.sparse as sp  # type: ignore[import-untyped]

  n = qp.P.shape[0]
  m = qp.A.shape[0]
  p_csc = sp.csc_matrix(np.triu(qp.P))
  q_vec = np.asarray(qp.q, dtype=np.float64)

  eq_a: list[np.ndarray] = []
  eq_b: list[float] = []
  ineq_a: list[np.ndarray] = []
  ineq_b: list[float] = []

  for i in range(m):
    lc, uc = float(qp.l_con[i]), float(qp.u_con[i])
    row = qp.A[i]
    if np.isfinite(lc) and np.isfinite(uc) and abs(uc - lc) <= 1e-12:
      eq_a.append(row)
      eq_b.append(lc)
    else:
      if np.isfinite(uc):
        ineq_a.append(row)
        ineq_b.append(uc)
      if np.isfinite(lc):
        ineq_a.append(-row)
        ineq_b.append(-lc)

  for j in range(n):
    lb, ub = float(qp.lb[j]), float(qp.ub[j])
    if np.isfinite(lb) and np.isfinite(ub) and abs(ub - lb) <= 1e-12:
      e = np.zeros(n)
      e[j] = 1.0
      eq_a.append(e)
      eq_b.append(lb)
    else:
      if np.isfinite(ub):
        e = np.zeros(n)
        e[j] = 1.0
        ineq_a.append(e)
        ineq_b.append(ub)
      if np.isfinite(lb):
        e = np.zeros(n)
        e[j] = -1.0
        ineq_a.append(e)
        ineq_b.append(-lb)

  a_blocks, b_blocks, cones = [], [], []
  if eq_a:
    a_blocks.append(np.vstack(eq_a))
    b_blocks.append(np.array(eq_b, dtype=np.float64))
    cones.append(clarabel.ZeroConeT(len(eq_b)))
  if ineq_a:
    a_blocks.append(np.vstack(ineq_a))
    b_blocks.append(np.array(ineq_b, dtype=np.float64))
    cones.append(clarabel.NonnegativeConeT(len(ineq_b)))

  a_cl = (
      sp.csc_matrix(np.vstack(a_blocks))
      if a_blocks
      else sp.csc_matrix((0, n))
  )
  b_cl = np.concatenate(b_blocks) if b_blocks else np.zeros(0, dtype=np.float64)

  settings = clarabel.DefaultSettings()
  settings.verbose = False
  settings.tol_gap_abs = opt_tol
  settings.tol_gap_rel = opt_tol
  settings.tol_feas = feas_tol

  times: list[float] = []
  sol = None
  for run_idx in range(6):
    solver = clarabel.DefaultSolver(p_csc, q_vec, a_cl, b_cl, cones, settings)
    sol = solver.solve()
    if run_idx > 0:
      times.append(float(sol.solve_time))

  assert sol is not None
  status = "CONVERGED" if str(sol.status) == "Solved" else str(sol.status)
  return harness.BenchmarkResult(
      suite="maros_meszaros",
      problem_name=f"{qp.name} (n={n},m={m})",
      solver_name="Clarabel",
      dtype="float64",
      status=status,
      converged=(status == "CONVERGED"),
      obj_val=float(sol.obj_val + qp.r),
      outer_iters=int(sol.iterations),
      inner_iters=0,
      smooth_evals=0,
      grad_evals=0,
      prox_evals=0,
      opt_residual=float(sol.r_dual),
      feas_residual=float(sol.r_prim),
      compile_s=0.0,
      run_s=float(np.median(times)),
  )


def main() -> int:
  """CLI entrypoint for the Maros-Meszaros QP benchmark suite."""
  parser = argparse.ArgumentParser(
      description="Run Proxax constrained solvers on Maros-Meszaros QPs."
  )
  parser.add_argument(
      "--problems",
      nargs="+",
      default=list(DEFAULT_PROBLEMS),
      help="Problem names (without .SIF extension).",
  )
  parser.add_argument(
      "--data-dir",
      type=pathlib.Path,
      default=None,
      help="Optional local directory containing .SIF files.",
  )
  parser.add_argument("--max-dim", type=int, default=200)
  parser.add_argument("--limit", type=int, default=None)
  parser.add_argument("--opt-tol", type=float, default=1e-4)
  parser.add_argument("--feas-tol", type=float, default=1e-4)
  parser.add_argument("--time-limit", type=float, default=None)
  parser.add_argument(
      "--solvers",
      nargs="+",
      default=None,
      help="Optional subset of Proxax solver names to run.",
  )
  parser.add_argument(
      "--precondition",
      action="store_true",
      help="Apply Modified Ruiz Equilibration before solving.",
  )
  parser.add_argument(
      "--clarabel",
      action="store_true",
      help="Also run Clarabel as an IPM baseline reference.",
  )
  parser.add_argument("--json-out", type=pathlib.Path, default=None)
  args = parser.parse_args()

  results = run_maros_meszaros(
      problems=tuple(args.problems),
      data_dir=args.data_dir,
      max_dim=args.max_dim,
      limit=args.limit,
      opt_tol=args.opt_tol,
      feas_tol=args.feas_tol,
      solver_names=None if args.solvers is None else tuple(args.solvers),
      time_limit=args.time_limit,
      precondition=args.precondition,
  )
  if args.clarabel:
    cache_dir = args.data_dir or (
        pathlib.Path.home() / ".cache" / "proxax" / "maros_meszaros"
    )
    for name in args.problems[: args.limit]:
      qp = qpsif.load_qpsif(_fetch_sif(name, cache_dir))
      if qp.P.shape[0] > args.max_dim:
        continue
      res = run_clarabel_baseline(
          qp, opt_tol=args.opt_tol, feas_tol=args.feas_tol
      )
      results.append(res)
      print(
          f"{res.problem_name:<24s} | {res.solver_name:<9s} | "
          f"{res.status:<17s} | iters={res.outer_iters:>4d} | "
          f"opt={res.opt_residual:.2e} | feas={res.feas_residual:.2e} | "
          f"{res.run_s * 1000:>7.2f}ms",
          flush=True,
      )
  if args.json_out is not None:
    args.json_out.write_text(
        json.dumps([dataclasses.asdict(r) for r in results], indent=2),
        encoding="utf-8",
    )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
