"""Shared Clarabel, SCS, and CVXPY runner helpers for external benchmarks."""

from __future__ import annotations

import time
import typing

import clarabel
import cvxpy as cp
import numpy as np
import scipy.sparse as sp
import scs

from benchmarks import harness


def solve_clarabel(
    *,
    suite: str,
    problem_name: str,
    p_sp: sp.csc_matrix,
    q_vec: np.ndarray,
    a_sp: sp.csc_matrix,
    b_vec: np.ndarray,
    cones: list[typing.Any],
    opt_tol: float,
    feas_tol: float,
) -> harness.BenchmarkResult:
  """Solves a conic QP in Clarabel and returns a `BenchmarkResult`."""
  settings = clarabel.DefaultSettings()
  settings.verbose = False
  settings.tol_gap_abs = opt_tol
  settings.tol_gap_rel = opt_tol
  settings.tol_feas = feas_tol
  t0 = time.perf_counter()
  sol = clarabel.DefaultSolver(
      p_sp, q_vec, a_sp, b_vec, cones, settings
  ).solve()
  run_s = time.perf_counter() - t0
  solved = str(sol.status) in {"Solved", "AlmostSolved"}
  return harness.BenchmarkResult(
      suite=suite,
      problem_name=problem_name,
      solver_name="Clarabel",
      dtype="float64",
      status="CONVERGED" if solved else str(sol.status).upper(),
      converged=solved,
      outer_iters=int(sol.iterations),
      inner_iters=0,
      grad_evals=0,
      smooth_evals=0,
      prox_evals=0,
      compile_s=0.0,
      run_s=float(run_s),
      obj_val=float(sol.obj_val),
      opt_residual=float(getattr(sol, "r_dual", np.nan)),
      feas_residual=float(getattr(sol, "r_prim", np.nan)),
  )


def solve_scs(
    *,
    suite: str,
    problem_name: str,
    p_sp: sp.csc_matrix | None,
    q_vec: np.ndarray,
    a_sp: sp.csc_matrix,
    b_vec: np.ndarray,
    cone_dict: dict[str, typing.Any],
    tol: float,
    use_indirect: bool = False,
    max_iters: int = 50_000,
) -> harness.BenchmarkResult:
  """Solves a conic QP in SCS (direct or indirect) and returns a result."""
  data: dict[str, typing.Any] = {
      "A": a_sp,
      "b": b_vec,
      "c": q_vec,
  }
  if p_sp is not None:
    data["P"] = p_sp

  t0 = time.perf_counter()
  linear_solver = (
      scs.LinearSolver.CPU_INDIRECT if use_indirect else scs.LinearSolver.QDLDL
  )
  sol = scs.solve(
      data,
      cone_dict,
      eps_abs=tol,
      eps_rel=tol,
      max_iters=max_iters,
      linear_solver=linear_solver,
      verbose=False,
  )
  run_s = time.perf_counter() - t0
  info = sol["info"]
  status_str = str(info["status"])
  solved = status_str in {"solved", "solved/inaccurate"}
  solver_label = "SCS (indirect)" if use_indirect else "SCS (direct)"
  return harness.BenchmarkResult(
      suite=suite,
      problem_name=problem_name,
      solver_name=solver_label,
      dtype="float64",
      status="CONVERGED" if solved else status_str.upper(),
      converged=solved,
      outer_iters=int(info["iter"]),
      inner_iters=0,
      grad_evals=0,
      smooth_evals=0,
      prox_evals=0,
      compile_s=0.0,
      run_s=float(run_s),
      obj_val=float(info["pobj"]),
      opt_residual=float(info.get("res_dual", np.nan)),
      feas_residual=float(info.get("res_pri", np.nan)),
  )


def solve_cvxpy(
    *,
    suite: str,
    problem_name: str,
    prob: cp.Problem,
    solver_name: str,
    cvxpy_solver: str,
    solver_kwargs: dict[str, typing.Any] | None = None,
) -> harness.BenchmarkResult:
  """Solves a CVXPY problem and records end-to-end DSL + solver runtime."""
  kwargs = dict(solver_kwargs or {})
  t0 = time.perf_counter()
  prob.solve(solver=cvxpy_solver, verbose=False, **kwargs)
  run_s = time.perf_counter() - t0
  solved = prob.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
  iters = 0
  if prob.solver_stats is not None and prob.solver_stats.num_iters is not None:
    iters = int(prob.solver_stats.num_iters)
  return harness.BenchmarkResult(
      suite=suite,
      problem_name=problem_name,
      solver_name=solver_name,
      dtype="float64",
      status="CONVERGED" if solved else str(prob.status).upper(),
      converged=solved,
      outer_iters=iters,
      inner_iters=0,
      grad_evals=0,
      smooth_evals=0,
      prox_evals=0,
      compile_s=0.0,
      run_s=float(run_s),
      obj_val=float(prob.value) if prob.value is not None else float("nan"),
      opt_residual=float("nan"),
      feas_residual=float("nan"),
  )
