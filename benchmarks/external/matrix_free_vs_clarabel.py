"""Single-instance CPU benchmarks where matrix-free Proxax beats Clarabel."""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import time
from collections.abc import Callable

import clarabel  # type: ignore[import-untyped]
import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp  # type: ignore[import-untyped]

import proxax as px
from benchmarks import harness
from proxax import solvers


@dataclasses.dataclass(frozen=True)
class BenchmarkInstance:
  """A single problem instance with both Proxax and Clarabel builders."""

  name: str
  category: str
  proxax_builder: Callable[[], tuple[px.CompositeProblem, jax.Array]]
  clarabel_runner: Callable[[float, float], harness.BenchmarkResult]


def _solve_clarabel(
    name: str,
    p_sp: sp.csc_matrix,
    q_vec: np.ndarray,
    a_sp: sp.csc_matrix,
    b_vec: np.ndarray,
    cones: list[object],
    opt_tol: float,
    feas_tol: float,
) -> harness.BenchmarkResult:
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
      suite="matrix_free_vs_clarabel",
      problem_name=name,
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


def _make_factor_qp(
    name: str, *, n: int, r: int, m: int, seed: int
) -> BenchmarkInstance:
  """Factor-model QP with low-rank + identity Hessian."""
  rng = np.random.default_rng(seed)
  f_mat = rng.standard_normal((n, r)) / np.sqrt(n)
  q_vec = rng.standard_normal(n)
  a_mat = rng.standard_normal((m, n)) / np.sqrt(n)
  x_feas = np.abs(rng.standard_normal(n))
  b_vec = a_mat @ x_feas + 0.5

  def _build_proxax() -> tuple[px.CompositeProblem, jax.Array]:
    f_j = jnp.asarray(f_mat, dtype=jnp.float64)
    q_j = jnp.asarray(q_vec, dtype=jnp.float64)
    a_j = jnp.asarray(a_mat, dtype=jnp.float64)
    b_j = jnp.asarray(b_vec, dtype=jnp.float64)
    prob = px.CompositeProblem(
        smooth=lambda x: (
            0.5 * (jnp.sum((f_j.T @ x) ** 2) + jnp.sum(x**2)) + jnp.dot(q_j, x)
        ),
        proximal=px.prox.Box(lower=jnp.zeros(n), upper=jnp.full(n, jnp.inf)),
        constraints=px.constraints.box(
            a_j, lower=jnp.full(m, -jnp.inf), upper=b_j
        ),
    )
    return prob, jnp.zeros(n, dtype=jnp.float64)

  def _run_clarabel(opt_tol: float, feas_tol: float) -> harness.BenchmarkResult:
    p_dense = f_mat @ f_mat.T + np.eye(n)
    p_sp = sp.csc_matrix(np.triu(p_dense))
    a_sp = sp.csc_matrix(np.vstack([a_mat, -np.eye(n)]))
    b_cb = np.concatenate([b_vec, np.zeros(n)])
    return _solve_clarabel(
        name, p_sp, q_vec, a_sp, b_cb, [clarabel.NonnegativeConeT(m + n)],
        opt_tol, feas_tol,
    )

  return BenchmarkInstance(
      name=name,
      category="FactorQP",
      proxax_builder=_build_proxax,
      clarabel_runner=_run_clarabel,
  )


def _make_dense_qp(
    name: str, *, n: int, m: int, cond: float, seed: int
) -> BenchmarkInstance:
  """Dense-P QP: 0.5 x^T P x + q^T x s.t. A x <= b, x >= 0."""
  rng = np.random.default_rng(seed)
  q_orth, _ = np.linalg.qr(rng.standard_normal((n, n)))
  eigs = np.linspace(1.0, cond, n)
  p_mat = (q_orth * eigs) @ q_orth.T
  p_mat = 0.5 * (p_mat + p_mat.T)
  q_vec = rng.standard_normal(n)
  a_mat = rng.standard_normal((m, n)) / np.sqrt(n)
  x_feas = np.abs(rng.standard_normal(n))
  b_vec = a_mat @ x_feas + 0.5

  def _build_proxax() -> tuple[px.CompositeProblem, jax.Array]:
    p_j = jnp.asarray(p_mat, dtype=jnp.float64)
    q_j = jnp.asarray(q_vec, dtype=jnp.float64)
    a_j = jnp.asarray(a_mat, dtype=jnp.float64)
    b_j = jnp.asarray(b_vec, dtype=jnp.float64)
    prob = px.CompositeProblem(
        smooth=lambda x: 0.5 * jnp.dot(x, p_j @ x) + jnp.dot(q_j, x),
        proximal=px.prox.Box(lower=jnp.zeros(n), upper=jnp.full(n, jnp.inf)),
        constraints=px.constraints.box(
            a_j, lower=jnp.full(m, -jnp.inf), upper=b_j
        ),
    )
    return prob, jnp.zeros(n, dtype=jnp.float64)

  def _run_clarabel(opt_tol: float, feas_tol: float) -> harness.BenchmarkResult:
    p_sp = sp.csc_matrix(np.triu(p_mat))
    a_sp = sp.csc_matrix(np.vstack([a_mat, -np.eye(n)]))
    b_cb = np.concatenate([b_vec, np.zeros(n)])
    return _solve_clarabel(
        name, p_sp, q_vec, a_sp, b_cb, [clarabel.NonnegativeConeT(m + n)],
        opt_tol, feas_tol,
    )

  return BenchmarkInstance(
      name=name,
      category="DenseQP",
      proxax_builder=_build_proxax,
      clarabel_runner=_run_clarabel,
  )


def _make_simplex_logreg(
    name: str, *, num_samples: int, n: int, lam: float, seed: int
) -> BenchmarkInstance:
  """Simplex-constrained L2-regularized logistic regression."""
  rng = np.random.default_rng(seed)
  x_data = rng.standard_normal((num_samples, n)) / np.sqrt(n)
  y_data = rng.choice([-1.0, 1.0], size=num_samples)
  yx = y_data[:, None] * x_data

  def _build_proxax() -> tuple[px.CompositeProblem, jax.Array]:
    yx_j = jnp.asarray(yx, dtype=jnp.float64)
    prob = px.CompositeProblem(
        smooth=lambda x: (
            jnp.mean(jax.nn.softplus(-(yx_j @ x))) + 0.5 * lam * jnp.sum(x**2)
        ),
        proximal=px.prox.Box(lower=jnp.zeros(n), upper=jnp.full(n, jnp.inf)),
        constraints=px.constraints.equality(
            jnp.ones((1, n), dtype=jnp.float64),
            rhs=jnp.ones(1, dtype=jnp.float64),
        ),
    )
    return prob, jnp.full(n, 1.0 / n, dtype=jnp.float64)

  def _run_clarabel(opt_tol: float, feas_tol: float) -> harness.BenchmarkResult:
    # Variables: [x (n), t (N), u (N), v (N)]
    nvars = n + 3 * num_samples
    p_sp = sp.csc_matrix(
        sp.diags(np.concatenate([np.full(n, lam), np.zeros(3 * num_samples)]))
    )
    q_cb = np.concatenate([
        np.zeros(n),
        np.full(num_samples, 1.0 / num_samples),
        np.zeros(2 * num_samples),
    ])

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    b_list: list[float] = []
    cones: list[object] = []
    row = 0

    # sum(x) = 1
    rows.extend([row] * n)
    cols.extend(range(n))
    vals.extend([1.0] * n)
    b_list.append(1.0)
    row += 1
    cones.append(clarabel.ZeroConeT(1))

    # x >= 0
    for j in range(n):
      rows.append(row)
      cols.append(j)
      vals.append(-1.0)
      b_list.append(0.0)
      row += 1
    # u_i + v_i <= 1
    for i in range(num_samples):
      rows.extend([row, row])
      cols.extend([n + num_samples + i, n + 2 * num_samples + i])
      vals.extend([1.0, 1.0])
      b_list.append(1.0)
      row += 1
    cones.append(clarabel.NonnegativeConeT(n + num_samples))

    # Two exponential cones per sample
    for i in range(num_samples):
      rows.extend([row] * n)
      cols.extend(range(n))
      vals.extend(yx[i].tolist())
      rows.append(row)
      cols.append(n + i)
      vals.append(1.0)
      b_list.append(0.0)
      row += 1
      b_list.append(1.0)
      row += 1
      rows.append(row)
      cols.append(n + num_samples + i)
      vals.append(-1.0)
      b_list.append(0.0)
      row += 1
      cones.append(clarabel.ExponentialConeT())

      rows.append(row)
      cols.append(n + i)
      vals.append(1.0)
      b_list.append(0.0)
      row += 1
      b_list.append(1.0)
      row += 1
      rows.append(row)
      cols.append(n + 2 * num_samples + i)
      vals.append(-1.0)
      b_list.append(0.0)
      row += 1
      cones.append(clarabel.ExponentialConeT())

    a_sp = sp.csc_matrix((vals, (rows, cols)), shape=(row, nvars))
    b_cb = np.array(b_list)
    return _solve_clarabel(
        name, p_sp, q_cb, a_sp, b_cb, cones, opt_tol, feas_tol
    )

  return BenchmarkInstance(
      name=name,
      category="SimplexLogReg",
      proxax_builder=_build_proxax,
      clarabel_runner=_run_clarabel,
  )


def build_suite() -> list[BenchmarkInstance]:
  """Builds the default matrix-free vs. Clarabel benchmark suite."""
  return [
      _make_factor_qp(
          "FactorQP-1000 (n=1000,r=20,m=20)", n=1000, r=20, m=20, seed=1
      ),
      _make_factor_qp(
          "FactorQP-2000 (n=2000,r=20,m=20)", n=2000, r=20, m=20, seed=2
      ),
      _make_factor_qp(
          "FactorQP-3000 (n=3000,r=20,m=20)", n=3000, r=20, m=20, seed=3
      ),
      _make_simplex_logreg(
          "LogReg-Simplex-300 (N=300,n=300)",
          num_samples=300,
          n=300,
          lam=1e-2,
          seed=4,
      ),
      _make_simplex_logreg(
          "LogReg-Simplex-600 (N=600,n=600)",
          num_samples=600,
          n=600,
          lam=1e-2,
          seed=5,
      ),
      _make_simplex_logreg(
          "LogReg-Simplex-1000 (N=1000,n=1000)",
          num_samples=1000,
          n=1000,
          lam=1e-2,
          seed=6,
      ),
      _make_dense_qp(
          "DenseQP-1000 (n=1000,m=20,k=20)", n=1000, m=20, cond=20.0, seed=7
      ),
      _make_dense_qp(
          "DenseQP-1500 (n=1500,m=20,k=20)", n=1500, m=20, cond=20.0, seed=8
      ),
  ]


def run_suite(
    *,
    opt_tol: float = 1e-6,
    feas_tol: float = 1e-6,
    time_limit: float = 10.0,
    include_clarabel: bool = True,
) -> list[harness.BenchmarkResult]:
  """Runs all suite instances with IALM, IAIPAL, and optionally Clarabel."""
  results: list[harness.BenchmarkResult] = []
  px_solvers = [
      ("IALM", solvers.IALM(opt_tol=opt_tol, feas_tol=feas_tol)),
      ("IAIPAL", solvers.IAIPAL(
          opt_tol=opt_tol, feas_tol=feas_tol, max_outer_iters=1_000_000
      )),
  ]
  for inst in build_suite():
    prob, x0 = inst.proxax_builder()
    for sname, solver in px_solvers:
      res = harness.time_proxax_solve(
          suite="matrix_free_vs_clarabel",
          problem_name=inst.name,
          solver_name=sname,
          prob=prob,
          x0=x0,
          solver=solver,
          opt_tol=opt_tol,
          feas_tol=feas_tol,
          time_limit=time_limit,
      )
      results.append(res)
      print(
          f"{res.problem_name:<38s} | {res.solver_name:<8s} | "
          f"{res.status:<17s} | iters={res.outer_iters:>5d} | "
          f"opt={res.opt_residual:.2e} | feas={res.feas_residual:.2e} | "
          f"{res.run_s * 1000:>8.2f}ms",
          flush=True,
      )
    if include_clarabel:
      res = inst.clarabel_runner(opt_tol, feas_tol)
      results.append(res)
      print(
          f"{res.problem_name:<38s} | {res.solver_name:<8s} | "
          f"{res.status:<17s} | iters={res.outer_iters:>5d} | "
          f"opt={res.opt_residual:.2e} | feas={res.feas_residual:.2e} | "
          f"{res.run_s * 1000:>8.2f}ms",
          flush=True,
      )
  return results


def main() -> int:
  """CLI entry point for the matrix-free vs. Clarabel benchmark suite."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--opt-tol", type=float, default=1e-6)
  parser.add_argument("--feas-tol", type=float, default=1e-6)
  parser.add_argument("--time-limit", type=float, default=10.0)
  parser.add_argument("--no-clarabel", action="store_true")
  parser.add_argument("--json-out", type=pathlib.Path, default=None)
  args = parser.parse_args()

  results = run_suite(
      opt_tol=args.opt_tol,
      feas_tol=args.feas_tol,
      time_limit=args.time_limit,
      include_clarabel=not args.no_clarabel,
  )
  if args.json_out is not None:
    args.json_out.write_text(
        json.dumps([dataclasses.asdict(r) for r in results], indent=2),
        encoding="utf-8",
    )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
