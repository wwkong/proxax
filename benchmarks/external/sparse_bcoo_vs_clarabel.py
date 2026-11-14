"""Large-scale sparse BCOO vs. dense vs. Clarabel CPU benchmark.

Compares `proxax-IALM(BCOO)` and `proxax-IALM(dense)` against `Clarabel` on
large-scale sparse constrained problems (`n in {1000, 2000, 4000}`, ~1% nnz):
1. `sparse_qp`: diagonal-Hessian QP with sparse inequalities `C x <= d, x >= 0`.
2. `sparse_logreg`: sparse L2-regularized logistic regression with sparse
   inequalities `C x <= d, -1 <= x <= 1`.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import time
import typing

import numpy as np
import scipy.sparse as sp


@dataclasses.dataclass(frozen=True)
class SparseCase:
  """Problem data for a large-scale sparse constrained benchmark."""

  suite: str
  name: str
  n: int
  m: int
  density: float
  diag_h: np.ndarray
  q: np.ndarray
  c_csc: sp.csc_matrix
  d: np.ndarray
  a_csc: sp.csc_matrix | None = None
  y_label: np.ndarray | None = None
  lam: float = 1e-2


@dataclasses.dataclass(frozen=True)
class SparseResult:
  """Single solver benchmark result."""

  suite: str
  problem_name: str
  solver_name: str
  n: int
  converged: bool
  objective: float
  compile_s: float
  run_s: float


def _make_sparse_matrix(
    rng: np.random.Generator, rows: int, cols: int, density: float
) -> sp.csc_matrix:
  nnz = max(rows, int(round(rows * cols * density)))
  r_idx = rng.integers(0, rows, size=nnz)
  c_idx = rng.integers(0, cols, size=nnz)
  vals = rng.standard_normal(nnz) / np.sqrt(max(1.0, density * cols))
  mat = sp.coo_matrix((vals, (r_idx, c_idx)), shape=(rows, cols)).tocsc()
  mat.sum_duplicates()
  return mat


def build_cases() -> list[SparseCase]:
  """Constructs sparse QP and sparse constrained logistic cases."""
  rng = np.random.default_rng(7)
  cases: list[SparseCase] = []
  density = 0.01

  for n in (1000, 2000, 4000):
    m = n // 2
    c_csc = _make_sparse_matrix(rng, m, n, density)
    diag_h = rng.uniform(0.5, 2.0, size=n)
    x_ref = rng.uniform(0.05, 0.3, size=n)
    d_vec = c_csc @ x_ref + 0.25
    q_vec = rng.normal(0.0, 0.5, size=n)
    cases.append(
        SparseCase(
            suite="sparse_qp",
            name=f"sparse_qp_n{n}",
            n=n,
            m=m,
            density=density,
            diag_h=diag_h,
            q=q_vec,
            c_csc=c_csc,
            d=d_vec,
        )
    )

  for n in (1000, 2000, 4000):
    m_data = n // 2
    m_con = n // 4
    a_csc = _make_sparse_matrix(rng, m_data, n, density)
    c_csc = _make_sparse_matrix(rng, m_con, n, density)
    w_true = rng.normal(0.0, 0.3, size=n)
    y_lbl = np.where((a_csc @ w_true) >= 0.0, 1.0, -1.0)
    x_feas = np.clip(w_true, -0.5, 0.5)
    d_vec = c_csc @ x_feas + 0.3
    cases.append(
        SparseCase(
            suite="sparse_logreg",
            name=f"sparse_logreg_n{n}",
            n=n,
            m=m_data,
            density=density,
            diag_h=np.ones(n, dtype=np.float64),
            q=np.zeros(n, dtype=np.float64),
            c_csc=c_csc,
            d=d_vec,
            a_csc=a_csc,
            y_label=y_lbl,
            lam=1e-2,
        )
    )
  return cases


def _scipy_to_bcoo(mat_csc: sp.csc_matrix) -> typing.Any:
  import jax.numpy as jnp
  from jax.experimental import sparse as jsparse

  coo = mat_csc.tocoo()
  indices = jnp.asarray(
      np.column_stack([coo.row, coo.col]), dtype=jnp.int32
  )
  data = jnp.asarray(coo.data, dtype=jnp.float64)
  return jsparse.BCOO((data, indices), shape=coo.shape).sum_duplicates()


def _solve_proxax(
    case: SparseCase, *, use_bcoo: bool, tol: float, repeats: int
) -> SparseResult:
  import jax
  import jax.numpy as jnp

  from proxax import px
  from proxax.solvers import IALM

  c_op: typing.Any = (
      _scipy_to_bcoo(case.c_csc)
      if use_bcoo
      else jnp.asarray(case.c_csc.toarray(), dtype=jnp.float64)
  )
  d_j = jnp.asarray(case.d, dtype=jnp.float64)
  constraints = px.constraints.less_equal(c_op, d_j)

  if case.suite == "sparse_qp":
    diag_j = jnp.asarray(case.diag_h, dtype=jnp.float64)
    q_j = jnp.asarray(case.q, dtype=jnp.float64)
    x0 = jnp.zeros(case.n, dtype=jnp.float64)
    prox_op: typing.Any = px.prox.NonNegative()

    def smooth_fn(x: jax.Array) -> jax.Array:
      return 0.5 * jnp.sum(diag_j * x * x) + jnp.dot(q_j, x)

  else:
    assert case.a_csc is not None and case.y_label is not None
    a_op: typing.Any = (
        _scipy_to_bcoo(case.a_csc)
        if use_bcoo
        else jnp.asarray(case.a_csc.toarray(), dtype=jnp.float64)
    )
    y_j = jnp.asarray(case.y_label, dtype=jnp.float64)
    x0 = jnp.zeros(case.n, dtype=jnp.float64)
    prox_op = px.prox.Box(lower=-1.0, upper=1.0)
    lam = float(case.lam)

    def smooth_fn(x: jax.Array) -> jax.Array:
      margins = y_j * (a_op @ x)
      return jnp.mean(jnp.logaddexp(0.0, -margins)) + 0.5 * lam * jnp.dot(x, x)

  prob = px.CompositeProblem(
      smooth=smooth_fn, proximal=prox_op, constraints=constraints
  )
  solver = IALM(opt_tol=tol, feas_tol=tol)

  @jax.jit
  def _run(init_x: jax.Array) -> tuple[jax.Array, jax.Array]:
    sol = solver.solve(prob, init_x, jit=True, time_limit=30.0)
    return sol.x, sol.status

  t0 = time.perf_counter()
  x_sol, status = _run(x0)
  x_sol.block_until_ready()
  compile_s = time.perf_counter() - t0

  run_times: list[float] = []
  for _ in range(max(1, repeats)):
    t1 = time.perf_counter()
    x_sol, status = _run(x0)
    x_sol.block_until_ready()
    run_times.append(time.perf_counter() - t1)

  obj_val = float(smooth_fn(x_sol))
  conv = int(status) == int(px.SolveStatus.CONVERGED)
  name = "proxax-IALM(BCOO)" if use_bcoo else "proxax-IALM(dense)"
  return SparseResult(
      suite=case.suite,
      problem_name=case.name,
      solver_name=name,
      n=case.n,
      converged=conv,
      objective=obj_val,
      compile_s=compile_s,
      run_s=min(run_times),
  )


def _solve_clarabel(
    case: SparseCase, *, tol: float, repeats: int
) -> SparseResult:
  import clarabel

  n = case.n
  if case.suite == "sparse_qp":
    p_sp = sp.diags(case.diag_h, format="csc", dtype=np.float64)
    q_np = case.q.astype(np.float64)
    m_c = case.c_csc.shape[0]
    a_sp = sp.vstack(
        [case.c_csc, -sp.eye(n, format="csc", dtype=np.float64)],
        format="csc",
    )
    b_np = np.concatenate([case.d, np.zeros(n, dtype=np.float64)])
    cones: list[typing.Any] = [clarabel.NonnegativeConeT(m_c + n)]
  else:
    assert case.a_csc is not None and case.y_label is not None
    m_d = case.a_csc.shape[0]
    m_c = case.c_csc.shape[0]
    # Variables: [x (n), t (m_d), u (m_d), v (m_d)]
    # Softplus: u + v <= 1, (-(y * Ax) - t, 1, u) in K_exp, (-t, 1, v) in K_exp
    p_sp = sp.block_diag(
        [
            case.lam * sp.eye(n, format="csc", dtype=np.float64),
            sp.csc_matrix((3 * m_d, 3 * m_d), dtype=np.float64),
        ],
        format="csc",
    )
    q_np = np.concatenate([
        np.zeros(n, dtype=np.float64),
        np.full(m_d, 1.0 / m_d, dtype=np.float64),
        np.zeros(2 * m_d, dtype=np.float64),
    ])
    row_c = sp.hstack(
        [case.c_csc, sp.csc_matrix((m_c, 3 * m_d), dtype=np.float64)],
        format="csc",
    )
    row_box_hi = sp.hstack(
        [
            sp.eye(n, format="csc", dtype=np.float64),
            sp.csc_matrix((n, 3 * m_d), dtype=np.float64),
        ],
        format="csc",
    )
    row_box_lo = sp.hstack(
        [
            -sp.eye(n, format="csc", dtype=np.float64),
            sp.csc_matrix((n, 3 * m_d), dtype=np.float64),
        ],
        format="csc",
    )
    i_m = sp.eye(m_d, format="csc", dtype=np.float64)
    z_m_n = sp.csc_matrix((m_d, n), dtype=np.float64)
    z_m_m = sp.csc_matrix((m_d, m_d), dtype=np.float64)
    row_uv = sp.hstack([z_m_n, z_m_m, i_m, i_m], format="csc")
    yx = sp.diags(case.y_label, format="csc") @ case.a_csc
    exp_rows: list[sp.csc_matrix] = []
    for i in range(m_d):
      e_i = sp.csc_matrix(
          ([1.0], ([0], [i])), shape=(1, m_d), dtype=np.float64
      )
      z1m = sp.csc_matrix((1, m_d), dtype=np.float64)
      z1n = sp.csc_matrix((1, n), dtype=np.float64)
      exp_rows.extend([
          sp.hstack([yx[i : i + 1, :], e_i, z1m, z1m], format="csc"),
          sp.csc_matrix((1, n + 3 * m_d), dtype=np.float64),
          sp.hstack([z1n, z1m, -e_i, z1m], format="csc"),
          sp.hstack([z1n, e_i, z1m, z1m], format="csc"),
          sp.csc_matrix((1, n + 3 * m_d), dtype=np.float64),
          sp.hstack([z1n, z1m, z1m, -e_i], format="csc"),
      ])
    a_sp = sp.vstack(
        [row_c, row_box_hi, row_box_lo, row_uv, *exp_rows], format="csc"
    )
    b_exp = np.tile(np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0]), m_d)
    b_np = np.concatenate([
        case.d,
        np.ones(n, dtype=np.float64),
        np.ones(n, dtype=np.float64),
        np.ones(m_d, dtype=np.float64),
        b_exp,
    ])
    cones = [
        clarabel.NonnegativeConeT(m_c + 2 * n + m_d),
        *[clarabel.ExponentialConeT() for _ in range(2 * m_d)],
    ]

  settings = clarabel.DefaultSettings()
  settings.verbose = False
  settings.tol_gap_abs = tol
  settings.tol_gap_rel = tol
  settings.tol_feas = tol
  settings.time_limit = 30.0

  run_times: list[float] = []
  sol_obj = float("nan")
  conv = False
  for _ in range(max(1, repeats)):
    t0 = time.perf_counter()
    solver = clarabel.DefaultSolver(p_sp, q_np, a_sp, b_np, cones, settings)
    sol = solver.solve()
    run_times.append(time.perf_counter() - t0)
    conv = sol.status == clarabel.SolverStatus.Solved
    sol_obj = float(sol.obj_val)

  return SparseResult(
      suite=case.suite,
      problem_name=case.name,
      solver_name="Clarabel",
      n=case.n,
      converged=conv,
      objective=sol_obj,
      compile_s=0.0,
      run_s=min(run_times),
  )


def format_markdown_table(results: list[SparseResult]) -> str:
  """Formats benchmark results as a Markdown table."""
  lines = [
      (
          "| Suite | Problem | Solver | n | Converged | Objective "
          "| Compile (s) | Run (ms) |"
      ),
      "| :--- | :--- | :--- | ---: | :--- | ---: | ---: | ---: |",
  ]
  for r in results:
    lines.append(
        f"| {r.suite} | {r.problem_name} | {r.solver_name} | {r.n} "
        f"| {r.converged} | {r.objective:.4f} | {r.compile_s:.2f} "
        f"| {r.run_s * 1000:.2f} |"
    )
  return "\n".join(lines)


def main() -> None:
  """CLI entry point for large-scale sparse BCOO vs. Clarabel benchmarks."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--tol", type=float, default=1e-6)
  parser.add_argument("--repeats", type=int, default=2)
  args = parser.parse_args()

  os.environ.setdefault("JAX_ENABLE_X64", "1")
  import jax

  jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

  results: list[SparseResult] = []
  for case in build_cases():
    results.append(
        _solve_proxax(case, use_bcoo=True, tol=args.tol, repeats=args.repeats)
    )
    results.append(
        _solve_proxax(case, use_bcoo=False, tol=args.tol, repeats=args.repeats)
    )
    results.append(
        _solve_clarabel(case, tol=args.tol, repeats=args.repeats)
    )
  print(format_markdown_table(results))


if __name__ == "__main__":
  main()
