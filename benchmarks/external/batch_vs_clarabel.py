"""Batched benchmark comparing Proxax (vmap / threads) vs. Clarabel (threads).

Covers two convex problem families across batch sizes B in {1, 16, 64}:
  1. ``factor_qp_batch``: Factor-model QP (n=500, r=20, m=20) with varying q_i.
  2. ``logreg_simplex_batch``: Simplex logistic regression (N=200, n=200) with
     varying label vectors y_i.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import time
import typing

import clarabel
import jax
import jax.numpy as jnp
import numpy as np
from scipy import sparse

from proxax import px
from proxax.solvers import IAIPAL
from proxax.solvers import IALM
from proxax.solvers import PANOC


@dataclasses.dataclass(frozen=True)
class BatchedClarabelCase:
  """Problem data for a batched convex problem."""

  suite: str
  name: str
  n: int
  m: int
  batch_size: int
  F: np.ndarray | None
  A: np.ndarray
  b: np.ndarray
  params_batch: np.ndarray


@dataclasses.dataclass(frozen=True)
class BatchClarabelResult:
  """Recorded timing and convergence metrics."""

  suite: str
  problem_name: str
  solver_name: str
  batch_size: int
  converged: bool
  mean_obj: float
  compile_s: float
  run_s: float


def build_cases(seed: int = 42) -> list[BatchedClarabelCase]:
  """Builds batched FactorQP and LogReg-Simplex cases."""
  rng = np.random.default_rng(seed)
  cases: list[BatchedClarabelCase] = []

  n_f, r_f, m_f = 500, 20, 20
  F = rng.standard_normal((n_f, r_f)) / np.sqrt(n_f)
  A_f = rng.standard_normal((m_f, n_f)) / np.sqrt(n_f)
  x_feas = rng.uniform(0.1, 0.5, size=n_f)
  b_f = A_f @ x_feas + 0.5
  for bs in (1, 16, 64):
    q_b = rng.standard_normal((bs, n_f))
    cases.append(
        BatchedClarabelCase(
            suite="factor_qp_batch",
            name=f"factor_qp_n{n_f}_B{bs}",
            n=n_f,
            m=m_f,
            batch_size=bs,
            F=F,
            A=A_f,
            b=b_f,
            params_batch=q_b,
        )
    )

  N_l, n_l = 200, 200
  X_l = rng.standard_normal((N_l, n_l)) / np.sqrt(n_l)
  x_true = np.full(n_l, 1.0 / n_l)
  logits = X_l @ x_true
  for bs in (1, 16, 64):
    flips = rng.choice([-1.0, 1.0], size=(bs, N_l), p=[0.15, 0.85])
    y_b = np.where(logits >= 0.0, 1.0, -1.0)[None, :] * flips
    cases.append(
        BatchedClarabelCase(
            suite="logreg_simplex_batch",
            name=f"logreg_simplex_n{n_l}_B{bs}",
            n=n_l,
            m=N_l,
            batch_size=bs,
            F=None,
            A=X_l,
            b=np.zeros(N_l, dtype=np.float64),
            params_batch=y_b,
        )
    )

  return cases


def _solve_clarabel_batch(
    case: BatchedClarabelCase,
    *,
    tol: float,
    repeats: int,
) -> BatchClarabelResult:
  """Solves a batch with Clarabel in parallel using ThreadPoolExecutor."""
  settings = clarabel.DefaultSettings()
  settings.verbose = False
  settings.tol_gap_abs = tol
  settings.tol_gap_rel = tol
  settings.tol_feas = tol

  if case.suite == "factor_qp_batch":
    assert case.F is not None
    P_dense = case.F @ case.F.T + np.eye(case.n, dtype=np.float64)
    P_sp = sparse.csc_matrix(np.triu(P_dense))
    A_sp = sparse.vstack(
        [
            sparse.csc_matrix(case.A),
            -sparse.eye(case.n, format="csc", dtype=np.float64),
        ],
        format="csc",
    )
    b_full = np.concatenate([case.b, np.zeros(case.n, dtype=np.float64)])
    cones = [clarabel.NonnegativeConeT(case.m + case.n)]

    def _one(idx: int) -> tuple[bool, float]:
      q_i = case.params_batch[idx]
      solver = clarabel.DefaultSolver(P_sp, q_i, A_sp, b_full, cones, settings)
      sol = solver.solve()
      ok = str(sol.status).endswith("Solved")
      return ok, float(sol.obj_val)

  else:
    N, n = case.m, case.n
    lam = 1e-2
    # Variables: [x (n), t (N), u (N), v (N)]
    P_sp = sparse.block_diag(
        [
            lam * sparse.eye(n, format="csc"),
            sparse.csc_matrix((3 * N, 3 * N)),
        ],
        format="csc",
    )
    q_base = np.concatenate(
        [np.zeros(n), np.full(N, 1.0 / N), np.zeros(2 * N)]
    )
    # 1^T x = 1, x >= 0, u + v <= 1, (1, u) in K_exp*(y*a^T x - t), etc.
    # Standard softplus: log(1 + exp(z)) <= t  <=>  u + v <= 1,
    # (z - t, 1, u) in K_exp, (-t, 1, v) in K_exp.
    row_eq = sparse.hstack(
        [sparse.csc_matrix(np.ones((1, n))), sparse.csc_matrix((1, 3 * N))]
    )
    row_x = sparse.hstack(
        [-sparse.eye(n, format="csc"), sparse.csc_matrix((n, 3 * N))]
    )
    row_uv = sparse.hstack(
        [
            sparse.csc_matrix((N, n + N)),
            sparse.eye(N, format="csc"),
            sparse.eye(N, format="csc"),
        ]
    )
    I_N = sparse.eye(N, format="csc")
    Z_N_n = sparse.csc_matrix((N, n))
    Z_N_N = sparse.csc_matrix((N, N))

    def _one(idx: int) -> tuple[bool, float]:
      y_i = case.params_batch[idx]
      YX = sparse.diags(y_i, format="csc") @ sparse.csc_matrix(case.A)
      # Build 2*N exponential cones interleaved per row
      exp_rows = []
      exp_b = []
      for j in range(N):
        # Cone 1: (-y_j a_j^T x - t_j, 1, u_j) -> s = b_sub - A_sub z in K_exp
        # i.e. s0 = -y_j a_j^T x - t_j, s1 = 1, s2 = u_j
        r0 = sparse.hstack(
            [
                YX[j : j + 1],
                I_N[j : j + 1],
                Z_N_N[j : j + 1],
                Z_N_N[j : j + 1],
            ]
        )
        r1 = sparse.csc_matrix((1, n + 3 * N))
        r2 = sparse.hstack(
            [
                Z_N_n[j : j + 1],
                Z_N_N[j : j + 1],
                -I_N[j : j + 1],
                Z_N_N[j : j + 1],
            ]
        )
        # Cone 2: (-t_j, 1, v_j)
        r3 = sparse.hstack(
            [
                Z_N_n[j : j + 1],
                I_N[j : j + 1],
                Z_N_N[j : j + 1],
                Z_N_N[j : j + 1],
            ]
        )
        r4 = sparse.csc_matrix((1, n + 3 * N))
        r5 = sparse.hstack(
            [
                Z_N_n[j : j + 1],
                Z_N_N[j : j + 1],
                Z_N_N[j : j + 1],
                -I_N[j : j + 1],
            ]
        )
        exp_rows.extend([r0, r1, r2, r3, r4, r5])
        exp_b.extend([0.0, 1.0, 0.0, 0.0, 1.0, 0.0])

      A_sp = sparse.vstack([row_eq, row_x, row_uv, *exp_rows], format="csc")
      b_full = np.concatenate(
          [
              np.array([1.0]),
              np.zeros(n),
              np.ones(N),
              np.asarray(exp_b, dtype=np.float64),
          ]
      )
      cones = [
          clarabel.ZeroConeT(1),
          clarabel.NonnegativeConeT(n + N),
          *[clarabel.ExponentialConeT() for _ in range(2 * N)],
      ]
      solver = clarabel.DefaultSolver(
          P_sp, q_base, A_sp, b_full, cones, settings
      )
      sol = solver.solve()
      ok = str(sol.status).endswith("Solved")
      return ok, float(sol.obj_val)

  def _run_pool() -> tuple[bool, float]:
    with concurrent.futures.ThreadPoolExecutor() as pool:
      outs = list(pool.map(_one, range(case.batch_size)))
    all_ok = all(ok for ok, _ in outs)
    mean_obj = float(np.mean([o for _, o in outs]))
    return all_ok, mean_obj

  t0 = time.perf_counter()
  all_ok, mean_obj = _run_pool()
  compile_s = time.perf_counter() - t0

  run_times: list[float] = []
  for _ in range(max(1, repeats)):
    t1 = time.perf_counter()
    all_ok, mean_obj = _run_pool()
    run_times.append(time.perf_counter() - t1)

  return BatchClarabelResult(
      suite=case.suite,
      problem_name=case.name,
      solver_name="Clarabel(threads)",
      batch_size=case.batch_size,
      converged=all_ok,
      mean_obj=mean_obj,
      compile_s=compile_s,
      run_s=min(run_times),
  )


def _solve_proxax_batch(
    case: BatchedClarabelCase,
    *,
    solver_name: str,
    solver: typing.Any,
    backend: str,
    tol: float,
    repeats: int,
) -> BatchClarabelResult:
  """Solves a batch with px.batch_solve."""
  A_j = jnp.asarray(case.A, dtype=jnp.float64)
  pb_j = jnp.asarray(case.params_batch, dtype=jnp.float64)

  if case.suite == "factor_qp_batch":
    assert case.F is not None
    F_j = jnp.asarray(case.F, dtype=jnp.float64)
    b_j = jnp.asarray(case.b, dtype=jnp.float64)
    x0 = jnp.full(case.n, 0.2, dtype=jnp.float64)
    constraints = px.constraints.less_equal(A_j, b_j)

    def prob_fn(param_i: typing.Any) -> px.CompositeProblem:
      def smooth_fn(x: typing.Any) -> typing.Any:
        ftx = F_j.T @ x
        return 0.5 * (jnp.dot(ftx, ftx) + jnp.dot(x, x)) + jnp.dot(param_i, x)

      return px.CompositeProblem(
          smooth=smooth_fn,
          proximal=px.prox.NonNegative(),
          constraints=constraints,
      )

  else:
    x0 = jnp.full(case.n, 1.0 / case.n, dtype=jnp.float64)
    lam = 1e-2

    def prob_fn(param_i: typing.Any) -> px.CompositeProblem:
      def smooth_fn(x: typing.Any) -> typing.Any:
        z = -param_i * (A_j @ x)
        return jnp.mean(jax.nn.softplus(z)) + 0.5 * lam * jnp.dot(x, x)

      return px.CompositeProblem(
          smooth=smooth_fn,
          proximal=px.prox.Simplex(),
      )

  def _run() -> px.Solution:
    sol = px.batch_solve(
        prob_fn,
        x0,
        pb_j,
        solver=solver,
        opt_tol=tol,
        feas_tol=tol,
        backend=backend,
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
  mean_obj = float(jnp.mean(sol.stats.obj_val))
  return BatchClarabelResult(
      suite=case.suite,
      problem_name=case.name,
      solver_name=solver_name,
      batch_size=case.batch_size,
      converged=converged,
      mean_obj=mean_obj,
      compile_s=compile_s,
      run_s=min(run_times),
  )


def format_markdown_table(results: list[BatchClarabelResult]) -> str:
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
  """CLI entry point for batched Proxax vs Clarabel benchmark."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--tol", type=float, default=1e-6)
  parser.add_argument("--repeats", type=int, default=3)
  args = parser.parse_args()

  jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]
  results: list[BatchClarabelResult] = []
  for case in build_cases():
    if case.suite == "factor_qp_batch":
      solvers: list[tuple[str, typing.Any, str]] = [
          (
              "proxax-IAIPAL(vmap)",
              IAIPAL(opt_tol=args.tol, feas_tol=args.tol),
              "vmap",
          ),
          (
              "proxax-IAIPAL(threads)",
              IAIPAL(opt_tol=args.tol, feas_tol=args.tol),
              "threads",
          ),
          (
              "proxax-IALM(threads)",
              IALM(opt_tol=args.tol, feas_tol=args.tol),
              "threads",
          ),
      ]
    else:
      solvers = [
          ("proxax-PANOC(vmap)", PANOC(opt_tol=args.tol), "vmap"),
          ("proxax-PANOC(threads)", PANOC(opt_tol=args.tol), "threads"),
      ]
    for s_name, solver, backend in solvers:
      results.append(
          _solve_proxax_batch(
              case,
              solver_name=s_name,
              solver=solver,
              backend=backend,
              tol=args.tol,
              repeats=args.repeats,
          )
      )
    results.append(
        _solve_clarabel_batch(case, tol=args.tol, repeats=args.repeats)
    )

  print(format_markdown_table(results))


if __name__ == "__main__":
  main()
