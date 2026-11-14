"""Convex-concave min-max benchmarks: Proxax vs dualized Clarabel, SCS, CVXPY.

Covers:
1. Zero-sum matrix games over `Simplex x Simplex` (dualized to LP).
2. Regularized bilinear saddle-point problems over `Box x Simplex` (dualized
   to QP).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import typing

import clarabel
import cvxpy as cp
import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

import proxax as px
from benchmarks import harness
from benchmarks.external import conic_helpers


@dataclasses.dataclass(frozen=True)
class MinMaxConvexBenchmark:
  """A convex-concave min-max benchmark with Proxax and dualized builders."""

  name: str
  proxax_builder: typing.Callable[
      [], tuple[px.MinMaxProblem, tuple[jax.Array, jax.Array]]
  ]
  clarabel_builder: typing.Callable[
      [float, float], harness.BenchmarkResult
  ]
  scs_builder: typing.Callable[[float, bool], harness.BenchmarkResult]
  cvxpy_builder: typing.Callable[[], cp.Problem] | None = None


def _make_matrix_game(
    n: int, m: int, *, seed: int = 101
) -> MinMaxConvexBenchmark:
  """Zero-sum matrix game `min_{x in Delta_n} max_{y in Delta_m} x^T A y`."""
  rng = np.random.default_rng(seed)
  a_mat = rng.standard_normal((n, m)) / np.sqrt(max(n, m))
  name = f"Matrix Game ({n}x{m})"

  def _proxax() -> tuple[px.MinMaxProblem, tuple[jax.Array, jax.Array]]:
    a_j = jnp.asarray(a_mat)

    def _coupling(x: jax.Array, y: jax.Array) -> jax.Array:
      return jnp.dot(x, a_j @ y)

    prob = px.MinMaxProblem(
        coupling=_coupling,
        prox_x=px.prox.Simplex(scale=1.0),
        prox_y=px.prox.Simplex(scale=1.0),
    )
    return prob, (jnp.ones(n) / n, jnp.ones(m) / m)

  # Dualized LP: min_{x >= 0, 1^T x = 1, t} t s.t. A^T x - t 1 <= 0
  p_sp = sp.csc_matrix((n + 1, n + 1))
  q_vec = np.zeros(n + 1)
  q_vec[-1] = 1.0

  a_eq = sp.csc_matrix(np.hstack([np.ones((1, n)), np.zeros((1, 1))]))
  a_game = sp.csc_matrix(np.hstack([a_mat.T, -np.ones((m, 1))]))
  a_nonneg = sp.hstack(
      [-sp.eye(n, format="csc"), sp.csc_matrix((n, 1))], format="csc"
  )
  a_ext = sp.vstack([a_eq, a_game, a_nonneg], format="csc")
  b_ext = np.concatenate([np.array([1.0]), np.zeros(m + n)])

  def _clarabel(opt_tol: float, feas_tol: float) -> harness.BenchmarkResult:
    cones = [clarabel.ZeroConeT(1), clarabel.NonnegativeConeT(m + n)]
    return conic_helpers.solve_clarabel(
        suite="minmax_convex_concave",
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
        suite="minmax_convex_concave",
        problem_name=name,
        p_sp=None,
        q_vec=q_vec,
        a_sp=a_ext,
        b_vec=b_ext,
        cone_dict={"z": 1, "l": m + n},
        tol=tol,
        use_indirect=indirect,
    )

  def _cvxpy() -> cp.Problem:
    x = cp.Variable(n, nonneg=True)
    t = cp.Variable()
    return cp.Problem(
        cp.Minimize(t),
        [cp.sum(x) == 1.0, a_mat.T @ x <= t],
    )

  return MinMaxConvexBenchmark(
      name=name,
      proxax_builder=_proxax,
      clarabel_builder=_clarabel,
      scs_builder=_scs,
      cvxpy_builder=_cvxpy,
  )


def _make_bilinear_saddle_qp(
    n: int, m: int, *, seed: int = 202
) -> MinMaxConvexBenchmark:
  """Convex-concave QP over `Box(-1, 1) x Simplex`."""
  rng = np.random.default_rng(seed)
  u_x = rng.standard_normal((n, n)) / np.sqrt(n)
  q_x = u_x.T @ u_x + 0.1 * np.eye(n)
  u_y = rng.standard_normal((m, m)) / np.sqrt(m)
  q_y = u_y.T @ u_y + 0.2 * np.eye(m)
  k_mat = rng.standard_normal((n, m)) / np.sqrt(max(n, m))
  c_x = rng.standard_normal(n) * 0.3
  c_y = rng.standard_normal(m) * 0.3
  name = f"Bilinear Saddle QP ({n}x{m})"

  def _proxax() -> tuple[px.MinMaxProblem, tuple[jax.Array, jax.Array]]:
    qx_j = jnp.asarray(q_x)
    qy_j = jnp.asarray(q_y)
    k_j = jnp.asarray(k_mat)
    cx_j = jnp.asarray(c_x)
    cy_j = jnp.asarray(c_y)

    def _coupling(x: jax.Array, y: jax.Array) -> jax.Array:
      return (
          0.5 * jnp.dot(x, qx_j @ x)
          + jnp.dot(x, k_j @ y)
          - 0.5 * jnp.dot(y, qy_j @ y)
          + jnp.dot(cx_j, x)
          - jnp.dot(cy_j, y)
      )

    prob = px.MinMaxProblem(
        coupling=_coupling,
        prox_x=px.prox.Box(lower=-1.0, upper=1.0),
        prox_y=px.prox.Simplex(scale=1.0),
    )
    return prob, (jnp.zeros(n), jnp.ones(m) / m)

  # Dualized QP variables: [x (n), z (m), nu (1), mu (m)]
  # min 0.5 x^T Q_x x + 0.5 ||z||^2 + c_x^T x + nu
  # s.t. K^T x - L_y z - nu 1 + mu = c_y,  -1 <= x <= 1,  mu >= 0
  l_y = np.linalg.cholesky(q_y)
  p_sp = sp.block_diag(
      [
          sp.triu(sp.csc_matrix(q_x), format="csc"),
          sp.eye(m, format="csc"),
          sp.csc_matrix((1 + m, 1 + m)),
      ],
      format="csc",
  )
  q_vec = np.concatenate([c_x, np.zeros(m), np.array([1.0]), np.zeros(m)])

  a_eq = sp.hstack(
      [
          sp.csc_matrix(k_mat.T),
          -sp.csc_matrix(l_y),
          -sp.csc_matrix(np.ones((m, 1))),
          sp.eye(m, format="csc"),
      ],
      format="csc",
  )
  a_ub_x = sp.hstack(
      [sp.eye(n, format="csc"), sp.csc_matrix((n, m + 1 + m))], format="csc"
  )
  a_lb_x = sp.hstack(
      [-sp.eye(n, format="csc"), sp.csc_matrix((n, m + 1 + m))], format="csc"
  )
  a_mu = sp.hstack(
      [sp.csc_matrix((m, n + m + 1)), -sp.eye(m, format="csc")], format="csc"
  )
  a_ext = sp.vstack([a_eq, a_ub_x, a_lb_x, a_mu], format="csc")
  b_ext = np.concatenate([c_y, np.ones(n), np.ones(n), np.zeros(m)])

  def _clarabel(opt_tol: float, feas_tol: float) -> harness.BenchmarkResult:
    cones = [
        clarabel.ZeroConeT(m),
        clarabel.NonnegativeConeT(2 * n + m),
    ]
    return conic_helpers.solve_clarabel(
        suite="minmax_convex_concave",
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
        suite="minmax_convex_concave",
        problem_name=name,
        p_sp=p_sp,
        q_vec=q_vec,
        a_sp=a_ext,
        b_vec=b_ext,
        cone_dict={"z": m, "l": 2 * n + m},
        tol=tol,
        use_indirect=indirect,
    )

  def _cvxpy() -> cp.Problem:
    x = cp.Variable(n)
    z = cp.Variable(m)
    nu = cp.Variable()
    mu = cp.Variable(m, nonneg=True)
    obj = cp.Minimize(
        0.5 * cp.quad_form(x, cp.psd_wrap(q_x))
        + 0.5 * cp.sum_squares(z)
        + c_x @ x
        + nu
    )
    cons = [
        k_mat.T @ x - l_y @ z - nu + mu == c_y,
        x >= -1.0,
        x <= 1.0,
    ]
    return cp.Problem(obj, cons)

  return MinMaxConvexBenchmark(
      name=name,
      proxax_builder=_proxax,
      clarabel_builder=_clarabel,
      scs_builder=_scs,
      cvxpy_builder=_cvxpy,
  )


def run_suite(
    *,
    opt_tol: float = 1e-4,
    include_cvxpy: bool = False,
    time_limit: float = 20.0,
) -> list[harness.BenchmarkResult]:
  """Runs convex-concave min-max benchmarks across Proxax and conic solvers."""
  benchmarks: list[MinMaxConvexBenchmark] = [
      _make_matrix_game(32, 32),
      _make_matrix_game(128, 128),
      _make_matrix_game(256, 256),
      _make_bilinear_saddle_qp(32, 32),
      _make_bilinear_saddle_qp(128, 128),
      _make_bilinear_saddle_qp(256, 256),
  ]

  proxax_solvers = [
      ("PDHG", px.solvers.PDHG(opt_tol=opt_tol, max_iters=10_000)),
      (
          "ExtraGradient",
          px.solvers.ExtraGradient(opt_tol=opt_tol, max_iters=10_000),
      ),
      (
          "ProxPointMinMax",
          px.solvers.ProxPointMinMax(opt_tol=opt_tol, max_outer_iters=500),
      ),
  ]

  results: list[harness.BenchmarkResult] = []
  for bench in benchmarks:
    prob, z0 = bench.proxax_builder()
    for solver_name, solver in proxax_solvers:
      results.append(
          harness.time_proxax_solve(
              suite="minmax_convex_concave",
              problem_name=bench.name,
              solver_name=solver_name,
              prob=prob,
              x0=z0,
              solver=solver,
              opt_tol=opt_tol,
              time_limit=time_limit,
          )
      )
    results.append(bench.clarabel_builder(opt_tol, opt_tol))
    results.append(bench.scs_builder(opt_tol, False))
    results.append(bench.scs_builder(opt_tol, True))
    if include_cvxpy and bench.cvxpy_builder is not None:
      cp_prob = bench.cvxpy_builder()
      results.append(
          conic_helpers.solve_cvxpy(
              suite="minmax_convex_concave",
              problem_name=bench.name,
              prob=cp_prob,
              solver_name="CVXPY (Clarabel)",
              cvxpy_solver=cp.CLARABEL,
          )
      )
  return results


def main() -> None:
  """CLI entry point for convex-concave min-max benchmarks."""
  parser = argparse.ArgumentParser(
      description=(
          "Benchmark Proxax convex-concave MinMax solvers vs Clarabel and SCS."
      )
  )
  parser.add_argument("--opt-tol", type=float, default=1e-4)
  parser.add_argument("--time-limit", type=float, default=20.0)
  parser.add_argument("--include-cvxpy", action="store_true")
  parser.add_argument("--json-out", type=pathlib.Path, default=None)
  args = parser.parse_args()

  results = run_suite(
      opt_tol=args.opt_tol,
      include_cvxpy=args.include_cvxpy,
      time_limit=args.time_limit,
  )
  print(harness.format_markdown_table(results))
  if args.json_out is not None:
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    payload = [dataclasses.asdict(r) for r in results]
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
  main()
