"""Conic optimization benchmarks (`ExpCone`, `SOC`, `Spectrahedron`)."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp

from proxax import px
from proxax.solvers.constrained import ialm
from proxax.solvers.constrained import qp_aipp
from proxax.solvers.constrained import qp_apd


def _make_exp_cone_problem() -> tuple[px.CompositeProblem, jax.Array]:
  # (x0, x1, x2) in ExpCone with linear budget/target constraints
  v = jnp.array([1.5, 1.2, 1.0])
  A = jnp.array([[1.0, 1.0, 0.0]])
  b = jnp.array([2.0])

  def _obj(x: jax.Array) -> jax.Array:
    return 0.5 * jnp.sum((x - v) ** 2) + 0.1 * x[2]

  prob = px.CompositeProblem(
      smooth=px.oracle.as_smooth_term(_obj),
      proximal=px.prox.ExpCone(),
      constraints=px.constraints.equality(A, b),
  )
  return prob, jnp.array([0.5, 1.5, 3.0])


def _make_socp_problem() -> tuple[px.CompositeProblem, jax.Array]:
  n = 8
  scales = jnp.geomspace(1.0, 50.0, n)
  x_ref = jnp.array([2.5, 0.5, -0.4, 0.3, -0.2, 0.6, -0.3, 0.2])
  A = jnp.ones((1, n))
  b = jnp.array([3.0])

  def _obj(x: jax.Array) -> jax.Array:
    return 0.5 * jnp.sum(scales * (x - x_ref) ** 2)

  prob = px.CompositeProblem(
      smooth=px.oracle.as_smooth_term(_obj),
      proximal=px.prox.SOC(),
      constraints=px.constraints.equality(A, b),
  )
  return prob, jnp.array([2.0, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1])


def _make_spectrahedron_sdp() -> tuple[px.CompositeProblem, jax.Array]:
  n = 4
  key = jax.random.PRNGKey(11)
  raw = jax.random.normal(key, (n, n))
  M = 0.5 * (raw + raw.T)

  def _obj(X: jax.Array) -> jax.Array:
    return 0.5 * jnp.sum((X - M) ** 2)

  def _diag_sum(X: jax.Array) -> jax.Array:
    return jnp.array([X[0, 0] + X[1, 1] - 0.5])

  prob = px.CompositeProblem(
      smooth=px.oracle.as_smooth_term(_obj),
      proximal=px.prox.Spectrahedron(scale=1.0),
      constraints=px.constraints.equality(_diag_sum, jnp.array([0.0])),
  )
  return prob, jnp.eye(n) / n


def _make_soc_portfolio() -> tuple[px.CompositeProblem, jax.Array]:
  n = 8
  key = jax.random.PRNGKey(22)
  k1, k2 = jax.random.split(key)
  F = jax.random.normal(k1, (n, 3)) * 0.3
  cov = F @ F.T + jnp.diag(jnp.geomspace(0.1, 10.0, n))
  mu = jax.random.uniform(k2, (n,), minval=0.05, maxval=0.25)
  A = jnp.vstack([jnp.array([[1.0] + [0.0] * (n - 1)]), jnp.ones((1, n))])
  b = jnp.array([2.0, 3.0])

  def _obj(x: jax.Array) -> jax.Array:
    return 0.5 * (x @ cov @ x) - jnp.dot(mu, x)

  prob = px.CompositeProblem(
      smooth=px.oracle.as_smooth_term(_obj),
      proximal=px.prox.SOC(),
      constraints=px.constraints.equality(A, b),
  )
  return prob, jnp.array([2.0] + [1.0 / (n - 1)] * (n - 1))


def main() -> None:
  """Runs conic benchmark suite and prints JSON summary."""
  problems = [
      ("ExpCone Entropy", *_make_exp_cone_problem()),
      ("Robust SOCP (cond=50)", *_make_socp_problem()),
      ("Spectrahedron SDP", *_make_spectrahedron_sdp()),
      ("SOC Portfolio", *_make_soc_portfolio()),
  ]
  solvers = [
      (
          "PANOC-ALM",
          ialm.IALM(
              opt_tol=1e-4,
              feas_tol=1e-4,
              max_alm_iters=40,
              subproblem_solver="panoc",
          ),
      ),
      (
          "IALM (APD)",
          ialm.IALM(
              opt_tol=1e-4,
              feas_tol=1e-4,
              max_alm_iters=40,
              subproblem_solver="apd",
          ),
      ),
      ("QP-APD", qp_apd.QPAPD(opt_tol=1e-4, feas_tol=1e-4)),
      ("QP-AIPP", qp_aipp.QPAIPP(opt_tol=1e-4, feas_tol=1e-4)),
  ]

  rows = []
  profile: dict[str, list[float]] = {name: [] for name, _ in solvers}
  for prob_name, prob, x0 in problems:
    for solver_name, solver in solvers:
      sol = solver.solve(prob, x0)
      evals = int(sol.stats.grad_evals + sol.stats.prox_evals)
      status = sol.status_enum.name
      rows.append({
          "problem": prob_name,
          "solver": solver_name,
          "status": status,
          "iters": int(sol.stats.outer_iters),
          "evals": evals,
      })
      profile[solver_name].append(
          float(evals) if status == "CONVERGED" else float("inf")
      )

  print(json.dumps({"rows": rows, "profile": profile}))


if __name__ == "__main__":
  main()
