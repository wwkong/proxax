"""Constrained benchmarks comparing `panoc_alm` against constrained solvers."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp

from proxax import px
from proxax.solvers.constrained import iaipal
from proxax.solvers.constrained import ialm
from proxax.solvers.constrained import qp_aipp
from proxax.solvers.constrained import qp_apd


def _make_constrained_lasso(
    cond: float,
) -> tuple[px.CompositeProblem, jax.Array]:
  n, k = 16, 3
  key = jax.random.PRNGKey(int(cond))
  scales = jnp.geomspace(1.0, cond, n)
  x_ref = jnp.linspace(-1.2, 1.2, n)
  C = jax.random.normal(key, (k, n)) / jnp.sqrt(n)
  d = C @ jnp.ones(n) * 0.25

  def _obj(
      x: jax.Array, s: jax.Array = scales, r: jax.Array = x_ref
  ) -> jax.Array:
    return 0.5 * jnp.sum(s * (x - r) ** 2)

  prob = px.CompositeProblem(
      smooth=px.oracle.as_smooth_term(_obj),
      proximal=px.prox.L1(weight=0.1),
      constraints=px.constraints.equality(C, d),
  )
  return prob, jnp.zeros(n)


def _make_sector_portfolio() -> tuple[px.CompositeProblem, jax.Array]:
  n = 16
  key = jax.random.PRNGKey(42)
  k1, k2 = jax.random.split(key)
  F = jax.random.normal(k1, (n, 4)) * 0.3
  cov = F @ F.T + jnp.diag(jnp.geomspace(0.05, 5.0, n))
  mu = jax.random.uniform(k2, (n,), minval=0.02, maxval=0.25)
  # Two sector neutrality/target constraints
  S = jnp.zeros((2, n))
  S = S.at[0, :8].set(1.0)
  S = S.at[1, 8:].set(1.0)
  rhs = jnp.array([0.55, 0.45])

  def _risk(x: jax.Array) -> jax.Array:
    return 0.5 * (x @ cov @ x) - jnp.dot(mu, x)

  prob = px.CompositeProblem(
      smooth=px.oracle.as_smooth_term(_risk),
      proximal=px.prox.Box(lower=0.0, upper=0.25),
      constraints=px.constraints.equality(S, rhs),
  )
  return prob, jnp.full(n, 1.0 / n)


def _make_constrained_logistic() -> tuple[px.CompositeProblem, jax.Array]:
  n, m, k = 16, 40, 2
  key = jax.random.PRNGKey(7)
  k1, k2, k3 = jax.random.split(key, 3)
  A_data = jax.random.normal(k1, (m, n)) / jnp.sqrt(n)
  y = jnp.where(jax.random.normal(k2, (m,)) > 0, 1.0, -1.0)
  C = jax.random.normal(k3, (k, n)) / jnp.sqrt(n)
  d = jnp.zeros(k)

  def _logreg(x: jax.Array) -> jax.Array:
    margins = -y * (A_data @ x)
    return jnp.mean(jnp.logaddexp(0.0, margins))

  prob = px.CompositeProblem(
      smooth=px.oracle.as_smooth_term(_logreg),
      proximal=px.prox.L1(weight=0.05),
      constraints=px.constraints.equality(C, d),
  )
  return prob, jnp.zeros(n)


def main() -> None:
  """Runs constrained benchmark suite and prints JSON summary."""
  problems = [
      ("Constrained Lasso (cond=100)", *_make_constrained_lasso(100.0)),
      ("Constrained Lasso (cond=1000)", *_make_constrained_lasso(1000.0)),
      ("Sector Portfolio", *_make_sector_portfolio()),
      ("Constrained Sparse Logistic", *_make_constrained_logistic()),
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
      ("IAIPAL", iaipal.IAIPAL(opt_tol=1e-4, feas_tol=1e-4)),
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
