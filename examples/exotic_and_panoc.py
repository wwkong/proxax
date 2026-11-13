"""Showcase of `PANOC` and `v0.2.1` exotic proximal operators."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from proxax import px


def main() -> None:
  """Compares `PANOC` on ill-conditioned Lasso and solves SDP/portfolio."""
  # 1. Ill-conditioned Lasso: PANOC vs APD vs AdapFISTA
  scales = jnp.geomspace(0.05, 50.0, 12)
  x_ref = jnp.linspace(-1.5, 1.5, 12)

  def _ill_cond_loss(x: jax.Array) -> jax.Array:
    return 0.5 * jnp.sum(scales * (x - x_ref) ** 2)

  lasso_prob = px.CompositeProblem(
      smooth=px.oracle.as_smooth_term(_ill_cond_loss),
      proximal=px.prox.L1(weight=0.1),
  )
  x0 = jnp.zeros(12)
  print("=== 1. Ill-Conditioned Lasso (cond=1000) ===")
  for label, method in [
      ("PANOC", "panoc"),
      ("APD", "apd"),
      ("AdapFISTA", "adap_fista"),
  ]:
    sol = px.solve(lasso_prob, x0, method=method, opt_tol=1e-5)
    print(
        f"  {label:10s} | status={sol.status_enum.name} "
        f"iters={int(sol.stats.outer_iters):4d} "
        f"evals={int(sol.stats.grad_evals + sol.stats.prox_evals):5d}"
    )

  # 2. Trace-1 PSD Matrix (Spectrahedron) approximation
  target_mat = jnp.array([[1.4, 0.6, -0.2], [0.6, 0.2, 0.3], [-0.2, 0.3, -0.4]])

  def _sdp_loss(x: jax.Array) -> jax.Array:
    return 0.5 * jnp.sum((x - target_mat) ** 2)

  sdp_prob = px.CompositeProblem(
      smooth=px.oracle.as_smooth_term(_sdp_loss),
      proximal=px.prox.Spectrahedron(scale=1.0),
  )
  sdp_sol = px.solve(
      sdp_prob, jnp.eye(3) / 3.0, method="panoc", opt_tol=1e-5
  )
  print("\n=== 2. Spectrahedron SDP ===")
  print(sdp_sol)

  # 3. Capped-Simplex Portfolio Allocation (max 35% per asset)
  cov = jnp.array([
      [0.4, 0.1, -0.05, 0.0],
      [0.1, 0.3, 0.05, 0.02],
      [-0.05, 0.05, 0.25, -0.03],
      [0.0, 0.02, -0.03, 0.2],
  ])
  ret = jnp.array([0.12, 0.08, 0.10, 0.05])

  def _portfolio_loss(w: jax.Array) -> jax.Array:
    return 0.5 * jnp.dot(w, cov @ w) - 0.5 * jnp.dot(ret, w)

  port_prob = px.CompositeProblem(
      smooth=px.oracle.as_smooth_term(_portfolio_loss),
      proximal=px.prox.CappedSimplex(scale=1.0, upper=0.35),
  )
  port_sol = px.solve(
      port_prob, jnp.ones(4) / 4.0, method="panoc", opt_tol=1e-6
  )
  print("\n=== 3. Capped-Simplex Portfolio ===")
  print("Weights:", jnp.round(port_sol.x, 4))


if __name__ == "__main__":
  main()
