"""Min-max and saddle-point optimization examples.

Covers two benchmark problems:
1. Bilinear two-player zero-sum matrix game over probability simplices
   (`ExtraGradient` and `PDHG`).
2. Weakly nonconvex-strongly concave minimax over box constraints
   (`ProxPointMinMax` via `m_x`).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from proxax import px


def run_bilinear_matrix_game() -> None:
  """Solves `min_{x in Delta^3} max_{y in Delta^3} x^T A y`."""
  print("=== 1. Bilinear Two-Player Matrix Game over Simplices ===")
  payoff = jnp.array([[2.0, -1.0, 0.5], [-1.0, 1.5, 0.2], [0.0, -0.5, 1.0]])
  l_xy = float(jnp.linalg.norm(payoff, ord=2))
  problem = px.MinMaxProblem(
      coupling=lambda x, y: x @ payoff @ y,
      prox_x=px.prox.Simplex(scale=1.0),
      prox_y=px.prox.Simplex(scale=1.0),
      L_xy=l_xy,
  )
  x0 = jnp.ones(3) / 3.0
  y0 = jnp.ones(3) / 3.0

  solvers = [
      ("ExtraGradient", px.solvers.ExtraGradient(opt_tol=1e-4)),
      ("PDHG", px.solvers.PDHG(opt_tol=1e-4)),
  ]
  for name, solver in solvers:
    sol = px.solve(problem, (x0, y0), solver=solver)
    x_hat, y_hat = sol.x
    game_val = float(x_hat @ payoff @ y_hat)
    print(
        f"{name:<14s} -> status={sol.status_enum.name}, "
        f"val={game_val:.6f}, ||v||={float(sol.stats.opt_residual):.2e}, "
        f"iters={int(sol.stats.outer_iters)}"
    )


def run_weakly_nonconvex_minimax() -> None:
  """Solves a weakly nonconvex-strongly concave minimax problem."""
  print("\n=== 2. Weakly Nonconvex-Strongly Concave Minimax ===")
  a_mat = jnp.array([[1.2, -0.4], [0.5, 1.0]])

  def coupling(x: jax.Array, y: jax.Array) -> jax.Array:
    return (
        0.5 * jnp.sum(x**2)
        + 0.3 * jnp.sum(jnp.sin(2.0 * x))
        + jnp.dot(y, a_mat @ x)
        - 0.5 * jnp.sum(y**2)
    )

  problem = px.MinMaxProblem(
      coupling=coupling,
      prox_x=px.prox.Box(lower=-2.0, upper=2.0),
      prox_y=px.prox.Box(lower=-2.0, upper=2.0),
      m_x=1.2,
  )
  x0 = jnp.array([0.5, -0.5])
  y0 = jnp.zeros(2)

  sol = px.solve(problem, (x0, y0), opt_tol=1e-4)
  x_hat, y_hat = sol.x
  print(
      f"ProxPointMinMax -> status={sol.status_enum.name}, "
      f"val={float(coupling(x_hat, y_hat)):.6f}, "
      f"||v||={float(sol.stats.opt_residual):.2e}, "
      f"outer={int(sol.stats.outer_iters)}, inner={int(sol.stats.inner_iters)}"
  )


def main() -> None:
  """Runs all min-max saddle-point examples."""
  run_bilinear_matrix_game()
  run_weakly_nonconvex_minimax()


if __name__ == "__main__":
  main()
