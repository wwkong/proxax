"""Tracking per-iteration objective values, residuals, and tolerances."""

from __future__ import annotations

import jax.numpy as jnp

from proxax import px


def main() -> None:
  """Solves a linearly constrained Lasso problem while logging history."""
  a_mat = jnp.array([[2.0, 1.0, -1.0], [0.5, -1.5, 2.0], [1.0, 1.0, 1.0]])
  b_vec = jnp.array([1.0, -1.5, 0.5])
  c_mat = jnp.array([[1.0, 1.0, 1.0]])
  d_vec = jnp.array([1.0])

  problem = px.CompositeProblem(
      smooth=lambda x: 0.5 * jnp.sum((a_mat @ x - b_vec) ** 2),
      proximal=px.prox.L1(weight=0.1),
      constraints=px.constraints.equality(c_mat, d_vec),
  )

  history = px.HistoryTracker(verbose=True, print_every=1)
  sol = px.solve(
      problem,
      jnp.zeros(3),
      method="ialm",
      opt_tol=1e-6,
      feas_tol=1e-6,
      jit=False,
      callback=history,
  )

  print("\nSolution summary:")
  print(sol)
  print("\nTracker summary:")
  print(history)
  print(f"Recorded {len(history)} outer iterations.")
  print("Final obj_val / opt_res / feas_res:",
        f"{history.obj_vals[-1]:.4e}",
        f"{history.opt_residuals[-1]:.2e}",
        f"{history.feas_residuals[-1]:.2e}")


if __name__ == "__main__":
  main()
