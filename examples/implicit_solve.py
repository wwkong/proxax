"""Implicit differentiation examples using `px.implicit_solve`.

Covers two small-scale bilevel / sensitivity examples:
1. Bilevel hyperparameter tuning of an L1 regularization weight with
   `optax.adam`.
2. Primal-dual KKT sensitivity (`return_dual=True`) on a constrained QP.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from proxax import px


def run_bilevel_l1_tuning() -> None:
  """Tunes an L1 regularization weight on a small validation loss."""
  print("=== 1. Bilevel L1 Hyperparameter Tuning ===")
  a_train = jnp.array([[2.0, 0.2, -0.1], [0.1, 1.8, 0.3], [-0.2, 0.1, 1.5]])
  b_train = jnp.array([2.2, -1.6, 0.2])
  a_val = jnp.array([[1.0, 0.0, 0.1], [0.0, 1.0, -0.1]])
  b_val = jnp.array([1.0, -0.9])

  def make_problem(log_lam: jax.Array) -> px.CompositeProblem:
    lam = jnp.exp(log_lam)
    return px.CompositeProblem(
        smooth=lambda x: 0.5 * jnp.sum((a_train @ x - b_train) ** 2),
        proximal=px.prox.L1(weight=lam),
    )

  def val_loss(log_lam: jax.Array) -> jax.Array:
    x_star = px.implicit_solve(
        make_problem,
        jnp.zeros(3),
        log_lam,
        solver=px.solvers.APD(opt_tol=1e-5),
        linear_tol=1e-6,
    )
    return 0.5 * jnp.sum((a_val @ x_star - b_val) ** 2)

  log_lam = jnp.array(0.5)
  optimizer = optax.adam(learning_rate=0.2)
  opt_state = optimizer.init(log_lam)
  step_fn = jax.jit(jax.value_and_grad(val_loss))

  for step in range(10):
    loss, grad = step_fn(log_lam)
    updates, opt_state = optimizer.update(grad, opt_state, log_lam)
    log_lam = optax.apply_updates(log_lam, updates)
    if step in (0, 4, 9):
      print(
          f"step={step:2d} | lam={float(jnp.exp(log_lam)):.4f} | "
          f"val_loss={float(loss):.6f} | grad={float(grad):+.4f}"
      )


def run_constrained_kkt_sensitivity() -> None:
  """Differentiates primal and dual solutions of a constrained QP."""
  print("\n=== 2. Constrained Primal-Dual KKT Sensitivity ===")
  c_vec = jnp.array([0.4, -0.2, 0.7])

  def make_problem(rhs: jax.Array) -> px.CompositeProblem:
    return px.CompositeProblem(
        smooth=lambda x: 0.5 * jnp.sum((x - c_vec) ** 2),
        constraints=px.constraints.equality(jnp.sum, rhs=rhs),
    )

  def dual_loss(rhs: jax.Array) -> jax.Array:
    x_hat, p_hat = px.implicit_solve(
        make_problem,
        jnp.zeros(3),
        rhs,
        solver=px.solvers.IALM(opt_tol=1e-5, feas_tol=1e-5),
        linear_tol=1e-6,
        return_dual=True,
    )
    return jnp.sum(x_hat) + 0.5 * jnp.sum(p_hat**2)

  rhs0 = jnp.array(1.2)
  grad_ad = jax.grad(dual_loss)(rhs0)
  eps = 1e-3
  grad_fd = (dual_loss(rhs0 + eps) - dual_loss(rhs0 - eps)) / (2.0 * eps)
  print(
      f"implicit grad={float(grad_ad):.6f} | "
      f"finite-diff={float(grad_fd):.6f} | "
      f"diff={float(jnp.abs(grad_ad - grad_fd)):.2e}"
  )


def main() -> None:
  """Runs all `px.implicit_solve` examples."""
  run_bilevel_l1_tuning()
  run_constrained_kkt_sensitivity()


if __name__ == "__main__":
  main()
