from __future__ import annotations

import jax
import jax.numpy as jnp

import proxax
from proxax import oracle
from proxax import prox
from proxax import solvers


def test_aipp_known_and_adaptive_certificates() -> None:
  q_mat = jnp.array([[2.0, -0.5], [-0.5, -1.0]])
  c_vec = jnp.array([0.3, -0.4])
  reg_h = prox.BoxL1(l1_weight=0.1, lower=-1.0, upper=1.0)

  def smooth_fn(x: jax.Array) -> jax.Array:
    return 0.5 * jnp.dot(x, q_mat @ x) + jnp.dot(c_vec, x)

  x0 = jnp.array([0.6, -0.5])
  tol = 1e-3
  cfg = solvers.AIPP(opt_tol=tol, max_outer_iters=30, max_inner_iters=40)

  prob_known = proxax.CompositeProblem(
      smooth=oracle.SmoothTerm(smooth_fn, lower_curvature=1.5),
      proximal=reg_h,
  )
  sol_known = proxax.solve(prob_known, x0, solver=cfg)
  assert sol_known.status_enum == proxax.SolveStatus.CONVERGED
  assert float(sol_known.stats.opt_residual) <= tol

  prob_adapt = proxax.CompositeProblem(smooth=smooth_fn, proximal=reg_h)
  sol_adapt = proxax.solve(prob_adapt, x0, solver=cfg)
  assert sol_adapt.status_enum == proxax.SolveStatus.CONVERGED
  assert float(sol_adapt.stats.opt_residual) <= tol


def test_aipp_eager_callback_and_vmap() -> None:
  prob = proxax.CompositeProblem(
      smooth=lambda x: 0.25 * jnp.sum(x**4) - 0.5 * jnp.sum(x**2),
      proximal=prox.Box(lower=-1.5, upper=1.5),
  )
  solver = solvers.AIPP(opt_tol=1e-3, max_outer_iters=25, max_inner_iters=30)

  seen: list[float] = []

  def on_step(st: solvers.AIPPState) -> None:
    seen.append(float(st.best_opt_res))

  sol = solver.solve(
      prob,
      jnp.array([1.2, -0.8]),
      jit=False,
      callback=on_step,
  )
  assert sol.status_enum == proxax.SolveStatus.CONVERGED
  assert len(seen) == int(sol.stats.outer_iters) + 1

  inits = jnp.array([[1.2, -0.8], [-1.1, 0.7]])
  batched = jax.jit(jax.vmap(lambda x_i: solver.solve(prob, x_i)))(inits)
  assert batched.x.shape == (2, 2)
  assert jnp.all(batched.status == int(proxax.SolveStatus.CONVERGED))
