from __future__ import annotations

import jax
import jax.numpy as jnp

import proxax
from proxax import prox
from proxax import solvers


def test_apd_nonconvex_and_convex_exact_certificates() -> None:
  # Indefinite (nonconvex) quadratic + Box-L1 composite problem, with no
  # curvature hints supplied to APD!
  q_ncvx = jnp.array([[3.5, -0.8], [-0.8, -2.5]])
  c_vec = jnp.array([0.4, -0.6])
  reg_h = prox.BoxL1(l1_weight=0.1, lower=-1.0, upper=1.0)

  def f_ncvx(x: jax.Array) -> jax.Array:
    return 0.5 * jnp.dot(x, q_ncvx @ x) + jnp.dot(c_vec, x)

  prob_ncvx = proxax.CompositeProblem(smooth=f_ncvx, proximal=reg_h)
  x0 = jnp.array([0.7, -0.6])
  tol = 1e-3

  # 1. Practical APD with curvature decay (Eq. (6.1)) via method="apd"
  sol_decay = proxax.solve(prob_ncvx, x0, method="apd", opt_tol=tol)
  assert sol_decay.status_enum == proxax.SolveStatus.CONVERGED
  assert float(sol_decay.stats.opt_residual) <= tol

  # Verify exact subdifferential membership v_hat - grad_f(x_hat) in dh(x_hat)
  grad_x = jax.grad(f_ncvx)(sol_decay.x)
  subg_h = sol_decay.subgrad - grad_x
  for x_i, g_i in zip(sol_decay.x.tolist(), subg_h.tolist(), strict=True):
    if abs(x_i) < 1.0 - 1e-5:
      if abs(x_i) > 1e-5:
        assert abs(g_i - 0.1 * (1.0 if x_i > 0 else -1.0)) < 5e-5
      else:
        assert abs(g_i) <= 0.1 + 5e-5

  # 2. Theoretical bitonic PF.APD (Algorithm 3.4, decay_curvature=False)
  sol_bitonic = proxax.solve(
      prob_ncvx,
      x0,
      solver=solvers.APD(
          opt_tol=tol,
          decay_curvature=False,
          lower_curvature_init=0.25,
          max_outer_iters=60,
          max_inner_iters=80,
      ),
  )
  assert sol_bitonic.status_enum == proxax.SolveStatus.CONVERGED
  assert float(sol_bitonic.stats.opt_residual) <= tol

  # 3. Convex composite problem without curvature hints (tests m_k decay)
  q_cvx = jnp.array([[4.0, 1.0], [1.0, 1.5]])
  prob_cvx = proxax.CompositeProblem(
      smooth=lambda x: 0.5 * jnp.dot(x, q_cvx @ x) + jnp.dot(c_vec, x),
      proximal=prox.L1(0.15),
  )
  sol_cvx = proxax.solve(prob_cvx, x0, method="apd", opt_tol=tol)
  assert sol_cvx.status_enum == proxax.SolveStatus.CONVERGED
  assert float(sol_cvx.stats.opt_residual) <= tol


def test_apd_eager_callback_and_vmap_on_pytree() -> None:
  prob = proxax.CompositeProblem(
      smooth=lambda p: 0.25 * jnp.sum(p["w"] ** 4)
      - 0.6 * jnp.sum(p["w"] ** 2)
      + 0.5 * jnp.sum((p["b"] - 0.3) ** 2),
      proximal=prox.MultiProx({
          "w": prox.Box(lower=-1.5, upper=1.5),
          "b": prox.L1(0.05),
      }),
  )
  solver = solvers.APD(
      opt_tol=1e-3,
      lower_curvature_init=0.2,
      max_outer_iters=40,
      max_inner_iters=50,
  )

  seen: list[float] = []

  def on_step(st: solvers.APDState) -> None:
    seen.append(float(st.best_opt_res))

  x0 = {"w": jnp.array([1.2, -0.7]), "b": jnp.array([-0.4])}
  sol = solver.solve(prob, x0, jit=False, callback=on_step)
  assert sol.status_enum == proxax.SolveStatus.CONVERGED
  assert len(seen) == int(sol.stats.outer_iters)

  batch_x0 = {
      "w": jnp.array([[1.2, -0.7], [-1.1, 0.8]]),
      "b": jnp.array([[-0.4], [0.5]]),
  }
  batched = jax.jit(jax.vmap(lambda init_i: solver.solve(prob, init_i)))(
      batch_x0
  )
  assert batched.x["w"].shape == (2, 2)
  assert jnp.all(batched.status == int(proxax.SolveStatus.CONVERGED))
