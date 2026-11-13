import jax
import jax.numpy as jnp

from proxax import px


def test_minmax_bilinear_simplex_game() -> None:
  payoff = jnp.array([[2.0, -1.0, 0.5], [-1.0, 1.5, 0.2], [0.0, -0.5, 1.0]])
  prob = px.MinMaxProblem(
      coupling=lambda x, y: x @ payoff @ y,
      prox_x=px.prox.Simplex(scale=1.0),
      prox_y=px.prox.Simplex(scale=1.0),
      L_xy=float(jnp.linalg.norm(payoff, ord=2)),
  )
  x0 = jnp.ones(3) / 3.0
  y0 = jnp.ones(3) / 3.0

  for method in ("extra_gradient", "pdhg"):
    sol = px.solve(prob, (x0, y0), method=method, opt_tol=1e-4)
    assert sol.status_enum == px.SolveStatus.CONVERGED
    assert float(sol.stats.opt_residual) <= 1e-4
    x_hat, y_hat = sol.x
    assert jnp.allclose(jnp.sum(x_hat), 1.0, atol=1e-5)
    assert jnp.allclose(jnp.sum(y_hat), 1.0, atol=1e-5)


def test_minmax_quadratic_l1_saddle() -> None:
  a_mat = jnp.array([[1.0, -0.5, 0.2], [0.3, 1.2, -0.4]])
  b_vec = jnp.array([0.4, -0.3])
  c_vec = jnp.array([0.8, -0.6, 0.5])

  def coupling(x: jax.Array, y: jax.Array) -> jax.Array:
    return (
        0.5 * jnp.sum((x - c_vec) ** 2)
        + jnp.dot(y, a_mat @ x - b_vec)
        - 0.25 * jnp.sum(y**2)
    )

  prob = px.MinMaxProblem(
      coupling=coupling,
      prox_x=px.prox.L1(weight=0.1),
      prox_y=px.prox.Zero(),
  )
  x0 = jnp.zeros(3)
  y0 = jnp.zeros(2)

  for solver in (
      px.solvers.ExtraGradient(opt_tol=1e-4),
      px.solvers.PDHG(opt_tol=1e-4),
      px.solvers.ProxPointMinMax(opt_tol=1e-4),
  ):
    sol = px.solve(prob, (x0, y0), solver=solver)
    assert sol.status_enum == px.SolveStatus.CONVERGED
    assert float(sol.stats.opt_residual) <= 1e-4


def test_minmax_nonconvex_concave_and_vmap() -> None:
  a_mat = jnp.array([[1.2, -0.4], [0.5, 1.0]])

  def coupling(x: jax.Array, y: jax.Array) -> jax.Array:
    # Weakly nonconvex in x, strongly concave in y.
    return (
        0.5 * jnp.sum(x**2)
        + 0.3 * jnp.sum(jnp.sin(2.0 * x))
        + jnp.dot(y, a_mat @ x)
        - 0.5 * jnp.sum(y**2)
    )

  prob = px.MinMaxProblem(
      coupling=coupling,
      prox_x=px.prox.Box(lower=-2.0, upper=2.0),
      prox_y=px.prox.Box(lower=-2.0, upper=2.0),
      m_x=1.2,
  )
  sol = px.solve(
      prob,
      (jnp.array([0.5, -0.5]), jnp.array([0.0, 0.0])),
      opt_tol=1e-4,
  )
  assert sol.status_enum == px.SolveStatus.CONVERGED
  assert float(sol.stats.opt_residual) <= 1e-4

  def _solve_one(x_init: jax.Array) -> jax.Array:
    s = px.solve(
        prob,
        (x_init, jnp.zeros(2)),
        solver=px.solvers.ExtraGradient(opt_tol=1e-3),
    )
    return s.stats.opt_residual

  x_batch = jnp.array([[0.3, -0.2], [-0.4, 0.5], [0.1, 0.1]])
  residuals = jax.jit(jax.vmap(_solve_one))(x_batch)
  assert residuals.shape == (3,)
  assert jnp.all(residuals <= 1e-3)
