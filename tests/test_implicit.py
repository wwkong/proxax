import jax
import jax.numpy as jnp

from proxax import px


def test_implicit_unconstrained_qp_and_l1_weight() -> None:
  a_mat = jnp.array([[2.0, 0.3], [0.3, 1.5]])
  x_target = jnp.array([1.0, -0.5])

  def loss_fn(theta: jax.Array, linear_solver: str) -> jax.Array:
    def make_prob(th: jax.Array) -> px.CompositeProblem:
      return px.CompositeProblem(
          smooth=lambda x: 0.5 * jnp.sum((a_mat @ x - th) ** 2),
          proximal=px.prox.Box(lower=-2.0, upper=2.0),
      )

    x_hat = px.implicit_solve(
        make_prob,
        jnp.zeros(2),
        theta,
        solver=px.solvers.APD(opt_tol=1e-5),
        linear_solver=linear_solver,
        linear_tol=1e-6,
    )
    return 0.5 * jnp.sum((x_hat - x_target) ** 2)

  theta0 = jnp.array([1.2, -0.8])
  eps = 1e-3
  fd = jnp.array([
      (
          loss_fn(theta0 + eps * e, "gmres")
          - loss_fn(theta0 - eps * e, "gmres")
      )
      / (2.0 * eps)
      for e in jnp.eye(2)
  ])

  for ls in ("gmres", "cg"):
    g = jax.jit(jax.grad(lambda th, l_s=ls: loss_fn(th, l_s)))(theta0)
    assert jnp.allclose(g, fd, atol=2e-2)

  # Analytical L1 weight derivative: x_hat(lam) = 2 - lam for 0 < lam < 2.
  def l1_loss(lam: jax.Array) -> jax.Array:
    x_hat = px.implicit_solve(
        lambda w: px.CompositeProblem(
            smooth=lambda x: 0.5 * jnp.sum((x - 2.0) ** 2),
            proximal=px.prox.L1(weight=w),
        ),
        jnp.array([0.5]),
        lam,
        solver=px.solvers.APD(opt_tol=1e-5),
        linear_tol=1e-6,
    )
    return 0.5 * jnp.sum(x_hat**2)

  lam0 = jnp.asarray(0.4)
  g_lam = jax.grad(l1_loss)(lam0)
  assert jnp.allclose(g_lam, -(2.0 - lam0), atol=2e-2)


def test_implicit_constrained_kkt_ialm_and_qpapd() -> None:
  c_vec = jnp.array([0.4, -0.2, 0.7])
  weights = jnp.array([1.5, -0.5, 2.0])

  def make_prob(rhs: jax.Array) -> px.CompositeProblem:
    return px.CompositeProblem(
        smooth=lambda x: 0.5 * jnp.sum((x - c_vec) ** 2),
        constraints=px.constraints.equality(jnp.sum, rhs=rhs),
    )

  rhs0 = jnp.asarray(1.2)
  expected = jnp.mean(weights)

  for solver in (
      px.solvers.IALM(opt_tol=1e-5, feas_tol=1e-5),
      px.solvers.QPAPD(opt_tol=1e-5, feas_tol=1e-5),
  ):
    g_rhs = jax.jit(
        jax.grad(
            lambda r, slv=solver: jnp.sum(
                weights
                * px.implicit_solve(
                    make_prob,
                    jnp.zeros(3),
                    r,
                    solver=slv,
                    linear_tol=1e-6,
                )
            )
        )
    )(rhs0)
    assert jnp.allclose(g_rhs, expected, atol=2e-2)

  # Also check return_dual=True differentiation.
  def dual_loss(r: jax.Array) -> jax.Array:
    x_hat, p_hat = px.implicit_solve(
        make_prob,
        jnp.zeros(3),
        r,
        solver=px.solvers.IALM(opt_tol=1e-5, feas_tol=1e-5),
        return_dual=True,
    )
    return jnp.sum(x_hat) + 0.5 * jnp.sum(p_hat**2)

  g_dual = jax.grad(dual_loss)(rhs0)
  assert jnp.isfinite(g_dual)
