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


def test_implicit_minmax_saddle_point() -> None:
  """Implicit differentiation through `MinMaxProblem` (`ExtraGradient`)."""

  def loss_fn(theta: jax.Array) -> jax.Array:
    def make_mm(th: jax.Array) -> px.MinMaxProblem:
      def coupling(x: jax.Array, y: jax.Array) -> jax.Array:
        return (
            0.5 * jnp.sum((x - th) ** 2)
            + jnp.dot(x, y)
            - 0.5 * jnp.sum((y + 0.5 * th) ** 2)
        )

      return px.MinMaxProblem(
          coupling=coupling,
          prox_x=px.prox.Box(lower=-3.0, upper=3.0),
          prox_y=px.prox.Box(lower=-3.0, upper=3.0),
          L_xx=1.0,
          L_xy=1.0,
          L_yy=1.0,
      )

    x_hat, y_hat = px.implicit_solve(
        make_mm,
        (jnp.zeros(2), jnp.zeros(2)),
        theta,
        method="extra_gradient",
        opt_tol=1e-5,
        linear_tol=1e-6,
    )
    return 0.5 * jnp.sum(x_hat**2) + jnp.sum(y_hat)

  theta0 = jnp.array([0.8, -0.4])
  eps = 1e-3
  fd = jnp.array([
      (loss_fn(theta0 + eps * e) - loss_fn(theta0 - eps * e)) / (2.0 * eps)
      for e in jnp.eye(2)
  ])
  grad_th = jax.jit(jax.grad(loss_fn))(theta0)
  assert jnp.allclose(grad_th, fd, atol=2e-2)


def test_implicit_block_problem() -> None:
  """Implicit differentiation through `BlockProblem` (`DPADMM`)."""
  w1 = jnp.array([1.2, -0.5])
  w2 = jnp.array([-0.3, 0.9])

  def make_block(rhs: jax.Array) -> px.BlockProblem:
    def s1(x: jax.Array) -> jax.Array:
      return 0.5 * jnp.sum((x - w1) ** 2)

    def s2(x: jax.Array) -> jax.Array:
      return 0.5 * jnp.sum((x - w2) ** 2)

    return px.BlockProblem(
        blocks=(
            px.BlockTerm(smooth=px.oracle.as_smooth_term(s1)),
            px.BlockTerm(smooth=px.oracle.as_smooth_term(s2)),
        ),
        rhs=rhs,
    )

  def block_loss(rhs: jax.Array) -> jax.Array:
    x_hat = px.implicit_solve(
        make_block,
        (jnp.zeros(2), jnp.zeros(2)),
        rhs,
        method="dp_admm",
        opt_tol=1e-5,
        feas_tol=1e-5,
        linear_tol=1e-6,
    )
    return 0.5 * jnp.sum(x_hat[0] ** 2) + jnp.sum(x_hat[1])

  rhs0 = jnp.array([0.4, -0.2])
  eps = 1e-3
  fd = jnp.array([
      (block_loss(rhs0 + eps * e) - block_loss(rhs0 - eps * e)) / (2.0 * eps)
      for e in jnp.eye(2)
  ])
  grad_rhs = jax.jit(jax.grad(block_loss))(rhs0)
  assert jnp.allclose(grad_rhs, fd, atol=2e-2)

  # Also check return_dual=True on BlockProblem.
  def block_dual_loss(rhs: jax.Array) -> jax.Array:
    x_hat, p_hat = px.implicit_solve(
        make_block,
        (jnp.zeros(2), jnp.zeros(2)),
        rhs,
        method="dp_admm",
        opt_tol=1e-5,
        feas_tol=1e-5,
        return_dual=True,
    )
    return jnp.sum(x_hat[0]) + 0.5 * jnp.sum(p_hat**2)

  assert jnp.all(jnp.isfinite(jax.grad(block_dual_loss)(rhs0)))
