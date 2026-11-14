from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import sparse

from proxax import certificates
from proxax import constraints
from proxax import implicit
from proxax import oracle
from proxax import problem as problem_mod
from proxax import prox
from proxax import px
from proxax import tree


def test_callable_linear_operator_auto_adjoint_on_pytree() -> None:
  x_ex = {"w": jnp.array([[1.0, 2.0], [3.0, 4.0]]), "b": jnp.array([0.5, -1.0])}

  def lin_fn(p: dict[str, jax.Array]) -> jax.Array:
    return jnp.sum(p["w"], axis=1) + 2.0 * p["b"]

  op = constraints.as_linear_operator(lin_fn).bind(x_ex)
  y = jnp.array([1.7, -0.9])
  # Verify adjoint identity <A(x), y> == <x, A^*(y)>
  lhs = tree.vdot(op.matvec(x_ex), y)
  rhs = tree.vdot(x_ex, op.rmatvec(y))
  assert jnp.isclose(lhs, rhs, atol=1e-6)

  sum_con = constraints.equality(jnp.sum, rhs=1.0).bind(jnp.array([0.2, 0.3]))
  assert jnp.isclose(sum_con.residual(jnp.array([0.2, 0.3])), -0.5)
  assert jnp.allclose(sum_con.adjoint(jnp.asarray(2.0)), jnp.array([2.0, 2.0]))

  id_con = constraints.box(constraints.identity, lower=0.0, upper=1.0).bind(
      jnp.array([-0.5, 1.5])
  )
  assert id_con.operator_norm_sq == 1.0
  assert jnp.allclose(
      id_con.residual(jnp.array([-0.5, 1.5])), jnp.array([-0.5, 0.5])
  )


def test_penalized_oracle_grad_matches_autodiff_product() -> None:
  a_eq = jnp.array([[1.0, -1.0, 0.5], [0.0, 2.0, 1.0]])
  b_eq = jnp.array([1.0, -0.5])
  c_box = jnp.array([[1.0, 0.0, -1.0], [0.5, 1.0, 0.5]])

  prob = problem_mod.CompositeProblem(
      smooth=lambda x: 0.25 * jnp.sum(x**4) - 0.5 * jnp.sum(x**2),
      proximal=prox.L1(0.2),
      constraints={
          "eq": constraints.equality(a_eq, b_eq),
          "box": constraints.box(c_box, lower=-0.2, upper=0.4),
      },
  )
  x = jnp.array([0.7, -0.8, 1.1])
  bound_c = prob.bind_constraints(x)
  assert bound_c is not None
  mult = {
      "eq": jnp.array([0.3, -0.4]),
      "box": jnp.array([-0.2, 0.5]),
  }
  pen_oracle = oracle.PenalizedOracle(
      base=prob.oracle,
      constraint=bound_c,
      multiplier=mult,
      penalty=jnp.asarray(3.5),
      dampen=0.25,
  )
  val, grad = pen_oracle.smooth_value_and_grad(x)
  ad_val, ad_grad = jax.value_and_grad(pen_oracle.smooth_value)(x)
  assert jnp.isclose(val, ad_val, atol=1e-6)
  assert jnp.allclose(grad, ad_grad, atol=1e-6)


def test_refine_produces_exact_subdifferential_certificate() -> None:
  q_mat = jnp.array([[3.0, -1.0], [-1.0, -2.0]])  # Indefinite (nonconvex)!
  c_vec = jnp.array([0.5, -1.2])
  weight = 0.4
  comp_oracle = oracle.CompositeOracle(
      smooth=lambda x: 0.5 * jnp.dot(x, q_mat @ x) + jnp.dot(c_vec, x),
      proximal=prox.L1(weight),
  )
  y = jnp.array([1.3, -0.7])
  res_jit = certificates.refine(comp_oracle, y, 0.5, jit=True)
  res_eager = certificates.refine(comp_oracle, y, 0.5, jit=False)
  assert jnp.allclose(res_jit.x_hat, res_eager.x_hat, atol=1e-6)
  assert jnp.allclose(res_jit.v_hat, res_eager.v_hat, atol=1e-6)
  assert res_jit.obj_val <= comp_oracle.value(y) + 1e-6

  # Verify v_hat - grad_f(x_hat) in d(weight * ||x_hat||_1)
  _, grad_x_hat = comp_oracle.smooth_value_and_grad(res_jit.x_hat)
  subg_h = res_jit.v_hat - grad_x_hat
  for x_i, g_i in zip(res_jit.x_hat.tolist(), subg_h.tolist(), strict=True):
    if abs(x_i) > 1e-6:
      assert abs(g_i - weight * (1.0 if x_i > 0 else -1.0)) < 1e-5
    else:
      assert abs(g_i) <= weight + 1e-5


def test_bcoo_sparse_matrix_operator_and_solve() -> None:
  a_dense = jnp.array([
      [1.5, 0.0, -0.5, 0.0],
      [0.0, 2.0, 0.0, -1.0],
      [-0.5, 0.0, 1.0, 0.5],
  ])
  a_bcoo = sparse.BCOO.fromdense(a_dense)
  x = jnp.array([0.4, -0.3, 0.8, 0.2])
  y = jnp.array([1.1, -0.6, 0.9])

  op = constraints.as_linear_operator(a_bcoo).bind(x)
  assert jnp.allclose(op.matvec(x), a_dense @ x, atol=1e-6)
  assert jnp.allclose(op.rmatvec(y), a_dense.T @ y, atol=1e-6)
  assert jnp.isclose(
      tree.vdot(op.matvec(x), y), tree.vdot(x, op.rmatvec(y)), atol=1e-6
  )

  b = jnp.array([0.5, 0.6, 0.4])
  q = jnp.array([0.2, -0.8, 0.5, -0.3])
  x0 = jnp.zeros(4)

  def _make_prob(
      mat: jax.Array | sparse.BCOO, q_vec: jax.Array
  ) -> problem_mod.CompositeProblem:
    return problem_mod.CompositeProblem(
        smooth=lambda z: 0.5 * jnp.dot(z, z) + jnp.dot(q_vec, z),
        proximal=prox.NonNegative(),
        constraints=constraints.less_equal(mat, b),
    )

  sol_dense = px.solve(
      _make_prob(a_dense, q), x0, method="ialm", opt_tol=1e-5, feas_tol=1e-5
  )
  sol_bcoo = px.solve(
      _make_prob(a_bcoo, q), x0, method="ialm", opt_tol=1e-5, feas_tol=1e-5
  )
  assert jnp.allclose(sol_dense.x, sol_bcoo.x, atol=1e-4)

  def _loss(mat: jax.Array | sparse.BCOO, q_vec: jax.Array) -> jax.Array:
    x_hat = implicit.implicit_solve(
        lambda qv: _make_prob(mat, qv),
        x0,
        q_vec,
        method="ialm",
        opt_tol=1e-5,
        feas_tol=1e-5,
        linear_tol=1e-5,
    )
    return 0.5 * jnp.sum(x_hat**2)

  v_d, g_d = jax.jit(jax.value_and_grad(lambda qv: _loss(a_dense, qv)))(q)
  v_s, g_s = jax.jit(jax.value_and_grad(lambda qv: _loss(a_bcoo, qv)))(q)
  assert jnp.isclose(v_d, v_s, atol=1e-4)
  assert jnp.allclose(g_d, g_s, atol=1e-3)
