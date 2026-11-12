"""Step-by-step Optax transforms and `jax.vmap` batching with `proxax`.

Demonstrates:
1. Single-inner-step trajectory tracking with `proxax.optax_compat.apd` and
   `proxax.optax_compat.ialm` inside a `jax.jit` training/optimization loop.
2. Solving a batch of 16 nonconvex simplex QPs simultaneously via `jax.vmap`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

import proxax
from proxax import constraints
from proxax import optax_compat
from proxax import prox
from proxax import solvers


def run_optax_apd_trajectory() -> None:
  """Tracks inner/outer progress step-by-step with `optax_compat.apd`."""
  print("=== 1. Step-by-Step Trajectory via proxax.optax_compat.apd ===")
  q_diag = jnp.array([6.0, -2.5, 4.0, -1.5])
  linear_c = jnp.array([-1.2, 0.5, -0.8, 0.3])

  def loss_fn(x: jax.Array) -> jax.Array:
    return 0.5 * jnp.sum(q_diag * (x**2)) + jnp.dot(linear_c, x)

  tx = optax_compat.apd(
      prox_op=prox.Box(lower=-1.0, upper=1.0),
      opt_tol=1e-5,
      max_inner_iters=50,
  )
  params = jnp.array([0.8, 0.8, -0.5, -0.5])
  state = tx.init(params)

  @jax.jit
  def step_fn(
      p: jax.Array,
      st: optax_compat.APDTransformState,
  ) -> tuple[jax.Array, optax_compat.APDTransformState]:
    val, grads = jax.value_and_grad(loss_fn)(p)
    updates, next_st = tx.update(grads, st, p, value=val, value_fn=loss_fn)
    return optax.apply_updates(p, updates), next_st

  for step_idx in range(1, 80):
    params, state = step_fn(params, state)
    if step_idx in (1, 3, 5, 10) or bool(state.converged):
      print(
          f"step={step_idx:2d}  outer={int(state.outer_iter):2d}  "
          f"loss={float(loss_fn(params)): .6f}  "
          f"||v_hat||_2={float(state.opt_residual):.2e}  "
          f"m_hat={float(state.lower_curvature):.3f}"
      )
    if bool(state.converged):
      break


def run_optax_ialm_loop() -> None:
  """Solves an equality-constrained problem via `optax_compat.ialm`."""
  print("\n=== 2. Constrained Step-by-Step Loop via optax_compat.ialm ===")

  def smooth_fn(p: dict[str, jax.Array]) -> jax.Array:
    w = p["w"]
    return 0.5 * jnp.sum(jnp.array([3.0, -1.0, 2.0]) * (w**2)) - w[0]

  tx = optax_compat.ialm(
      constraints=constraints.equality(
          lambda p: jnp.sum(p["w"]), rhs=jnp.asarray(1.0)
      ),
      prox_op=prox.Box(lower=0.0, upper=1.0),
      opt_tol=1e-4,
      feas_tol=1e-4,
  )
  params = {"w": jnp.array([0.2, 0.3, 0.1])}
  state = tx.init(params)

  @jax.jit
  def step_fn(
      p: dict[str, jax.Array],
      st: optax_compat.IALMTransformState,
  ) -> tuple[dict[str, jax.Array], optax_compat.IALMTransformState]:
    val, grads = jax.value_and_grad(smooth_fn)(p)
    updates, next_st = tx.update(grads, st, p, value=val, value_fn=smooth_fn)
    return optax.apply_updates(p, updates), next_st

  for _ in range(300):
    params, state = step_fn(params, state)
    if bool(state.converged):
      break

  print(
      f"converged={bool(state.converged)}  w*={params['w']}  "
      f"stat={float(state.opt_residual):.2e}  "
      f"feas={float(state.feas_residual):.2e}"
  )


def run_batched_vmap_qps() -> None:
  """Solves 16 nonconvex simplex QPs simultaneously with `jax.vmap`."""
  print("\n=== 3. Batched Nonconvex Simplex QPs via jax.jit(jax.vmap(...)) ===")
  batch_size, dim_n = 16, 10
  key = jax.random.key(123)
  k_q, k_c = jax.random.split(key)
  raw = jax.random.normal(k_q, (batch_size, dim_n, dim_n)) / jnp.sqrt(dim_n)
  q_batch = 0.5 * (raw + jnp.swapaxes(raw, -1, -2)) - 0.8 * jnp.eye(dim_n)
  c_batch = jax.random.normal(k_c, (batch_size, dim_n))
  x0 = jnp.ones(dim_n) / dim_n
  solver = solvers.APD(opt_tol=1e-4)

  def solve_single(q_mat: jax.Array, c_vec: jax.Array) -> solvers.Solution:
    prob = proxax.CompositeProblem(
        smooth=lambda x: 0.5 * jnp.dot(x, q_mat @ x) + jnp.dot(c_vec, x),
        proximal=prox.Simplex(scale=1.0),
    )
    return solver.solve(prob, x0, jit=True)

  batched_sol = jax.jit(jax.vmap(solve_single))(q_batch, c_batch)
  conv_code = int(solvers.SolveStatus.CONVERGED)
  num_conv = int(jnp.sum(batched_sol.status == conv_code))
  max_res = float(jnp.max(batched_sol.stats.opt_residual))
  mean_iters = float(jnp.mean(batched_sol.stats.inner_iters))
  print(
      f"Solved {num_conv}/{batch_size} instances in parallel: "
      f"max ||v_hat||_2 = {max_res:.2e}, mean inner iters = {mean_iters:.1f}"
  )


if __name__ == "__main__":
  run_optax_apd_trajectory()
  run_optax_ialm_loop()
  run_batched_vmap_qps()
