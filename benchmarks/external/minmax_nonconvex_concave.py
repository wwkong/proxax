"""Nonconvex-concave min-max benchmarks: Proxax vs Optax GDA baseline.

Replicates and scales the Kong & Monteiro (SIOPT 2021) nonconvex-concave QP
over `Box(-1, 1) x Simplex` across dual concavity `m_y` and dimension `(n, m)`.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import time
import typing

import jax
import jax.numpy as jnp
import optax

import proxax as px
from benchmarks import harness


def _make_nonconvex_concave_qp(
    *, m_y: float, n: int, m: int, seed: int = 33
) -> tuple[px.MinMaxProblem, tuple[jax.Array, jax.Array]]:
  """Constructs a nonconvex-concave QP (`Q_x` indefinite, `Q_y >= m_y I`)."""
  key = jax.random.PRNGKey(seed)
  k1, k2, k3, k4 = jax.random.split(key, 4)
  u_mat, _ = jnp.linalg.qr(jax.random.normal(k1, (n, n)))
  n_neg = max(1, n // 3)
  eigs_x = jnp.concatenate([
      -jnp.geomspace(0.5, 2.0, n_neg),
      jnp.geomspace(0.5, 8.0, n - n_neg),
  ])
  q_x = u_mat @ jnp.diag(eigs_x) @ u_mat.T
  v_mat = jax.random.normal(k2, (m, m)) / jnp.sqrt(m)
  q_y = v_mat.T @ v_mat + float(m_y) * jnp.eye(m)
  k_mat = jax.random.normal(k3, (n, m)) / jnp.sqrt(n)
  c_x = jax.random.normal(k4, (n,)) * 0.3
  c_y = jax.random.normal(k4, (m,)) * 0.3

  def _coupling(x: jax.Array, y: jax.Array) -> jax.Array:
    return (
        0.5 * jnp.dot(x, q_x @ x)
        + jnp.dot(x, k_mat @ y)
        - 0.5 * jnp.dot(y, q_y @ y)
        + jnp.dot(c_x, x)
        - jnp.dot(c_y, y)
    )

  prob = px.MinMaxProblem(
      coupling=_coupling,
      prox_x=px.prox.Box(lower=-1.0, upper=1.0),
      prox_y=px.prox.Simplex(scale=1.0),
  )
  return prob, (jnp.zeros(n), jnp.ones(m) / m)


def _time_optax_gda(
    *,
    problem_name: str,
    prob: px.MinMaxProblem,
    z0: tuple[jax.Array, jax.Array],
    opt_tol: float,
    lr: float = 0.05,
    max_iters: int = 5000,
) -> harness.BenchmarkResult:
  """Times a projected gradient descent-ascent baseline using Optax SGD."""
  dtype_str = "float64" if jax.config.jax_enable_x64 else "float32"
  tx_x = optax.sgd(lr)
  tx_y = optax.sgd(lr)
  grad_fn = jax.grad(prob.coupling, argnums=(0, 1))

  @jax.jit
  def _run(
      x_init: jax.Array, y_init: jax.Array
  ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    opt_x = tx_x.init(x_init)
    opt_y = tx_y.init(y_init)

    def _cond(
        state: tuple[
            jax.Array, jax.Array, jax.Array, typing.Any, typing.Any, jax.Array
        ],
    ) -> jax.Array:
      it, _, _, _, _, res = state
      return (it < max_iters) & (res > opt_tol)

    def _body(
        state: tuple[
            jax.Array, jax.Array, jax.Array, typing.Any, typing.Any, jax.Array
        ],
    ) -> tuple[
        jax.Array, jax.Array, jax.Array, typing.Any, typing.Any, jax.Array
    ]:
      it, x, y, ox, oy, _ = state
      gx, gy = grad_fn(x, y)
      ux, ox = tx_x.update(gx, ox, x)
      uy, oy = tx_y.update(-gy, oy, y)
      x_next = prob.prox_x.prox(x + ux, lr)
      y_next = prob.prox_y.prox(y + uy, lr)
      # Projected gradient residual norm at (x_next, y_next)
      gx_n, gy_n = grad_fn(x_next, y_next)
      rx = (x_next - prob.prox_x.prox(x_next - gx_n, 1.0))
      ry = (y_next - prob.prox_y.prox(y_next + gy_n, 1.0))
      res = jnp.sqrt(jnp.sum(rx**2) + jnp.sum(ry**2))
      return (it + 1, x_next, y_next, ox, oy, res)

    init_res = jnp.asarray(1e9, dtype=x_init.dtype)
    it, xf, yf, _, _, res = jax.lax.while_loop(
        _cond, _body, (jnp.asarray(0), x_init, y_init, opt_x, opt_y, init_res)
    )
    obj = prob.coupling(xf, yf)
    return it, xf, yf, res, obj

  t0 = time.perf_counter()
  it, xf, _, res, obj = _run(z0[0], z0[1])
  xf.block_until_ready()
  compile_s = time.perf_counter() - t0

  t1 = time.perf_counter()
  it, xf, _, res, obj = _run(z0[0], z0[1])
  xf.block_until_ready()
  run_s = time.perf_counter() - t1

  iters = int(it)
  res_f = float(res)
  converged = res_f <= opt_tol
  return harness.BenchmarkResult(
      suite="minmax_nonconvex_concave",
      problem_name=problem_name,
      solver_name="Optax GDA (lr=0.05)",
      dtype=dtype_str,
      status="CONVERGED" if converged else "MAX_ITERS",
      converged=converged,
      obj_val=float(obj),
      opt_residual=res_f,
      feas_residual=0.0,
      outer_iters=iters,
      inner_iters=0,
      smooth_evals=iters,
      grad_evals=2 * iters,
      prox_evals=2 * iters,
      compile_s=compile_s,
      run_s=run_s,
  )


def run_suite(
    *,
    opt_tol: float = 1e-4,
    time_limit: float = 20.0,
) -> list[harness.BenchmarkResult]:
  """Runs nonconvex-concave min-max benchmarks across `m_y` and `(n, m)`."""
  configs = [
      (0.05, 32, 32),
      (0.20, 32, 32),
      (1.00, 32, 32),
      (0.20, 64, 64),
      (0.20, 256, 256),
  ]

  proxax_solvers = [
      (
          "ProxPointMinMax",
          px.solvers.ProxPointMinMax(opt_tol=opt_tol, max_outer_iters=500),
      ),
      (
          "ExtraGradient",
          px.solvers.ExtraGradient(opt_tol=opt_tol, max_iters=5000),
      ),
      ("PDHG", px.solvers.PDHG(opt_tol=opt_tol, max_iters=5000)),
  ]

  results: list[harness.BenchmarkResult] = []
  for m_y, n, m in configs:
    name = f"NC-C QP (n={n}, m_y={m_y})"
    prob, z0 = _make_nonconvex_concave_qp(m_y=m_y, n=n, m=m)
    for solver_name, solver in proxax_solvers:
      results.append(
          harness.time_proxax_solve(
              suite="minmax_nonconvex_concave",
              problem_name=name,
              solver_name=solver_name,
              prob=prob,
              x0=z0,
              solver=solver,
              opt_tol=opt_tol,
              time_limit=time_limit,
          )
      )
    results.append(
        _time_optax_gda(
            problem_name=name,
            prob=prob,
            z0=z0,
            opt_tol=opt_tol,
        )
    )
  return results


def main() -> None:
  """CLI entry point for nonconvex-concave min-max benchmarks."""
  parser = argparse.ArgumentParser(
      description="Benchmark Proxax nonconvex-concave MinMax solvers vs GDA."
  )
  parser.add_argument("--opt-tol", type=float, default=1e-4)
  parser.add_argument("--time-limit", type=float, default=20.0)
  parser.add_argument("--json-out", type=pathlib.Path, default=None)
  args = parser.parse_args()

  results = run_suite(opt_tol=args.opt_tol, time_limit=args.time_limit)
  print(harness.format_markdown_table(results))
  if args.json_out is not None:
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    payload = [dataclasses.asdict(r) for r in results]
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
  main()
