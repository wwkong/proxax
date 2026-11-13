"""Small-scale replication of Kong & Monteiro (SIOPT 2021): nonconvex-concave min-max.

Compares `ProxPointMinMax` against `ExtraGradient` and `PDHG` on small
(`n=16, m=16`) box/simplex-constrained nonconvex-concave saddle-point QPs.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import json
import time

import jax
import jax.numpy as jnp

from proxax import px


def _make_minmax_qp(n: int = 16, m: int = 16, seed: int = 33) -> tuple[px.MinMaxProblem, tuple[jax.Array, jax.Array]]:
  key = jax.random.PRNGKey(seed)
  k1, k2, k3, k4 = jax.random.split(key, 4)
  U, _ = jnp.linalg.qr(jax.random.normal(k1, (n, n)))
  eigs_x = jnp.concatenate([-jnp.geomspace(0.5, 2.0, n // 3), jnp.geomspace(0.5, 8.0, n - n // 3)])
  Q_x = U @ jnp.diag(eigs_x) @ U.T
  V = jax.random.normal(k2, (m, m)) / jnp.sqrt(m)
  Q_y = V.T @ V + 0.2 * jnp.eye(m)
  K = jax.random.normal(k3, (n, m)) / jnp.sqrt(n)
  c_x = jax.random.normal(k4, (n,)) * 0.3
  c_y = jax.random.normal(k4, (m,)) * 0.3

  def _coupling(x: jax.Array, y: jax.Array) -> jax.Array:
    return (
        0.5 * jnp.dot(x, Q_x @ x)
        + jnp.dot(x, K @ y)
        - 0.5 * jnp.dot(y, Q_y @ y)
        + jnp.dot(c_x, x)
        - jnp.dot(c_y, y)
    )

  prob = px.MinMaxProblem(
      coupling=_coupling,
      prox_x=px.prox.Box(lower=-1.0, upper=1.0),
      prox_y=px.prox.Simplex(scale=1.0),
  )
  return prob, (jnp.zeros(n), jnp.ones(m) / m)


def main() -> None:
  tol = 1e-4
  prob, z0 = _make_minmax_qp()
  solvers = [
      ("ProxPointMinMax", px.solvers.ProxPointMinMax(opt_tol=tol, max_outer_iters=200)),
      ("ExtraGradient", px.solvers.ExtraGradient(opt_tol=tol, max_iters=2000)),
      ("PDHG", px.solvers.PDHG(opt_tol=tol, max_iters=2000)),
  ]

  rows = []
  for name, solver in solvers:
    t0 = time.perf_counter()
    sol = px.solve(prob, z0, solver=solver)
    jax.block_until_ready(sol.x)
    wall_s = time.perf_counter() - t0
    oracle_evals = int(sol.stats.smooth_evals) + int(sol.stats.grad_evals) + int(sol.stats.prox_evals)
    rows.append({
        "problem": "Nonconvex-Concave Min-Max QP (16x16)",
        "solver": name,
        "status": sol.status_enum.name,
        "outer_iters": int(sol.stats.outer_iters),
        "oracle_evals": oracle_evals,
        "obj_val": float(sol.stats.obj_val),
        "opt_residual": float(sol.stats.opt_residual),
        "wall_s": round(wall_s, 3),
    })
  print(json.dumps(rows, indent=2))


if __name__ == "__main__":
  main()
