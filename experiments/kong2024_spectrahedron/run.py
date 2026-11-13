"""Small-scale replication of Kong (SIOPT 2024): curvature-free nonconvex QPs.

Sweeps condition number kappa in {10, 40, 100, 250, 500} on:
1. Nonconvex Spectrahedron QP (10x10)
2. Nonconvex Simplex QP (n=40)
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import json
import time

import jax
import jax.numpy as jnp

from proxax import px


def _make_spectrahedron_qp(
    kappa: float, n: int = 10, seed: int = 7
) -> tuple[px.CompositeProblem, jax.Array]:
  key = jax.random.PRNGKey(seed)
  k1, k2, k3 = jax.random.split(key, 3)
  A = jax.random.normal(k1, (n, n)) / jnp.sqrt(n)
  B = jax.random.normal(k2, (n, n)) / jnp.sqrt(n)
  C_raw = jax.random.normal(k3, (n, n))
  C = 0.5 * (C_raw + C_raw.T)
  scale_pos = float(kappa) / 10.0

  def _obj(X: jax.Array) -> jax.Array:
    QX = scale_pos * (A @ X @ A.T) - 1.5 * (B @ X @ B.T)
    return 0.5 * jnp.sum(X * QX) + jnp.sum(C * X)

  prob = px.CompositeProblem(
      smooth=px.oracle.as_smooth_term(_obj),
      proximal=px.prox.Spectrahedron(scale=1.0),
  )
  return prob, jnp.eye(n) / n


def _make_simplex_qp(
    kappa: float, n: int = 40, seed: int = 19
) -> tuple[px.CompositeProblem, jax.Array]:
  key = jax.random.PRNGKey(seed)
  k1, k2, k3 = jax.random.split(key, 3)
  U, _ = jnp.linalg.qr(jax.random.normal(k1, (n, n)))
  eigs = jnp.concatenate([
      -jnp.geomspace(0.5, 4.0, n // 3),
      jnp.geomspace(1.0, float(kappa), n - n // 3),
  ])
  Q = U @ jnp.diag(eigs) @ U.T
  c = jax.random.normal(k2, (n,)) * 0.5
  x0 = jax.random.uniform(k3, (n,))
  x0 = x0 / jnp.sum(x0)

  def _obj(x: jax.Array) -> jax.Array:
    return 0.5 * jnp.dot(x, Q @ x) + jnp.dot(c, x)

  prob = px.CompositeProblem(
      smooth=px.oracle.as_smooth_term(_obj),
      proximal=px.prox.Simplex(scale=1.0),
  )
  return prob, x0


def main() -> None:
  tol = 1e-4
  kappas = [10, 40, 100, 250, 500]
  solvers = [
      ("APD", px.solvers.APD(opt_tol=tol, max_outer_iters=250)),
      ("AdapAPG", px.solvers.AdapAPG(opt_tol=tol, max_iters=3000)),
      ("PANOC", px.solvers.PANOC(opt_tol=tol, max_iters=3000)),
      ("AIPP", px.solvers.AIPP(opt_tol=tol, max_outer_iters=250)),
      ("AG", px.solvers.AG(opt_tol=tol, max_iters=3000)),
  ]

  rows = []
  for family, builder in [
      ("Spectrahedron QP (10x10)", _make_spectrahedron_qp),
      ("Simplex QP (n=40)", _make_simplex_qp),
  ]:
    for kappa in kappas:
      prob, x0 = builder(kappa)
      for name, solver in solvers:
        t0 = time.perf_counter()
        sol = px.solve(prob, x0, solver=solver)
        jax.block_until_ready(sol.x)
        wall_s = time.perf_counter() - t0
        oracle_evals = (
            int(sol.stats.smooth_evals)
            + int(sol.stats.grad_evals)
            + int(sol.stats.prox_evals)
        )
        rows.append({
            "family": family,
            "kappa": kappa,
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
