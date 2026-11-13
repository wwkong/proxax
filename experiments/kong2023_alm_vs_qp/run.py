"""Small-scale replication of Kong, Monteiro, Melo (SIOPT 2023) & Kong, Monteiro (COAP 2023).

Sweeps tolerance eps in {1e-2, 3e-3, 1e-3, 3e-4} on:
1. Linearly constrained nonconvex box QP (`n=24, m=4`)
2. Linearly constrained k-Fantope sparse PCA (`8x8, k=2`)
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import json
import time

import jax
import jax.numpy as jnp

from proxax import px


def _make_lin_constr_qp(n: int = 24, m: int = 4, seed: int = 42) -> tuple[px.CompositeProblem, jax.Array]:
  key = jax.random.PRNGKey(seed)
  k1, k2, k3 = jax.random.split(key, 3)
  U, _ = jnp.linalg.qr(jax.random.normal(k1, (n, n)))
  eigs = jnp.concatenate([
      -jnp.geomspace(0.5, 5.0, n // 3),
      jnp.geomspace(0.5, 25.0, n - n // 3),
  ])
  Q = U @ jnp.diag(eigs) @ U.T
  c = jax.random.normal(k2, (n,)) * 0.4
  A = jax.random.normal(k3, (m, n)) / jnp.sqrt(n)
  x_feas = jnp.full((n,), 0.5)
  b = A @ x_feas

  def _obj(x: jax.Array) -> jax.Array:
    return 0.5 * jnp.dot(x, Q @ x) + jnp.dot(c, x)

  prob = px.CompositeProblem(
      smooth=px.oracle.as_smooth_term(_obj),
      proximal=px.prox.Box(lower=0.0, upper=1.0),
      constraints=px.constraints.equality(A, b),
  )
  return prob, jnp.zeros(n)


def _make_fantope_spca(n: int = 8, k: int = 2, seed: int = 77) -> tuple[px.CompositeProblem, jax.Array]:
  key = jax.random.PRNGKey(seed)
  k1, k2 = jax.random.split(key)
  Z = jax.random.normal(k1, (30, n))
  Sigma = (Z.T @ Z) / 30.0
  A_mat = jax.random.normal(k2, (2, n * n)) / n
  X_feas = (k / n) * jnp.eye(n)
  b_vec = A_mat @ X_feas.reshape(-1)

  def _obj(X: jax.Array) -> jax.Array:
    return -jnp.sum(Sigma * X) + 0.05 * jnp.sum(jnp.sqrt(X ** 2 + 1e-4))

  def _lin(X: jax.Array) -> jax.Array:
    return A_mat @ X.reshape(-1)

  prob = px.CompositeProblem(
      smooth=px.oracle.as_smooth_term(_obj),
      proximal=px.prox.Fantope(k=k),
      constraints=px.constraints.equality(_lin, b_vec),
  )
  return prob, X_feas


def main() -> None:
  tols = [1e-2, 3e-3, 1e-3, 3e-4]
  problems = [
      ("Linearly Constrained QP (n=24, m=4)", *_make_lin_constr_qp()),
      ("Constrained k-Fantope SPCA (8x8, k=2)", *_make_fantope_spca()),
  ]

  rows = []
  for prob_name, prob, x0 in problems:
    for tol in tols:
      solvers = [
          ("IALM (APD)", px.solvers.IALM(opt_tol=tol, feas_tol=tol, subproblem_solver="apd", max_alm_iters=50, max_inner_iters=500)),
          ("IALM (PANOC)", px.solvers.IALM(opt_tol=tol, feas_tol=tol, subproblem_solver="panoc", max_alm_iters=50, max_inner_iters=500)),
          ("SProxALM", px.solvers.SProxALM(opt_tol=tol, feas_tol=tol, max_iters=2500)),
          ("QP-APD", px.solvers.QPAPD(opt_tol=tol, feas_tol=tol, max_penalty_iters=25, max_inner_iters=500)),
          ("QP-AIPP", px.solvers.QPAIPP(opt_tol=tol, feas_tol=tol, max_penalty_iters=25, max_inner_iters=500)),
      ]
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
            "problem": prob_name,
            "tol": tol,
            "solver": name,
            "status": sol.status_enum.name,
            "outer_iters": int(sol.stats.outer_iters),
            "oracle_evals": oracle_evals,
            "opt_residual": float(sol.stats.opt_residual),
            "feas_residual": float(sol.stats.feas_residual),
            "wall_s": round(wall_s, 3),
        })
  print(json.dumps(rows, indent=2))


if __name__ == "__main__":
  main()
