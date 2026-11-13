"""Small-scale replication of Kong & Monteiro (SIOPT 2021): nonconvex-concave min-max.

Records convergence trajectories on a 16x16 nonconvex-concave QP (m_y=0.2)
and sweeps dual concavity regularization m_y in {0.05, 0.2, 0.5, 1.0}.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import json
import time

import jax
import jax.numpy as jnp

from proxax import px


def _make_minmax_qp(
    m_y: float, n: int = 16, m: int = 16, seed: int = 33
) -> tuple[px.MinMaxProblem, tuple[jax.Array, jax.Array]]:
  key = jax.random.PRNGKey(seed)
  k1, k2, k3, k4 = jax.random.split(key, 4)
  U, _ = jnp.linalg.qr(jax.random.normal(k1, (n, n)))
  eigs_x = jnp.concatenate([-jnp.geomspace(0.5, 2.0, n // 3), jnp.geomspace(0.5, 8.0, n - n // 3)])
  Q_x = U @ jnp.diag(eigs_x) @ U.T
  V = jax.random.normal(k2, (m, m)) / jnp.sqrt(m)
  Q_y = V.T @ V + float(m_y) * jnp.eye(m)
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
  m_ys = [0.05, 0.2, 0.5, 1.0]
  solvers = [
      ("ProxPointMinMax", px.solvers.ProxPointMinMax(opt_tol=tol, max_outer_iters=250)),
      ("ExtraGradient", px.solvers.ExtraGradient(opt_tol=tol, max_iters=2500)),
      ("PDHG", px.solvers.PDHG(opt_tol=tol, max_iters=2500)),
  ]

  rows = []
  trajectories = {}
  for m_y in m_ys:
    prob, z0 = _make_minmax_qp(m_y)
    for name, solver in solvers:
      track = (m_y == 0.2)
      t0 = time.perf_counter()
      sol = px.solve(prob, z0, solver=solver, track_history=track)
      jax.block_until_ready(sol.x)
      wall_s = time.perf_counter() - t0
      oracle_evals = (
          int(sol.stats.smooth_evals)
          + int(sol.stats.grad_evals)
          + int(sol.stats.prox_evals)
      )
      rows.append({
          "m_y": m_y,
          "solver": name,
          "status": sol.status_enum.name,
          "outer_iters": int(sol.stats.outer_iters),
          "oracle_evals": oracle_evals,
          "obj_val": float(sol.stats.obj_val),
          "opt_residual": float(sol.stats.opt_residual),
          "wall_s": round(wall_s, 3),
      })
      if track and sol.history is not None:
        h = sol.history
        evals = [
            int(s + g + p_ev)
            for s, g, p_ev in zip(h.smooth_evals, h.grad_evals, h.prox_evals)
        ]
        trajectories[name] = {
            "outer_iters": [int(x) for x in h.outer_iters],
            "oracle_evals": evals,
            "opt_residuals": [float(x) for x in h.opt_residuals],
            "obj_vals": [float(x) for x in h.obj_vals],
        }
  print(json.dumps({"rows": rows, "trajectories": trajectories}, indent=2))


if __name__ == "__main__":
  main()
