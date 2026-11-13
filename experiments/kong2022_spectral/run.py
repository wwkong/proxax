"""Small-scale replication of Kong & Monteiro (COAP 2022): spectral NCO.

Compares doubly-accelerated (`DAICG`) vs inner-accelerated (`IAICG`) inexact
composite gradient methods on small (`20 x 20`) noisy matrix completion with
nonconvex log-sum singular-value regularization.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import dataclasses
import json
import time

import jax
import jax.numpy as jnp
import optax

from proxax import px


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _BallL1VectorProx:
  weight: float
  radius: float

  @property
  def strong_convexity(self) -> float:
    return 0.0

  def value(self, x: px.typing.PyTree) -> jax.Array:
    return self.weight * jnp.sum(jnp.abs(jnp.asarray(x)))

  def prox(self, x: px.typing.PyTree, step_size: px.typing.Scalar) -> px.typing.PyTree:
    s_vec = jnp.asarray(x)
    shrunk = jnp.sign(s_vec) * jnp.maximum(jnp.abs(s_vec) - step_size * self.weight, 0.0)
    return optax.projections.projection_l2_ball(shrunk, self.radius)

  def domain_project(self, x: px.typing.PyTree) -> px.typing.PyTree:
    return optax.projections.projection_l2_ball(jnp.asarray(x), self.radius)


def _make_spectral_mc(m: int = 20, n: int = 20, rank: int = 3, seed: int = 55) -> tuple[px.SpectralProblem, jax.Array]:
  key = jax.random.PRNGKey(seed)
  k1, k2, k3, k4 = jax.random.split(key, 4)
  U = jax.random.normal(k1, (m, rank))
  V = jax.random.normal(k2, (n, rank))
  M_true = U @ V.T
  mask = (jax.random.uniform(k3, (m, n)) < 0.45).astype(jnp.float32)
  M_obs = mask * (M_true + 0.05 * jax.random.normal(k4, (m, n)))

  alpha = 0.05
  mu = 0.2
  beta = 1.0
  theta = 0.5
  kappa0 = beta / theta
  radius = float(jnp.linalg.norm(M_true) * 1.5)

  def _f1(X: jax.Array) -> jax.Array:
    return 0.5 * jnp.sum(mask * (X - M_obs) ** 2)

  def _f2_vec(s: jax.Array) -> jax.Array:
    s_abs = jnp.abs(s)
    k_tilde = beta * (1.0 - jnp.exp(-jnp.sum(s_abs ** 2) / (2.0 * theta)))
    k_sum = jnp.sum(beta * jnp.log1p(s_abs / theta) - kappa0 * s_abs)
    return alpha * k_tilde + mu * k_sum

  prob = px.SpectralProblem(
      f1=px.oracle.as_smooth_term(_f1),
      f2_vec=px.oracle.as_smooth_term(_f2_vec),
      prox_vec=_BallL1VectorProx(weight=mu * kappa0, radius=radius),
  )
  return prob, jnp.zeros((m, n))


def main() -> None:
  tol = 1e-3
  prob, x0 = _make_spectral_mc()
  solvers = [
      ("DAICG (monotone)", px.solvers.DAICG(opt_tol=tol, monotone=True, max_outer_iters=600)),
      ("DAICG (nonmonotone)", px.solvers.DAICG(opt_tol=tol, monotone=False, max_outer_iters=600)),
      ("IAICG", px.solvers.IAICG(opt_tol=tol, max_outer_iters=600)),
  ]

  rows = []
  for name, solver in solvers:
    t0 = time.perf_counter()
    sol = px.solve(prob, x0, solver=solver)
    jax.block_until_ready(sol.x)
    wall_s = time.perf_counter() - t0
    oracle_evals = int(sol.stats.smooth_evals) + int(sol.stats.grad_evals) + int(sol.stats.prox_evals)
    rows.append({
        "problem": "Spectral Matrix Completion (20x20, rank=3)",
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
