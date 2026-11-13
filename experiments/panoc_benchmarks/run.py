"""Benchmark comparing `PANOC`, `APD`, `AdapFISTA`, and `ECG`."""

from __future__ import annotations

import json
import math

import jax
import jax.numpy as jnp

from proxax import px


def _build_problems() -> list[tuple[str, px.CompositeProblem, jax.Array]]:
  """Constructs 6 ill-conditioned and exotic-prox composite benchmarks."""
  problems: list[tuple[str, px.CompositeProblem, jax.Array]] = []

  for cond in (100.0, 1000.0):
    scales = jnp.geomspace(1.0 / cond, 1.0, 16)
    x_ref = jnp.linspace(-1.5, 1.5, 16)

    def _lasso(
        x: jax.Array, s: jax.Array = scales, r: jax.Array = x_ref
    ) -> jax.Array:
      return 0.5 * jnp.sum(s * (x - r) ** 2)

    problems.append((
        f"Lasso (cond={int(cond)})",
        px.CompositeProblem(
            smooth=px.oracle.as_smooth_term(_lasso),
            proximal=px.prox.L1(weight=0.05),
        ),
        jnp.zeros(16),
    ))

  key = jax.random.PRNGKey(7)
  a_log = jax.random.normal(key, (30, 10)) * jnp.geomspace(0.2, 3.0, 10)
  y_log = jnp.where(jnp.arange(30) % 2 == 0, 1.0, -1.0)

  def _logistic(x: jax.Array) -> jax.Array:
    margins = y_log * (a_log @ x)
    return jnp.mean(jnp.logaddexp(0.0, -margins))

  problems.append((
      "Sparse Logistic",
      px.CompositeProblem(
          smooth=px.oracle.as_smooth_term(_logistic),
          proximal=px.prox.L1(weight=0.02),
      ),
      jnp.zeros(10),
  ))

  target_mat = jnp.array(
      [[1.5, 0.7, -0.3], [0.7, 0.2, 0.4], [-0.3, 0.4, -0.5]]
  )

  def _sdp(x: jax.Array) -> jax.Array:
    return 0.5 * jnp.sum((x - target_mat) ** 2)

  problems.append((
      "Spectrahedron SDP",
      px.CompositeProblem(
          smooth=px.oracle.as_smooth_term(_sdp),
          proximal=px.prox.Spectrahedron(scale=1.0),
      ),
      jnp.eye(3) / 3.0,
  ))

  cov = jnp.array([
      [0.4, 0.1, -0.05, 0.0],
      [0.1, 0.3, 0.05, 0.02],
      [-0.05, 0.05, 0.25, -0.03],
      [0.0, 0.02, -0.03, 0.2],
  ])
  ret = jnp.array([0.12, 0.08, 0.10, 0.05])

  def _portfolio(w: jax.Array) -> jax.Array:
    return 0.5 * jnp.dot(w, cov @ w) - 0.5 * jnp.dot(ret, w)

  problems.append((
      "CappedSimplex Portfolio",
      px.CompositeProblem(
          smooth=px.oracle.as_smooth_term(_portfolio),
          proximal=px.prox.CappedSimplex(scale=1.0, upper=0.35),
      ),
      jnp.ones(4) / 4.0,
  ))

  scales_o = jnp.geomspace(0.05, 5.0, 10)
  ref_o = jnp.linspace(-1.0, 1.0, 10)

  def _oscar_loss(x: jax.Array) -> jax.Array:
    return 0.5 * jnp.sum(scales_o * (x - ref_o) ** 2)

  problems.append((
      "OSCAR Regression",
      px.CompositeProblem(
          smooth=px.oracle.as_smooth_term(_oscar_loss),
          proximal=px.prox.OSCAR(alpha=0.05, beta=0.01),
      ),
      jnp.zeros(10),
  ))
  return problems


def main() -> None:
  """Runs all 4 solvers across the 6 problems and prints JSON summary."""
  methods = [
      ("PANOC", "panoc"),
      ("APD", "apd"),
      ("AdapFISTA", "adap_fista"),
      ("ECG", "ecg"),
  ]
  rows: list[dict[str, object]] = []
  profile: dict[str, list[float]] = {name: [] for name, _ in methods}

  for pname, prob, x0 in _build_problems():
    for sname, method in methods:
      sol = px.solve(prob, x0, method=method, opt_tol=1e-5)
      evals = int(sol.stats.grad_evals + sol.stats.prox_evals)
      ok = sol.status_enum == px.SolveStatus.CONVERGED
      profile[sname].append(float(evals) if ok else math.inf)
      rows.append({
          "problem": pname,
          "solver": sname,
          "status": sol.status_enum.name,
          "iters": int(sol.stats.outer_iters),
          "evals": evals,
      })
  print(json.dumps({"rows": rows, "profile": profile}))


if __name__ == "__main__":
  main()
