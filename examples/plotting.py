"""End-to-end diagnostic plotting examples using `proxax.px.plotting`."""

from __future__ import annotations

import pathlib
import time

import jax
import jax.numpy as jnp

from proxax import px


def _make_lasso_problem() -> tuple[px.CompositeProblem, jax.Array]:
  """Builds a small constrained Lasso problem and initial point."""
  a_mat = jnp.array([[2.0, 1.0, -1.0], [0.5, -1.5, 2.0], [1.0, 1.0, 1.0]])
  b_vec = jnp.array([1.0, -1.5, 0.5])
  c_mat = jnp.array([[1.0, 1.0, 1.0]])
  d_vec = jnp.array([1.0])

  def _loss(x: jax.Array) -> jax.Array:
    return 0.5 * jnp.sum((a_mat @ x - b_vec) ** 2)

  problem = px.CompositeProblem(
      smooth=px.oracle.as_smooth_term(_loss),
      proximal=px.prox.L1(weight=0.1),
      constraints=px.constraints.equality(c_mat, d_vec),
  )
  return problem, jnp.zeros(3)


def _bilinear(x: jax.Array, y: jax.Array) -> jax.Array:
  """Bilinear saddle-point coupling."""
  return jnp.sum(x * y)


def main(out_dir: str = "plots_out") -> None:
  """Runs solver comparisons and writes diagnostic plots to `out_dir`."""
  dest = pathlib.Path(out_dir)
  dest.mkdir(parents=True, exist_ok=True)
  problem, x0 = _make_lasso_problem()

  histories: dict[str, px.HistoryTracker] = {}
  solutions: dict[str, px.Solution] = {}
  runtimes: dict[str, float] = {}

  for label, method in [("IALM", "ialm"), ("QP-APD", "qp_apd")]:
    sol = px.solve(
        problem,
        x0,
        method=method,
        opt_tol=1e-4,
        feas_tol=1e-4,
        track_history=True,
    )
    assert sol.history is not None
    histories[label] = sol.history
    solutions[label] = sol

    t0 = time.perf_counter()
    _ = px.solve(problem, x0, method=method, opt_tol=1e-4, feas_tol=1e-4)
    runtimes[label] = time.perf_counter() - t0

  fig_hist, _ = px.plotting.plot_histories(
      histories,
      x_axis="evals",
      log_x=True,
      title="Constrained Lasso Convergence",
  )
  fig_hist.savefig(dest / "histories.svg", format="svg")

  fig_brk, _ = px.plotting.plot_eval_breakdown(
      solutions,
      runtimes=runtimes,
      title="Oracle Breakdown & Warm-Started Runtime",
  )
  fig_brk.savefig(dest / "eval_breakdown.svg", format="svg")

  tols = [1e-2, 1e-3, 1e-4]
  evals_by_solver: dict[str, list[float]] = {"IALM": [], "QP-APD": []}
  for tol in tols:
    for label, method in [("IALM", "ialm"), ("QP-APD", "qp_apd")]:
      res = px.solve(problem, x0, method=method, opt_tol=tol, feas_tol=tol)
      evals_by_solver[label].append(
          float(res.stats.grad_evals + res.stats.prox_evals)
      )
  fig_tol, _ = px.plotting.plot_tolerance_scaling(
      tols,
      evals_by_solver,
      title="Oracle Complexity vs. Tolerance",
  )
  fig_tol.savefig(dest / "tolerance_scaling.svg", format="svg")

  mm_problem = px.MinMaxProblem(
      coupling=_bilinear, L_xx=0.0, L_xy=1.0, L_yy=0.0
  )
  mm_x0 = (jnp.array([1.2]), jnp.array([1.0]))
  trajs: dict[str, tuple[list[float], list[float]]] = {}
  for label, method in [("ExtraGradient", "extra_gradient"), ("PDHG", "pdhg")]:
    tracker = px.HistoryTracker(record_iterates=True)
    px.solve(mm_problem, mm_x0, method=method, opt_tol=1e-4, callback=tracker)
    xs = [float(mm_x0[0][0])] + [float(it[0][0]) for it in tracker.iterates]
    ys = [float(mm_x0[1][0])] + [float(it[1][0]) for it in tracker.iterates]
    trajs[label] = (xs, ys)
  fig_phase, _ = px.plotting.plot_phase_trajectory(
      trajs,
      optimum=(0.0, 0.0),
      title="Bilinear Min-Max Phase Portrait",
  )
  fig_phase.savefig(dest / "phase.svg", format="svg")
  print(f"Wrote diagnostic plots to {dest.resolve()}")


if __name__ == "__main__":
  main()
