"""Collect JIT-compiled convergence trajectories on Table-1 DQP (B=2, m=4, n=6)."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

from proxax import px
from experiments.maia2026_table1_dqp.run import _initial_scales, make_dqp_instance


def main() -> None:
  problem, x0 = make_dqp_instance(num_blocks=2, m=4, n=6, omega=100.0, seed=0)
  g_scale, f_scale = _initial_scales(problem, x0)
  rel_tol = 1e-3
  opt_tol = rel_tol * g_scale
  feas_tol = rel_tol * f_scale
  c0 = 1.0 / f_scale

  trajs: dict[str, dict[str, list[float]]] = {
      "PAdaptADMM-prac": {"grad_evals": [], "rel_opt": [], "rel_feas": []},
      "PAdaptADMM-theo": {"grad_evals": [], "rel_opt": [], "rel_feas": []},
      "DPADMM": {"grad_evals": [], "rel_opt": [], "rel_feas": []},
      "ADMM": {"grad_evals": [], "rel_opt": [], "rel_feas": []},
  }

  for adaptive, key in [(True, "PAdaptADMM-prac"), (False, "PAdaptADMM-theo")]:
    for s_cap in [1, 2, 3, 5, 8, 12, 18, 25, 30]:
      sol = px.solve(
          problem,
          x0,
          solver=px.solvers.PAdaptADMM(
              opt_tol=opt_tol,
              feas_tol=feas_tol,
              lmt_bound=1e3 * opt_tol,
              gamma_init=10.0,
              max_stages=s_cap,
              max_stage_iters=5000,
              max_inner_iters=200,
              adaptive_tau=adaptive,
          ),
      )
      trajs[key]["grad_evals"].append(float(sol.stats.grad_evals))
      trajs[key]["rel_opt"].append(float(sol.stats.opt_residual) / g_scale)
      trajs[key]["rel_feas"].append(float(sol.stats.feas_residual) / f_scale)
      if sol.status_enum.name == "CONVERGED":
        break

  for s_cap in [1, 2, 3, 5, 8, 12, 18, 25, 30]:
    sol = px.solve(
        problem,
        x0,
        solver=px.solvers.DPADMM(
            opt_tol=opt_tol,
            feas_tol=feas_tol,
            theta=0.0,
            chi=1.0,
            lam_init=0.5,
            c_init=c0,
            max_stages=s_cap,
            max_stage_iters=1000,
        ),
    )
    trajs["DPADMM"]["grad_evals"].append(float(sol.stats.grad_evals))
    trajs["DPADMM"]["rel_opt"].append(float(sol.stats.opt_residual) / g_scale)
    trajs["DPADMM"]["rel_feas"].append(float(sol.stats.feas_residual) / f_scale)
    if sol.status_enum.name == "CONVERGED":
      break

  for it_cap in [50, 200, 500, 1500, 4000, 10000, 20000]:
    sol = px.solve(
        problem,
        x0,
        solver=px.solvers.ADMM(
            opt_tol=opt_tol,
            feas_tol=feas_tol,
            rho=c0,
            max_iters=it_cap,
        ),
    )
    trajs["ADMM"]["grad_evals"].append(float(sol.stats.grad_evals))
    trajs["ADMM"]["rel_opt"].append(float(sol.stats.opt_residual) / g_scale)
    trajs["ADMM"]["rel_feas"].append(float(sol.stats.feas_residual) / f_scale)
    if sol.status_enum.name == "CONVERGED":
      break

  out = Path("/tmp/proxax_repl/dqp_trajectories.json")
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(trajs, indent=2))
  print("Saved", out)


if __name__ == "__main__":
  main()
