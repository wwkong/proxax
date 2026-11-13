"""Collect JIT-compiled convergence trajectories on Table-1 DQP instances.

1. (B=2, m=4, n=6) where PAdaptADMM-prac wins.
2. (B=4, m=8, n=12) where DPADMM wins.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
from experiments.maia2026_table1_dqp.run import _initial_scales
from experiments.maia2026_table1_dqp.run import make_dqp_instance

from proxax import px

jax.config.update("jax_enable_x64", True)


def _collect_instance(
    num_blocks: int,
    m: int,
    n: int,
    *,
    rel_tol: float = 1e-3,
    seed: int = 0,
    admm_caps: list[int],
) -> dict[str, dict[str, list[float]]]:
    problem, x0 = make_dqp_instance(
        num_blocks=num_blocks, m=m, n=n, omega=100.0, seed=seed
    )
    g_scale, f_scale = _initial_scales(problem, x0)
    opt_tol = rel_tol * g_scale
    feas_tol = rel_tol * f_scale
    c0 = 1.0 / f_scale

    trajs: dict[str, dict[str, list[float]]] = {
        "PAdaptADMM-prac": {"grad_evals": [], "rel_opt": [], "rel_feas": []},
        "PAdaptADMM-theo": {"grad_evals": [], "rel_opt": [], "rel_feas": []},
        "DPADMM": {"grad_evals": [], "rel_opt": [], "rel_feas": []},
        "ADMM": {"grad_evals": [], "rel_opt": [], "rel_feas": []},
    }

    for adaptive, key in [
        (True, "PAdaptADMM-prac"),
        (False, "PAdaptADMM-theo"),
    ]:
        for s_cap in [1, 2, 3, 5, 8, 12, 18, 25]:
            sol = px.solve(
                problem,
                x0,
                solver=px.solvers.PAdaptADMM(
                    opt_tol=opt_tol,
                    feas_tol=feas_tol,
                    lmt_bound=1e3 * opt_tol,
                    gamma_init=10.0,
                    max_stages=s_cap,
                    max_stage_iters=2500,
                    max_inner_iters=200,
                    adaptive_tau=adaptive,
                ),
            )
            trajs[key]["grad_evals"].append(float(sol.stats.grad_evals))
            trajs[key]["rel_opt"].append(
                float(sol.stats.opt_residual) / g_scale
            )
            trajs[key]["rel_feas"].append(
                float(sol.stats.feas_residual) / f_scale
            )
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
                max_inner_iters=200,
            ),
        )
        trajs["DPADMM"]["grad_evals"].append(float(sol.stats.grad_evals))
        trajs["DPADMM"]["rel_opt"].append(
            float(sol.stats.opt_residual) / g_scale
        )
        trajs["DPADMM"]["rel_feas"].append(
            float(sol.stats.feas_residual) / f_scale
        )
        if sol.status_enum.name == "CONVERGED":
            break

    for it_cap in admm_caps:
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
        trajs["ADMM"]["rel_feas"].append(
            float(sol.stats.feas_residual) / f_scale
        )
        if sol.status_enum.name == "CONVERGED":
            break

    return trajs


def main() -> None:
    """Collects trajectories for (2, 4, 6) and (4, 8, 12) DQP instances."""
    out_dir = Path("/tmp/proxax_repl")
    out_dir.mkdir(parents=True, exist_ok=True)

    t_small = _collect_instance(
        2, 4, 6, admm_caps=[50, 200, 500, 1500, 4000, 10000, 20000]
    )
    (out_dir / "dqp_trajectories.json").write_text(
        json.dumps(t_small, indent=2)
    )

    t_med = _collect_instance(
        4, 8, 12, seed=2, admm_caps=[50, 200, 500, 1500, 4000, 10000, 25000]
    )
    (out_dir / "dqp_trajectories_dpadmm_wins.json").write_text(
        json.dumps(t_med, indent=2)
    )
    print("Saved both DQP trajectory JSONs")


if __name__ == "__main__":
    main()
