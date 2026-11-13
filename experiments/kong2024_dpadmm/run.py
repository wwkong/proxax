"""Small-scale replication of Kong & Monteiro (SIOPT 2024): DP.ADMM.

Records JIT-compiled stage/iteration trajectories on a rank-deficient (r=2, m=6)
3-block nonconvex QP (n=24) and sweeps coupling rank r in {1, 2, 4, 6}.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import json
import time

import jax
import jax.numpy as jnp

from proxax import px


def _make_multiblock_qp(
    rank_a3: int, seed: int = 101
) -> tuple[px.BlockProblem, tuple[jax.Array, ...]]:
    key = jax.random.PRNGKey(seed)
    n_b = 8
    m = 6
    keys = jax.random.split(key, 7)

    blocks = []
    x_feas = []
    rhs = jnp.zeros(m)
    for i in range(3):
        U, _ = jnp.linalg.qr(jax.random.normal(keys[i], (n_b, n_b)))
        eigs = jnp.concatenate(
            [-jnp.array([1.5, 0.5]), jnp.linspace(0.5, 10.0, n_b - 2)]
        )
        Q_i = U @ jnp.diag(eigs) @ U.T
        c_i = jax.random.normal(keys[i + 3], (n_b,)) * 0.4
        A_i = jax.random.normal(keys[6], (m, n_b)) / jnp.sqrt(n_b)
        if i == 2 and rank_a3 < m:
            u_r = jax.random.normal(keys[i], (m, rank_a3))
            v_r = jax.random.normal(keys[i + 3], (rank_a3, n_b))
            A_i = (u_r @ v_r) / jnp.sqrt(n_b)
        x_i = jnp.full((n_b,), 0.25)
        rhs = rhs + A_i @ x_i
        x_feas.append(jnp.zeros(n_b))

        def _make_smooth(Q: jax.Array, c: jax.Array):
            return lambda x: 0.5 * jnp.dot(x, Q @ x) + jnp.dot(c, x)

        blocks.append(
            px.BlockTerm(
                smooth=_make_smooth(Q_i, c_i),
                proximal=px.prox.Box(lower=-1.0, upper=1.0),
                linear_op=A_i,
            )
        )

    return px.BlockProblem(blocks=tuple(blocks), rhs=rhs), tuple(x_feas)


def main() -> None:
    """Runs the rank sweep and rank-deficient trajectory collection."""
    tol = 1e-3
    ranks = [1, 2, 4, 6]
    solvers = [
        (
            "DPADMM",
            px.solvers.DPADMM(
                opt_tol=tol, feas_tol=tol, c_init=2.0, max_stages=80
            ),
        ),
        (
            "PAdaptADMM",
            px.solvers.PAdaptADMM(
                opt_tol=tol,
                feas_tol=tol,
                gamma_init=25.0,
                lmt_bound=10.0,
                adaptive_tau=True,
                max_stages=40,
            ),
        ),
        (
            "ADMM (rho=5)",
            px.solvers.ADMM(opt_tol=tol, feas_tol=tol, rho=5.0, max_iters=1500),
        ),
    ]

    rows = []
    for r in ranks:
        prob, x0 = _make_multiblock_qp(rank_a3=r)
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
            rows.append(
                {
                    "rank": r,
                    "solver": name,
                    "status": sol.status_enum.name,
                    "outer_iters": int(sol.stats.outer_iters),
                    "oracle_evals": oracle_evals,
                    "opt_residual": float(sol.stats.opt_residual),
                    "feas_residual": float(sol.stats.feas_residual),
                    "wall_s": round(wall_s, 3),
                }
            )

    # JIT-compiled trajectory checkpoints on the rank-deficient r=2 instance
    prob_r2, x0_r2 = _make_multiblock_qp(rank_a3=2)
    trajectories: dict[str, dict[str, list[float | int]]] = {
        "DPADMM": {
            "outer_iters": [],
            "opt_residuals": [],
            "feas_residuals": [],
        },
        "PAdaptADMM": {
            "outer_iters": [],
            "opt_residuals": [],
            "feas_residuals": [],
        },
        "ADMM (rho=5)": {
            "outer_iters": [],
            "opt_residuals": [],
            "feas_residuals": [],
        },
    }
    for s_cap in [1, 2, 5, 10, 20, 40, 80]:
        sol = px.solve(
            prob_r2,
            x0_r2,
            solver=px.solvers.DPADMM(
                opt_tol=tol, feas_tol=tol, c_init=2.0, max_stages=s_cap
            ),
        )
        trajectories["DPADMM"]["outer_iters"].append(int(sol.stats.outer_iters))
        trajectories["DPADMM"]["opt_residuals"].append(
            float(sol.stats.opt_residual)
        )
        trajectories["DPADMM"]["feas_residuals"].append(
            float(sol.stats.feas_residual)
        )
    for s_cap in [1, 2, 5, 10, 20, 40]:
        sol = px.solve(
            prob_r2,
            x0_r2,
            solver=px.solvers.PAdaptADMM(
                opt_tol=tol,
                feas_tol=tol,
                gamma_init=25.0,
                lmt_bound=10.0,
                adaptive_tau=True,
                max_stages=s_cap,
            ),
        )
        trajectories["PAdaptADMM"]["outer_iters"].append(
            int(sol.stats.outer_iters)
        )
        trajectories["PAdaptADMM"]["opt_residuals"].append(
            float(sol.stats.opt_residual)
        )
        trajectories["PAdaptADMM"]["feas_residuals"].append(
            float(sol.stats.feas_residual)
        )
    for it_cap in [25, 50, 100, 200, 300, 415]:
        sol = px.solve(
            prob_r2,
            x0_r2,
            solver=px.solvers.ADMM(
                opt_tol=tol, feas_tol=tol, rho=5.0, max_iters=it_cap
            ),
        )
        trajectories["ADMM (rho=5)"]["outer_iters"].append(
            int(sol.stats.outer_iters)
        )
        trajectories["ADMM (rho=5)"]["opt_residuals"].append(
            float(sol.stats.opt_residual)
        )
        trajectories["ADMM (rho=5)"]["feas_residuals"].append(
            float(sol.stats.feas_residual)
        )

    print(json.dumps({"rows": rows, "trajectories": trajectories}, indent=2))


if __name__ == "__main__":
    main()
