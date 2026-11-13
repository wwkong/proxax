"""Small Table-1 DQP variants from Maia-Monteiro-Silva (arXiv:2611.00570)."""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import jax
import jax.numpy as jnp

from proxax import px
from proxax import tree


def _sample_l1_interior(
    key: jax.Array,
    dim: int,
    omega: float,
) -> jax.Array:
  """Samples a vector strictly inside the L1 ball of radius ``omega``."""
  raw = jax.random.normal(key, (dim,), dtype=jnp.float64)
  l1 = jnp.sum(jnp.abs(raw))
  scale = jnp.where(
      l1 >= 0.9 * omega, (0.9 * omega) / jnp.maximum(l1, 1e-12), 1.0
  )
  return raw * scale


def make_dqp_instance(
    *,
    num_blocks: int,
    m: int,
    n: int,
    omega: float = 100.0,
    seed: int = 0,
) -> tuple[px.BlockProblem, tuple[jax.Array, ...]]:
  """Builds a nonconvex Distributed QP (eq. 72 of arXiv:2611.00570)."""
  key = jax.random.PRNGKey(seed)
  blocks: list[px.BlockTerm] = []
  x_feas: list[jax.Array] = []
  x0_list: list[jax.Array] = []
  b_total = jnp.zeros((m,), dtype=jnp.float64)

  for _i in range(num_blocks):
    key, k_a, k_xb, k_x0, k_q, k_d, k_r = jax.random.split(key, 7)
    a_i = jax.random.normal(k_a, (m, n), dtype=jnp.float64)
    xb_i = _sample_l1_interior(k_xb, n, omega)
    x0_i = _sample_l1_interior(k_x0, n, omega)
    b_total = b_total + a_i @ xb_i

    q_mat, _ = jnp.linalg.qr(jax.random.normal(k_q, (n, n), dtype=jnp.float64))
    n_zero = n // 3
    d_nonzero = jax.random.uniform(
        k_d, (n - n_zero,), minval=-10.0, maxval=10.0, dtype=jnp.float64
    )
    # Guarantee at least one negative eigenvalue for nonconvexity.
    if bool(jnp.all(d_nonzero >= 0.0)):
      d_nonzero = d_nonzero.at[0].set(-abs(float(d_nonzero[0])) - 1.0)
    diag = jnp.concatenate(
        [jnp.zeros((n_zero,), dtype=jnp.float64), d_nonzero]
    )
    p_i = (q_mat.T * diag) @ q_mat
    p_i = 0.5 * (p_i + p_i.T)
    r_i = jax.random.normal(k_r, (n,), dtype=jnp.float64)

    def _make_smooth(p_mat: jax.Array, r_vec: jax.Array):
      def _smooth(u: jax.Array) -> jax.Array:
        return 0.5 * jnp.dot(u, p_mat @ u) + jnp.dot(r_vec, u)

      return _smooth

    blocks.append(
        px.BlockTerm(
            smooth=_make_smooth(p_i, r_i),
            proximal=px.prox.L1Ball(radius=omega),
            linear_op=a_i,
        )
    )
    x_feas.append(xb_i)
    x0_list.append(x0_i)

  problem = px.BlockProblem(blocks=tuple(blocks), rhs=b_total)
  return problem, tuple(x0_list)


def _initial_scales(
    problem: px.BlockProblem,
    x0: tuple[jax.Array, ...],
) -> tuple[float, float]:
  """Computes ``1 + ||grad f(x0)||_2`` and ``1 + ||A x0 - b||_2``."""
  bound_ops = problem.bind_operators(x0)
  grads = tuple(
      blk.smooth.value_and_grad(x_i)[1]
      for blk, x_i in zip(problem.blocks, x0, strict=True)
  )
  grad_norm = float(tree.norm(grads))
  res = tree.sub(
      sum(
          (op.matvec(x_i) for op, x_i in zip(bound_ops, x0, strict=True)),
          jnp.zeros_like(problem.rhs),
      ),
      problem.rhs,
  )
  feas_norm = float(tree.norm(res))
  return 1.0 + grad_norm, 1.0 + feas_norm


def run_suite(
    configs: list[tuple[int, int, int]],
    *,
    omega: float = 100.0,
    rel_tol: float = 1e-4,
    seed: int = 0,
    time_limit: float = 120.0,
) -> list[dict[str, object]]:
  """Runs ADMM, DPADMM, and PAdaptADMM on small Table-1 DQP variants."""
  rows: list[dict[str, object]] = []
  for num_blocks, m, n in configs:
    problem, x0 = make_dqp_instance(
        num_blocks=num_blocks, m=m, n=n, omega=omega, seed=seed
    )
    g_scale, f_scale = _initial_scales(problem, x0)
    opt_tol = rel_tol * g_scale
    feas_tol = rel_tol * f_scale
    c0 = 1.0 / f_scale

    solvers_to_run = [
        (
            "PAdaptADMM-prac",
            px.solvers.PAdaptADMM(
                opt_tol=opt_tol,
                feas_tol=feas_tol,
                lmt_bound=1e3 * opt_tol,
                gamma_init=10.0,
                max_stages=30,
                max_stage_iters=5000,
                max_inner_iters=200,
                adaptive_tau=True,
            ),
        ),
        (
            "PAdaptADMM-theo",
            px.solvers.PAdaptADMM(
                opt_tol=opt_tol,
                feas_tol=feas_tol,
                lmt_bound=1e3 * opt_tol,
                gamma_init=10.0,
                max_stages=30,
                max_stage_iters=5000,
                max_inner_iters=200,
                adaptive_tau=False,
            ),
        ),
        (
            "DPADMM",
            px.solvers.DPADMM(
                opt_tol=opt_tol,
                feas_tol=feas_tol,
                theta=0.0,
                chi=1.0,
                lam_init=0.5,
                c_init=c0,
                max_stages=30,
                max_stage_iters=1000,
                max_inner_iters=200,
            ),
        ),
        (
            "ADMM",
            px.solvers.ADMM(
                opt_tol=opt_tol,
                feas_tol=feas_tol,
                rho=c0,
                max_iters=50000,
            ),
        ),
    ]

    for name, solver in solvers_to_run:
      t0 = time.perf_counter()
      # Single-stage ADMM falls back to a Python loop when time_limit is set.
      solver_tl = None if name == "ADMM" else time_limit
      sol = solver.solve(problem, x0, jit=True, time_limit=solver_tl)
      sol.stats.opt_residual.block_until_ready()
      elapsed = time.perf_counter() - t0
      rel_opt = float(sol.stats.opt_residual) / g_scale
      rel_feas = float(sol.stats.feas_residual) / f_scale
      row = {
          "B": num_blocks,
          "m": m,
          "n": n,
          "total_dim": num_blocks * n,
          "omega": omega,
          "solver": name,
          "status": sol.status_enum.name,
          "iters": int(sol.stats.outer_iters),
          "inner_iters": int(sol.stats.inner_iters),
          "grad_evals": int(sol.stats.grad_evals),
          "obj": float(sol.stats.obj_val),
          "rel_opt": rel_opt,
          "rel_feas": rel_feas,
          "time_s": elapsed,
      }
      rows.append(row)
      print(
          f"B={num_blocks:>2d} (m,n)=({m:>2d},{n:>2d}) | {name:<15s} | "
          f"{row['status']:<9s} | iters={row['iters']:>5d} | "
          f"inner={row['inner_iters']:>6d} | grads={row['grad_evals']:>6d} | "
          f"obj={row['obj']:>11.4e} | rel_opt={rel_opt:.2e} | "
          f"rel_feas={rel_feas:.2e} | {elapsed:>5.2f}s",
          flush=True,
      )
    instance_rows = rows[-len(solvers_to_run) :]
    if all(r["status"] != "CONVERGED" for r in instance_rows):
      print(
          f"Stopping ladder at B={num_blocks}, (m,n)=({m},{n}): "
          "all methods failed to converge.",
          flush=True,
      )
      break
  return rows


def main() -> int:
  """CLI entrypoint for the small Table-1 DQP experiment."""
  parser = argparse.ArgumentParser(
      description="Run small Table-1 DQP variants from arXiv:2611.00570."
  )
  parser.add_argument("--omega", type=float, default=100.0)
  parser.add_argument("--rel-tol", type=float, default=1e-3)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--time-limit", type=float, default=60.0)
  args = parser.parse_args()

  jax.config.update("jax_enable_x64", True)
  configs = [
      (2, 2, 4),
      (2, 4, 6),
      (4, 4, 6),
      (4, 4, 8),
      (4, 8, 12),
      (8, 8, 16),
      (8, 16, 16),
      (16, 16, 16),
  ]
  rows = run_suite(
      configs,
      omega=args.omega,
      rel_tol=args.rel_tol,
      seed=args.seed,
      time_limit=args.time_limit,
  )
  out_dir = pathlib.Path(__file__).resolve().parent / "results"
  out_dir.mkdir(parents=True, exist_ok=True)
  out_path = out_dir / "results.json"
  out_path.write_text(json.dumps(rows, indent=2))
  print(f"\nSaved {len(rows)} rows to {out_path}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
