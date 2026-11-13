"""Multi-block ADMM splitting optimization examples.

Covers two benchmark families using `px.BlockProblem`:
1. Two-block Lasso consensus splitting across `ADMM`, `DPADMM`, and
   `PAdaptADMM`.
2. Three-block nonconvex quadratic splitting over box-constrained blocks.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from proxax import px


def run_two_block_lasso_splitting() -> None:
  """Solves `min 0.5 ||A x - b||_2^2 + lam ||z||_1 s.t. x - z = 0`."""
  print("=== 1. Two-Block Lasso Splitting ===")
  key = jax.random.key(0)
  a_mat = jax.random.normal(key, (6, 4))
  b_vec = jnp.array([1.0, -0.5, 0.2, 0.8, -1.2, 0.3])
  lam = 0.25

  block1 = px.BlockTerm(
      smooth=lambda x: 0.5 * jnp.sum((a_mat @ x - b_vec) ** 2),
      linear_op=px.constraints.identity,
  )
  block2 = px.BlockTerm(
      proximal=px.prox.L1(weight=lam),
      linear_op=-jnp.eye(4),
  )
  problem = px.BlockProblem(blocks=(block1, block2), rhs=0.0)
  x0 = (jnp.zeros(4), jnp.zeros(4))

  solvers = [
      ("ADMM", px.solvers.ADMM(opt_tol=1e-4, feas_tol=1e-4, rho=2.0)),
      ("DPADMM", px.solvers.DPADMM(opt_tol=1e-4, feas_tol=1e-4, c_init=1.0)),
      (
          "PAdaptADMM",
          px.solvers.PAdaptADMM(
              opt_tol=1e-4,
              feas_tol=1e-4,
              gamma_init=10.0,
              lmt_bound=1e-1,
              adaptive_tau=True,
          ),
      ),
  ]
  for name, solver in solvers:
    sol = px.solve(problem, x0, solver=solver)
    print(
        f"{name:<11s} -> status={sol.status_enum.name}, "
        f"obj={float(sol.stats.obj_val):.6f}, "
        f"||v||={float(sol.stats.opt_residual):.2e}, "
        f"||Ax-b||={float(sol.stats.feas_residual):.2e}, "
        f"outer={int(sol.stats.outer_iters)}"
    )


def run_three_block_nonconvex_splitting() -> None:
  """Solves a 3-block nonconvex quadratic program with linear coupling."""
  print("\n=== 2. Three-Block Nonconvex Quadratic Splitting ===")
  rhs = jnp.array([1.0, -0.5, 0.25])
  a1 = jnp.eye(3)
  a2 = jnp.array([[1.0, 0.5, 0.0], [0.0, 1.0, -0.5], [0.5, 0.0, 1.0]])
  a3 = -jnp.eye(3)

  b1 = px.BlockTerm(
      smooth=lambda x: (
          0.5 * jnp.sum(x**2) - 0.8 * jnp.sum(jnp.sin(x)) - jnp.sum(x)
      ),
      proximal=px.prox.Box(lower=-1.5, upper=1.5),
      linear_op=a1,
  )
  b2 = px.BlockTerm(
      smooth=lambda x: 0.5 * jnp.sum((x + 0.5) ** 2),
      proximal=px.prox.Box(lower=-1.5, upper=1.5),
      linear_op=a2,
  )
  b3 = px.BlockTerm(
      proximal=px.prox.Box(lower=-1.5, upper=1.5),
      linear_op=a3,
  )
  problem = px.BlockProblem(blocks=(b1, b2, b3), rhs=rhs)
  x0 = (jnp.zeros(3), jnp.zeros(3), jnp.zeros(3))

  solvers = [
      ("DPADMM", px.solvers.DPADMM(opt_tol=1e-4, feas_tol=1e-4, c_init=1.0)),
      (
          "PAdaptADMM",
          px.solvers.PAdaptADMM(
              opt_tol=1e-4,
              feas_tol=1e-4,
              gamma_init=10.0,
              lmt_bound=1e-1,
              adaptive_tau=True,
          ),
      ),
      ("ADMM", px.solvers.ADMM(opt_tol=1e-4, feas_tol=1e-4, rho=2.0)),
  ]
  for name, solver in solvers:
    sol = px.solve(problem, x0, solver=solver)
    print(
        f"{name:<11s} -> status={sol.status_enum.name}, "
        f"obj={float(sol.stats.obj_val):.6f}, "
        f"||v||={float(sol.stats.opt_residual):.2e}, "
        f"||Ax-b||={float(sol.stats.feas_residual):.2e}, "
        f"outer={int(sol.stats.outer_iters)}"
    )


def main() -> None:
  """Runs all multi-block ADMM examples."""
  run_two_block_lasso_splitting()
  run_three_block_nonconvex_splitting()


if __name__ == "__main__":
  main()
