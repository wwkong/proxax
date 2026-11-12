"""Set- and cone-constrained nonconvex examples ported from `wwkong/nc_opt`.

Covers three classic constrained benchmark families from `nc_opt/examples/`:
1. Linearly-constrained nonconvex simplex QP (`nonconvex_lin_constr_qp.m`).
2. Consensus Sparse PCA on the `k`-Fantope (`nonconvex_sparse_pca.m`).
3. Box-constrained nonconvex matrix completion (`nonconvex_bounded_mc.m`).
"""

from __future__ import annotations

import dataclasses
import typing

import jax
import jax.numpy as jnp

import proxax
from proxax import _typing
from proxax import constraints
from proxax import prox
from proxax import solvers
from proxax.oracle import terms


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class FantopeProx:
  """Indicator and Euclidean projector of the `k`-Fantope.

  Projects onto `{X in S^p_+ : 0 <= eig(X) <= 1, tr(X) = rank_k}` via symmetric
  eigendecomposition and capped-simplex bisection.
  """

  rank_k: float

  @property
  def strong_convexity(self) -> float:
    """Strong convexity constant (`0.0`)."""
    return 0.0

  def value(self, x: _typing.PyTree) -> jax.Array:
    """Returns `0.0` at feasible `k`-Fantope matrices."""
    mat = jnp.asarray(x)
    return jnp.asarray(0.0, dtype=mat.dtype)

  def _project(self, x: _typing.PyTree) -> jax.Array:
    mat = jnp.asarray(x)
    sym = 0.5 * (mat + mat.T)
    eigvals, eigvecs = jnp.linalg.eigh(sym)
    lo = jnp.min(eigvals) - 1.0
    hi = jnp.max(eigvals)

    def body_fn(_: int, bounds: tuple[jax.Array, jax.Array]) -> tuple[
        jax.Array, jax.Array
    ]:
      low, high = bounds
      mid = 0.5 * (low + high)
      clipped_sum = jnp.sum(jnp.clip(eigvals - mid, 0.0, 1.0))
      res = jax.lax.cond(
          clipped_sum > self.rank_k,
          lambda: (mid, high),
          lambda: (low, mid),
      )
      return typing.cast(tuple[jax.Array, jax.Array], res)

    lo_fin, hi_fin = jax.lax.fori_loop(0, 36, body_fn, (lo, hi))
    theta = 0.5 * (lo_fin + hi_fin)
    proj_eigs = jnp.clip(eigvals - theta, 0.0, 1.0)
    proj_mat = (eigvecs * proj_eigs[None, :]) @ eigvecs.T
    return typing.cast(jax.Array, proj_mat)

  def prox(
      self,
      x: _typing.PyTree,
      step_size: _typing.Scalar,
  ) -> _typing.PyTree:
    """Projects `x` onto the `k`-Fantope."""
    del step_size
    return self._project(x)

  def domain_project(self, x: _typing.PyTree) -> _typing.PyTree:
    """Projects `x` onto the `k`-Fantope."""
    return self._project(x)


def run_linearly_constrained_qp() -> None:
  """Solves linearly-constrained nonconvex simplex QP across 5 solvers."""
  print("=== 1. Linearly-Constrained Nonconvex Simplex QP ===")
  key = jax.random.key(777)
  k1, k2, k3, k4, k5 = jax.random.split(key, 5)
  dim_eq, dim_n = 4, 12
  d_diag = jax.random.randint(k1, (dim_n,), 1, 10).astype(jnp.float32)
  c_mat = jax.random.uniform(k2, (dim_eq, dim_n))
  b_mat = jax.random.uniform(k3, (dim_n, dim_n))
  a_eq = jax.random.uniform(k4, (dim_eq, dim_n))
  d_vec = jax.random.uniform(k5, (dim_eq,))

  db_mat = d_diag[:, None] * b_mat
  hp = c_mat.T @ c_mat
  hn = db_mat.T @ db_mat
  xi = 2.0 / float(jnp.max(jnp.linalg.eigvalsh(hn)))
  tau = 20.0 / float(jnp.max(jnp.linalg.eigvalsh(hp)))
  hess = tau * hp - xi * hn
  eigs = jnp.linalg.eigvalsh(0.5 * (hess + hess.T))
  m_upper = float(jnp.maximum(0.1, jnp.max(eigs)))
  m_lower = float(jnp.maximum(0.1, -jnp.min(eigs)))
  rhs_b = a_eq @ (jnp.ones(dim_n) / dim_n)

  def smooth_fn(x: jax.Array) -> jax.Array:
    res = c_mat @ x - d_vec
    dbx = db_mat @ x
    return 0.5 * tau * jnp.dot(res, res) - 0.5 * xi * jnp.dot(dbx, dbx)

  problem = proxax.CompositeProblem(
      smooth=terms.SmoothTerm(
          smooth_fn, upper_curvature=m_upper, lower_curvature=m_lower
      ),
      proximal=prox.Simplex(scale=1.0),
      constraints=constraints.equality(a_eq, rhs=rhs_b),
  )
  x0 = jnp.linspace(0.1, 1.0, dim_n)
  x0 = x0 / jnp.sum(x0)

  suite: list[tuple[str, solvers.Solver]] = [
      ("QPAPD (param-free)", solvers.QPAPD(opt_tol=5e-4, feas_tol=5e-4)),
      ("QPAIPP", solvers.QPAIPP(opt_tol=5e-4, feas_tol=5e-4)),
      ("IALM (NL-IAPIAL)", solvers.IALM(opt_tol=5e-4, feas_tol=5e-4)),
      (
          "IAIPAL",
          solvers.IAIPAL(
              opt_tol=5e-4, feas_tol=5e-4, penalty_init=5.0, max_outer_iters=60
          ),
      ),
      (
          "SProxALM",
          solvers.SProxALM(
              opt_tol=5e-4, feas_tol=5e-4, penalty=5.0, max_iters=3000
          ),
      ),
  ]
  print(
      f"{'Solver':<20} {'Status':<11} {'Obj':>10} "
      f"{'Stat':>10} {'Feas':>10} {'Grad':>6}"
  )
  for name, solver in suite:
    sol = proxax.solve(problem, x0, solver=solver)
    print(
        f"{name:<20} {sol.status_enum.name:<11} "
        f"{float(sol.stats.obj_val):>10.5f} "
        f"{float(sol.stats.opt_residual):>10.2e} "
        f"{float(sol.stats.feas_residual):>10.2e} "
        f"{int(sol.stats.grad_evals):>6d}"
    )


def run_sparse_pca_fantope() -> None:
  """Solves MCP-regularized Sparse PCA on the `k`-Fantope (`spca_01`)."""
  print("\n=== 2. Consensus Sparse PCA on the k-Fantope (PyTree IALM) ===")
  dim_p, support_s, rank_k = 8, 3, 1
  v_true = jnp.zeros((dim_p,))
  v_true = v_true.at[:support_s].set(1.0 / jnp.sqrt(support_s))
  sigma_cov = 8.0 * jnp.outer(v_true, v_true) + 0.25 * jnp.eye(dim_p)

  nu = 0.25
  b_mcp = 2.0

  def mcp_concave_part(phi: jax.Array) -> jax.Array:
    abs_phi = jnp.abs(phi)
    quad = -(phi**2) / (2.0 * b_mcp)
    lin = 0.5 * b_mcp * (nu**2) - nu * abs_phi
    return jnp.sum(jnp.where(abs_phi <= b_mcp * nu, quad, lin))

  def smooth_spca(z: dict[str, jax.Array]) -> jax.Array:
    return -jnp.sum(sigma_cov * z["pi"]) + mcp_concave_part(z["phi"])

  problem = proxax.CompositeProblem(
      smooth=terms.SmoothTerm(
          smooth_spca, upper_curvature=0.1, lower_curvature=1.0 / b_mcp
      ),
      proximal=prox.MultiProx(
          {
              "pi": FantopeProx(rank_k=float(rank_k)),
              "phi": prox.L1(weight=nu),
          }
      ),
      constraints=constraints.equality(lambda z: z["pi"] - z["phi"]),
  )
  z0 = {
      "pi": jnp.eye(dim_p) * (rank_k / dim_p),
      "phi": jnp.zeros((dim_p, dim_p)),
  }
  sol = proxax.solve(
      problem,
      z0,
      solver=solvers.IALM(opt_tol=1e-3, feas_tol=1e-3, penalty_init=2.0),
  )
  eigvals, eigvecs = jnp.linalg.eigh(0.5 * (sol.x["pi"] + sol.x["pi"].T))
  lead_vec = jnp.abs(eigvecs[:, jnp.argmax(eigvals)])
  overlap = float(jnp.abs(jnp.dot(lead_vec, v_true)))
  nnz_phi = int(jnp.sum(jnp.abs(sol.x["phi"]) > 1e-2))
  print(
      f"IALM -> status={sol.status_enum.name}, "
      f"stat={float(sol.stats.opt_residual):.2e}, "
      f"||Pi - Phi||_F={float(sol.stats.feas_residual):.2e}, "
      f"subspace_cos={overlap:.4f}, nnz(Phi)={nnz_phi}/{dim_p * dim_p}"
  )


def run_bounded_matrix_completion() -> None:
  """Solves box-constrained nonconvex matrix completion (`bounded_mc`)."""
  print("\n=== 3. Box-Constrained Nonconvex Matrix Completion ===")
  key = jax.random.key(99)
  k1, k2, k3 = jax.random.split(key, 3)
  rows, cols = 12, 10
  u_mat = jax.random.uniform(k1, (rows, 2))
  v_mat = jax.random.uniform(k2, (cols, 2))
  target = jnp.clip(u_mat @ v_mat.T, 0.0, 2.0)
  mask = (jax.random.uniform(k3, (rows, cols)) < 0.6).astype(jnp.float32)
  obs = mask * target

  beta, theta, mu = 0.5, float(jnp.sqrt(2.0)), 0.15

  def smooth_bmc(x_mat: jax.Array) -> jax.Array:
    fit = 0.5 * jnp.sum((mask * (x_mat - obs)) ** 2)
    smooth_abs = jnp.sqrt(x_mat * x_mat + 1e-4)
    reg = mu * beta * jnp.sum(jnp.log(1.0 + smooth_abs / theta))
    return fit + reg

  problem = proxax.CompositeProblem(
      smooth=smooth_bmc,
      constraints=constraints.box(lambda x: x, lower=0.0, upper=2.0),
  )
  x0 = jnp.full((rows, cols), 0.5)
  sol = proxax.solve(
      problem,
      x0,
      solver=solvers.QPAPD(opt_tol=1e-3, feas_tol=1e-4),
  )
  rmse = float(jnp.sqrt(jnp.mean((sol.x - target) ** 2)))
  print(
      f"QPAPD -> status={sol.status_enum.name}, "
      f"stat={float(sol.stats.opt_residual):.2e}, "
      f"box_viol={float(sol.stats.feas_residual):.2e}, "
      f"full_matrix_RMSE={rmse:.4f}"
  )


if __name__ == "__main__":
  run_linearly_constrained_qp()
  run_sparse_pca_fantope()
  run_bounded_matrix_completion()
