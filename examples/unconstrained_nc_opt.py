"""Unconstrained nonconvex composite examples ported from `wwkong/nc_opt`.

Covers three classic benchmark families from `nc_opt/examples/unconstrained/`:
1. Basic univariate convex QP (`basic_convex_qp.m`).
2. Nonconvex QP over the unit simplex (`nonconvex_qp.m` / `test_fn_unconstr`).
3. Nonconvex hyperbolic-tangent SVM (`nonconvex_svm.m` / `test_fn_svm_01`).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import proxax
from proxax import prox
from proxax import solvers
from proxax.oracle import terms


def calibrate_qp_curvatures(
    a_mat: jax.Array,
    db_mat: jax.Array,
    target_m_upper: float,
    target_m_lower: float,
) -> tuple[float, float, float, float]:
  """Finds `(tau, xi)` so `H = tau A^T A - xi (DB)^T (DB)` hits `(M, m)`."""
  hp = 0.5 * (a_mat.T @ a_mat + (a_mat.T @ a_mat).T)
  hn = 0.5 * (db_mat.T @ db_mat + (db_mat.T @ db_mat).T)
  lam_max_hn = float(jnp.max(jnp.linalg.eigvalsh(hn)))
  xi = target_m_lower / max(lam_max_hn, 1e-8)

  lo, hi = 0.0, 1.0
  while float(jnp.max(jnp.linalg.eigvalsh(hi * hp - xi * hn))) < target_m_upper:
    hi *= 2.0
  for _ in range(45):
    mid = 0.5 * (lo + hi)
    cur_max = float(jnp.max(jnp.linalg.eigvalsh(mid * hp - xi * hn)))
    if cur_max < target_m_upper:
      lo = mid
    else:
      hi = mid
  tau = 0.5 * (lo + hi)
  eigs = jnp.linalg.eigvalsh(tau * hp - xi * hn)
  m_upper = float(jnp.max(eigs))
  m_lower = float(jnp.maximum(0.0, -jnp.min(eigs)))
  return tau, xi, m_upper, m_lower


def run_basic_convex_qp() -> None:
  """Solves `min_{x in R} 0.5 * x^2 - x + 0.5` (`basic_convex_qp.m`)."""
  print("=== 1. Basic Convex Univariate QP (basic_convex_qp.m) ===")
  problem = proxax.CompositeProblem(
      smooth=lambda x: 0.5 * jnp.sum(x**2) - jnp.sum(x) + 0.5,
  )
  x0 = jnp.array([-3.0])
  sol = proxax.solve(problem, x0, solver=solvers.APD(opt_tol=1e-6))
  print(
      f"APD -> status={sol.status_enum.name}, x*={float(sol.x[0]):.6f}, "
      f"obj={float(sol.stats.obj_val):.2e}, "
      f"||v||={float(sol.stats.opt_residual):.2e}, "
      f"inner_iters={int(sol.stats.inner_iters)}"
  )


def run_nonconvex_simplex_qp() -> None:
  """Solves nonconvex simplex QP (`nonconvex_qp.m` / `test_fn_unconstr_01`)."""
  print("\n=== 2. Nonconvex Simplex QP (nonconvex_qp.m) ===")
  key = jax.random.key(777)
  k1, k2, k3, k4 = jax.random.split(key, 4)
  dim_m, dim_n = 10, 20
  d_diag = jax.random.randint(k1, (dim_n,), 1, 15).astype(jnp.float32)
  a_mat = jax.random.uniform(k2, (dim_m, dim_n))
  b_mat = jax.random.uniform(k3, (dim_n, dim_n))
  b_vec = jax.random.uniform(k4, (dim_m,))
  db_mat = d_diag[:, None] * b_mat

  tau, xi, m_upper, m_lower = calibrate_qp_curvatures(
      a_mat, db_mat, target_m_upper=50.0, target_m_lower=5.0
  )
  print(f"Calibrated curvatures: M = {m_upper:.3f}, m = {m_lower:.3f}")

  def smooth_qp(x: jax.Array) -> jax.Array:
    res = a_mat @ x - b_vec
    dbx = db_mat @ x
    return 0.5 * tau * jnp.dot(res, res) - 0.5 * xi * jnp.dot(dbx, dbx)

  problem = proxax.CompositeProblem(
      smooth=terms.SmoothTerm(
          smooth_qp,
          upper_curvature=m_upper,
          lower_curvature=m_lower,
      ),
      proximal=prox.Simplex(scale=1.0),
  )
  x0 = jnp.ones(dim_n) / dim_n
  tol = 1e-4

  suite: list[tuple[str, solvers.Solver]] = [
      ("APD (param-free)", solvers.APD(opt_tol=tol)),
      ("AIPP", solvers.AIPP(opt_tol=tol)),
      ("NCFISTA", solvers.NCFISTA(opt_tol=tol)),
      ("AdapFISTA", solvers.AdapFISTA(opt_tol=tol)),
      ("ACACG", solvers.ACACG(opt_tol=tol)),
      ("UPFAG", solvers.UPFAG(opt_tol=tol)),
      ("ECG", solvers.ECG(opt_tol=tol)),
  ]
  header = (
      f"{'Solver':<18} {'Status':<11} {'Obj':>11} {'||v||_2':>10} {'Grad':>6}"
  )
  print(header)
  for name, solver in suite:
    sol = proxax.solve(problem, x0, solver=solver)
    print(
        f"{name:<18} {sol.status_enum.name:<11} "
        f"{float(sol.stats.obj_val):>11.5f} "
        f"{float(sol.stats.opt_residual):>10.2e} "
        f"{int(sol.stats.grad_evals):>6d}"
    )


def run_nonconvex_svm() -> None:
  """Solves nonconvex sigmoid/tanh SVM (`nonconvex_svm.m`)."""
  print("\n=== 3. Nonconvex Hyperbolic-Tangent SVM (nonconvex_svm.m) ===")
  key = jax.random.key(42)
  k1, k2 = jax.random.split(key)
  dim_n, num_samples = 25, 60
  u_mat = jax.random.normal(k1, (dim_n, num_samples)) / jnp.sqrt(dim_n)
  x_true = jax.random.normal(k2, (dim_n,))
  labels = jnp.where(u_mat.T @ x_true >= 0.0, 1.0, -1.0)
  reg = 1.0 / num_samples

  def svm_loss(x: jax.Array) -> jax.Array:
    margins = labels * (u_mat.T @ x)
    return jnp.mean(1.0 - jnp.tanh(margins)) + 0.5 * reg * jnp.dot(x, x)

  m_bound = float(
      (4.0 * jnp.sqrt(3.0) / 9.0) * (jnp.sum(u_mat**2) / num_samples) + reg
  )
  problem_free = proxax.CompositeProblem(smooth=svm_loss)
  problem_curv = proxax.CompositeProblem(
      smooth=terms.SmoothTerm(
          svm_loss, upper_curvature=m_bound, lower_curvature=m_bound
      )
  )
  x0 = jnp.zeros(dim_n)
  tol = 1e-4

  svm_suite: list[tuple[str, proxax.CompositeProblem, solvers.Solver]] = [
      ("APD (no M, m)", problem_free, solvers.APD(opt_tol=tol)),
      ("UPFAG (no M, m)", problem_free, solvers.UPFAG(opt_tol=tol)),
      ("AdapAPG (no M, m)", problem_free, solvers.AdapAPG(opt_tol=tol)),
      ("AG (uses M)", problem_curv, solvers.AG(opt_tol=tol)),
  ]
  for name, prob, solver in svm_suite:
    sol = proxax.solve(prob, x0, solver=solver)
    acc = float(jnp.mean(jnp.sign(u_mat.T @ sol.x) == labels)) * 100.0
    print(
        f"{name:<18} status={sol.status_enum.name:<9} "
        f"loss={float(sol.stats.obj_val):.5f} "
        f"||grad||={float(sol.stats.opt_residual):.2e} "
        f"acc={acc:.1f}% grad_evals={int(sol.stats.grad_evals)}"
    )


if __name__ == "__main__":
  run_basic_convex_qp()
  run_nonconvex_simplex_qp()
  run_nonconvex_svm()
