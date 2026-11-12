from __future__ import annotations

import jax.numpy as jnp

import proxax
from proxax import prox
from proxax import solvers


def test_da_icg_nonconvex_spectral_monotone_and_nonmonotone() -> None:
  target = jnp.array([[2.0, -0.5, 0.3], [0.4, 1.2, -0.7], [-0.2, 0.6, 0.8]])
  w_mat = jnp.array([[1.5, 0.8, 1.2], [1.1, 1.4, 0.9], [0.7, 1.3, 1.6]])

  spec_prob = proxax.SpectralProblem(
      f1=lambda x: 0.5 * jnp.sum(w_mat * (x - target) ** 2),
      f2_vec=lambda s: -0.15 * jnp.sum(s**2 / (1.0 + s**2)),
      prox_vec=prox.Box(lower=-1.5, upper=1.5),
      domain_proj=lambda x: jnp.clip(x, -2.0, 2.0),
  )
  x0 = jnp.zeros((3, 3))
  sol = proxax.solve(spec_prob, x0, method="da_icg", opt_tol=1e-4)
  assert sol.status_enum == proxax.SolveStatus.CONVERGED
  assert sol.subgrad is not None
  kkt = spec_prob.evaluate_residuals(sol.x, sol.subgrad)
  assert float(kkt.stationarity_norm) <= 1e-4

  _, grad_x = spec_prob.oracle.smooth_value_and_grad(sol.x)
  fp_x = spec_prob.oracle.prox(sol.x + sol.subgrad - grad_x, 1.0)
  assert float(jnp.linalg.norm(sol.x - fp_x)) <= 1e-5

  sol_nm = proxax.solve(
      spec_prob,
      x0,
      solver=solvers.DAICG(opt_tol=1e-4, monotone=False),
  )
  assert sol_nm.status_enum == proxax.SolveStatus.CONVERGED
  assert float(sol_nm.stats.opt_residual) <= 1e-4
