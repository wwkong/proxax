"""Spectral nuclear-norm matrix completion: Proxax vs Clarabel, SCS, CVXPY.

Compares `DAICG` and `IAICG` (in `m x n` matrix space) against lifted SDP
formulations in `Clarabel`, `SCS`, and `CVXPY` (`(m+n) x (m+n)` PSD cone).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
import typing

import clarabel
import cvxpy as cp
import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

import proxax as px
from benchmarks import harness
from benchmarks.external import conic_helpers


@dataclasses.dataclass(frozen=True)
class SpectralBenchmark:
  """A single spectral matrix completion benchmark instance."""

  name: str
  proxax_builder: typing.Callable[[], tuple[px.SpectralProblem, jax.Array]]
  clarabel_builder: typing.Callable[
      [float, float], harness.BenchmarkResult
  ]
  scs_builder: typing.Callable[[float, bool], harness.BenchmarkResult]
  cvxpy_builder: typing.Callable[[], cp.Problem]


def _svec_pairs(d: int, *, upper: bool) -> list[tuple[int, int]]:
  if upper:
    return [(i, j) for j in range(d) for i in range(j + 1)]
  return [(i, j) for j in range(d) for i in range(j, d)]


def _make_nuclear_mc(
    m: int, n: int, *, rank: int = 3, obs_prob: float = 0.5, seed: int = 55
) -> SpectralBenchmark:
  """Convex nuclear-norm regularized matrix completion."""
  rng = np.random.default_rng(seed)
  u_true = rng.standard_normal((m, rank))
  v_true = rng.standard_normal((n, rank))
  m_true = u_true @ v_true.T
  mask = (rng.uniform(size=(m, n)) < obs_prob).astype(float)
  m_obs = mask * (m_true + 0.05 * rng.standard_normal((m, n)))
  lam = 0.5 * float(np.sqrt(max(m, n)))
  name = f"Nuclear MC ({m}x{n})"

  def _proxax() -> tuple[px.SpectralProblem, jax.Array]:
    mask_j = jnp.asarray(mask)
    obs_j = jnp.asarray(m_obs)

    def _f1(mat_x: jax.Array) -> jax.Array:
      return 0.5 * jnp.sum(mask_j * (mat_x - obs_j) ** 2)

    def _f2_vec(s: jax.Array) -> jax.Array:
      return jnp.asarray(0.0, dtype=s.dtype)

    prob = px.SpectralProblem(
        f1=px.oracle.as_smooth_term(_f1),
        f2_vec=px.oracle.as_smooth_term(_f2_vec),
        prox_vec=px.prox.L1(weight=lam),
    )
    return prob, jnp.zeros((m, n))

  # Lifted SDP for Clarabel / SCS:
  # Variables z = svec(Z) where Z = [[W1, X], [X^T, W2]] in S_+^{m+n}.
  # Minimize 0.5 * sum_{obs} (X_ij - M_ij)^2 + 0.5 * lam * (Tr(W1) + Tr(W2))
  # Note: in svec, X_ij corresponds to Z[i, m+j] which has scale sqrt(2).
  d = m + n
  dim_s = d * (d + 1) // 2
  const_term = float(0.5 * np.sum(mask * (m_obs**2)))

  def _build_sdp(
      *, upper: bool
  ) -> tuple[sp.csc_matrix, np.ndarray, sp.csc_matrix, np.ndarray]:
    pairs = _svec_pairs(d, upper=upper)
    idx_map = {pair: idx for idx, pair in enumerate(pairs)}

    p_diag = np.zeros(dim_s)
    q_vec = np.zeros(dim_s)
    for i in range(d):
      q_vec[idx_map[(i, i)]] = 0.5 * lam
    for i in range(m):
      for j in range(n):
        if mask[i, j] > 0.0:
          pair = (i, m + j) if upper else (m + j, i)
          k = idx_map[pair]
          # svec entry is sqrt(2) * X_ij -> X_ij^2 = 0.5 * z_k^2
          p_diag[k] = 0.5
          q_vec[k] = -m_obs[i, j] / np.sqrt(2.0)

    p_sp = sp.diags(p_diag, format="csc")
    a_ext = -sp.eye(dim_s, format="csc")
    b_ext = np.zeros(dim_s)
    return p_sp, q_vec, a_ext, b_ext

  def _clarabel(opt_tol: float, feas_tol: float) -> harness.BenchmarkResult:
    p_sp, q_vec, a_ext, b_ext = _build_sdp(upper=True)
    res = conic_helpers.solve_clarabel(
        suite="spectral_vs_clarabel_scs",
        problem_name=name,
        p_sp=p_sp,
        q_vec=q_vec,
        a_sp=a_ext,
        b_vec=b_ext,
        cones=[clarabel.PSDTriangleConeT(d)],
        opt_tol=opt_tol,
        feas_tol=feas_tol,
    )
    return dataclasses.replace(res, obj_val=res.obj_val + const_term)

  def _scs(tol: float, indirect: bool) -> harness.BenchmarkResult:
    p_sp, q_vec, a_ext, b_ext = _build_sdp(upper=False)
    res = conic_helpers.solve_scs(
        suite="spectral_vs_clarabel_scs",
        problem_name=name,
        p_sp=p_sp,
        q_vec=q_vec,
        a_sp=a_ext,
        b_vec=b_ext,
        cone_dict={"s": [d]},
        tol=tol,
        use_indirect=indirect,
    )
    return dataclasses.replace(res, obj_val=res.obj_val + const_term)

  def _cvxpy() -> cp.Problem:
    x = cp.Variable((m, n))
    obj = cp.Minimize(
        0.5 * cp.sum_squares(cp.multiply(mask, x - m_obs))
        + lam * cp.normNuc(x)
    )
    return cp.Problem(obj)

  return SpectralBenchmark(
      name=name,
      proxax_builder=_proxax,
      clarabel_builder=_clarabel,
      scs_builder=_scs,
      cvxpy_builder=_cvxpy,
  )


def run_suite(
    *,
    opt_tol: float = 1e-4,
    include_cvxpy: bool = False,
    time_limit: float = 15.0,
) -> list[harness.BenchmarkResult]:
  """Runs spectral matrix completion benchmarks."""
  benchmarks = [
      _make_nuclear_mc(16, 16),
      _make_nuclear_mc(32, 32),
      _make_nuclear_mc(48, 48),
  ]

  proxax_solvers = [
      (
          "DAICG (monotone)",
          px.solvers.DAICG(opt_tol=opt_tol, monotone=True, max_outer_iters=600),
      ),
      (
          "DAICG (nonmonotone)",
          px.solvers.DAICG(
              opt_tol=opt_tol, monotone=False, max_outer_iters=600
          ),
      ),
      (
          "IAICG",
          px.solvers.IAICG(opt_tol=opt_tol, max_outer_iters=600),
      ),
  ]

  def _log_done(res: harness.BenchmarkResult) -> None:
    print(
        f"  -> {res.solver_name}: {res.status} "
        f"(run={1e3 * res.run_s:.2f}ms, compile={res.compile_s:.2f}s)",
        file=sys.stderr,
        flush=True,
    )

  results: list[harness.BenchmarkResult] = []
  total = len(benchmarks)
  for idx, bench in enumerate(benchmarks, start=1):
    print(f"[{idx}/{total}] {bench.name}", file=sys.stderr, flush=True)
    prob, x0 = bench.proxax_builder()
    for solver_name, solver in proxax_solvers:
      res = harness.time_proxax_solve(
          suite="spectral_vs_clarabel_scs",
          problem_name=bench.name,
          solver_name=solver_name,
          prob=prob,
          x0=x0,
          solver=solver,
          opt_tol=opt_tol,
          time_limit=time_limit,
      )
      _log_done(res)
      results.append(res)
    for ext_res in (
        bench.clarabel_builder(opt_tol, opt_tol),
        bench.scs_builder(opt_tol, False),
        bench.scs_builder(opt_tol, True),
    ):
      _log_done(ext_res)
      results.append(ext_res)
    if include_cvxpy:
      cp_res = conic_helpers.solve_cvxpy(
          suite="spectral_vs_clarabel_scs",
          problem_name=bench.name,
          prob=bench.cvxpy_builder(),
          solver_name="CVXPY (SCS)",
          cvxpy_solver=cp.SCS,
          solver_kwargs={"eps": opt_tol},
      )
      _log_done(cp_res)
      results.append(cp_res)
  return results


def main() -> None:
  """CLI entry point for spectral matrix completion benchmarks."""
  parser = argparse.ArgumentParser(
      description="Benchmark Proxax SpectralProblem vs Clarabel, SCS, CVXPY."
  )
  parser.add_argument("--opt-tol", type=float, default=1e-4)
  parser.add_argument("--time-limit", type=float, default=15.0)
  parser.add_argument("--include-cvxpy", action="store_true")
  parser.add_argument("--json-out", type=pathlib.Path, default=None)
  args = parser.parse_args()

  results = run_suite(
      opt_tol=args.opt_tol,
      include_cvxpy=args.include_cvxpy,
      time_limit=args.time_limit,
  )
  print(harness.format_markdown_table(results))
  if args.json_out is not None:
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    payload = [dataclasses.asdict(r) for r in results]
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
  main()
