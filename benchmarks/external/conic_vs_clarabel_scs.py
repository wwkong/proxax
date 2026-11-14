"""Conic optimization benchmarks: Proxax vs Clarabel, SCS, and CVXPY.

Covers Second-Order Cone (`SOC`), Exponential Cone (`ExpCone`), and
Spectrahedron SDP (`Spectrahedron`) problems across increasing dimensions.
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
from proxax.solvers.constrained import ialm
from proxax.solvers.constrained import qp_aipp
from proxax.solvers.constrained import qp_apd


@dataclasses.dataclass(frozen=True)
class ConicBenchmark:
  """A single conic benchmark instance with Proxax and external builders."""

  name: str
  category: str
  proxax_builder: typing.Callable[[], tuple[px.CompositeProblem, typing.Any]]
  clarabel_builder: typing.Callable[
      [float, float], harness.BenchmarkResult
  ]
  scs_builder: typing.Callable[[float, bool], harness.BenchmarkResult]
  cvxpy_builder: typing.Callable[[], cp.Problem] | None = None


def _make_socp_portfolio(n: int, *, seed: int = 22) -> ConicBenchmark:
  """Markowitz-style SOC-constrained QP with `n` variables."""
  rng = np.random.default_rng(seed)
  f_mat = rng.standard_normal((n, 4)) * 0.3
  diag_cov = np.geomspace(0.1, 10.0, n)
  cov = f_mat @ f_mat.T + np.diag(diag_cov)
  mu = rng.uniform(0.05, 0.25, size=n)
  a_mat = np.vstack([
      np.array([[1.0] + [0.0] * (n - 1)]),
      np.ones((1, n)),
  ])
  b_vec = np.array([2.0, 3.0])
  x0_np = np.array([2.0] + [1.0 / (n - 1)] * (n - 1))

  def _proxax() -> tuple[px.CompositeProblem, jax.Array]:
    cov_j = jnp.asarray(cov)
    mu_j = jnp.asarray(mu)
    a_j = jnp.asarray(a_mat)
    b_j = jnp.asarray(b_vec)

    def _obj(x: jax.Array) -> jax.Array:
      return 0.5 * jnp.dot(x, cov_j @ x) - jnp.dot(mu_j, x)

    prob = px.CompositeProblem(
        smooth=px.oracle.as_smooth_term(_obj),
        proximal=px.prox.SOC(),
        constraints=px.constraints.equality(a_j, b_j),
    )
    return prob, jnp.asarray(x0_np)

  p_sp = sp.triu(sp.csc_matrix(cov), format="csc")
  a_ext = sp.vstack(
      [sp.csc_matrix(a_mat), -sp.eye(n, format="csc")], format="csc"
  )
  b_ext = np.concatenate([b_vec, np.zeros(n)])

  def _clarabel(opt_tol: float, feas_tol: float) -> harness.BenchmarkResult:
    cones = [
        clarabel.ZeroConeT(2),
        clarabel.SecondOrderConeT(n),
    ]
    return conic_helpers.solve_clarabel(
        suite="conic_vs_clarabel_scs",
        problem_name=f"SOC Portfolio (n={n})",
        p_sp=p_sp,
        q_vec=-mu,
        a_sp=a_ext,
        b_vec=b_ext,
        cones=cones,
        opt_tol=opt_tol,
        feas_tol=feas_tol,
    )

  def _scs(tol: float, indirect: bool) -> harness.BenchmarkResult:
    return conic_helpers.solve_scs(
        suite="conic_vs_clarabel_scs",
        problem_name=f"SOC Portfolio (n={n})",
        p_sp=p_sp,
        q_vec=-mu,
        a_sp=a_ext,
        b_vec=b_ext,
        cone_dict={"z": 2, "q": [n]},
        tol=tol,
        use_indirect=indirect,
    )

  def _cvxpy() -> cp.Problem:
    x = cp.Variable(n)
    obj = cp.Minimize(0.5 * cp.quad_form(x, cp.psd_wrap(cov)) - mu @ x)
    cons = [a_mat @ x == b_vec, cp.SOC(x[0], x[1:])]
    return cp.Problem(obj, cons)

  return ConicBenchmark(
      name=f"SOC Portfolio (n={n})",
      category="SOCP",
      proxax_builder=_proxax,
      clarabel_builder=_clarabel,
      scs_builder=_scs,
      cvxpy_builder=_cvxpy,
  )


class _BatchExpConeProx:
  """Vectorized product of `k` 3D exponential cones on a `(k, 3)` array."""

  def __init__(self) -> None:
    self._single = px.prox.ExpCone()

  @property
  def strong_convexity(self) -> float:
    return 0.0

  def value(self, x: jax.Array) -> jax.Array:
    return jnp.asarray(0.0, dtype=x.dtype)

  def prox(
      self, x: jax.Array, step_size: float | jax.Array
  ) -> jax.Array:
    del step_size
    return jnp.asarray(jax.vmap(lambda row: self._single.prox(row, 1.0))(x))

  def domain_project(self, x: jax.Array) -> jax.Array:
    return self.prox(x, 1.0)


def _make_exp_cone_blocks(k: int, *, seed: int = 7) -> ConicBenchmark:
  """Product of `k` 3D exponential cones with linear budget constraints."""
  rng = np.random.default_rng(seed)
  n = 3 * k
  v_blocks = rng.uniform(0.5, 2.0, size=(k, 3))
  c_blocks = rng.uniform(0.05, 0.2, size=(k, 3))
  a_mat = np.zeros((k, n))
  b_vec = np.full(k, 2.0)
  for j in range(k):
    a_mat[j, 3 * j] = 1.0
    a_mat[j, 3 * j + 1] = 1.0

  def _proxax() -> tuple[px.CompositeProblem, jax.Array]:
    v_j = jnp.asarray(v_blocks)
    c_j = jnp.asarray(c_blocks)

    def _obj(xs: jax.Array) -> jax.Array:
      return 0.5 * jnp.sum((xs - v_j) ** 2) + jnp.sum(c_j * xs)

    def _eq_fn(xs: jax.Array) -> jax.Array:
      return xs[:, 0] + xs[:, 1] - 2.0

    prob = px.CompositeProblem(
        smooth=px.oracle.as_smooth_term(_obj),
        proximal=_BatchExpConeProx(),
        constraints=px.constraints.equality(_eq_fn, jnp.zeros(k)),
    )
    x0 = jnp.broadcast_to(jnp.array([0.5, 1.5, 3.0]), (k, 3)).copy()
    return prob, x0

  p_sp = sp.eye(n, format="csc")
  q_vec = c_blocks.reshape(-1) - v_blocks.reshape(-1)
  const_term = float(0.5 * np.sum(v_blocks**2))
  a_ext = sp.vstack(
      [sp.csc_matrix(a_mat), -sp.eye(n, format="csc")], format="csc"
  )
  b_ext = np.concatenate([b_vec, np.zeros(n)])

  def _clarabel(opt_tol: float, feas_tol: float) -> harness.BenchmarkResult:
    cones = [clarabel.ZeroConeT(k)] + [
        clarabel.ExponentialConeT() for _ in range(k)
    ]
    res = conic_helpers.solve_clarabel(
        suite="conic_vs_clarabel_scs",
        problem_name=f"ExpCone Blocks (k={k}, n={n})",
        p_sp=p_sp,
        q_vec=q_vec,
        a_sp=a_ext,
        b_vec=b_ext,
        cones=cones,
        opt_tol=opt_tol,
        feas_tol=feas_tol,
    )
    return dataclasses.replace(res, obj_val=res.obj_val + const_term)

  def _scs(tol: float, indirect: bool) -> harness.BenchmarkResult:
    res = conic_helpers.solve_scs(
        suite="conic_vs_clarabel_scs",
        problem_name=f"ExpCone Blocks (k={k}, n={n})",
        p_sp=p_sp,
        q_vec=q_vec,
        a_sp=a_ext,
        b_vec=b_ext,
        cone_dict={"z": k, "ep": k},
        tol=tol,
        use_indirect=indirect,
    )
    return dataclasses.replace(res, obj_val=res.obj_val + const_term)

  def _cvxpy() -> cp.Problem:
    x = cp.Variable((k, 3))
    obj = cp.Minimize(
        0.5 * cp.sum_squares(x - v_blocks) + cp.sum(cp.multiply(c_blocks, x))
    )
    cons = [
        x[:, 0] + x[:, 1] == 2.0,
        cp.constraints.exponential.ExpCone(x[:, 0], x[:, 1], x[:, 2]),
    ]
    return cp.Problem(obj, cons)

  return ConicBenchmark(
      name=f"ExpCone Blocks (k={k}, n={n})",
      category="ExpCone",
      proxax_builder=_proxax,
      clarabel_builder=_clarabel,
      scs_builder=_scs,
      cvxpy_builder=_cvxpy,
  )


def _svec_indices(
    n: int, *, upper: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Returns column-major `(rows, cols, scale)` for symmetric `svec` packing."""
  if upper:
    pairs = [(i, j) for j in range(n) for i in range(j + 1)]
  else:
    pairs = [(i, j) for j in range(n) for i in range(j, n)]
  r_idx = np.array([p[0] for p in pairs])
  c_idx = np.array([p[1] for p in pairs])
  scale = np.where(r_idx == c_idx, 1.0, np.sqrt(2.0))
  return r_idx, c_idx, scale


def _make_spectrahedron_sdp(n: int, *, seed: int = 11) -> ConicBenchmark:
  """Spectrahedron projection with an extra trace-like equality constraint."""
  rng = np.random.default_rng(seed)
  raw = rng.standard_normal((n, n))
  m_mat = 0.5 * (raw + raw.T)

  def _proxax() -> tuple[px.CompositeProblem, jax.Array]:
    m_j = jnp.asarray(m_mat)

    def _obj(mat_x: jax.Array) -> jax.Array:
      return 0.5 * jnp.sum((mat_x - m_j) ** 2)

    def _diag_sum(mat_x: jax.Array) -> jax.Array:
      return jnp.array([mat_x[0, 0] + mat_x[1, 1] - 0.5])

    prob = px.CompositeProblem(
        smooth=px.oracle.as_smooth_term(_obj),
        proximal=px.prox.Spectrahedron(scale=1.0),
        constraints=px.constraints.equality(_diag_sum, jnp.array([0.0])),
    )
    return prob, jnp.eye(n) / n

  dim_s = n * (n + 1) // 2
  const_term = float(0.5 * np.sum(m_mat**2))

  def _build_svec_mats(
      *, upper: bool
  ) -> tuple[sp.csc_matrix, np.ndarray, sp.csc_matrix, np.ndarray]:
    r_idx, c_idx, scale = _svec_indices(n, upper=upper)
    q_svec = -m_mat[r_idx, c_idx] * scale
    p_sp = sp.eye(dim_s, format="csc")

    # Row 0: Tr(X) = 1; Row 1: X[0,0] + X[1,1] = 0.5
    diag_mask = (r_idx == c_idx).astype(float)
    row_diag = np.zeros(dim_s)
    for idx, (ri, ci) in enumerate(zip(r_idx, c_idx, strict=True)):
      if ri == ci and ri in (0, 1):
        row_diag[idx] = 1.0
    a_eq = sp.csc_matrix(np.vstack([diag_mask, row_diag]))
    a_ext = sp.vstack([a_eq, -sp.eye(dim_s, format="csc")], format="csc")
    b_ext = np.concatenate([np.array([1.0, 0.5]), np.zeros(dim_s)])
    return p_sp, q_svec, a_ext, b_ext

  def _clarabel(opt_tol: float, feas_tol: float) -> harness.BenchmarkResult:
    p_sp, q_svec, a_ext, b_ext = _build_svec_mats(upper=True)
    cones = [
        clarabel.ZeroConeT(2),
        clarabel.PSDTriangleConeT(n),
    ]
    res = conic_helpers.solve_clarabel(
        suite="conic_vs_clarabel_scs",
        problem_name=f"Spectrahedron SDP (n={n})",
        p_sp=p_sp,
        q_vec=q_svec,
        a_sp=a_ext,
        b_vec=b_ext,
        cones=cones,
        opt_tol=opt_tol,
        feas_tol=feas_tol,
    )
    return dataclasses.replace(res, obj_val=res.obj_val + const_term)

  def _scs(tol: float, indirect: bool) -> harness.BenchmarkResult:
    p_sp, q_svec, a_ext, b_ext = _build_svec_mats(upper=False)
    res = conic_helpers.solve_scs(
        suite="conic_vs_clarabel_scs",
        problem_name=f"Spectrahedron SDP (n={n})",
        p_sp=p_sp,
        q_vec=q_svec,
        a_sp=a_ext,
        b_vec=b_ext,
        cone_dict={"z": 2, "s": [n]},
        tol=tol,
        use_indirect=indirect,
    )
    return dataclasses.replace(res, obj_val=res.obj_val + const_term)

  def _cvxpy() -> cp.Problem:
    x = cp.Variable((n, n), PSD=True)
    obj = cp.Minimize(0.5 * cp.sum_squares(x - m_mat))
    cons = [cp.trace(x) == 1.0, x[0, 0] + x[1, 1] == 0.5]
    return cp.Problem(obj, cons)

  return ConicBenchmark(
      name=f"Spectrahedron SDP (n={n})",
      category="SDP",
      proxax_builder=_proxax,
      clarabel_builder=_clarabel,
      scs_builder=_scs,
      cvxpy_builder=_cvxpy,
  )


def run_suite(
    *,
    opt_tol: float = 1e-4,
    feas_tol: float = 1e-4,
    include_cvxpy: bool = False,
    time_limit: float = 10.0,
    category: str | None = None,
) -> list[harness.BenchmarkResult]:
  """Runs all conic benchmarks across Proxax, Clarabel, SCS, and CVXPY."""
  benchmarks: list[ConicBenchmark] = [
      _make_socp_portfolio(32),
      _make_socp_portfolio(128),
      _make_socp_portfolio(512),
      _make_exp_cone_blocks(1),
      _make_exp_cone_blocks(16),
      _make_exp_cone_blocks(64),
      _make_spectrahedron_sdp(8),
      _make_spectrahedron_sdp(24),
      _make_spectrahedron_sdp(48),
  ]
  if category is not None and category.lower() != "all":
    benchmarks = [
        b for b in benchmarks if b.category.lower() == category.lower()
    ]

  proxax_solvers = [
      (
          "PANOC-ALM",
          ialm.IALM(
              opt_tol=opt_tol,
              feas_tol=feas_tol,
              max_alm_iters=60,
              subproblem_solver="panoc",
          ),
      ),
      (
          "IALM (APD)",
          ialm.IALM(
              opt_tol=opt_tol,
              feas_tol=feas_tol,
              max_alm_iters=60,
              subproblem_solver="apd",
          ),
      ),
      ("QP-APD", qp_apd.QPAPD(opt_tol=opt_tol, feas_tol=feas_tol)),
      ("QP-AIPP", qp_aipp.QPAIPP(opt_tol=opt_tol, feas_tol=feas_tol)),
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
    print(
        f"[{idx}/{total}] {bench.name} ({bench.category})",
        file=sys.stderr,
        flush=True,
    )
    prob, x0 = bench.proxax_builder()
    for solver_name, solver in proxax_solvers:
      res = harness.time_proxax_solve(
          suite="conic_vs_clarabel_scs",
          problem_name=bench.name,
          solver_name=solver_name,
          prob=prob,
          x0=x0,
          solver=solver,
          opt_tol=opt_tol,
          feas_tol=feas_tol,
          time_limit=time_limit,
      )
      _log_done(res)
      results.append(res)
    for ext_res in (
        bench.clarabel_builder(opt_tol, feas_tol),
        bench.scs_builder(min(opt_tol, feas_tol), False),
        bench.scs_builder(min(opt_tol, feas_tol), True),
    ):
      _log_done(ext_res)
      results.append(ext_res)
    if include_cvxpy and bench.cvxpy_builder is not None:
      cp_prob = bench.cvxpy_builder()
      cp_res = conic_helpers.solve_cvxpy(
          suite="conic_vs_clarabel_scs",
          problem_name=bench.name,
          prob=cp_prob,
          solver_name="CVXPY (SCS)",
          cvxpy_solver=cp.SCS,
          solver_kwargs={"eps": min(opt_tol, feas_tol)},
      )
      _log_done(cp_res)
      results.append(cp_res)
  return results


def main() -> None:
  """CLI entry point for conic benchmarks."""
  parser = argparse.ArgumentParser(
      description="Benchmark Proxax conic solvers vs Clarabel, SCS, and CVXPY."
  )
  parser.add_argument("--opt-tol", type=float, default=1e-4)
  parser.add_argument("--feas-tol", type=float, default=1e-4)
  parser.add_argument("--time-limit", type=float, default=10.0)
  parser.add_argument(
      "--category",
      type=str,
      default="all",
      choices=["all", "SOCP", "ExpCone", "SDP"],
  )
  parser.add_argument("--include-cvxpy", action="store_true")
  parser.add_argument("--json-out", type=pathlib.Path, default=None)
  args = parser.parse_args()

  results = run_suite(
      opt_tol=args.opt_tol,
      feas_tol=args.feas_tol,
      include_cvxpy=args.include_cvxpy,
      time_limit=args.time_limit,
      category=args.category,
  )
  print(harness.format_markdown_table(results))
  if args.json_out is not None:
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    payload = [dataclasses.asdict(r) for r in results]
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
  main()
