"""Large-scale matrix-free operator benchmark: proxax vs. Clarabel.

Tests O(n log n) FFT-based operators where forming and factoring an explicit
matrix in an interior-point solver hits O(n^1.5) - O(n^2) KKT fill-in walls:
  1. ``circulant_lasso``: min_x 0.5 ||C x - y||_2^2 + lam ||x||_1
     with circulant kernel bandwidth W=64 for n in {1024, 4096, 16384, 65536}.
  2. ``deblur_2d_box``: 2D FFT Gaussian deblurring with box bounds [0, 1] and
     smooth Huber-TV regularization across L in {32, 64, 128, 256}
     (n = L^2 in {1024, 4096, 16384, 65536}).
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import pathlib
import sys
import time

import numpy as np


@dataclasses.dataclass(frozen=True)
class LargeOpCase:
  """Problem data for a large-scale matrix-free operator benchmark."""

  suite: str
  name: str
  n: int
  grid_side: int
  kernel: np.ndarray
  target: np.ndarray
  reg: float
  bandwidth: int


@dataclasses.dataclass(frozen=True)
class LargeOpResult:
  """Timing and objective result for one solver on a large operator case."""

  suite: str
  problem_name: str
  solver_name: str
  n: int
  converged: bool
  obj_val: float
  compile_s: float
  run_s: float


def build_cases() -> list[LargeOpCase]:
  """Constructs large-scale circulant LASSO and 2D FFT deblurring cases."""
  rng = np.random.default_rng(19)
  cases: list[LargeOpCase] = []

  # 1. Circulant LASSO (bandwidth W=64)
  bw = 64
  for n in (1024, 4096, 16384, 65536):
    k_1d = np.zeros(n, dtype=np.float64)
    idx = np.arange(bw)
    raw_k = np.exp(-0.08 * idx) * np.cos(0.25 * idx)
    k_1d[:bw] = raw_k / np.sum(np.abs(raw_k))
    x_true = np.zeros(n, dtype=np.float64)
    nnz_idx = rng.choice(n, size=max(16, n // 32), replace=False)
    x_true[nnz_idx] = rng.standard_normal(len(nnz_idx))
    y = np.fft.irfft(np.fft.rfft(k_1d) * np.fft.rfft(x_true), n=n)
    y = y + 0.01 * rng.standard_normal(n)
    cases.append(
        LargeOpCase(
            suite="circulant_lasso",
            name=f"circ_lasso_n{n}",
            n=n,
            grid_side=0,
            kernel=k_1d,
            target=y,
            reg=0.02,
            bandwidth=bw,
        )
    )

  # 2. 2D Gaussian deblurring with box [0, 1] + smooth quadratic gradient
  for side in (32, 64, 128, 256):
    n = side * side
    ax = np.fft.fftfreq(side) * side
    gx, gy = np.meshgrid(ax, ax, indexing="ij")
    psf = np.exp(-0.15 * (gx * gx + gy * gy))
    psf /= np.sum(psf)
    # Smooth synthetic target image in [0, 1]
    u_grid = np.linspace(0.0, 1.0, side)
    xx, yy = np.meshgrid(u_grid, u_grid, indexing="ij")
    img_true = np.clip(
        0.5
        + 0.35 * np.sin(2.0 * np.pi * xx) * np.cos(2.0 * np.pi * yy)
        + 0.2 * ((xx - 0.5) ** 2 + (yy - 0.5) ** 2 < 0.08),
        0.0,
        1.0,
    )
    blur = np.fft.irfft2(
        np.fft.rfft2(psf) * np.fft.rfft2(img_true), s=(side, side)
    )
    blur = blur + 0.005 * rng.standard_normal((side, side))
    cases.append(
        LargeOpCase(
            suite="deblur_2d_box",
            name=f"deblur_2d_n{n}",
            n=n,
            grid_side=side,
            kernel=psf,
            target=blur,
            reg=1e-2,
            bandwidth=0,
        )
    )

  return cases


def _solve_proxax_large(
    case: LargeOpCase, *, tol: float, repeats: int
) -> LargeOpResult:
  """Solves a large-scale FFT operator problem matrix-free with proxax-PANOC."""
  import jax
  import jax.numpy as jnp

  from proxax import px
  from proxax.solvers import PANOC

  x0 = jnp.zeros(case.n, dtype=jnp.float64)
  solver = PANOC(opt_tol=tol, max_iters=2000)

  if case.suite == "circulant_lasso":
    k_fft = jnp.fft.rfft(jnp.asarray(case.kernel, dtype=jnp.float64))
    y_j = jnp.asarray(case.target, dtype=jnp.float64)

    def smooth_lasso(x: jax.Array) -> jax.Array:
      res = jnp.fft.irfft(k_fft * jnp.fft.rfft(x), n=case.n) - y_j
      return 0.5 * jnp.dot(res, res)

    prob = px.CompositeProblem(
        smooth=smooth_lasso,
        proximal=px.prox.L1(case.reg),
    )
  else:
    side = case.grid_side
    psf_fft = jnp.fft.rfft2(jnp.asarray(case.kernel, dtype=jnp.float64))
    b_j = jnp.asarray(case.target, dtype=jnp.float64)

    def smooth_deblur(x: jax.Array) -> jax.Array:
      x_img = x.reshape((side, side))
      res = jnp.fft.irfft2(psf_fft * jnp.fft.rfft2(x_img), s=(side, side)) - b_j
      dx = jnp.roll(x_img, -1, axis=0) - x_img
      dy = jnp.roll(x_img, -1, axis=1) - x_img
      return 0.5 * jnp.sum(res * res) + 0.5 * case.reg * (
          jnp.sum(dx * dx) + jnp.sum(dy * dy)
      )

    prob = px.CompositeProblem(
        smooth=smooth_deblur,
        proximal=px.prox.Box(lower=0.0, upper=1.0),
    )

  def _run() -> px.Solution:
    sol = px.solve(prob, x0, solver=solver, opt_tol=tol, jit=True)
    sol.x.block_until_ready()
    return sol

  t0 = time.perf_counter()
  sol = _run()
  compile_s = time.perf_counter() - t0

  run_times: list[float] = []
  for _ in range(max(1, repeats)):
    t1 = time.perf_counter()
    sol = _run()
    run_times.append(time.perf_counter() - t1)

  return LargeOpResult(
      suite=case.suite,
      problem_name=case.name,
      solver_name="proxax-PANOC",
      n=case.n,
      converged=bool(sol.status == int(px.SolveStatus.CONVERGED.value)),
      obj_val=float(sol.stats.obj_val),
      compile_s=compile_s,
      run_s=min(run_times),
  )


def _solve_clarabel_large(
    case: LargeOpCase, *, tol: float, repeats: int, max_n: int
) -> LargeOpResult | None:
  """Solves the explicit sparse-matrix reformulation with Clarabel."""
  if case.n > max_n:
    return None

  import clarabel
  import scipy.sparse as sp

  n = case.n
  settings = clarabel.DefaultSettings()
  settings.verbose = False
  settings.tol_gap_abs = tol
  settings.tol_gap_rel = tol
  settings.tol_feas = tol

  if case.suite == "circulant_lasso":
    # Epigraph: min 0.5 ||r||^2 + lam sum(t) s.t. C x - r = y, -t <= x <= t
    bw = case.bandwidth
    diags = [np.full(n, case.kernel[d], dtype=np.float64) for d in range(bw)]
    offsets = [-d for d in range(bw)]
    C_sparse = sp.diags(diags, offsets, shape=(n, n), format="csc")
    # Also wrap around top-right corner for true circular convolution
    wrap_rows: list[int] = []
    wrap_cols: list[int] = []
    wrap_vals: list[float] = []
    for d in range(1, bw):
      val = float(case.kernel[d])
      if val != 0.0:
        r_idx = np.arange(d)
        c_idx = n - d + r_idx
        wrap_rows.extend(r_idx.tolist())
        wrap_cols.extend(c_idx.tolist())
        wrap_vals.extend([val] * d)
    if wrap_rows:
      C_sparse = (
          C_sparse
          + sp.csc_matrix((wrap_vals, (wrap_rows, wrap_cols)), shape=(n, n))
      ).tocsc()

    # Variables: [x (n), t (n), r (n)]
    P_csc = sp.block_diag(
        [sp.csc_matrix((2 * n, 2 * n)), sp.eye(n, format="csc")], format="csc"
    )
    q_vec = np.concatenate([
        np.zeros(n, dtype=np.float64),
        np.full(n, case.reg, dtype=np.float64),
        np.zeros(n, dtype=np.float64),
    ])
    eye_n = sp.eye(n, format="csc")
    zero_n = sp.csc_matrix((n, n))
    # C x - r = y  => ZeroConeT(n)
    A_eq = sp.hstack([C_sparse, zero_n, -eye_n], format="csc")
    # x - t <= 0 and -x - t <= 0 => NonnegativeConeT(2n)
    A_ineq1 = sp.hstack([eye_n, -eye_n, zero_n], format="csc")
    A_ineq2 = sp.hstack([-eye_n, -eye_n, zero_n], format="csc")
    A_csc = sp.vstack([A_eq, A_ineq1, A_ineq2], format="csc")
    b_vec = np.concatenate([case.target, np.zeros(2 * n, dtype=np.float64)])
    cones = [clarabel.ZeroConeT(n), clarabel.NonnegativeConeT(2 * n)]
  else:
    # 2D deblur: 0.5||r||^2 + 0.5*reg*(||dx||^2+||dy||^2), Kx-r=b, 0<=x<=1
    side = case.grid_side
    # Build explicit sparse K from non-negligible PSF entries
    psf = case.kernel
    nz_r, nz_c = np.where(psf > 1e-10)
    rows_l: list[np.ndarray] = []
    cols_l: list[np.ndarray] = []
    vals_l: list[np.ndarray] = []
    base_i, base_j = np.meshgrid(
        np.arange(side), np.arange(side), indexing="ij"
    )
    out_idx = (base_i * side + base_j).ravel()
    for dr, dc in zip(nz_r, nz_c, strict=True):
      in_i = (base_i - dr) % side
      in_j = (base_j - dc) % side
      rows_l.append(out_idx)
      cols_l.append((in_i * side + in_j).ravel())
      vals_l.append(np.full(n, float(psf[dr, dc]), dtype=np.float64))
    K_sparse = sp.csc_matrix(
        (
            np.concatenate(vals_l),
            (np.concatenate(rows_l), np.concatenate(cols_l)),
        ),
        shape=(n, n),
    )

    # Smoothness Laplacian L_2d = Dx^T Dx + Dy^T Dy
    idx_grid = np.arange(n).reshape((side, side))
    dx_mat = sp.eye(n, format="csc") - sp.csc_matrix(
        (np.ones(n), (np.arange(n), np.roll(idx_grid, -1, axis=0).ravel())),
        shape=(n, n),
    )
    dy_mat = sp.eye(n, format="csc") - sp.csc_matrix(
        (np.ones(n), (np.arange(n), np.roll(idx_grid, -1, axis=1).ravel())),
        shape=(n, n),
    )
    lap = (dx_mat.T @ dx_mat + dy_mat.T @ dy_mat).tocsc()

    # Variables: [x (n), r (n)]
    P_csc = sp.block_diag(
        [sp.triu(case.reg * lap, format="csc"), sp.eye(n, format="csc")],
        format="csc",
    )
    q_vec = np.zeros(2 * n, dtype=np.float64)
    eye_n = sp.eye(n, format="csc")
    zero_n = sp.csc_matrix((n, n))
    A_eq = sp.hstack([K_sparse, -eye_n], format="csc")
    A_ub = sp.hstack([eye_n, zero_n], format="csc")
    A_lb = sp.hstack([-eye_n, zero_n], format="csc")
    A_csc = sp.vstack([A_eq, A_ub, A_lb], format="csc")
    b_vec = np.concatenate([
        case.target.ravel(),
        np.ones(n, dtype=np.float64),
        np.zeros(n, dtype=np.float64),
    ])
    cones = [clarabel.ZeroConeT(n), clarabel.NonnegativeConeT(2 * n)]

  def _run() -> clarabel.DefaultSolution:
    solver = clarabel.DefaultSolver(P_csc, q_vec, A_csc, b_vec, cones, settings)
    return solver.solve()

  t0 = time.perf_counter()
  sol = _run()
  first_s = time.perf_counter() - t0

  run_times = [first_s]
  if first_s < 10.0:
    for _ in range(max(1, repeats)):
      t1 = time.perf_counter()
      sol = _run()
      run_times.append(time.perf_counter() - t1)

  return LargeOpResult(
      suite=case.suite,
      problem_name=case.name,
      solver_name="Clarabel",
      n=n,
      converged=sol.status == clarabel.SolverStatus.Solved,
      obj_val=float(sol.obj_val),
      compile_s=0.0,
      run_s=min(run_times),
  )


def format_markdown_table(results: list[LargeOpResult]) -> str:
  """Formats large-scale operator benchmark results as Markdown."""
  lines = [
      (
          "| Suite | Problem | Solver | n | Converged | Obj "
          "| Compile (s) | Run (ms) |"
      ),
      "| :--- | :--- | :--- | ---: | :--- | ---: | ---: | ---: |",
  ]
  for r in results:
    lines.append(
        f"| {r.suite} | {r.problem_name} | {r.solver_name} "
        f"| {r.n} | {r.converged} | {r.obj_val:.4f} "
        f"| {r.compile_s:.2f} | {r.run_s * 1000:.2f} |"
    )
  return "\n".join(lines)


def main() -> None:
  """CLI entry point for large-scale matrix-free operator benchmarks."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--tol", type=float, default=1e-6)
  parser.add_argument("--repeats", type=int, default=3)
  parser.add_argument(
      "--clarabel-max-n",
      type=int,
      default=16384,
      help="Skip Clarabel above this dimension to avoid OOM/timeouts.",
  )
  args = parser.parse_args()

  repo_root = pathlib.Path(__file__).resolve().parents[2]
  src_dir = repo_root / "src"
  if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
  os.environ.setdefault("JAX_ENABLE_X64", "1")

  import jax

  jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

  results: list[LargeOpResult] = []
  for case in build_cases():
    print(f"Running {case.name}...", file=sys.stderr, flush=True)
    results.append(
        _solve_proxax_large(case, tol=args.tol, repeats=args.repeats)
    )
    suite_max_n = (
        args.clarabel_max_n
        if case.suite == "circulant_lasso"
        else min(args.clarabel_max_n, 4096)
    )
    clar_res = _solve_clarabel_large(
        case, tol=args.tol, repeats=args.repeats, max_n=suite_max_n
    )
    if clar_res is not None:
      results.append(clar_res)

  print(format_markdown_table(results))


if __name__ == "__main__":
  main()
