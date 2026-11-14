"""Closed-loop warm-started MPC rollout benchmark: proxax vs. Clarabel & alpaqa.

Simulates a T=20 step receding-horizon MPC controller on a coupled 6-mass
oscillator system (n_x=12, n_u=4) across horizons N in {10, 20} and batch
sizes B in {1, 16}:
  1. ``mpc_box``: Condensed input-constrained MPC (u_min <= u_k <= u_max),
     comparing `proxax-PANOC` (cold & warm `lax.scan` inside `jax.jit`) vs.
     `Clarabel` and `alpaqa-PANOC` (in-process or worker).
  2. ``mpc_state_con``: State- and input-constrained condensed MPC
     (|x_k| <= x_max, |u_k| <= u_max), comparing `proxax-IAIPAL` (cold & warm
     `lax.scan` inside `jax.jit`) vs. `Clarabel`.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import datetime
import json
import os
import pathlib
import subprocess
import sys
import time
import typing

import numpy as np


@dataclasses.dataclass(frozen=True)
class MpcCase:
  """Problem data for a closed-loop MPC rollout."""

  suite: str
  name: str
  nx: int
  nu: int
  horizon: int
  sim_steps: int
  batch_size: int
  A_dyn: np.ndarray
  B_dyn: np.ndarray
  Q_stage: np.ndarray
  R_stage: np.ndarray
  Q_term: np.ndarray
  u_bound: float
  x_bound: float | None
  x0_batch: np.ndarray


@dataclasses.dataclass(frozen=True)
class MpcResult:
  """Timing and trajectory accuracy result for one MPC solver."""

  suite: str
  problem_name: str
  solver_name: str
  horizon: int
  batch_size: int
  converged: bool
  mean_cost: float
  compile_s: float
  run_s: float


def _make_oscillator_system(
    nx: int = 12, nu: int = 4
) -> tuple[np.ndarray, np.ndarray]:
  """Builds a discretized coupled chain of 6 mass-spring oscillators."""
  n_mass = nx // 2
  dt = 0.1
  K = np.zeros((n_mass, n_mass), dtype=np.float64)
  for i in range(n_mass):
    K[i, i] = 2.0
    if i > 0:
      K[i, i - 1] = -1.0
    if i + 1 < n_mass:
      K[i, i + 1] = -1.0
  A_c = np.zeros((nx, nx), dtype=np.float64)
  A_c[:n_mass, n_mass:] = np.eye(n_mass)
  A_c[n_mass:, :n_mass] = -K
  A_c[n_mass:, n_mass:] = -0.15 * np.eye(n_mass)
  B_c = np.zeros((nx, nu), dtype=np.float64)
  for j in range(nu):
    B_c[n_mass + j, j] = 1.0
  # Forward Euler / 2-step RK2 discretization for clean stability
  A_d = np.eye(nx) + dt * A_c + 0.5 * (dt * dt) * (A_c @ A_c)
  B_d = dt * B_c + 0.5 * dt * dt * (A_c @ B_c)
  return A_d, B_d


def _condense_mpc(
    A_d: np.ndarray,
    B_d: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    Qf: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Builds condensed MPC matrices: X = S_x x0 + S_u U."""
  import scipy.linalg as la

  nx, nu = B_d.shape
  S_x = np.zeros((horizon * nx, nx), dtype=np.float64)
  S_u = np.zeros((horizon * nx, horizon * nu), dtype=np.float64)
  A_pow = np.eye(nx, dtype=np.float64)
  for k in range(horizon):
    A_pow = A_d @ A_pow
    S_x[k * nx : (k + 1) * nx, :] = A_pow
    cur = B_d.copy()
    for j in range(k, -1, -1):
      S_u[k * nx : (k + 1) * nx, j * nu : (j + 1) * nu] = cur
      cur = A_d @ cur
  Q_blocks = [Q] * (horizon - 1) + [Qf]
  Q_bar = la.block_diag(*Q_blocks)
  R_bar = np.kron(np.eye(horizon), R)
  H = S_u.T @ Q_bar @ S_u + R_bar
  H = 0.5 * (H + H.T)
  F = S_u.T @ Q_bar @ S_x
  return H, F, S_x, S_u, Q_bar


def build_cases() -> list[MpcCase]:
  """Constructs closed-loop MPC rollout benchmark cases."""
  rng = np.random.default_rng(7)
  nx, nu = 12, 4
  A_d, B_d = _make_oscillator_system(nx, nu)
  Q = np.eye(nx, dtype=np.float64)
  Q[nx // 2 :, nx // 2 :] *= 0.2
  R = 0.1 * np.eye(nu, dtype=np.float64)
  Qf = 5.0 * Q

  cases: list[MpcCase] = []
  for bs in (1, 16):
    x0_b = rng.uniform(-0.35, 0.35, size=(bs, nx))
    for horizon in (10, 20):
      cases.append(
          MpcCase(
              suite="mpc_box",
              name=f"mpc_box_N{horizon}_B{bs}",
              nx=nx,
              nu=nu,
              horizon=horizon,
              sim_steps=20,
              batch_size=bs,
              A_dyn=A_d,
              B_dyn=B_d,
              Q_stage=Q,
              R_stage=R,
              Q_term=Qf,
              u_bound=0.5,
              x_bound=None,
              x0_batch=x0_b,
          )
      )
      cases.append(
          MpcCase(
              suite="mpc_state_con",
              name=f"mpc_state_N{horizon}_B{bs}",
              nx=nx,
              nu=nu,
              horizon=horizon,
              sim_steps=20,
              batch_size=bs,
              A_dyn=A_d,
              B_dyn=B_d,
              Q_stage=Q,
              R_stage=R,
              Q_term=Qf,
              u_bound=0.8,
              x_bound=1.5,
              x0_batch=x0_b,
          )
      )
  return cases


def _solve_proxax_rollout(
    case: MpcCase,
    *,
    warm_start: bool,
    tol: float,
    repeats: int,
    use_threads: bool = False,
) -> MpcResult:
  """Runs closed-loop MPC via compiled `jax.lax.scan` (vmap over batch)."""
  import jax
  import jax.numpy as jnp

  from proxax import px
  from proxax.solvers import IAIPAL
  from proxax.solvers import PANOC

  H, F, S_x, S_u, _ = _condense_mpc(
      case.A_dyn,
      case.B_dyn,
      case.Q_stage,
      case.R_stage,
      case.Q_term,
      case.horizon,
  )
  n_z = case.horizon * case.nu
  H_j = jnp.asarray(H, dtype=jnp.float64)
  F_j = jnp.asarray(F, dtype=jnp.float64)
  A_d_j = jnp.asarray(case.A_dyn, dtype=jnp.float64)
  B_d_j = jnp.asarray(case.B_dyn, dtype=jnp.float64)
  Q_j = jnp.asarray(case.Q_stage, dtype=jnp.float64)
  R_j = jnp.asarray(case.R_stage, dtype=jnp.float64)
  x0_b_j = jnp.asarray(case.x0_batch, dtype=jnp.float64)

  lb = jnp.full(n_z, -case.u_bound, dtype=jnp.float64)
  ub = jnp.full(n_z, case.u_bound, dtype=jnp.float64)
  prox_op = px.prox.Box(lower=lb, upper=ub)

  if case.x_bound is not None:
    A_con = np.vstack([S_u, -S_u])
    A_con_j = jnp.asarray(A_con, dtype=jnp.float64)
    S_x_j = jnp.asarray(S_x, dtype=jnp.float64)
    xb_vec = jnp.full(case.horizon * case.nx, case.x_bound, dtype=jnp.float64)
    solver: typing.Any = IAIPAL(opt_tol=tol, feas_tol=tol)
  else:
    A_con_j = None
    S_x_j = None
    xb_vec = None
    solver = PANOC(opt_tol=tol)

  def _rollout_one(x_init: jax.Array) -> tuple[jax.Array, jax.Array]:
    z_zero = jnp.zeros(n_z, dtype=jnp.float64)

    def _step(
        carry: tuple[jax.Array, jax.Array, jax.Array],
        _: None,
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
      x_cur, z_prev, total_cost = carry
      q_vec = F_j @ x_cur
      z_init = (
          jnp.concatenate([z_prev[case.nu :], z_prev[-case.nu :]])
          if warm_start
          else z_zero
      )
      constraints = None
      if A_con_j is not None and S_x_j is not None and xb_vec is not None:
        Sx_x = S_x_j @ x_cur
        u_vec = jnp.concatenate([xb_vec - Sx_x, xb_vec + Sx_x])
        constraints = px.constraints.less_equal(A_con_j, u_vec)
      prob = px.CompositeProblem(
          smooth=lambda z: 0.5 * jnp.dot(z, H_j @ z) + jnp.dot(q_vec, z),
          proximal=prox_op,
          constraints=constraints,
      )
      sol = px.solve(
          prob,
          z_init,
          solver=solver,
          opt_tol=tol,
          feas_tol=tol,
          jit=True,
      )
      u_cur = sol.x[: case.nu]
      stage_cost = 0.5 * (
          jnp.dot(x_cur, Q_j @ x_cur) + jnp.dot(u_cur, R_j @ u_cur)
      )
      x_next = A_d_j @ x_cur + B_d_j @ u_cur
      ok = sol.status == int(px.SolveStatus.CONVERGED.value)
      return (x_next, sol.x, total_cost + stage_cost), ok

    init_carry = (x_init, z_zero, jnp.asarray(0.0, dtype=jnp.float64))
    (_, _, final_cost), oks = jax.lax.scan(
        _step, init_carry, None, length=case.sim_steps
    )
    return final_cost, jnp.all(oks)

  if use_threads and case.batch_size > 1:
    compiled_one = jax.jit(_rollout_one)
    x0_list = [x0_b_j[i] for i in range(case.batch_size)]
    pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=min(case.batch_size, os.cpu_count() or 8)
    )

    def _run() -> tuple[jax.Array, jax.Array]:
      outs = list(pool.map(compiled_one, x0_list))
      costs_t = jnp.stack([c for c, _ in outs])
      oks_t = jnp.stack([o for _, o in outs])
      costs_t.block_until_ready()
      return costs_t, oks_t

    try:
      t0 = time.perf_counter()
      costs, oks = _run()
      compile_s = time.perf_counter() - t0

      run_times: list[float] = []
      for _ in range(max(1, repeats)):
        t1 = time.perf_counter()
        costs, oks = _run()
        run_times.append(time.perf_counter() - t1)
    finally:
      pool.shutdown(wait=True)
  else:
    compiled_rollout = jax.jit(jax.vmap(_rollout_one))

    def _run() -> tuple[jax.Array, jax.Array]:
      costs, oks = compiled_rollout(x0_b_j)
      costs.block_until_ready()
      return costs, oks

    t0 = time.perf_counter()
    costs, oks = _run()
    compile_s = time.perf_counter() - t0

    run_times = []
    for _ in range(max(1, repeats)):
      t1 = time.perf_counter()
      costs, oks = _run()
      run_times.append(time.perf_counter() - t1)

  tag = "warm,threads" if (warm_start and use_threads) else (
      "warm" if warm_start else "cold"
  )
  base_name = "proxax-PANOC" if case.x_bound is None else "proxax-IAIPAL"
  return MpcResult(
      suite=case.suite,
      problem_name=case.name,
      solver_name=f"{base_name}({tag})",
      horizon=case.horizon,
      batch_size=case.batch_size,
      converged=bool(jnp.all(oks)),
      mean_cost=float(jnp.mean(costs)),
      compile_s=compile_s,
      run_s=min(run_times),
  )


def _solve_clarabel_rollout(
    case: MpcCase, *, tol: float, repeats: int
) -> MpcResult:
  """Runs closed-loop MPC in Python with Clarabel."""
  import clarabel
  import scipy.sparse as sp

  H, F, S_x, S_u, _ = _condense_mpc(
      case.A_dyn,
      case.B_dyn,
      case.Q_stage,
      case.R_stage,
      case.Q_term,
      case.horizon,
  )
  n_z = case.horizon * case.nu
  P_csc = sp.csc_matrix(np.triu(H))

  eye_z = sp.eye(n_z, format="csc")
  u_con = sp.vstack([eye_z, -eye_z], format="csc")
  u_rhs = np.full(2 * n_z, case.u_bound, dtype=np.float64)

  if case.x_bound is not None:
    A_x = sp.csc_matrix(np.vstack([S_u, -S_u]))
    A_csc = sp.vstack([u_con, A_x], format="csc")
    m_tot = 2 * n_z + 2 * case.horizon * case.nx
    xb_vec = np.full(case.horizon * case.nx, case.x_bound, dtype=np.float64)
  else:
    A_csc = u_con
    m_tot = 2 * n_z
    xb_vec = None

  cones = [clarabel.NonnegativeConeT(m_tot)]
  settings = clarabel.DefaultSettings()
  settings.verbose = False
  settings.tol_gap_abs = tol
  settings.tol_gap_rel = tol
  settings.tol_feas = tol

  def _rollout_one(idx: int) -> tuple[bool, float]:
    x_cur = case.x0_batch[idx].copy()
    total_cost = 0.0
    all_ok = True
    for _ in range(case.sim_steps):
      q_vec = F @ x_cur
      if xb_vec is not None:
        Sx_x = S_x @ x_cur
        b_vec = np.concatenate([u_rhs, xb_vec - Sx_x, xb_vec + Sx_x])
      else:
        b_vec = u_rhs
      solver = clarabel.DefaultSolver(
          P_csc, q_vec, A_csc, b_vec, cones, settings
      )
      sol = solver.solve()
      if sol.status != clarabel.SolverStatus.Solved:
        all_ok = False
      z_opt = np.asarray(sol.x, dtype=np.float64)
      u_cur = z_opt[: case.nu]
      total_cost += 0.5 * (
          float(x_cur @ case.Q_stage @ x_cur)
          + float(u_cur @ case.R_stage @ u_cur)
      )
      x_cur = case.A_dyn @ x_cur + case.B_dyn @ u_cur
    return all_ok, total_cost

  def _run_batch() -> tuple[bool, float]:
    if case.batch_size == 1:
      ok, cost = _rollout_one(0)
      return ok, cost
    with concurrent.futures.ThreadPoolExecutor() as pool:
      outs = list(pool.map(_rollout_one, range(case.batch_size)))
    return all(ok for ok, _ in outs), float(np.mean([c for _, c in outs]))

  t0 = time.perf_counter()
  all_ok, mean_cost = _run_batch()
  compile_s = time.perf_counter() - t0

  run_times: list[float] = []
  for _ in range(max(1, repeats)):
    t1 = time.perf_counter()
    all_ok, mean_cost = _run_batch()
    run_times.append(time.perf_counter() - t1)

  return MpcResult(
      suite=case.suite,
      problem_name=case.name,
      solver_name="Clarabel",
      horizon=case.horizon,
      batch_size=case.batch_size,
      converged=all_ok,
      mean_cost=mean_cost,
      compile_s=compile_s,
      run_s=min(run_times),
  )


def _run_alpaqa_mpc_worker(
    payload: dict[str, typing.Any],
) -> dict[str, typing.Any]:
  """Runs closed-loop box-MPC rollout with alpaqa PANOC (sequential for B=1)."""
  import alpaqa as pa

  H = np.asarray(payload["H"], dtype=np.float64)
  F = np.asarray(payload["F"], dtype=np.float64)
  A_dyn = np.asarray(payload["A_dyn"], dtype=np.float64)
  B_dyn = np.asarray(payload["B_dyn"], dtype=np.float64)
  Q_stage = np.asarray(payload["Q_stage"], dtype=np.float64)
  R_stage = np.asarray(payload["R_stage"], dtype=np.float64)
  x0_batch = np.asarray(payload["x0_batch"], dtype=np.float64)
  nu = int(payload["nu"])
  sim_steps = int(payload["sim_steps"])
  u_bound = float(payload["u_bound"])
  tol = float(payload["tol"])
  repeats = int(payload["repeats"])
  n_z = H.shape[0]
  bs = x0_batch.shape[0]

  box = pa.Box(n_z)
  box.lowerbound = np.full(n_z, -u_bound, dtype=np.float64)
  box.upperbound = np.full(n_z, u_bound, dtype=np.float64)
  panoc_params = pa.PANOCParams(
      max_iter=2000,
      max_time=datetime.timedelta(seconds=30.0),
      stop_crit=pa.PANOCStopCrit.FPRNorm,
  )
  lbfgs_params = pa.LBFGSParams(memory=10)

  def _rollout_one(idx: int) -> tuple[bool, float]:
    x_cur = x0_batch[idx].copy()
    z_prev = np.zeros(n_z, dtype=np.float64)
    total_cost = 0.0
    all_ok = True
    for _ in range(sim_steps):
      q_vec = F @ x_cur
      z_init = np.concatenate([z_prev[nu:], z_prev[-nu:]])

      def f(z: np.ndarray, qv: np.ndarray = q_vec) -> float:
        return float(0.5 * (z @ (H @ z)) + qv @ z)

      def grad_f(z: np.ndarray, qv: np.ndarray = q_vec) -> np.ndarray:
        return np.asarray(H @ z + qv, dtype=np.float64)

      z_sol, stats = pa.panoc(
          f, grad_f, box, z_init, tol, panoc_params, lbfgs_params
      )
      if stats["status"] != pa.SolverStatus.Converged:
        all_ok = False
      z_prev = np.asarray(z_sol, dtype=np.float64)
      u_cur = z_prev[:nu]
      total_cost += 0.5 * (
          float(x_cur @ Q_stage @ x_cur) + float(u_cur @ R_stage @ u_cur)
      )
      x_cur = A_dyn @ x_cur + B_dyn @ u_cur
    return all_ok, total_cost

  def _run_batch() -> tuple[bool, float]:
    outs = [_rollout_one(i) for i in range(bs)]
    return all(ok for ok, _ in outs), float(np.mean([c for _, c in outs]))

  t0 = time.perf_counter()
  all_ok, mean_cost = _run_batch()
  compile_s = time.perf_counter() - t0

  run_times: list[float] = []
  for _ in range(max(1, repeats)):
    t1 = time.perf_counter()
    all_ok, mean_cost = _run_batch()
    run_times.append(time.perf_counter() - t1)

  return {
      "converged": all_ok,
      "mean_cost": mean_cost,
      "compile_s": compile_s,
      "run_s": min(run_times),
  }


def _solve_alpaqa_rollout(
    case: MpcCase,
    *,
    tol: float,
    repeats: int,
    alpaqa_python: str | None,
) -> MpcResult:
  """Solves box-constrained MPC rollout with warm-started alpaqa PANOC."""
  H, F, _, _, _ = _condense_mpc(
      case.A_dyn,
      case.B_dyn,
      case.Q_stage,
      case.R_stage,
      case.Q_term,
      case.horizon,
  )
  payload = {
      "H": H.tolist(),
      "F": F.tolist(),
      "A_dyn": case.A_dyn.tolist(),
      "B_dyn": case.B_dyn.tolist(),
      "Q_stage": case.Q_stage.tolist(),
      "R_stage": case.R_stage.tolist(),
      "x0_batch": case.x0_batch.tolist(),
      "nu": case.nu,
      "sim_steps": case.sim_steps,
      "u_bound": case.u_bound,
      "tol": tol,
      "repeats": repeats,
  }
  if alpaqa_python is None:
    raw = _run_alpaqa_mpc_worker(payload)
  else:
    proc = subprocess.run(
        [alpaqa_python, __file__, "--alpaqa-worker"],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=True,
    )
    raw = json.loads(proc.stdout)

  return MpcResult(
      suite=case.suite,
      problem_name=case.name,
      solver_name="alpaqa-PANOC(warm)",
      horizon=case.horizon,
      batch_size=case.batch_size,
      converged=bool(raw["converged"]),
      mean_cost=float(raw["mean_cost"]),
      compile_s=float(raw["compile_s"]),
      run_s=float(raw["run_s"]),
  )


def _detect_alpaqa_python(cli_value: str | None) -> str | None:
  if cli_value:
    return cli_value
  try:
    import alpaqa  # noqa: F401

    return None
  except ImportError:
    pass
  candidate = pathlib.Path("/tmp/alpaqa_env/bin/python")
  if candidate.exists():
    return str(candidate)
  return None


def format_markdown_table(results: list[MpcResult]) -> str:
  """Formats MPC rollout benchmark results as Markdown."""
  lines = [
      (
          "| Suite | Problem | Solver | N | B | Converged | Mean Cost "
          "| Compile (s) | Run (ms) |"
      ),
      "| :--- | :--- | :--- | ---: | ---: | :--- | ---: | ---: | ---: |",
  ]
  for r in results:
    lines.append(
        f"| {r.suite} | {r.problem_name} | {r.solver_name} "
        f"| {r.horizon} | {r.batch_size} | {r.converged} "
        f"| {r.mean_cost:.4f} | {r.compile_s:.2f} | {r.run_s * 1000:.2f} |"
    )
  return "\n".join(lines)


def main() -> None:
  """CLI entry point for closed-loop MPC rollout benchmarks."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--alpaqa-worker", action="store_true")
  parser.add_argument("--alpaqa-python", type=str, default=None)
  parser.add_argument("--tol", type=float, default=1e-6)
  parser.add_argument("--repeats", type=int, default=3)
  args = parser.parse_args()

  if args.alpaqa_worker:
    payload = json.loads(sys.stdin.read())
    print(json.dumps(_run_alpaqa_mpc_worker(payload)))
    return

  repo_root = pathlib.Path(__file__).resolve().parents[2]
  src_dir = repo_root / "src"
  if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
  os.environ.setdefault("JAX_ENABLE_X64", "1")

  import jax

  jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

  alpaqa_python = _detect_alpaqa_python(args.alpaqa_python)
  results: list[MpcResult] = []
  for case in build_cases():
    print(f"Running {case.name}...", file=sys.stderr, flush=True)
    results.append(
        _solve_proxax_rollout(
            case, warm_start=False, tol=args.tol, repeats=args.repeats
        )
    )
    results.append(
        _solve_proxax_rollout(
            case, warm_start=True, tol=args.tol, repeats=args.repeats
        )
    )
    if case.batch_size > 1:
      results.append(
          _solve_proxax_rollout(
              case,
              warm_start=True,
              tol=args.tol,
              repeats=args.repeats,
              use_threads=True,
          )
      )
    if case.suite == "mpc_box" and case.batch_size == 1:
      results.append(
          _solve_alpaqa_rollout(
              case,
              tol=args.tol,
              repeats=args.repeats,
              alpaqa_python=alpaqa_python,
          )
      )
    results.append(
        _solve_clarabel_rollout(case, tol=args.tol, repeats=args.repeats)
    )

  print(format_markdown_table(results))


if __name__ == "__main__":
  main()
