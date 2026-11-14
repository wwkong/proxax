"""End-to-end 50-step differentiable optimization training loop benchmark.

Compares `proxax` (`px.batch_implicit_solve` with `jit(lax.scan)` over `vmap`
and Python loop over `threads`) against `cvxpylayers` (`CvxpyLayer` via
`diffcp`) on a 50-step Adam loop learning cost parameters of batched QP and
simplex layers (`n=50/100`, `B in {16, 64}`, `tol=1e-6`).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import subprocess
import sys
import time
import typing

import numpy as np


@dataclasses.dataclass(frozen=True)
class TrainCase:
  """Problem data for a 50-step differentiable layer training benchmark."""

  suite: str
  name: str
  n: int
  m: int
  batch_size: int
  steps: int
  lr: float
  P: np.ndarray
  A: np.ndarray | None
  b: np.ndarray | None
  q_batch: np.ndarray
  x_target: np.ndarray
  theta0: np.ndarray


@dataclasses.dataclass(frozen=True)
class TrainResult:
  """Training benchmark result for one solver."""

  suite: str
  problem_name: str
  solver_name: str
  batch_size: int
  steps: int
  init_loss: float
  final_loss: float
  compile_s: float
  train_s: float


def build_cases() -> list[TrainCase]:
  """Constructs QP and simplex 50-step training benchmark cases."""
  rng = np.random.default_rng(11)
  cases: list[TrainCase] = []

  n_qp, m_qp = 50, 25
  M_qp = rng.standard_normal((n_qp, n_qp)) / np.sqrt(n_qp)
  P_qp = M_qp.T @ M_qp + 0.5 * np.eye(n_qp)
  A_qp = rng.standard_normal((m_qp, n_qp)) / np.sqrt(n_qp)
  b_qp = A_qp @ np.full(n_qp, 0.25) + 0.5
  theta_true_qp = rng.normal(0.0, 0.5, size=(n_qp,))

  n_sx = 100
  M_sx = rng.standard_normal((n_sx, n_sx)) / np.sqrt(n_sx)
  P_sx = M_sx.T @ M_sx + 0.5 * np.eye(n_sx)
  theta_true_sx = rng.normal(0.0, 0.5, size=(n_sx,))

  for bs in (16, 64):
    q_b = rng.standard_normal((bs, n_qp))
    x_tgt_qp = np.clip(rng.uniform(0.0, 0.4, size=(bs, n_qp)), 0.0, 1.0)
    cases.append(
        TrainCase(
            suite="qp_train",
            name=f"qp_train_n{n_qp}_B{bs}",
            n=n_qp,
            m=m_qp,
            batch_size=bs,
            steps=50,
            lr=0.05,
            P=P_qp,
            A=A_qp,
            b=b_qp,
            q_batch=q_b + theta_true_qp[None, :],
            x_target=x_tgt_qp,
            theta0=np.zeros(n_qp, dtype=np.float64),
        )
    )

    mu_b = rng.standard_normal((bs, n_sx))
    x_tgt_sx = np.full((bs, n_sx), 1.0 / n_sx, dtype=np.float64)
    cases.append(
        TrainCase(
            suite="simplex_train",
            name=f"simplex_train_n{n_sx}_B{bs}",
            n=n_sx,
            m=0,
            batch_size=bs,
            steps=50,
            lr=0.05,
            P=P_sx,
            A=None,
            b=None,
            q_batch=mu_b + theta_true_sx[None, :],
            x_target=x_tgt_sx,
            theta0=np.zeros(n_sx, dtype=np.float64),
        )
    )
  return cases


def _solve_proxax_train(
    case: TrainCase,
    *,
    mode: str,
    tol: float,
    repeats: int,
) -> TrainResult:
  """Runs 50-step Adam training with proxax (`jit_scan` or `threads`)."""
  import jax
  import jax.numpy as jnp

  from proxax import px
  from proxax.solvers import IALM
  from proxax.solvers import PANOC

  P_j = jnp.asarray(case.P, dtype=jnp.float64)
  q_j = jnp.asarray(case.q_batch, dtype=jnp.float64)
  x_tgt_j = jnp.asarray(case.x_target, dtype=jnp.float64)
  theta0_j = jnp.asarray(case.theta0, dtype=jnp.float64)

  if case.suite == "qp_train" and case.A is not None and case.b is not None:
    A_j = jnp.asarray(case.A, dtype=jnp.float64)
    b_j = jnp.asarray(case.b, dtype=jnp.float64)
    x0_j = jnp.full(case.n, 0.25, dtype=jnp.float64)
    constraints = px.constraints.less_equal(A_j, b_j)
    prox_op: typing.Any = px.prox.NonNegative()
    solver: typing.Any = IALM(opt_tol=tol, feas_tol=tol)

    def prob_fn(q_vec: jax.Array) -> px.CompositeProblem:
      return px.CompositeProblem(
          smooth=lambda x: 0.5 * jnp.dot(x, P_j @ x) + jnp.dot(q_vec, x),
          proximal=prox_op,
          constraints=constraints,
      )

  else:
    x0_j = jnp.full(case.n, 1.0 / case.n, dtype=jnp.float64)
    prox_op = px.prox.Simplex(1.0)
    solver = PANOC(opt_tol=tol)

    def prob_fn(q_vec: jax.Array) -> px.CompositeProblem:
      return px.CompositeProblem(
          smooth=lambda x: 0.5 * jnp.dot(x, P_j @ x) - jnp.dot(q_vec, x),
          proximal=prox_op,
      )

  backend = "vmap" if mode == "jit_scan" else "threads"

  def loss_fn(theta: jax.Array) -> jax.Array:
    xs = px.batch_implicit_solve(
        prob_fn,
        x0_j,
        q_j + theta[None, :],
        solver=solver,
        opt_tol=tol,
        feas_tol=tol,
        linear_tol=tol,
        backend=backend,
    )
    diff = xs - x_tgt_j
    return 0.5 * jnp.mean(jnp.sum(diff * diff, axis=-1))

  val_and_grad = jax.value_and_grad(loss_fn)

  @jax.jit
  def adam_update(
      theta: jax.Array,
      m: jax.Array,
      v: jax.Array,
      grad: jax.Array,
      step: jax.Array,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    b1, b2, eps = 0.9, 0.999, 1e-8
    m_new = b1 * m + (1.0 - b1) * grad
    v_new = b2 * v + (1.0 - b2) * (grad * grad)
    m_hat = m_new / (1.0 - b1**step)
    v_hat = v_new / (1.0 - b2**step)
    theta_new = theta - case.lr * m_hat / (jnp.sqrt(v_hat) + eps)
    return theta_new, m_new, v_new

  init_loss = float(loss_fn(theta0_j))

  if mode == "jit_scan":

    @jax.jit
    def run_scan(th0: jax.Array) -> tuple[jax.Array, jax.Array]:
      def _body(
          carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
          _: None,
      ) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]:
        th, m_s, v_s, step_idx = carry
        loss_val, g = val_and_grad(th)
        th_n, m_n, v_n = adam_update(th, m_s, v_s, g, step_idx)
        return (th_n, m_n, v_n, step_idx + 1.0), loss_val

      zeros = jnp.zeros_like(th0)
      (th_f, _, _, _), _ = jax.lax.scan(
          _body,
          (th0, zeros, zeros, jnp.asarray(1.0, dtype=jnp.float64)),
          None,
          length=case.steps,
      )
      return th_f, loss_fn(th_f)

    def _exec() -> float:
      _, fin = run_scan(theta0_j)
      fin.block_until_ready()
      return float(fin)

  else:
    jit_val_and_grad = jax.jit(val_and_grad)

    def _exec() -> float:
      th = theta0_j
      m_s = jnp.zeros_like(th)
      v_s = jnp.zeros_like(th)
      for s in range(1, case.steps + 1):
        _, g = jit_val_and_grad(th)
        th, m_s, v_s = adam_update(
            th, m_s, v_s, g, jnp.asarray(float(s), dtype=jnp.float64)
        )
      fin = loss_fn(th)
      fin.block_until_ready()
      return float(fin)

  t0 = time.perf_counter()
  final_loss = _exec()
  compile_s = time.perf_counter() - t0

  run_times: list[float] = []
  for _ in range(max(1, repeats)):
    t1 = time.perf_counter()
    final_loss = _exec()
    run_times.append(time.perf_counter() - t1)

  s_name = "proxax(jit-scan)" if mode == "jit_scan" else "proxax(threads)"
  return TrainResult(
      suite=case.suite,
      problem_name=case.name,
      solver_name=s_name,
      batch_size=case.batch_size,
      steps=case.steps,
      init_loss=init_loss,
      final_loss=final_loss,
      compile_s=compile_s,
      train_s=min(run_times),
  )


def _run_cvxpylayers_train_worker(
    payload: dict[str, typing.Any],
) -> dict[str, typing.Any]:
  """Runs 50-step Adam training loop with cvxpylayers in Python."""
  import cvxpy as cp
  import jax
  import jax.numpy as jnp
  from cvxpylayers.jax import CvxpyLayer

  jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

  suite = str(payload["suite"])
  n = int(payload["n"])
  steps = int(payload["steps"])
  lr = float(payload["lr"])
  tol = float(payload["tol"])
  repeats = int(payload["repeats"])
  P = np.asarray(payload["P"], dtype=np.float64)
  q_batch = jnp.asarray(payload["q_batch"], dtype=jnp.float64)
  x_tgt = jnp.asarray(payload["x_target"], dtype=jnp.float64)
  theta0 = jnp.asarray(payload["theta0"], dtype=jnp.float64)

  x_var = cp.Variable(n)
  q_param = cp.Parameter(n)
  if suite == "qp_train":
    A = np.asarray(payload["A"], dtype=np.float64)
    b = np.asarray(payload["b"], dtype=np.float64)
    obj = cp.Minimize(0.5 * cp.quad_form(x_var, P) + q_param @ x_var)
    cons = [x_var >= 0, A @ x_var <= b]
  else:
    obj = cp.Minimize(0.5 * cp.quad_form(x_var, P) - q_param @ x_var)
    cons = [x_var >= 0, cp.sum(x_var) == 1.0]

  prob = cp.Problem(obj, cons)
  t_setup = time.perf_counter()
  layer = CvxpyLayer(prob, parameters=[q_param], variables=[x_var])
  setup_s = time.perf_counter() - t_setup

  solver_args = {"eps": tol, "eps_abs": tol, "eps_rel": tol}

  def loss_fn(theta: jax.Array) -> jax.Array:
    (xs,) = layer(q_batch + theta[None, :], solver_args=solver_args)
    diff = xs - x_tgt
    return 0.5 * jnp.mean(jnp.sum(diff * diff, axis=-1))

  val_and_grad = jax.value_and_grad(loss_fn)

  @jax.jit
  def adam_update(
      theta: jax.Array,
      m: jax.Array,
      v: jax.Array,
      grad: jax.Array,
      step: jax.Array,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    b1, b2, eps = 0.9, 0.999, 1e-8
    m_new = b1 * m + (1.0 - b1) * grad
    v_new = b2 * v + (1.0 - b2) * (grad * grad)
    m_hat = m_new / (1.0 - b1**step)
    v_hat = v_new / (1.0 - b2**step)
    theta_new = theta - lr * m_hat / (jnp.sqrt(v_hat) + eps)
    return theta_new, m_new, v_new

  init_loss = float(loss_fn(theta0))

  def _exec() -> float:
    th = theta0
    m_s = jnp.zeros_like(th)
    v_s = jnp.zeros_like(th)
    for s in range(1, steps + 1):
      _, g = val_and_grad(th)
      th, m_s, v_s = adam_update(
          th, m_s, v_s, g, jnp.asarray(float(s), dtype=jnp.float64)
      )
    fin = loss_fn(th)
    fin.block_until_ready()
    return float(fin)

  t0 = time.perf_counter()
  final_loss = _exec()
  compile_s = setup_s + (time.perf_counter() - t0)

  run_times: list[float] = []
  for _ in range(max(1, repeats)):
    t1 = time.perf_counter()
    final_loss = _exec()
    run_times.append(time.perf_counter() - t1)

  return {
      "init_loss": init_loss,
      "final_loss": final_loss,
      "compile_s": compile_s,
      "train_s": min(run_times),
  }


def _solve_cvxpylayers_train(
    case: TrainCase,
    *,
    tol: float,
    repeats: int,
    cvx_python: str | None,
) -> TrainResult:
  """Runs cvxpylayers training loop in-process or via worker Python."""
  payload = {
      "suite": case.suite,
      "n": case.n,
      "steps": case.steps,
      "lr": case.lr,
      "tol": tol,
      "repeats": repeats,
      "P": case.P.tolist(),
      "A": case.A.tolist() if case.A is not None else None,
      "b": case.b.tolist() if case.b is not None else None,
      "q_batch": case.q_batch.tolist(),
      "x_target": case.x_target.tolist(),
      "theta0": case.theta0.tolist(),
  }
  if cvx_python is None:
    raw = _run_cvxpylayers_train_worker(payload)
  else:
    proc = subprocess.run(
        [cvx_python, __file__, "--cvxpylayers-worker"],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=True,
    )
    raw = json.loads(proc.stdout)

  return TrainResult(
      suite=case.suite,
      problem_name=case.name,
      solver_name="cvxpylayers",
      batch_size=case.batch_size,
      steps=case.steps,
      init_loss=float(raw["init_loss"]),
      final_loss=float(raw["final_loss"]),
      compile_s=float(raw["compile_s"]),
      train_s=float(raw["train_s"]),
  )


def _detect_cvxpylayers_python(cli_value: str | None) -> str | None:
  if cli_value:
    return cli_value
  try:
    from cvxpylayers.jax import CvxpyLayer  # noqa: F401

    return None
  except ImportError:
    pass
  candidate = pathlib.Path("/tmp/cvxpylayers_env/bin/python")
  if candidate.exists():
    return str(candidate)
  return None


def format_markdown_table(results: list[TrainResult]) -> str:
  """Formats training loop benchmark results as Markdown."""
  lines = [
      (
          "| Suite | Problem | Solver | B | Init Loss | Final Loss "
          "| Compile (s) | 50-Step Train (ms) |"
      ),
      "| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: |",
  ]
  for r in results:
    lines.append(
        f"| {r.suite} | {r.problem_name} | {r.solver_name} "
        f"| {r.batch_size} | {r.init_loss:.5f} | {r.final_loss:.5f} "
        f"| {r.compile_s:.2f} | {r.train_s * 1000:.2f} |"
    )
  return "\n".join(lines)


def main() -> None:
  """CLI entry point for end-to-end differentiable training benchmarks."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--cvxpylayers-worker", action="store_true")
  parser.add_argument("--cvxpylayers-python", type=str, default=None)
  parser.add_argument("--tol", type=float, default=1e-6)
  parser.add_argument("--repeats", type=int, default=2)
  args = parser.parse_args()

  if args.cvxpylayers_worker:
    payload = json.loads(sys.stdin.read())
    print(json.dumps(_run_cvxpylayers_train_worker(payload)))
    return

  repo_root = pathlib.Path(__file__).resolve().parents[2]
  src_dir = repo_root / "src"
  if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
  os.environ.setdefault("JAX_ENABLE_X64", "1")

  import jax

  jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

  cvx_python = _detect_cvxpylayers_python(args.cvxpylayers_python)
  results: list[TrainResult] = []
  for case in build_cases():
    print(f"Running {case.name}...", file=sys.stderr, flush=True)
    results.append(
        _solve_proxax_train(
            case, mode="jit_scan", tol=args.tol, repeats=args.repeats
        )
    )
    results.append(
        _solve_cvxpylayers_train(
            case,
            tol=args.tol,
            repeats=args.repeats,
            cvx_python=cvx_python,
        )
    )

  print(format_markdown_table(results))


if __name__ == "__main__":
  main()
