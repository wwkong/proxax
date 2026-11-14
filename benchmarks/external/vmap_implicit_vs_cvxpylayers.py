"""Benchmark comparing Proxax (vmap + implicit_solve) against cvxpylayers.

Covers batched differentiable optimization (forward solve + backward VJP via
``jax.value_and_grad``) on CPU ``float64`` across two problem families:
  1. ``qp_layer``: min_{x >= 0} 0.5 x^T P x + q^T x s.t. A x <= b.
  2. ``simplex_layer``: min_{x in Delta^n} 0.5 x^T P x - mu^T x.

Because ``cvxpylayers`` is an optional external dependency, this script runs
``cvxpylayers`` in-process when importable or delegates to a separate Python
environment via ``--cvxpylayers-python`` (using ``--cvxpylayers-worker``).
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
class BatchedLayerCase:
  """Problem data for a batched differentiable optimization layer."""

  suite: str
  name: str
  n: int
  m: int
  batch_size: int
  P: np.ndarray
  P_sqrt: np.ndarray
  A: np.ndarray | None
  b: np.ndarray | None
  q_batch: np.ndarray


@dataclasses.dataclass(frozen=True)
class LayerBenchmarkResult:
  """Recorded timing and gradient-accuracy metrics for one batched run."""

  suite: str
  problem_name: str
  solver_name: str
  batch_size: int
  loss_val: float
  grad_norm: float
  rel_grad_err: float
  compile_s: float
  run_s: float


def _build_case(
    rng: np.random.Generator,
    *,
    suite: str,
    n: int,
    m: int,
    batch_size: int,
) -> BatchedLayerCase:
  """Builds a reproducible SPD QP or simplex layer batch."""
  rank = min(n, 15)
  F = rng.standard_normal((n, rank)) / np.sqrt(rank)
  P = F @ F.T + 0.5 * np.eye(n, dtype=np.float64)
  P = 0.5 * (P + P.T)
  eigvals, eigvecs = np.linalg.eigh(P)
  P_sqrt = (eigvecs * np.sqrt(np.maximum(eigvals, 1e-12))) @ eigvecs.T

  if m > 0:
    A = rng.standard_normal((m, n)) / np.sqrt(n)
    x_feas = np.full(n, 0.2, dtype=np.float64)
    b = A @ x_feas + 0.5
  else:
    A = None
    b = None

  q_batch = rng.standard_normal((batch_size, n))
  return BatchedLayerCase(
      suite=suite,
      name=f"{suite}_n{n}_B{batch_size}",
      n=n,
      m=m,
      batch_size=batch_size,
      P=P,
      P_sqrt=P_sqrt,
      A=A,
      b=b,
      q_batch=q_batch,
  )


def build_cases(seed: int = 42) -> list[BatchedLayerCase]:
  """Builds QP and simplex batched layer benchmarks."""
  rng = np.random.default_rng(seed)
  cases: list[BatchedLayerCase] = []
  for batch_size in (1, 16, 64, 256):
    cases.append(
        _build_case(rng, suite="qp_layer", n=50, m=25, batch_size=batch_size)
    )
  for batch_size in (1, 16, 64, 256):
    cases.append(
        _build_case(
            rng, suite="simplex_layer", n=100, m=0, batch_size=batch_size
        )
    )
  for batch_size in (1, 16, 64, 256):
    cases.append(
        _build_case(
            rng, suite="l2ball_layer", n=100, m=0, batch_size=batch_size
        )
    )
  return cases


def _run_cvxpylayers_solve(
    payload: dict[str, typing.Any],
) -> dict[str, typing.Any]:
  """Runs cvxpylayers.jax.CvxpyLayer forward+backward in-process."""
  import cvxpy as cp
  import jax
  import jax.numpy as jnp
  from cvxpylayers.jax import CvxpyLayer

  jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

  suite = str(payload["suite"])
  n = int(payload["n"])
  m = int(payload["m"])
  batch_size = int(payload["batch_size"])
  P_sqrt_np = np.asarray(payload["P_sqrt"], dtype=np.float64)
  q_batch_np = np.asarray(payload["q_batch"], dtype=np.float64)
  tol = float(payload["tol"])
  repeats = int(payload["repeats"])

  t0 = time.perf_counter()
  x = cp.Variable(n)
  P_sqrt_param = cp.Parameter((n, n))
  q_param = cp.Parameter(n)

  if suite == "qp_layer":
    A_np = np.asarray(payload["A"], dtype=np.float64)
    b_np = np.asarray(payload["b"], dtype=np.float64)
    A_param = cp.Parameter((m, n))
    b_param = cp.Parameter(m)
    prob = cp.Problem(
        cp.Minimize(0.5 * cp.sum_squares(P_sqrt_param @ x) + q_param @ x),
        [A_param @ x <= b_param, x >= 0],
    )
    layer = CvxpyLayer(
        prob,
        parameters=[P_sqrt_param, q_param, A_param, b_param],
        variables=[x],
    )
    P_b = jnp.broadcast_to(jnp.asarray(P_sqrt_np), (batch_size, n, n))
    A_b = jnp.broadcast_to(jnp.asarray(A_np), (batch_size, m, n))
    b_b = jnp.broadcast_to(jnp.asarray(b_np), (batch_size, m))
    q_b = jnp.asarray(q_batch_np)

    def loss_fn(q_in: typing.Any) -> typing.Any:
      (x_sol,) = layer(
          P_b,
          q_in,
          A_b,
          b_b,
          solver_args={"eps": tol, "eps_abs": tol, "eps_rel": tol},
      )
      return 0.5 * jnp.sum(x_sol**2)

  elif suite == "simplex_layer":
    prob = cp.Problem(
        cp.Minimize(0.5 * cp.sum_squares(P_sqrt_param @ x) - q_param @ x),
        [cp.sum(x) == 1, x >= 0],
    )
    layer = CvxpyLayer(
        prob,
        parameters=[P_sqrt_param, q_param],
        variables=[x],
    )
    P_b = jnp.broadcast_to(jnp.asarray(P_sqrt_np), (batch_size, n, n))
    q_b = jnp.asarray(q_batch_np)

    def loss_fn(q_in: typing.Any) -> typing.Any:
      (x_sol,) = layer(
          P_b,
          q_in,
          solver_args={"eps": tol, "eps_abs": tol, "eps_rel": tol},
      )
      return 0.5 * jnp.sum((x_sol - 0.5) ** 2)

  else:
    prob = cp.Problem(
        cp.Minimize(0.5 * cp.sum_squares(P_sqrt_param @ x) + q_param @ x),
        [cp.norm2(x) <= 1.0],
    )
    layer = CvxpyLayer(
        prob,
        parameters=[P_sqrt_param, q_param],
        variables=[x],
    )
    P_b = jnp.broadcast_to(jnp.asarray(P_sqrt_np), (batch_size, n, n))
    q_b = jnp.asarray(q_batch_np)

    def loss_fn(q_in: typing.Any) -> typing.Any:
      (x_sol,) = layer(
          P_b,
          q_in,
          solver_args={"eps": tol, "eps_abs": tol, "eps_rel": tol},
      )
      return 0.5 * jnp.sum((x_sol - 0.5) ** 2)

  val, grad = jax.value_and_grad(loss_fn)(q_b)
  grad.block_until_ready()
  compile_s = time.perf_counter() - t0

  run_times: list[float] = []
  for _ in range(max(1, repeats)):
    t1 = time.perf_counter()
    val, grad = jax.value_and_grad(loss_fn)(q_b)
    grad.block_until_ready()
    run_times.append(time.perf_counter() - t1)

  return {
      "loss_val": float(val),
      "grad": np.asarray(grad, dtype=np.float64).tolist(),
      "compile_s": compile_s,
      "run_s": min(run_times),
  }


def _solve_with_cvxpylayers(
    case: BatchedLayerCase,
    *,
    tol: float,
    repeats: int,
    cvxpylayers_python: str | None,
) -> tuple[LayerBenchmarkResult, np.ndarray]:
  """Runs cvxpylayers (in-process or worker) and returns result + gradient."""
  payload = {
      "suite": case.suite,
      "n": case.n,
      "m": case.m,
      "batch_size": case.batch_size,
      "P_sqrt": case.P_sqrt.tolist(),
      "A": case.A.tolist() if case.A is not None else None,
      "b": case.b.tolist() if case.b is not None else None,
      "q_batch": case.q_batch.tolist(),
      "tol": tol,
      "repeats": repeats,
  }
  if cvxpylayers_python is None:
    raw = _run_cvxpylayers_solve(payload)
  else:
    proc = subprocess.run(
        [cvxpylayers_python, __file__, "--cvxpylayers-worker"],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=True,
    )
    raw = json.loads(proc.stdout)

  grad_np = np.asarray(raw["grad"], dtype=np.float64)
  return (
      LayerBenchmarkResult(
          suite=case.suite,
          problem_name=case.name,
          solver_name="cvxpylayers",
          batch_size=case.batch_size,
          loss_val=float(raw["loss_val"]),
          grad_norm=float(np.linalg.norm(grad_np)),
          rel_grad_err=0.0,
          compile_s=float(raw["compile_s"]),
          run_s=float(raw["run_s"]),
      ),
      grad_np,
  )


def _solve_with_proxax(
    case: BatchedLayerCase,
    *,
    solver_name: str,
    solver: typing.Any,
    tol: float,
    repeats: int,
    ref_grad: np.ndarray,
) -> LayerBenchmarkResult:
  """Runs proxax vmap(implicit_solve) + jit(value_and_grad)."""
  import jax
  import jax.numpy as jnp

  from proxax import px

  P_j = jnp.asarray(case.P, dtype=jnp.float64)
  q_b = jnp.asarray(case.q_batch, dtype=jnp.float64)

  if case.suite == "qp_layer":
    assert case.A is not None and case.b is not None
    A_j = jnp.asarray(case.A, dtype=jnp.float64)
    b_j = jnp.asarray(case.b, dtype=jnp.float64)
    x0 = jnp.full(case.n, 0.2, dtype=jnp.float64)

    def solve_one(q_vec: typing.Any) -> typing.Any:
      def prob_fn(q_in: typing.Any) -> px.CompositeProblem:
        return px.CompositeProblem(
            smooth=lambda x: 0.5 * jnp.dot(x, P_j @ x) + jnp.dot(q_in, x),
            proximal=px.prox.NonNegative(),
            constraints=px.constraints.less_equal(A_j, b_j),
        )

      return px.implicit_solve(
          prob_fn,
          x0,
          q_vec,
          solver=solver,
          opt_tol=tol,
          feas_tol=tol,
          linear_tol=tol,
      )

  elif case.suite == "simplex_layer":
    x0 = jnp.full(case.n, 1.0 / case.n, dtype=jnp.float64)

    def solve_one(q_vec: typing.Any) -> typing.Any:
      def prob_fn(q_in: typing.Any) -> px.CompositeProblem:
        return px.CompositeProblem(
            smooth=lambda x: 0.5 * jnp.dot(x, P_j @ x) - jnp.dot(q_in, x),
            proximal=px.prox.Simplex(),
        )

      return px.implicit_solve(
          prob_fn,
          x0,
          q_vec,
          solver=solver,
          opt_tol=tol,
          linear_tol=tol,
      )

  else:
    x0 = jnp.zeros(case.n, dtype=jnp.float64)

    def solve_one(q_vec: typing.Any) -> typing.Any:
      def prob_fn(q_in: typing.Any) -> px.CompositeProblem:
        return px.CompositeProblem(
            smooth=lambda x: 0.5 * jnp.dot(x, P_j @ x) + jnp.dot(q_in, x),
            proximal=px.prox.L2Ball(radius=1.0),
        )

      return px.implicit_solve(
          prob_fn,
          x0,
          q_vec,
          solver=solver,
          opt_tol=tol,
          linear_tol=tol,
      )

  @jax.jit
  def loss_and_grad(qb: typing.Any) -> tuple[typing.Any, typing.Any]:
    def loss(q_in: typing.Any) -> typing.Any:
      xs = jax.vmap(solve_one)(q_in)
      if case.suite in ("simplex_layer", "l2ball_layer"):
        return 0.5 * jnp.sum((xs - 0.5) ** 2)
      return 0.5 * jnp.sum(xs**2)

    return jax.value_and_grad(loss)(qb)

  t0 = time.perf_counter()
  val, grad = loss_and_grad(q_b)
  grad.block_until_ready()
  compile_s = time.perf_counter() - t0

  run_times: list[float] = []
  for _ in range(max(1, repeats)):
    t1 = time.perf_counter()
    val, grad = loss_and_grad(q_b)
    grad.block_until_ready()
    run_times.append(time.perf_counter() - t1)

  grad_np = np.asarray(grad, dtype=np.float64)
  ref_norm = max(1e-12, float(np.linalg.norm(ref_grad)))
  rel_err = float(np.linalg.norm(grad_np - ref_grad) / ref_norm)
  return LayerBenchmarkResult(
      suite=case.suite,
      problem_name=case.name,
      solver_name=solver_name,
      batch_size=case.batch_size,
      loss_val=float(val),
      grad_norm=float(np.linalg.norm(grad_np)),
      rel_grad_err=rel_err,
      compile_s=compile_s,
      run_s=min(run_times),
  )


def _detect_cvxpylayers_python(cli_value: str | None) -> str | None:
  """Returns None if cvxpylayers imports natively, or path to worker."""
  if cli_value:
    return cli_value
  try:
    import cvxpylayers  # noqa: F401

    return None
  except ImportError:
    pass
  env_path = os.environ.get("CVXPYLAYERS_PYTHON")
  if env_path and pathlib.Path(env_path).exists():
    return env_path
  default_path = pathlib.Path("/tmp/cvxpylayers_env/bin/python")
  if default_path.exists():
    return str(default_path)
  raise RuntimeError(
      "cvxpylayers is not importable and no --cvxpylayers-python worker "
      "was found."
  )


def format_markdown_table(results: list[LayerBenchmarkResult]) -> str:
  """Formats layer benchmark results as a GitHub-flavored Markdown table."""
  header = (
      "| Suite | Problem | Solver | B | Loss | Rel Grad Err "
      "| Compile (s) | Run (ms) |"
  )
  sep = (
      "| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: |"
  )
  rows = [header, sep]
  for r in results:
    rows.append(
        f"| {r.suite} | {r.problem_name} | {r.solver_name} | {r.batch_size} "
        f"| {r.loss_val:.4f} | {r.rel_grad_err:.2e} | {r.compile_s:.2f} "
        f"| {1e3 * r.run_s:.2f} |"
    )
  return "\n".join(rows)


def run_benchmark(
    *,
    tol: float,
    repeats: int,
    cvxpylayers_python: str | None,
) -> list[LayerBenchmarkResult]:
  """Runs all batched differentiable layer benchmarks."""
  from proxax.solvers import IALM
  from proxax.solvers import PANOC

  results: list[LayerBenchmarkResult] = []
  for case in build_cases():
    print(f"Running {case.name}...", file=sys.stderr, flush=True)
    cvx_res, ref_grad = _solve_with_cvxpylayers(
        case,
        tol=tol,
        repeats=repeats,
        cvxpylayers_python=cvxpylayers_python,
    )
    print(
        f"  cvxpylayers: {1e3 * cvx_res.run_s:.2f} ms",
        file=sys.stderr,
        flush=True,
    )
    if case.suite == "qp_layer":
      solvers: list[tuple[str, typing.Any]] = [
          ("proxax-IALM", IALM(opt_tol=tol, feas_tol=tol)),
      ]
    else:
      solvers = [
          ("proxax-PANOC", PANOC(opt_tol=tol)),
      ]
    for s_name, solver in solvers:
      res = _solve_with_proxax(
          case,
          solver_name=s_name,
          solver=solver,
          tol=tol,
          repeats=repeats,
          ref_grad=ref_grad,
      )
      print(
          f"  {s_name}: {1e3 * res.run_s:.2f} ms "
          f"(compile {res.compile_s:.2f}s, rel_err={res.rel_grad_err:.2e})",
          file=sys.stderr,
          flush=True,
      )
      results.append(res)
    results.append(cvx_res)
  return results


def main() -> None:
  """CLI entry point for the Proxax vs cvxpylayers benchmark."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--cvxpylayers-worker",
      action="store_true",
      help=(
          "Internal mode: read JSON case on stdin, run cvxpylayers, print JSON."
      ),
  )
  parser.add_argument(
      "--cvxpylayers-python",
      type=str,
      default=None,
      help="Python executable with cvxpylayers installed.",
  )
  parser.add_argument(
      "--tol",
      type=float,
      default=1e-6,
      help="Forward and implicit-linear-solve tolerance.",
  )
  parser.add_argument(
      "--repeats",
      type=int,
      default=3,
      help="Number of steady-state timing repeats after warmup.",
  )
  args = parser.parse_args()

  if args.cvxpylayers_worker:
    payload = json.load(sys.stdin)
    out = _run_cvxpylayers_solve(payload)
    json.dump(out, sys.stdout)
    return

  import jax

  jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]
  cvx_python = _detect_cvxpylayers_python(args.cvxpylayers_python)
  results = run_benchmark(
      tol=args.tol,
      repeats=args.repeats,
      cvxpylayers_python=cvx_python,
  )
  print(format_markdown_table(results))


if __name__ == "__main__":
  main()
