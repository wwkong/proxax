"""Experiment: QPALM with QDLDL FFI, BCOO sparsity, and Ruiz equilibration.

Benchmarks 4 configurations on the 23 medium-scale Maros-Meszaros QPs:
  1. Clarabel (Rust IPM baseline)
  2. QPALM (dense, unscaled)
  3. QPALM (qdldl_ffi, BCOO, unscaled)
  4. QPALM (qdldl_ffi, BCOO, built-in Ruiz equilibration)
"""

from __future__ import annotations

import dataclasses
import json
import math
import pathlib
import sys

import jax

jax.config.update("jax_enable_x64", True)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from benchmarks import harness  # noqa: E402
from benchmarks.external import maros_meszaros as mm  # noqa: E402
from benchmarks.external import maros_meszaros_medium as mmm  # noqa: E402
from benchmarks.external import qpsif  # noqa: E402

from proxax.solvers.constrained import qpalm  # noqa: E402

RESULTS_DIR = pathlib.Path(__file__).resolve().parent / "results"
CACHE_DIR = pathlib.Path.home() / ".cache" / "proxax" / "maros_meszaros"


def _sgm(times_ms: list[float], shift_ms: float = 10.0) -> float:
  """Computes the shifted geometric mean in milliseconds."""
  if not times_ms:
    return 0.0
  logs = [math.log(max(t, 0.0) + shift_ms) for t in times_ms]
  return math.exp(sum(logs) / len(logs)) - shift_ms


def main() -> int:
  """Runs the QPALM QDLDL-FFI + Ruiz equilibration experiment."""
  opt_tol = 1e-6
  feas_tol = 1e-6
  time_limit = 10.0
  time_limit_ms = time_limit * 1000.0
  repeats = 3

  configs = [
      ("QPALM (dense)", False, "dense", False),
      ("QPALM (qdldl_ffi)", True, "qdldl_ffi", False),
      ("QPALM (qdldl_ffi+Ruiz)", True, "qdldl_ffi", True),
  ]

  all_results: list[dict[str, object]] = []

  for name in mmm.MEDIUM_PROBLEMS:
    sif_path = mm._fetch_sif(name, CACHE_DIR)
    qp = qpsif.load_qpsif_sparse(sif_path)
    n, m = qp.P.shape[0], qp.A.shape[0]
    prob_label = f"{qp.name} (n={n},m={m})"

    clarabel_res = mm.run_clarabel_baseline(
        qp, opt_tol=opt_tol, feas_tol=feas_tol
    )
    all_results.append(dataclasses.asdict(clarabel_res))

    prob_dense, x0_dense = mm.to_proxax_problem(qp, sparse=False)
    prob_sparse, x0_sparse = mm.to_proxax_problem(qp, sparse=True)

    for solver_name, use_sparse, linear_solver, equilibrate in configs:
      prob = prob_sparse if use_sparse else prob_dense
      x0 = x0_sparse if use_sparse else x0_dense
      solver = qpalm.QPALM(
          opt_tol=opt_tol,
          feas_tol=feas_tol,
          linear_solver=linear_solver,
          equilibrate=equilibrate,
      )
      res = harness.time_proxax_solve(
          suite="maros_meszaros_medium",
          problem_name=prob_label,
          solver_name=solver_name,
          prob=prob,
          x0=x0,
          solver=solver,
          opt_tol=opt_tol,
          feas_tol=None if prob.constraints is None else feas_tol,
          repeats=repeats,
          time_limit=time_limit,
      )
      all_results.append(dataclasses.asdict(res))
      print(
          f"{prob_label:<24s} | {solver_name:<22s} | "
          f"{res.status:<17s} | "
          f"it={res.outer_iters:>4d}({res.inner_iters:>4d}) | "
          f"opt={res.opt_residual:.2e} feas={res.feas_residual:.2e} | "
          f"{res.run_s * 1000:>7.2f} ms (compile {res.compile_s:.2f} s)",
          flush=True,
      )

  solver_order = [
      "Clarabel",
      "QPALM (dense)",
      "QPALM (qdldl_ffi)",
      "QPALM (qdldl_ffi+Ruiz)",
  ]
  summary: list[dict[str, object]] = []
  print("\n=== Summary (23 Maros-Meszaros Medium QPs) ===", flush=True)
  for sname in solver_order:
    rows = [r for r in all_results if r["solver_name"] == sname]
    solved = sum(1 for r in rows if r["converged"])
    run_ms = [float(r["run_s"]) * 1000.0 for r in rows]
    sgm_times = [
        float(r["run_s"]) * 1000.0 if r["converged"] else time_limit_ms
        for r in rows
    ]
    sgm10 = _sgm(sgm_times, shift_ms=10.0)
    avg_ms = sum(run_ms) / len(run_ms)
    max_ms = max(run_ms)
    avg_compile_s = sum(float(r["compile_s"]) for r in rows) / len(rows)
    entry = {
        "solver_name": sname,
        "solved": solved,
        "total": len(rows),
        "sgm10_ms": round(sgm10, 3),
        "avg_ms": round(avg_ms, 3),
        "max_ms": round(max_ms, 3),
        "avg_compile_s": round(avg_compile_s, 3),
    }
    summary.append(entry)
    print(
        f"{sname:<22s} | solved={solved:>2d}/{len(rows)} | "
        f"SGM(10)={sgm10:>7.2f} ms | avg={avg_ms:>7.2f} ms | "
        f"max={max_ms:>7.2f} ms | compile={avg_compile_s:.2f} s",
        flush=True,
    )

  RESULTS_DIR.mkdir(parents=True, exist_ok=True)
  out_path = RESULTS_DIR / "results.json"
  out_path.write_text(
      json.dumps({"summary": summary, "results": all_results}, indent=2) + "\n",
      encoding="utf-8",
  )
  print(f"Wrote {out_path}", flush=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
