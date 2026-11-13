"""CLI entrypoint for running Proxax benchmark suites."""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib

import jax

from benchmarks import harness
from benchmarks import suites
from proxax import solvers
from proxax.solvers import base


def _unconstrained_solvers(
    opt_tol: float,
) -> list[tuple[str, base.Solver]]:
  return [
      ("APD", solvers.APD(opt_tol=opt_tol)),
      ("AIPP", solvers.AIPP(opt_tol=opt_tol)),
      ("ACACG", solvers.ACACG(opt_tol=opt_tol)),
      ("AdapFISTA", solvers.AdapFISTA(opt_tol=opt_tol)),
      ("UPFAG", solvers.UPFAG(opt_tol=opt_tol)),
  ]


def _constrained_solvers(
    opt_tol: float,
    feas_tol: float,
) -> list[tuple[str, base.Solver]]:
  return [
      ("IALM", solvers.IALM(opt_tol=opt_tol, feas_tol=feas_tol)),
      ("IAIPAL", solvers.IAIPAL(opt_tol=opt_tol, feas_tol=feas_tol)),
      ("QPAPD", solvers.QPAPD(opt_tol=opt_tol, feas_tol=feas_tol)),
      ("QPAIPP", solvers.QPAIPP(opt_tol=opt_tol, feas_tol=feas_tol)),
      ("SProxALM", solvers.SProxALM(opt_tol=opt_tol, feas_tol=feas_tol)),
  ]


def _spectral_solvers(
    opt_tol: float,
) -> list[tuple[str, base.Solver]]:
  return [
      ("DAICG", solvers.DAICG(opt_tol=opt_tol)),
      ("IAICG", solvers.IAICG(opt_tol=opt_tol)),
  ]


def main(argv: list[str] | None = None) -> int:
  """Runs selected benchmark suites and prints formatted summaries."""
  parser = argparse.ArgumentParser(description="Run Proxax benchmarks.")
  parser.add_argument(
      "--suite",
      choices=("all", "unconstrained", "constrained", "spectral", "vmap"),
      default="all",
      help="Benchmark suite to execute.",
  )
  parser.add_argument("--opt-tol", type=float, default=1e-3)
  parser.add_argument("--feas-tol", type=float, default=1e-3)
  parser.add_argument("--json-out", type=pathlib.Path, default=None)
  args = parser.parse_args(argv)

  dtype_str = "float64" if jax.config.jax_enable_x64 else "float32"
  print(
      f"Proxax Benchmarks | backend={jax.default_backend()} | dtype={dtype_str}"
  )

  results: list[harness.BenchmarkResult] = []

  if args.suite in ("all", "unconstrained"):
    unc_probs = [
        suites.indefinite_box_qp(dim=100, m_ratio=0.25, seed=0),
        suites.sparse_nonconvex_margin(n_samples=120, n_features=40, seed=1),
    ]
    for bp in unc_probs:
      for s_name, solver in _unconstrained_solvers(args.opt_tol):
        results.append(
            harness.time_proxax_solve(
                suite=bp.suite,
                problem_name=bp.name,
                solver_name=s_name,
                prob=bp.prob,
                x0=bp.x0,
                solver=solver,
                opt_tol=args.opt_tol,
            )
        )
      print(f"\n### Optax Baseline ({bp.name})")
      for use_adam in (False, True):
        base_res = suites.run_optax_prox_baseline(bp, use_adam=use_adam)
        print(
            f"- {base_res['solver']}: obj={base_res['obj_val']:.4f}, "
            f"pg_res={base_res['pg_residual']:.2e}, "
            f"grad_evals={base_res['grad_evals']}, "
            f"run={1e3 * float(base_res['run_s']):.2f} ms"
        )

  if args.suite in ("all", "constrained"):
    con_probs = [
        suites.simplex_linear_qp(n_eq=4, dim=24, seed=777),
        suites.fantope_sparse_pca(dim=8, k_rank=2, seed=42),
    ]
    for bp in con_probs:
      for s_name, solver in _constrained_solvers(args.opt_tol, args.feas_tol):
        results.append(
            harness.time_proxax_solve(
                suite=bp.suite,
                problem_name=bp.name,
                solver_name=s_name,
                prob=bp.prob,
                x0=bp.x0,
                solver=solver,
                opt_tol=args.opt_tol,
                feas_tol=args.feas_tol,
            )
        )

  if args.suite in ("all", "spectral"):
    spec_probs = [
        suites.spectral_matrix_completion(rows=8, cols=8, rank=2, seed=13),
    ]
    for bp in spec_probs:
      for s_name, solver in _spectral_solvers(args.opt_tol):
        results.append(
            harness.time_proxax_solve(
                suite=bp.suite,
                problem_name=bp.name,
                solver_name=s_name,
                prob=bp.prob,
                x0=bp.x0,
                solver=solver,
                opt_tol=args.opt_tol,
            )
        )

  if results:
    print("\n## Benchmark Results\n")
    print(harness.format_markdown_table(results))
    print("\n## Dolan-More Profile (Grad Evals)\n")
    print(harness.dolan_more_summary(results, metric="grad_evals"))

  vmap_rows: list[dict[str, float | int]] = []
  if args.suite in ("all", "vmap"):
    vmap_rows = suites.run_vmap_throughput(opt_tol=args.opt_tol)
    print("\n## Batched `jax.vmap` Throughput\n")
    print("| Batch Size | Compile (s) | Run (ms) | Solves / sec |")
    print("| ---: | ---: | ---: | ---: |")
    for row in vmap_rows:
      print(
          f"| {row['batch_size']} | {row['compile_s']:.3f} "
          f"| {1e3 * float(row['run_s']):.2f} "
          f"| {row['throughput_per_s']:.1f} |"
      )

  if args.json_out is not None:
    payload = {
        "results": [dataclasses.asdict(r) for r in results],
        "vmap": vmap_rows,
    }
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n")

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
