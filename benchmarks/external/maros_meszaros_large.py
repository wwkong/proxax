"""Large-scale Maros-Meszaros sparse QP benchmark runner (n = 1000..10197)."""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib

from benchmarks.external import maros_meszaros as mm
from benchmarks.external import qpsif

LARGE_PROBLEMS = (
    "CVXQP1_M",
    "CVXQP2_M",
    "CVXQP3_M",
    "MOSARQP1",
    "STCQP1",
    "STCQP2",
    "CONT-050",
    "CONT-100",
)


def main() -> int:
  """CLI entrypoint for the large-scale Maros-Meszaros sparse QP suite."""
  parser = argparse.ArgumentParser(
      description=(
          "Run Proxax sparse (BCOO) solvers on large-scale Maros-Meszaros QPs."
      )
  )
  parser.add_argument(
      "--problems",
      nargs="+",
      default=list(LARGE_PROBLEMS),
      help="Problem names (without .SIF extension).",
  )
  parser.add_argument(
      "--data-dir",
      type=pathlib.Path,
      default=None,
      help="Optional local directory containing .SIF files.",
  )
  parser.add_argument("--max-dim", type=int, default=20000)
  parser.add_argument("--limit", type=int, default=None)
  parser.add_argument("--opt-tol", type=float, default=1e-6)
  parser.add_argument("--feas-tol", type=float, default=1e-6)
  parser.add_argument("--time-limit", type=float, default=10.0)
  parser.add_argument(
      "--precondition",
      action="store_true",
      default=True,
      help="Apply Modified Ruiz Equilibration before solving (default: True).",
  )
  parser.add_argument(
      "--no-precondition",
      dest="precondition",
      action="store_false",
      help="Disable Modified Ruiz Equilibration.",
  )
  parser.add_argument(
      "--solvers",
      nargs="+",
      default=["IALM"],
      help="Proxax solver names to run.",
  )
  parser.add_argument(
      "--clarabel",
      action="store_true",
      help="Also run Clarabel as an IPM baseline reference.",
  )
  parser.add_argument("--json-out", type=pathlib.Path, default=None)
  args = parser.parse_args()

  results = mm.run_maros_meszaros(
      problems=tuple(args.problems),
      data_dir=args.data_dir,
      max_dim=args.max_dim,
      limit=args.limit,
      opt_tol=args.opt_tol,
      feas_tol=args.feas_tol,
      solver_names=tuple(args.solvers),
      time_limit=args.time_limit,
      precondition=args.precondition,
      sparse=True,
  )
  if args.clarabel:
    cache_dir = args.data_dir or (
        pathlib.Path.home() / ".cache" / "proxax" / "maros_meszaros"
    )
    for name in args.problems[: args.limit]:
      qp = qpsif.load_qpsif_sparse(mm._fetch_sif(name, cache_dir))
      if qp.P.shape[0] > args.max_dim:
        continue
      res = mm.run_clarabel_baseline(
          qp, opt_tol=args.opt_tol, feas_tol=args.feas_tol
      )
      results.append(res)
      print(
          f"{res.problem_name:<24s} | {res.solver_name:<9s} | "
          f"{res.status:<17s} | iters={res.outer_iters:>4d} | "
          f"opt={res.opt_residual:.2e} | feas={res.feas_residual:.2e} | "
          f"{res.run_s * 1000:>7.2f}ms",
          flush=True,
      )
  if args.json_out is not None:
    args.json_out.write_text(
        json.dumps([dataclasses.asdict(r) for r in results], indent=2),
        encoding="utf-8",
    )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
