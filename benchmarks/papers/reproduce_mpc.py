"""Single-command reproduction script for Proxax paper benchmarks and plots.

Runs the core CPU and GPU external benchmark suites, writes JSON artifacts to
``--json-dir``, and regenerates all Mocha-themed SVG plots in ``--plots-dir``.

Examples:
  # Preview all commands without executing
  python -m benchmarks.papers.reproduce_mpc --dry-run

  # Reproduce CPU suites and plots
  python -m benchmarks.papers.reproduce_mpc --suite cpu

  # Reproduce GPU suites and plots (requires CUDA-enabled jaxlib)
  python -m benchmarks.papers.reproduce_mpc --suite gpu
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_PLOTS_DIR = REPO_ROOT / "docs-experiments" / "_static" / "plots"
DEFAULT_JSON_DIR = pathlib.Path("/tmp/proxax_paper_results")


def _run(
    cmd: list[str],
    *,
    env_overrides: dict[str, str],
    dry_run: bool,
) -> None:
  """Runs a subprocess command with environment overrides."""
  env_str = " ".join(f"{k}={v}" for k, v in env_overrides.items())
  print(f"+ {env_str} {' '.join(cmd)}", flush=True)
  if dry_run:
    return
  env = os.environ.copy()
  env.update(env_overrides)
  subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def run_cpu_suites(
    python_exe: str,
    json_dir: pathlib.Path,
    plots_dir: pathlib.Path,
    *,
    skip_plots: bool,
    dry_run: bool,
) -> None:
  """Runs core CPU external benchmarks (`float64`) and generates SVG plots."""
  cpu_env = {"JAX_PLATFORMS": "cpu", "JAX_ENABLE_X64": "1"}
  suites: list[tuple[str, list[str], str | None]] = [
      (
          "conic_vs_clarabel_scs",
          ["--include-cvxpy"],
          "benchmarks.external.plot_conic_vs_clarabel_scs",
      ),
      (
          "minmax_convex_concave",
          [],
          "benchmarks.external.plot_minmax_convex_concave",
      ),
      (
          "minmax_nonconvex_concave",
          [],
          "benchmarks.external.plot_minmax_nonconvex_concave",
      ),
      (
          "spectral_vs_clarabel_scs",
          ["--include-cvxpy"],
          "benchmarks.external.plot_spectral_vs_clarabel_scs",
      ),
      (
          "block_vs_clarabel_scs",
          [],
          "benchmarks.external.plot_block_vs_clarabel_scs",
      ),
      (
          "batch_conic_minmax",
          [
              "--backend",
              "threads",
              "--batch-sizes",
              "1",
              "16",
              "64",
              "256",
              "1024",
          ],
          "benchmarks.external.plot_batch_conic_minmax",
      ),
      (
          "maros_meszaros",
          [
              "--opt-tol",
              "1e-6",
              "--feas-tol",
              "1e-6",
              "--solvers",
              "IALM",
              "QPALM",
              "--sparse",
              "--clarabel",
              "--time-limit",
              "10.0",
          ],
          None,
      ),
      (
          "maros_meszaros_medium",
          [
              "--solvers",
              "IALM",
              "QPALM",
              "--sparse",
              "--clarabel",
              "--time-limit",
              "10.0",
          ],
          None,
      ),
      (
          "maros_meszaros_large",
          [
              "--solvers",
              "IALM",
              "IAIPAL",
              "QPALM",
              "--clarabel",
              "--time-limit",
              "10.0",
          ],
          "benchmarks.external.plot_maros_meszaros_large",
      ),
  ]

  for name, extra_args, plot_mod in suites:
    json_out = json_dir / f"cpu_{name}.json"
    _run(
        [
            python_exe,
            "-m",
            f"benchmarks.external.{name}",
            *extra_args,
            "--json-out",
            str(json_out),
        ],
        env_overrides=cpu_env,
        dry_run=dry_run,
    )
    if not skip_plots and plot_mod is not None:
      _run(
          [
              python_exe,
              "-m",
              plot_mod,
              "--json-in",
              str(json_out),
              "--out-dir",
              str(plots_dir),
          ],
          env_overrides=cpu_env,
          dry_run=dry_run,
      )


def run_gpu_suites(
    python_exe: str,
    json_dir: pathlib.Path,
    plots_dir: pathlib.Path,
    *,
    skip_plots: bool,
    dry_run: bool,
) -> None:
  """Runs GPU external benchmarks (`vmap`, `Optax` GPU) and generates plots."""
  gpu_f64_env = {
      "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
      "JAX_ENABLE_X64": "1",
  }
  gpu_f32_env = {
      "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
      "JAX_ENABLE_X64": "0",
  }
  cpu_env = {"JAX_PLATFORMS": "cpu", "JAX_ENABLE_X64": "1"}

  gpu_batch_json = json_dir / "gpu_batch_vmap.json"
  cpu_batch_json = json_dir / "cpu_batch_conic_minmax.json"
  gpu_mm_json = json_dir / "gpu_minmax_nonconvex_concave.json"
  gpu_mm_f32_json = json_dir / "gpu_minmax_nonconvex_concave_f32.json"

  _run(
      [
          python_exe,
          "-m",
          "benchmarks.external.batch_conic_minmax",
          "--backend",
          "vmap",
          "--batch-sizes",
          "1",
          "16",
          "64",
          "256",
          "1024",
          "--json-out",
          str(gpu_batch_json),
      ],
      env_overrides=gpu_f64_env,
      dry_run=dry_run,
  )
  if not cpu_batch_json.exists() or dry_run:
    _run(
        [
            python_exe,
            "-m",
            "benchmarks.external.batch_conic_minmax",
            "--backend",
            "threads",
            "--batch-sizes",
            "1",
            "16",
            "64",
            "256",
            "1024",
            "--json-out",
            str(cpu_batch_json),
        ],
        env_overrides=cpu_env,
        dry_run=dry_run,
    )
  _run(
      [
          python_exe,
          "-m",
          "benchmarks.external.minmax_nonconvex_concave",
          "--json-out",
          str(gpu_mm_json),
      ],
      env_overrides=gpu_f64_env,
      dry_run=dry_run,
  )
  _run(
      [
          python_exe,
          "-m",
          "benchmarks.external.minmax_nonconvex_concave",
          "--opt-tol",
          "1e-4",
          "--time-limit",
          "5.0",
          "--json-out",
          str(gpu_mm_f32_json),
      ],
      env_overrides=gpu_f32_env,
      dry_run=dry_run,
  )

  if not skip_plots:
    _run(
        [
            python_exe,
            "-m",
            "benchmarks.external.plot_gpu_benchmarks",
            "--gpu-batch",
            str(gpu_batch_json),
            "--cpu-batch",
            str(cpu_batch_json),
            "--gpu-minmax",
            str(gpu_mm_json),
            "--out-dir",
            str(plots_dir),
        ],
        env_overrides=cpu_env,
        dry_run=dry_run,
    )


def main() -> int:
  """CLI entry point for paper benchmark reproduction."""
  parser = argparse.ArgumentParser(
      description="Reproduce Proxax paper benchmarks and SVG plots."
  )
  parser.add_argument(
      "--suite",
      choices=("all", "cpu", "gpu"),
      default="all",
      help="Which benchmark group to run (default: all).",
  )
  parser.add_argument("--json-dir", type=pathlib.Path, default=DEFAULT_JSON_DIR)
  parser.add_argument(
      "--plots-dir", type=pathlib.Path, default=DEFAULT_PLOTS_DIR
  )
  parser.add_argument(
      "--skip-plots",
      action="store_true",
      help="Skip SVG plot generation.",
  )
  parser.add_argument(
      "--dry-run",
      action="store_true",
      help="Print commands without executing them.",
  )
  args = parser.parse_args()

  if not args.dry_run:
    args.json_dir.mkdir(parents=True, exist_ok=True)
    args.plots_dir.mkdir(parents=True, exist_ok=True)

  python_exe = sys.executable
  if args.suite in ("all", "cpu"):
    run_cpu_suites(
        python_exe,
        args.json_dir,
        args.plots_dir,
        skip_plots=args.skip_plots,
        dry_run=args.dry_run,
    )
  if args.suite in ("all", "gpu"):
    run_gpu_suites(
        python_exe,
        args.json_dir,
        args.plots_dir,
        skip_plots=args.skip_plots,
        dry_run=args.dry_run,
    )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
