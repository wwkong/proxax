"""Generate Mocha-themed SVG plots for Batched Conic & MinMax benchmarks."""

from __future__ import annotations

import argparse
import json
import typing
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLOTS_DIR = REPO_ROOT / "docs-experiments" / "_static" / "plots"

SOLVER_COLORS = {
    "proxax-IALM (threads)": "#a6e3a1",
    "proxax-PDHG (threads)": "#89b4fa",
    "Clarabel (threads)": "#94e2d5",
    "SCS (threads)": "#f9e2af",
}
SOLVER_MARKERS = {
    "proxax-IALM (threads)": "o",
    "proxax-PDHG (threads)": "s",
    "Clarabel (threads)": "P",
    "SCS (threads)": "D",
}


def _apply_mocha_style() -> None:
  plt.rcParams.update({
      "figure.facecolor": "#1e1e2e",
      "axes.facecolor": "#181825",
      "axes.edgecolor": "#45475a",
      "axes.labelcolor": "#cdd6f4",
      "axes.titlecolor": "#cdd6f4",
      "text.color": "#cdd6f4",
      "xtick.color": "#bac2de",
      "ytick.color": "#bac2de",
      "grid.color": "#313244",
      "grid.linestyle": "--",
      "grid.linewidth": 0.7,
      "legend.facecolor": "#181825",
      "legend.edgecolor": "#45475a",
      "font.size": 10,
  })


def plot_batch_conic_minmax(
    rows: list[dict[str, typing.Any]], out_path: Path
) -> None:
  """Generates a 2-panel batch-size scaling plot for SOCP and Saddle QP."""
  _apply_mocha_style()
  fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)

  by_key = {(r["problem_name"], r["solver_name"]): r for r in rows}
  batches = [1, 16, 64]
  panels = [
      (
          "Batched SOC Portfolio (n=64)",
          [f"Batched SOCP (n=64, B={b})" for b in batches],
          ["proxax-IALM (threads)", "Clarabel (threads)", "SCS (threads)"],
      ),
      (
          "Batched Bilinear Saddle QP (n=64)",
          [f"Batched Saddle QP (n=64, B={b})" for b in batches],
          ["proxax-PDHG (threads)", "Clarabel (threads)", "SCS (threads)"],
      ),
  ]

  for ax, (title, pnames, solvers) in zip(axes, panels, strict=True):
    for s in solvers:
      xs = []
      ys = []
      for b, pname in zip(batches, pnames, strict=True):
        row = by_key.get((pname, s))
        if row is not None and row["converged"]:
          xs.append(b)
          ys.append(1e3 * float(row["run_s"]))
      if xs:
        ax.plot(
            xs,
            ys,
            label=s,
            color=SOLVER_COLORS[s],
            marker=SOLVER_MARKERS[s],
            linewidth=2.0,
            markersize=6.5,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(batches)
    ax.set_xticklabels([f"B={b}" for b in batches])
    ax.minorticks_off()
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Batch Size B")
    ax.set_ylabel("Wall Time (ms)")
    ax.grid(True, which="major")
    ax.legend(loc="upper left", frameon=True)

  fig.suptitle(
      "Batched Conic & MinMax Solves (CPU threads, float64, tol = 1e-4)",
      fontweight="bold",
      fontsize=12,
  )
  out_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_path, format="svg", bbox_inches="tight")
  plt.close(fig)


def main() -> None:
  """CLI entry point for batched conic & minmax plots."""
  parser = argparse.ArgumentParser()
  parser.add_argument("--json-in", type=Path, required=True)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_PLOTS_DIR)
  args = parser.parse_args()

  rows = json.loads(args.json_in.read_text(encoding="utf-8"))
  plot_batch_conic_minmax(rows, args.out_dir / "batch_conic_minmax.svg")


if __name__ == "__main__":
  main()
