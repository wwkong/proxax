"""Generate ggplot SVG plots for Batched Conic & MinMax benchmarks."""

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
    "proxax-IALM (threads)": "#009E73",
    "proxax-PDHG (threads)": "#0072B2",
    "Clarabel (threads)": "#1B9E77",
    "SCS (threads)": "#B8860B",
}
SOLVER_MARKERS = {
    "proxax-IALM (threads)": "o",
    "proxax-PDHG (threads)": "s",
    "Clarabel (threads)": "P",
    "SCS (threads)": "D",
}


def _apply_mocha_style() -> None:
  plt.style.use("ggplot")
  plt.rcParams.update({
      "figure.facecolor": "#ffffff",
      "savefig.facecolor": "#ffffff",
      "savefig.edgecolor": "#ffffff",
      "axes.facecolor": "#ebebeb",
      "axes.edgecolor": "#ffffff",
      "axes.labelcolor": "#222222",
      "axes.titlecolor": "#111111",
      "text.color": "#222222",
      "xtick.color": "#444444",
      "ytick.color": "#444444",
      "grid.color": "#ffffff",
      "grid.linestyle": "-",
      "grid.linewidth": 0.9,
      "legend.facecolor": "#ffffff",
      "legend.edgecolor": "#cccccc",
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
