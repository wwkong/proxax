"""Generate Mocha-themed SVG plots for Proxax vs. alpaqa benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLOTS_DIR = REPO_ROOT / "docs-experiments" / "_static" / "plots"

SOLVER_COLORS = {
    "alpaqa-PANOC": "#94e2d5",
    "alpaqa-ALM(PANOC)": "#94e2d5",
    "proxax-PANOC": "#89b4fa",
    "proxax-IALM(apd)": "#89b4fa",
    "proxax-IALM(panoc)": "#cba6f7",
    "proxax-IAIPAL": "#a6e3a1",
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


def plot_runtime(out_path: Path) -> None:
  """Generates a 3-panel grouped runtime bar chart (tol=1e-6, CPU float64)."""
  _apply_mocha_style()
  fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), constrained_layout=True)

  panels = [
      (
          "Box Nonconvex QPs (box_qp)",
          ["n=50", "n=200", "n=500"],
          {
              "alpaqa-PANOC": [0.58, 0.69, 5.64],
              "proxax-PANOC": [0.53, 1.21, 10.31],
          },
      ),
      (
          "Linearly Constrained QPs (lin_qp)",
          ["n=50, m=25", "n=200, m=100"],
          {
              "alpaqa-ALM(PANOC)": [3.72, 48.67],
              "proxax-IALM(apd)": [3.15, 36.89],
              "proxax-IALM(panoc)": [6.02, 49.06],
              "proxax-IAIPAL": [2.85, 333.44],
          },
      ),
      (
          "l2-Ball + Linear QPs (nl_ball_qp)",
          ["n=50, m=25", "n=200, m=100"],
          {
              "alpaqa-ALM(PANOC)": [1.68, 9.17],
              "proxax-IALM(apd)": [2.79, 7.04],
              "proxax-IALM(panoc)": [4.69, 9.87],
              "proxax-IAIPAL": [0.77, 2.18],
          },
      ),
  ]

  for ax, (title, xlabels, series) in zip(axes, panels, strict=True):
    x = np.arange(len(xlabels), dtype=float)
    n_solvers = len(series)
    width = 0.72 / n_solvers
    for idx, (solver, vals) in enumerate(series.items()):
      offset = (idx - 0.5 * (n_solvers - 1)) * width
      bars = ax.bar(
          x + offset,
          vals,
          width=width * 0.92,
          color=SOLVER_COLORS[solver],
          label=solver,
          zorder=3,
      )
      for bar, v in zip(bars, vals, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            v * 1.08,
            f"{v:.2f}" if v < 10 else f"{v:.1f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#cdd6f4",
        )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_title(title, fontsize=10.5, pad=8)
    ax.set_ylabel("Runtime (ms, log scale)")
    ax.grid(True, axis="y", alpha=0.6, zorder=0)
    ax.legend(fontsize=8, loc="upper left")
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.8)

  fig.suptitle(
      "Proxax vs. alpaqa Runtime (CPU float64, tol = 1e-6)",
      fontsize=12,
      y=1.03,
  )
  out_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_path, format="svg", bbox_inches="tight")
  plt.close(fig)


def plot_speedup(out_path: Path) -> None:
  """Generates a horizontal speedup chart (alpaqa / best proxax)."""
  _apply_mocha_style()
  fig, ax = plt.subplots(figsize=(10.5, 4.4), constrained_layout=True)

  labels = [
      "box_qp_n50 (PANOC)",
      "box_qp_n200 (PANOC)",
      "box_qp_n500 (PANOC)",
      "lin_qp_n50_m25 (IAIPAL)",
      "lin_qp_n200_m100 (IALM-apd)",
      "nl_ball_qp_n50_m25 (IAIPAL)",
      "nl_ball_qp_n200_m100 (IAIPAL)",
  ]
  speedups = [
      0.58 / 0.53,
      0.69 / 1.21,
      5.64 / 10.31,
      3.72 / 2.85,
      48.67 / 36.89,
      1.68 / 0.77,
      9.17 / 2.18,
  ]
  colors = ["#a6e3a1" if s >= 1.0 else "#f38ba8" for s in speedups]

  y = np.arange(len(labels))
  bars = ax.barh(y, speedups, color=colors, height=0.6, zorder=3)
  ax.axvline(1.0, color="#f9e2af", linestyle="--", linewidth=1.2, zorder=4)

  for bar, s in zip(bars, speedups, strict=True):
    ax.text(
        s + 0.08,
        bar.get_y() + bar.get_height() / 2.0,
        f"{s:.2f}x",
        va="center",
        ha="left",
        fontsize=9,
        color="#cdd6f4",
    )

  ax.set_yticks(y)
  ax.set_yticklabels(labels)
  ax.invert_yaxis()
  ax.set_xlabel(
      "Speedup Ratio (alpaqa time / best proxax time, >1 = proxax faster)"
  )
  ax.set_title(
      "Best Proxax Solver Speedup vs. alpaqa (CPU float64, tol = 1e-6)"
  )
  ax.set_xlim(0, max(speedups) + 0.7)
  ax.grid(True, axis="x", alpha=0.6, zorder=0)

  out_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_path, format="svg", bbox_inches="tight")
  plt.close(fig)


def main() -> None:
  """CLI entry point for generating Proxax vs alpaqa Mocha plots."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--out-dir",
      type=Path,
      default=DEFAULT_PLOTS_DIR,
      help="Directory where SVG plots are written.",
  )
  args = parser.parse_args()
  plot_runtime(args.out_dir / "panoc_vs_alpaqa_runtime.svg")
  plot_speedup(args.out_dir / "panoc_vs_alpaqa_speedup.svg")


if __name__ == "__main__":
  main()
