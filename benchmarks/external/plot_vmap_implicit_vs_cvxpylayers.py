"""Generate Mocha-themed SVG plots for Proxax vs. cvxpylayers benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLOTS_DIR = REPO_ROOT / "docs-experiments" / "_static" / "plots"

SOLVER_COLORS = {
    "cvxpylayers": "#94e2d5",
    "proxax-IALM(vmap)": "#89b4fa",
    "proxax-IALM(threads)": "#a6e3a1",
    "proxax-PANOC(vmap)": "#cba6f7",
    "proxax-PANOC(threads)": "#f9e2af",
}
SOLVER_MARKERS = {
    "cvxpylayers": "P",
    "proxax-IALM(vmap)": "o",
    "proxax-IALM(threads)": "s",
    "proxax-PANOC(vmap)": "^",
    "proxax-PANOC(threads)": "D",
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


def plot_scaling(out_path: Path) -> None:
  """Generates a 3-panel log-log batch scaling plot (CPU float64, tol=1e-6)."""
  _apply_mocha_style()
  fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5), constrained_layout=True)

  batches = [1, 16, 64, 256]
  panels = [
      (
          "Batched QP Layer (n=50, m=25)",
          {
              "cvxpylayers": [14.05, 70.81, 227.66, 806.73],
              "proxax-IALM(vmap)": [0.96, 37.49, 252.14, 1108.99],
              "proxax-IALM(threads)": [1.64, 9.01, 42.93, 156.20],
          },
      ),
      (
          "Batched Simplex Layer (n=100)",
          {
              "cvxpylayers": [26.29, 114.52, 344.08, 1566.05],
              "proxax-PANOC(vmap)": [0.80, 8.47, 37.84, 193.69],
              "proxax-PANOC(threads)": [1.12, 7.80, 26.34, 88.86],
          },
      ),
      (
          "Batched l2-Ball Layer (n=100)",
          {
              "cvxpylayers": [22.92, 127.11, 349.87, 1574.06],
              "proxax-PANOC(vmap)": [0.29, 2.50, 10.46, 44.52],
              "proxax-PANOC(threads)": [0.69, 6.29, 26.27, 87.87],
          },
      ),
  ]

  for ax, (title, series) in zip(axes, panels, strict=True):
    for solver, vals in series.items():
      ax.plot(
          batches,
          vals,
          marker=SOLVER_MARKERS[solver],
          color=SOLVER_COLORS[solver],
          linewidth=2.2,
          markersize=7,
          label=solver,
      )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(batches)
    ax.set_xticklabels([str(b) for b in batches])
    ax.set_title(title, fontsize=10.5, pad=8)
    ax.set_xlabel("Batch Size B")
    ax.set_ylabel("Steady-State Runtime (ms, log scale)")
    ax.grid(True, which="both", alpha=0.55)
    ax.legend(fontsize=9, loc="upper left")

  fig.suptitle(
      "Batched Differentiable Layers: Proxax (vmap + implicit_solve) vs. "
      "cvxpylayers (CPU float64)",
      fontsize=11.5,
      y=1.03,
  )
  out_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_path, format="svg", bbox_inches="tight")
  plt.close(fig)


def main() -> None:
  """CLI entry point for generating Proxax vs cvxpylayers Mocha plots."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--out-dir",
      type=Path,
      default=DEFAULT_PLOTS_DIR,
      help="Directory where SVG plots are written.",
  )
  args = parser.parse_args()
  plot_scaling(args.out_dir / "vmap_implicit_vs_cvxpylayers_scaling.svg")


if __name__ == "__main__":
  main()
