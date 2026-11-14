"""Generate Mocha-themed SVG plots for differentiable training benchmarks."""

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
    "proxax(jit-scan)": "#a6e3a1",
    "proxax(threads)": "#89b4fa",
}
SOLVER_MARKERS = {
    "cvxpylayers": "P",
    "proxax(jit-scan)": "o",
    "proxax(threads)": "D",
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
  """Generates a 2-panel 50-step Adam training runtime comparison."""
  _apply_mocha_style()
  fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)

  batches = [16, 64]
  panels = [
      (
          "Batched QP Layer (n=50, m=25, 50 Adam Steps)",
          {
              "cvxpylayers": [3270.79, 10498.29],
              "proxax(jit-scan)": [1164.90, 5784.87],
              "proxax(threads)": [189.04, 691.31],
          },
      ),
      (
          "Batched Simplex Layer (n=100, 50 Adam Steps)",
          {
              "cvxpylayers": [3776.53, 12400.86],
              "proxax(jit-scan)": [250.13, 1196.43],
              "proxax(threads)": [168.18, 581.59],
          },
      ),
  ]

  for ax, (title, series) in zip(axes, panels, strict=True):
    for name, vals in series.items():
      ax.plot(
          batches,
          vals,
          marker=SOLVER_MARKERS[name],
          color=SOLVER_COLORS[name],
          linewidth=2.2,
          markersize=7.0,
          label=name,
      )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(batches)
    ax.set_xticklabels([str(b) for b in batches])
    ax.set_xlabel("Batch Size B")
    ax.set_ylabel("50-Step Training Time (ms)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.6)
    ax.legend(fontsize=9, loc="upper left")

  fig.suptitle(
      "End-to-End 50-Step Differentiable Training: proxax vs. cvxpylayers",
      fontsize=12.5,
      fontweight="bold",
  )
  out_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_path, format="svg", bbox_inches="tight")
  plt.close(fig)


def plot_dolan_more_profile(out_path: Path) -> None:
  """Generates a Mocha-themed Dolan-More profile across the 4 training cases."""
  _apply_mocha_style()
  fig, ax = plt.subplots(figsize=(7.6, 4.4), constrained_layout=True)

  # (proxax_threads_ms, proxax_jit_scan_ms, cvxpylayers_ms)
  cases = [
      (189.04, 1164.90, 3270.79),
      (168.18, 250.13, 3776.53),
      (691.31, 5784.87, 10498.29),
      (581.59, 1196.43, 12400.86),
  ]
  series_raw = {
      "proxax (threads)": [t for t, _, _ in cases],
      "proxax (jit-scan)": [p for _, p, _ in cases],
      "cvxpylayers (eager Python)": [c for _, _, c in cases],
  }
  colors = {
      "proxax (threads)": "#89b4fa",
      "proxax (jit-scan)": "#a6e3a1",
      "cvxpylayers (eager Python)": "#94e2d5",
  }

  best = [min(t, p, c) for t, p, c in cases]
  tau_max = 32.0
  n_probs = len(cases)

  for label, vals in series_raw.items():
    ratios = sorted(max(v / b, 1.0) for v, b in zip(vals, best, strict=True))
    xs = [1.0]
    ys = [sum(1 for r in ratios if r <= 1.0) / n_probs]
    for r in ratios:
      if r > xs[-1]:
        xs.append(r)
        ys.append(sum(1 for rv in ratios if rv <= r) / n_probs)
    if xs[-1] < tau_max:
      xs.append(tau_max)
      ys.append(ys[-1])
    ax.step(
        xs,
        ys,
        where="post",
        linewidth=2.2,
        color=colors[label],
        label=label,
    )

  ax.set_xscale("log", base=2)
  ax.set_xlim(1.0, tau_max)
  ax.set_ylim(0.0, 1.05)
  ax.set_xlabel(r"Performance Ratio $\tau$ (log$_2$ scale)")
  ax.set_ylabel("Fraction of Problems Solved")
  ax.set_title(
      "Dolan-Moré Runtime Profile (4 End-to-End Training Cases, CPU float64)"
  )
  ax.grid(True, which="both", alpha=0.55)
  ax.legend(fontsize=9, loc="lower right")

  out_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_path, format="svg", bbox_inches="tight")
  plt.close(fig)


def main() -> None:
  """CLI entry point for generating training loop Mocha plots."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--out-dir",
      type=Path,
      default=DEFAULT_PLOTS_DIR,
      help="Directory to write SVG plots into.",
  )
  args = parser.parse_args()
  plot_scaling(args.out_dir / "diff_training_loop_scaling.svg")
  plot_dolan_more_profile(args.out_dir / "diff_training_loop_profile.svg")
  print(f"Wrote differentiable training Mocha SVG plots to {args.out_dir}")


if __name__ == "__main__":
  main()
