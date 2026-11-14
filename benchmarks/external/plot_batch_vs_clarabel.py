"""Generate Mocha-themed SVG plots for Proxax vs. Clarabel batch benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLOTS_DIR = REPO_ROOT / "docs-experiments" / "_static" / "plots"

SOLVER_COLORS = {
    "Clarabel(threads)": "#94e2d5",
    "proxax-IAIPAL(vmap)": "#89b4fa",
    "proxax-IAIPAL(threads)": "#a6e3a1",
    "proxax-IALM(threads)": "#fab387",
    "proxax-PANOC(vmap)": "#cba6f7",
    "proxax-PANOC(threads)": "#f9e2af",
}
SOLVER_MARKERS = {
    "Clarabel(threads)": "P",
    "proxax-IAIPAL(vmap)": "o",
    "proxax-IAIPAL(threads)": "s",
    "proxax-IALM(threads)": "X",
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
  """Generates a 2-panel log-log batch scaling plot (CPU float64, tol=1e-6)."""
  _apply_mocha_style()
  fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)

  batches = [1, 16, 64]
  panels = [
      (
          "Factor-Model QP (n=500, r=20, m=20)",
          {
              "Clarabel(threads)": [135.51, 634.22, 1863.06],
              "proxax-IAIPAL(vmap)": [1.13, 17.78, 87.33],
              "proxax-IAIPAL(threads)": [1.45, 6.41, 25.49],
              "proxax-IALM(threads)": [2.62, 7.94, 25.79],
          },
      ),
      (
          "Simplex Logistic Regression (N=200, n=200)",
          {
              "Clarabel(threads)": [113.58, 2835.77, 10959.67],
              "proxax-PANOC(vmap)": [4.54, 85.93, 413.27],
              "proxax-PANOC(threads)": [3.80, 12.05, 32.48],
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
          linewidth=2.0,
          markersize=6.5,
          label=name,
      )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(batches)
    ax.set_xticklabels([str(b) for b in batches])
    ax.set_xlabel("Batch Size B")
    ax.set_ylabel("Steady-State Runtime (ms)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.6)
    ax.legend(fontsize=8.5, loc="upper left")

  fig.suptitle(
      "Batched Convex Solves: proxax vs. Clarabel (CPU float64, tol=1e-6)",
      fontsize=12.5,
      fontweight="bold",
  )
  out_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_path, format="svg", bbox_inches="tight")
  plt.close(fig)


def plot_dolan_more_profile(out_path: Path) -> None:
  """Generates a Mocha-themed Dolan-More runtime profile across all 6 cases."""
  _apply_mocha_style()
  fig, ax = plt.subplots(figsize=(7.6, 4.4), constrained_layout=True)

  # All 6 (suite, B) cases: (vmap_ms, threads_ms, clarabel_ms)
  cases = [
      (1.13, 1.45, 135.51),
      (17.78, 6.41, 634.22),
      (87.33, 25.49, 1863.06),
      (4.54, 3.80, 113.58),
      (85.93, 12.05, 2835.77),
      (413.27, 32.48, 10959.67),
  ]
  series_raw = {
      "proxax (auto: min of vmap/threads)": [min(v, t) for v, t, _ in cases],
      "proxax (threads)": [t for _, t, _ in cases],
      "proxax (vmap)": [v for v, _, _ in cases],
      "Clarabel (threads)": [c for _, _, c in cases],
  }
  colors = {
      "proxax (auto: min of vmap/threads)": "#a6e3a1",
      "proxax (threads)": "#f9e2af",
      "proxax (vmap)": "#89b4fa",
      "Clarabel (threads)": "#94e2d5",
  }

  best = [min(v, t, c) for v, t, c in cases]
  tau_max = 512.0
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
      "Dolan-Moré Runtime Profile (6 Batched Convex Cases, CPU float64)"
  )
  ax.grid(True, which="both", alpha=0.55)
  ax.legend(fontsize=9, loc="lower right")

  out_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_path, format="svg", bbox_inches="tight")
  plt.close(fig)


def main() -> None:
  """CLI entry point for generating batched Proxax vs Clarabel Mocha plots."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--out-dir",
      type=Path,
      default=DEFAULT_PLOTS_DIR,
      help="Directory to write SVG plots into.",
  )
  args = parser.parse_args()
  plot_scaling(args.out_dir / "batch_vs_clarabel_scaling.svg")
  plot_dolan_more_profile(args.out_dir / "batch_vs_clarabel_profile.svg")
  print(f"Wrote batched Clarabel Mocha SVG plots to {args.out_dir}")


if __name__ == "__main__":
  main()
