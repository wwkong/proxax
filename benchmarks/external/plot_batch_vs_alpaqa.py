"""Generate Mocha-themed SVG plots for batched Proxax vs. alpaqa benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLOTS_DIR = REPO_ROOT / "docs-experiments" / "_static" / "plots"

SOLVER_COLORS = {
    "alpaqa-PANOC(procs)": "#94e2d5",
    "alpaqa-ALM(procs)": "#94e2d5",
    "proxax-PANOC(vmap)": "#89b4fa",
    "proxax-PANOC(threads)": "#a6e3a1",
    "proxax-PANOC(threads,c=4)": "#fab387",
    "proxax-IALM(vmap)": "#cba6f7",
    "proxax-IALM(threads)": "#f9e2af",
    "proxax-IALM(threads,c=4)": "#fab387",
}
SOLVER_MARKERS = {
    "alpaqa-PANOC(procs)": "P",
    "alpaqa-ALM(procs)": "P",
    "proxax-PANOC(vmap)": "o",
    "proxax-PANOC(threads)": "s",
    "proxax-PANOC(threads,c=4)": "*",
    "proxax-IALM(vmap)": "^",
    "proxax-IALM(threads)": "D",
    "proxax-IALM(threads,c=4)": "*",
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

  batches = [1, 16, 64]
  panels = [
      (
          "Box-Constrained Nonconvex QP (n=100)",
          {
              "alpaqa-PANOC(procs)": [78.94, 456.67, 531.07],
              "proxax-PANOC(vmap)": [0.27, 6.69, 42.91],
              "proxax-PANOC(threads)": [0.52, 4.66, 17.51],
              "proxax-PANOC(threads,c=4)": [0.52, 3.37, 8.56],
          },
      ),
      (
          "Linearly-Constrained Nonconvex QP (n=50, m=25)",
          {
              "alpaqa-ALM(procs)": [3.33, 10.78, 27.91],
              "proxax-IALM(vmap)": [1.29, 110.24, 562.05],
              "proxax-IALM(threads)": [1.45, 7.74, 19.94],
              "proxax-IALM(threads,c=4)": [1.45, 37.60, 70.07],
          },
      ),
      (
          "Ball + Linear Nonconvex QP (n=50, m=25)",
          {
              "alpaqa-ALM(procs)": [4.02, 4.29, 11.84],
              "proxax-IALM(vmap)": [0.62, 11.12, 61.36],
              "proxax-IALM(threads)": [0.73, 5.21, 18.88],
              "proxax-IALM(threads,c=4)": [0.73, 5.20, 15.26],
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
      "Batched Nonconvex Solves: proxax vs. alpaqa (CPU float64, tol=1e-6)",
      fontsize=12.5,
      fontweight="bold",
  )
  out_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_path, format="svg", bbox_inches="tight")
  plt.close(fig)


def plot_dolan_more_profile(out_path: Path) -> None:
  """Generates a Mocha-themed Dolan-More runtime profile across all 9 cases."""
  _apply_mocha_style()
  fig, ax = plt.subplots(figsize=(7.6, 4.4), constrained_layout=True)

  # All 9 (suite, B) cases: (vmap_ms, threads_ms, chunk4_ms, alpaqa_ms)
  cases = [
      (0.27, 0.52, 0.52, 78.94),
      (6.69, 4.66, 3.37, 456.67),
      (42.91, 17.51, 8.56, 531.07),
      (1.29, 1.45, 1.45, 3.33),
      (110.24, 7.74, 37.60, 10.78),
      (562.05, 19.94, 70.07, 27.91),
      (0.62, 0.73, 0.73, 4.02),
      (11.12, 5.21, 5.20, 4.29),
      (61.36, 18.88, 15.26, 11.84),
  ]
  series_raw = {
      "proxax (best of vmap/threads/c=4)": [
          min(v, t, c4) for v, t, c4, _ in cases
      ],
      "proxax (threads, c=1)": [t for _, t, _, _ in cases],
      "proxax (threads, c=4)": [c4 for _, _, c4, _ in cases],
      "proxax (vmap)": [v for v, _, _, _ in cases],
      "alpaqa (procs)": [a for _, _, _, a in cases],
  }
  colors = {
      "proxax (best of vmap/threads/c=4)": "#a6e3a1",
      "proxax (threads, c=1)": "#f9e2af",
      "proxax (threads, c=4)": "#fab387",
      "proxax (vmap)": "#89b4fa",
      "alpaqa (procs)": "#94e2d5",
  }

  best = [min(v, t, c4, a) for v, t, c4, a in cases]
  tau_max = 64.0
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
      "Dolan-Moré Runtime Profile (9 Batched Nonconvex Cases, CPU float64)"
  )
  ax.grid(True, which="both", alpha=0.55)
  ax.legend(fontsize=9, loc="lower right")

  out_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_path, format="svg", bbox_inches="tight")
  plt.close(fig)


def main() -> None:
  """CLI entry point for generating batched Proxax vs alpaqa Mocha plots."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--out-dir",
      type=Path,
      default=DEFAULT_PLOTS_DIR,
      help="Directory to write SVG plots into.",
  )
  args = parser.parse_args()
  plot_scaling(args.out_dir / "batch_vs_alpaqa_scaling.svg")
  plot_dolan_more_profile(args.out_dir / "batch_vs_alpaqa_profile.svg")
  print(f"Wrote batched alpaqa Mocha SVG plots to {args.out_dir}")


if __name__ == "__main__":
  main()
