"""Generate Mocha-themed SVG plots for closed-loop MPC rollout benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLOTS_DIR = REPO_ROOT / "docs-experiments" / "_static" / "plots"

SOLVER_COLORS = {
    "Clarabel": "#94e2d5",
    "alpaqa-PANOC(warm)": "#cba6f7",
    "proxax-PANOC(cold)": "#89b4fa",
    "proxax-PANOC(warm)": "#a6e3a1",
    "proxax-IAIPAL(cold)": "#f9e2af",
    "proxax-IAIPAL(warm)": "#a6e3a1",
}
SOLVER_MARKERS = {
    "Clarabel": "P",
    "alpaqa-PANOC(warm)": "^",
    "proxax-PANOC(cold)": "o",
    "proxax-PANOC(warm)": "s",
    "proxax-IAIPAL(cold)": "D",
    "proxax-IAIPAL(warm)": "s",
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
  """Generates a 2-panel horizon/batch MPC rollout runtime plot (CPU f64)."""
  _apply_mocha_style()
  fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), constrained_layout=True)

  labels = ["N=10, B=1", "N=20, B=1", "N=10, B=16", "N=20, B=16"]
  xs = list(range(len(labels)))

  panels = [
      (
          "Box-Constrained MPC (20-Step Rollout)",
          {
              "Clarabel": [5.75, 22.35, 56.11, 146.30],
              "proxax-PANOC(cold)": [0.36, 1.30, 7.55, 41.58],
              "proxax-PANOC(warm)": [0.26, 1.04, 6.39, 30.59],
          },
          ([0, 1], [1.66, 3.25]),
      ),
      (
          "State + Input Constrained MPC (20-Step Rollout)",
          {
              "Clarabel": [39.97, 207.44, 215.75, 825.60],
              "proxax-IAIPAL(cold)": [10.76, 70.44, 267.95, 1805.12],
              "proxax-IAIPAL(warm)": [7.88, 54.98, 212.49, 1477.95],
          },
          None,
      ),
  ]

  for ax, (title, series, alpaqa_pts) in zip(axes, panels, strict=True):
    for name, vals in series.items():
      ax.plot(
          xs,
          vals,
          marker=SOLVER_MARKERS[name],
          color=SOLVER_COLORS[name],
          linewidth=2.0,
          markersize=6.5,
          label=name,
      )
    if alpaqa_pts is not None:
      ax.plot(
          alpaqa_pts[0],
          alpaqa_pts[1],
          marker=SOLVER_MARKERS["alpaqa-PANOC(warm)"],
          color=SOLVER_COLORS["alpaqa-PANOC(warm)"],
          linewidth=2.0,
          markersize=6.5,
          label="alpaqa-PANOC(warm) (B=1)",
      )
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("20-Step Closed-Loop Runtime (ms)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.6)
    ax.legend(fontsize=8.5, loc="upper left")

  fig.suptitle(
      "Closed-Loop MPC Rollouts: Compiled lax.scan vs. Python Loop (CPU f64)",
      fontsize=12.5,
      fontweight="bold",
  )
  out_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_path, format="svg", bbox_inches="tight")
  plt.close(fig)


def plot_dolan_more_profile(out_path: Path) -> None:
  """Generates a Mocha-themed Dolan-More profile across all 8 MPC cases."""
  _apply_mocha_style()
  fig, ax = plt.subplots(figsize=(7.6, 4.4), constrained_layout=True)

  # All 8 (suite, N, B) cases: (proxax_cold_ms, proxax_warm_ms, clarabel_ms)
  cases = [
      (0.36, 0.26, 5.75),
      (1.30, 1.04, 22.35),
      (7.55, 6.39, 56.11),
      (41.58, 30.59, 146.30),
      (10.76, 7.88, 39.97),
      (70.44, 54.98, 207.44),
      (267.95, 212.49, 215.75),
      (1805.12, 1477.95, 825.60),
  ]
  series_raw = {
      "proxax (warm lax.scan)": [w for _, w, _ in cases],
      "proxax (cold lax.scan)": [c for c, _, _ in cases],
      "Clarabel (Python loop)": [cl for _, _, cl in cases],
  }
  colors = {
      "proxax (warm lax.scan)": "#a6e3a1",
      "proxax (cold lax.scan)": "#89b4fa",
      "Clarabel (Python loop)": "#94e2d5",
  }

  best = [min(c, w, cl) for c, w, cl in cases]
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
      "Dolan-Moré Runtime Profile (8 Closed-Loop MPC Cases, CPU float64)"
  )
  ax.grid(True, which="both", alpha=0.55)
  ax.legend(fontsize=9, loc="lower right")

  out_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_path, format="svg", bbox_inches="tight")
  plt.close(fig)


def main() -> None:
  """CLI entry point for generating closed-loop MPC rollout Mocha plots."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--out-dir",
      type=Path,
      default=DEFAULT_PLOTS_DIR,
      help="Directory to write SVG plots into.",
  )
  args = parser.parse_args()
  plot_scaling(args.out_dir / "mpc_rollout_scaling.svg")
  plot_dolan_more_profile(args.out_dir / "mpc_rollout_profile.svg")
  print(f"Wrote MPC rollout Mocha SVG plots to {args.out_dir}")


if __name__ == "__main__":
  main()
