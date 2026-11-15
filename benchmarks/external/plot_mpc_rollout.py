"""Generate ggplot SVG plots for closed-loop MPC rollout benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLOTS_DIR = REPO_ROOT / "docs-experiments" / "_static" / "plots"

SOLVER_COLORS = {
    "Clarabel": "#1B9E77",
    "alpaqa-PANOC(warm)": "#88419D",
    "proxax-PANOC(cold)": "#0072B2",
    "proxax-PANOC(warm)": "#009E73",
    "proxax-PANOC(warm,threads)": "#E69F00",
    "proxax-IAIPAL(cold)": "#B8860B",
    "proxax-IAIPAL(warm)": "#009E73",
    "proxax-IAIPAL(warm,threads)": "#E69F00",
}
SOLVER_MARKERS = {
    "Clarabel": "P",
    "alpaqa-PANOC(warm)": "^",
    "proxax-PANOC(cold)": "o",
    "proxax-PANOC(warm)": "s",
    "proxax-PANOC(warm,threads)": "*",
    "proxax-IAIPAL(cold)": "D",
    "proxax-IAIPAL(warm)": "s",
    "proxax-IAIPAL(warm,threads)": "*",
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
              "Clarabel": [5.74, 22.56, 52.09, 141.94],
              "proxax-PANOC(cold)": [0.37, 1.38, 7.50, 43.26],
              "proxax-PANOC(warm)": [0.25, 1.01, 6.45, 31.26],
              "proxax-PANOC(warm,threads)": [0.25, 1.01, 1.28, 4.25],
          },
          ([0, 1], [1.87, 3.33]),
      ),
      (
          "State + Input Constrained MPC (20-Step Rollout)",
          {
              "Clarabel": [39.54, 202.46, 220.65, 767.58],
              "proxax-IAIPAL(cold)": [10.57, 70.61, 268.73, 1754.45],
              "proxax-IAIPAL(warm)": [7.78, 53.67, 212.13, 1420.66],
              "proxax-IAIPAL(warm,threads)": [7.78, 53.67, 41.43, 217.94],
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
  """Generates a ggplot Dolan-More profile across all 8 MPC cases."""
  _apply_mocha_style()
  fig, ax = plt.subplots(figsize=(7.6, 4.4), constrained_layout=True)

  # All 8 (suite, N, B) cases: (cold_ms, warm_vmap_ms, warm_threads_ms, clar_ms)
  cases = [
      (0.37, 0.25, 0.25, 5.74),
      (1.38, 1.01, 1.01, 22.56),
      (7.50, 6.45, 1.28, 52.09),
      (43.26, 31.26, 4.25, 141.94),
      (10.57, 7.78, 7.78, 39.54),
      (70.61, 53.67, 53.67, 202.46),
      (268.73, 212.13, 41.43, 220.65),
      (1754.45, 1420.66, 217.94, 767.58),
  ]
  series_raw = {
      "proxax (warm, threads for B>1)": [wt for _, _, wt, _ in cases],
      "proxax (warm, vmap)": [w for _, w, _, _ in cases],
      "proxax (cold, vmap)": [c for c, _, _, _ in cases],
      "Clarabel (Python loop)": [cl for _, _, _, cl in cases],
  }
  colors = {
      "proxax (warm, threads for B>1)": "#E69F00",
      "proxax (warm, vmap)": "#009E73",
      "proxax (cold, vmap)": "#0072B2",
      "Clarabel (Python loop)": "#1B9E77",
  }

  best = [min(c, w, wt, cl) for c, w, wt, cl in cases]
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
