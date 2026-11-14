"""Generate Mocha-themed SVG plots for large matrix-free operator benchmarks."""

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
    "proxax-PANOC": "#a6e3a1",
}
SOLVER_MARKERS = {
    "Clarabel": "P",
    "proxax-PANOC": "o",
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
  """Generates a 2-panel log-log dimension scaling plot (CPU float64)."""
  _apply_mocha_style()
  fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)

  ns_all = [1024, 4096, 16384, 65536]
  panels = [
      (
          "1D Circulant LASSO (W=64)",
          [7.56, 40.98, 188.92, 1072.64],
          [1024, 4096, 16384],
          [157.97, 963.75, 9037.82],
      ),
      (
          "2D FFT Gaussian Deblurring (Box [0, 1])",
          [4.70, 13.18, 62.48, 352.29],
          [1024, 4096],
          [3388.18, 31474.61],
      ),
  ]

  for ax, (title, px_vals, cl_ns, cl_vals) in zip(axes, panels, strict=True):
    ax.plot(
        ns_all,
        px_vals,
        marker=SOLVER_MARKERS["proxax-PANOC"],
        color=SOLVER_COLORS["proxax-PANOC"],
        linewidth=2.2,
        markersize=7.0,
        label="proxax-PANOC (FFT matrix-free)",
    )
    ax.plot(
        cl_ns,
        cl_vals,
        marker=SOLVER_MARKERS["Clarabel"],
        color=SOLVER_COLORS["Clarabel"],
        linewidth=2.2,
        markersize=7.0,
        label="Clarabel (explicit sparse KKT)",
    )
    ax.set_xscale("log", base=4)
    ax.set_yscale("log")
    ax.set_xticks(ns_all)
    ax.set_xticklabels([f"{n:,}" for n in ns_all])
    ax.set_xlabel("Problem Dimension n")
    ax.set_ylabel("Steady-State Runtime (ms)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.6)
    ax.legend(fontsize=8.5, loc="upper left")

  fig.suptitle(
      "Large-Scale Matrix-Free Operators: proxax vs. Clarabel (CPU f64)",
      fontsize=12.5,
      fontweight="bold",
  )
  out_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_path, format="svg", bbox_inches="tight")
  plt.close(fig)


def plot_dolan_more_profile(out_path: Path) -> None:
  """Generates a Mocha-themed Dolan-More profile across all 8 cases."""
  _apply_mocha_style()
  fig, ax = plt.subplots(figsize=(7.6, 4.4), constrained_layout=True)

  # All 8 cases: (proxax_ms, clarabel_ms or None if skipped/timed out)
  cases: list[tuple[float, float | None]] = [
      (7.56, 157.97),
      (40.98, 963.75),
      (188.92, 9037.82),
      (1072.64, None),
      (4.70, 3388.18),
      (13.18, 31474.61),
      (62.48, None),
      (352.29, None),
  ]
  tau_max = 4096.0
  n_probs = len(cases)
  best = [p if c is None else min(p, c) for p, c in cases]

  series_list: list[tuple[str, str, list[float | None]]] = [
      ("proxax-PANOC (matrix-free)", "#a6e3a1", [p for p, _ in cases]),
      ("Clarabel (explicit KKT)", "#94e2d5", [c for _, c in cases]),
  ]
  for label, color, vals in series_list:
    ratios = sorted(
        max(v / b, 1.0)
        for v, b in zip(vals, best, strict=True)
        if v is not None
    )
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
        color=color,
        label=label,
    )

  ax.set_xscale("log", base=2)
  ax.set_xlim(1.0, tau_max)
  ax.set_ylim(0.0, 1.05)
  ax.set_xlabel(r"Performance Ratio $\tau$ (log$_2$ scale)")
  ax.set_ylabel("Fraction of Problems Solved")
  ax.set_title(
      "Dolan-Moré Runtime Profile (8 Large Operator Cases, CPU float64)"
  )
  ax.grid(True, which="both", alpha=0.55)
  ax.legend(fontsize=9, loc="lower right")

  out_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_path, format="svg", bbox_inches="tight")
  plt.close(fig)


def main() -> None:
  """CLI entry point for generating large matrix-free Mocha plots."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--out-dir",
      type=Path,
      default=DEFAULT_PLOTS_DIR,
      help="Directory to write SVG plots into.",
  )
  args = parser.parse_args()
  plot_scaling(args.out_dir / "large_matrix_free_scaling.svg")
  plot_dolan_more_profile(args.out_dir / "large_matrix_free_profile.svg")
  print(f"Wrote large matrix-free Mocha SVG plots to {args.out_dir}")


if __name__ == "__main__":
  main()
