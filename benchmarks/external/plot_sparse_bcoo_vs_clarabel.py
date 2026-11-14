"""Generate Mocha-themed SVG plots for sparse BCOO vs. Clarabel benchmarks."""

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
    "proxax-IALM(dense)": "#f38ba8",
    "proxax-IALM(BCOO)": "#a6e3a1",
}
SOLVER_MARKERS = {
    "Clarabel": "P",
    "proxax-IALM(dense)": "s",
    "proxax-IALM(BCOO)": "o",
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
  """Generates a 2-panel runtime scaling plot for sparse BCOO vs. Clarabel."""
  _apply_mocha_style()
  fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)

  dims = [1000, 2000, 4000]
  panels = [
      (
          "Sparse Inequality QP (1% nnz)",
          {
              "Clarabel": [72.56, 238.32, 982.29],
              "proxax-IALM(dense)": [639.55, 3959.61, 13614.60],
              "proxax-IALM(BCOO)": [94.61, 311.35, 645.90],
          },
      ),
      (
          "Sparse Constrained Logistic Regression (1% nnz)",
          {
              "Clarabel": [194.21, 615.59, 1980.60],
              "proxax-IALM(dense)": [124.40, 901.13, 4945.23],
              "proxax-IALM(BCOO)": [23.74, 98.55, 227.87],
          },
      ),
  ]

  for ax, (title, series) in zip(axes, panels, strict=True):
    for name, vals in series.items():
      ax.plot(
          dims,
          vals,
          marker=SOLVER_MARKERS[name],
          color=SOLVER_COLORS[name],
          linewidth=2.2,
          markersize=7.0,
          label=name,
      )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(dims)
    ax.set_xticklabels([f"{d:,}" for d in dims])
    ax.set_xlabel("Dimension n")
    ax.set_ylabel("Solve Time (ms)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.6)
    ax.legend(fontsize=9, loc="upper left")

  fig.suptitle(
      "Large-Scale Sparse BCOO vs. Dense vs. Clarabel (CPU float64)",
      fontsize=12.5,
      fontweight="bold",
  )
  out_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_path, format="svg", bbox_inches="tight")
  plt.close(fig)


def plot_dolan_more_profile(out_path: Path) -> None:
  """Generates a Mocha-themed Dolan-More profile across the 6 sparse cases."""
  _apply_mocha_style()
  fig, ax = plt.subplots(figsize=(7.6, 4.4), constrained_layout=True)

  # (bcoo_ms, dense_ms, clarabel_ms)
  cases = [
      (94.61, 639.55, 72.56),
      (311.35, 3959.61, 238.32),
      (645.90, 13614.60, 982.29),
      (23.74, 124.40, 194.21),
      (98.55, 901.13, 615.59),
      (227.87, 4945.23, 1980.60),
  ]
  series_raw = {
      "proxax-IALM(BCOO)": [b for b, _, _ in cases],
      "Clarabel": [c for _, _, c in cases],
      "proxax-IALM(dense)": [d for _, d, _ in cases],
  }

  best = [min(b, d, c) for b, d, c in cases]
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
        color=SOLVER_COLORS[label],
        label=label,
    )

  ax.set_xscale("log", base=2)
  ax.set_xlim(1.0, tau_max)
  ax.set_ylim(0.0, 1.05)
  ax.set_xlabel(r"Performance Ratio $\tau$ (log$_2$ scale)")
  ax.set_ylabel("Fraction of Problems Solved")
  ax.set_title(
      "Dolan-Moré Runtime Profile (6 Sparse Cases, CPU float64)"
  )
  ax.grid(True, which="both", alpha=0.55)
  ax.legend(fontsize=9, loc="lower right")

  out_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_path, format="svg", bbox_inches="tight")
  plt.close(fig)


def main() -> None:
  """CLI entry point for generating sparse BCOO benchmark plots."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_PLOTS_DIR)
  args = parser.parse_args()

  plot_scaling(args.out_dir / "sparse_bcoo_scaling.svg")
  plot_dolan_more_profile(args.out_dir / "sparse_bcoo_profile.svg")
  print(f"Wrote sparse BCOO Mocha SVG plots to {args.out_dir}")


if __name__ == "__main__":
  main()
