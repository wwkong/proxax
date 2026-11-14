"""Generate Mocha-themed SVG plots for Convex-Concave MinMax benchmarks."""

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
    "PDHG": "#a6e3a1",
    "ExtraGradient": "#89b4fa",
    "ProxPointMinMax": "#fab387",
    "Clarabel": "#94e2d5",
    "SCS (direct)": "#f9e2af",
    "CVXPY (Clarabel)": "#cba6f7",
}
SOLVER_MARKERS = {
    "PDHG": "o",
    "ExtraGradient": "s",
    "ProxPointMinMax": "^",
    "Clarabel": "P",
    "SCS (direct)": "D",
    "CVXPY (Clarabel)": "X",
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


def plot_minmax_cc(
    rows: list[dict[str, typing.Any]], out_path: Path
) -> None:
  """Generates a 2-panel runtime scaling plot for convex-concave MinMax."""
  _apply_mocha_style()
  fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)

  by_key = {(r["problem_name"], r["solver_name"]): r for r in rows}
  dims = [32, 128, 256]
  panels = [
      (
          "Zero-Sum Matrix Game (Simplex x Simplex)",
          [f"Matrix Game ({d}x{d})" for d in dims],
      ),
      (
          "Bilinear Saddle QP (Box x Simplex)",
          [f"Bilinear Saddle QP ({d}x{d})" for d in dims],
      ),
  ]
  solvers = [
      "PDHG",
      "ExtraGradient",
      "ProxPointMinMax",
      "Clarabel",
      "SCS (direct)",
      "CVXPY (Clarabel)",
  ]

  for ax, (title, pnames) in zip(axes, panels, strict=True):
    for s in solvers:
      xs = []
      ys = []
      for d, pname in zip(dims, pnames, strict=True):
        row = by_key.get((pname, s))
        if row is not None and row["converged"]:
          xs.append(d)
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
    ax.set_xticks(dims)
    ax.set_xticklabels([f"{d}x{d}" for d in dims])
    ax.minorticks_off()
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Dimension (n x m)")
    ax.set_ylabel("Solve Time (ms)")
    ax.grid(True, which="major")

  handles, labels = axes[0].get_legend_handles_labels()
  fig.legend(
      handles,
      labels,
      loc="lower center",
      ncol=len(solvers),
      bbox_to_anchor=(0.5, -0.11),
      frameon=True,
  )
  fig.suptitle(
      "Convex-Concave MinMax vs. Dualized Conic Solvers (CPU, float64)",
      fontweight="bold",
      fontsize=12,
  )
  out_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_path, format="svg", bbox_inches="tight")
  plt.close(fig)


def main() -> None:
  """CLI entry point for convex-concave MinMax plots."""
  parser = argparse.ArgumentParser()
  parser.add_argument("--json-in", type=Path, required=True)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_PLOTS_DIR)
  args = parser.parse_args()

  rows = json.loads(args.json_in.read_text(encoding="utf-8"))
  plot_minmax_cc(rows, args.out_dir / "minmax_convex_concave.svg")


if __name__ == "__main__":
  main()
