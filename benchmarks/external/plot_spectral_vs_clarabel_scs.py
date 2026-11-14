"""Generate Mocha-themed SVG plots for Spectral MC vs. Clarabel/SCS/CVXPY."""

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
    "DAICG (monotone)": "#a6e3a1",
    "DAICG (nonmonotone)": "#89b4fa",
    "IAICG": "#fab387",
    "Clarabel": "#94e2d5",
    "SCS (direct)": "#f9e2af",
    "CVXPY (SCS)": "#cba6f7",
}
SOLVER_MARKERS = {
    "DAICG (monotone)": "o",
    "DAICG (nonmonotone)": "s",
    "IAICG": "^",
    "Clarabel": "P",
    "SCS (direct)": "D",
    "CVXPY (SCS)": "X",
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


def plot_spectral(rows: list[dict[str, typing.Any]], out_path: Path) -> None:
  """Generates a 2-panel scaling + Dolan-More profile plot for Spectral MC."""
  _apply_mocha_style()
  fig, (ax0, ax1) = plt.subplots(
      1, 2, figsize=(11.5, 4.4), constrained_layout=True
  )

  by_key = {(r["problem_name"], r["solver_name"]): r for r in rows}
  dims = [16, 32, 48]
  pnames = [f"Nuclear MC ({d}x{d})" for d in dims]
  solvers = [
      "DAICG (monotone)",
      "DAICG (nonmonotone)",
      "IAICG",
      "Clarabel",
      "SCS (direct)",
      "CVXPY (SCS)",
  ]

  for s in solvers:
    xs = []
    ys = []
    for d, pname in zip(dims, pnames, strict=True):
      row = by_key.get((pname, s))
      if row is not None and row["converged"]:
        xs.append(d)
        ys.append(1e3 * float(row["run_s"]))
    if xs:
      ax0.plot(
          xs,
          ys,
          label=s,
          color=SOLVER_COLORS[s],
          marker=SOLVER_MARKERS[s],
          linewidth=2.0,
          markersize=6.5,
      )
  ax0.set_xscale("log")
  ax0.set_yscale("log")
  ax0.set_xticks(dims)
  ax0.set_xticklabels([f"{d}x{d}" for d in dims])
  ax0.minorticks_off()
  ax0.set_title("Runtime Scaling", fontweight="bold")
  ax0.set_xlabel("Matrix Size (m x n)")
  ax0.set_ylabel("Solve Time (ms)")
  ax0.grid(True, which="major")

  best_time = {
      p: min(
          float(by_key[(p, s)]["run_s"])
          for s in solvers
          if (p, s) in by_key and by_key[(p, s)]["converged"]
      )
      for p in pnames
  }
  taus = [1.0, 2.0, 4.0, 8.0, 16.0, 64.0, 256.0]
  for s in solvers:
    ratios = [
        float(by_key[(p, s)]["run_s"]) / best_time[p]
        for p in pnames
        if (p, s) in by_key and by_key[(p, s)]["converged"]
    ]
    ys = [sum(1 for r in ratios if r <= t) / len(pnames) for t in taus]
    ax1.step(
        taus,
        ys,
        where="post",
        label=s,
        color=SOLVER_COLORS[s],
        marker=SOLVER_MARKERS[s],
        linewidth=2.0,
        markersize=5.5,
    )
  ax1.set_xscale("log")
  ax1.set_xticks(taus)
  ax1.set_xticklabels([f"{t:g}x" for t in taus])
  ax1.minorticks_off()
  ax1.set_ylim(-0.02, 1.05)
  ax1.set_title("Dolan-More Runtime Profile", fontweight="bold")
  ax1.set_xlabel("Runtime Ratio tau")
  ax1.set_ylabel("Fraction Solved")
  ax1.grid(True, which="major")

  handles, labels = ax0.get_legend_handles_labels()
  fig.legend(
      handles,
      labels,
      loc="lower center",
      ncol=len(solvers),
      bbox_to_anchor=(0.5, -0.11),
      frameon=True,
  )
  fig.suptitle(
      "Nuclear-Norm Matrix Completion (CPU, float64, tol = 1e-4)",
      fontweight="bold",
      fontsize=12,
  )
  out_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_path, format="svg", bbox_inches="tight")
  plt.close(fig)


def main() -> None:
  """CLI entry point for spectral benchmark plots."""
  parser = argparse.ArgumentParser()
  parser.add_argument("--json-in", type=Path, required=True)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_PLOTS_DIR)
  args = parser.parse_args()

  rows = json.loads(args.json_in.read_text(encoding="utf-8"))
  plot_spectral(rows, args.out_dir / "spectral_vs_clarabel_scs.svg")


if __name__ == "__main__":
  main()
