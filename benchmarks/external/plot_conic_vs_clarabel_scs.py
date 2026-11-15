"""Generate ggplot SVG plots for Conic vs. Clarabel/SCS/CVXPY."""

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
    "IALM (APD)": "#009E73",
    "PANOC-ALM": "#0072B2",
    "QP-APD": "#E69F00",
    "Clarabel": "#1B9E77",
    "SCS (direct)": "#B8860B",
    "SCS (indirect)": "#D55E00",
    "CVXPY (SCS)": "#88419D",
}
SOLVER_MARKERS = {
    "IALM (APD)": "o",
    "PANOC-ALM": "s",
    "QP-APD": "^",
    "Clarabel": "P",
    "SCS (direct)": "D",
    "SCS (indirect)": "v",
    "CVXPY (SCS)": "X",
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


def plot_conic_scaling(
    rows: list[dict[str, typing.Any]], out_path: Path
) -> None:
  """Generates a 3-panel runtime scaling plot for SOCP, ExpCone, and SDP."""
  _apply_mocha_style()
  fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.3), constrained_layout=True)

  by_key = {(r["problem_name"], r["solver_name"]): r for r in rows}
  panels = [
      (
          "SOC Portfolio",
          "Dimension n",
          [32, 128, 512],
          [f"SOC Portfolio (n={n})" for n in (32, 128, 512)],
      ),
      (
          "ExpCone Blocks",
          "Blocks k (dim = 3k)",
          [1, 16, 64],
          [f"ExpCone Blocks (k={k}, n={3 * k})" for k in (1, 16, 64)],
      ),
      (
          "Spectrahedron SDP",
          "Matrix size n (n x n)",
          [8, 24, 48],
          [f"Spectrahedron SDP (n={n})" for n in (8, 24, 48)],
      ),
  ]

  solvers = [
      "IALM (APD)",
      "PANOC-ALM",
      "QP-APD",
      "Clarabel",
      "SCS (direct)",
      "CVXPY (SCS)",
  ]

  for ax, (title, xlabel, xvals, pnames) in zip(axes, panels, strict=True):
    for s in solvers:
      ys = []
      xs = []
      for xv, pname in zip(xvals, pnames, strict=True):
        row = by_key.get((pname, s))
        if row is not None and row["converged"]:
          xs.append(xv)
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
    ax.set_xticks(xvals)
    ax.set_xticklabels([str(v) for v in xvals])
    ax.minorticks_off()
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlabel)
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
      "Conic Optimization Runtime Scaling (CPU, float64, tol = 1e-4)",
      fontweight="bold",
      fontsize=12,
  )
  out_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_path, format="svg", bbox_inches="tight")
  plt.close(fig)


def plot_dolan_more_profile(
    rows: list[dict[str, typing.Any]], out_path: Path
) -> None:
  """Generates a Dolan-More runtime performance profile across 9 problems."""
  _apply_mocha_style()
  fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)

  solvers = [
      "IALM (APD)",
      "PANOC-ALM",
      "QP-APD",
      "Clarabel",
      "SCS (direct)",
      "SCS (indirect)",
      "CVXPY (SCS)",
  ]
  problems = sorted({str(r["problem_name"]) for r in rows})
  by_key = {(r["problem_name"], r["solver_name"]): r for r in rows}

  best_time: dict[str, float] = {}
  for p in problems:
    times = [
        float(by_key[(p, s)]["run_s"])
        for s in solvers
        if (p, s) in by_key and by_key[(p, s)]["converged"]
    ]
    best_time[p] = min(times) if times else float("inf")

  taus = [1.0, 1.5, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
  for s in solvers:
    ratios = []
    for p in problems:
      row = by_key.get((p, s))
      if row is not None and row["converged"] and best_time[p] > 0:
        ratios.append(float(row["run_s"]) / best_time[p])
    ys = [
        sum(1 for r in ratios if r <= t) / max(1, len(problems))
        for t in taus
    ]
    ax.step(
        taus,
        ys,
        where="post",
        label=s,
        color=SOLVER_COLORS[s],
        marker=SOLVER_MARKERS[s],
        linewidth=2.0,
        markersize=5.5,
    )

  ax.set_xscale("log")
  ax.set_xticks(taus)
  ax.set_xticklabels([f"{t:g}x" for t in taus])
  ax.minorticks_off()
  ax.set_ylim(-0.02, 1.05)
  ax.set_xlabel("Runtime Ratio tau (vs. fastest per problem)")
  ax.set_ylabel("Fraction of Problems Solved")
  ax.set_title(
      "Conic Optimization Dolan-More Runtime Profile (9 Problems)",
      fontweight="bold",
  )
  ax.grid(True, which="major")
  ax.legend(loc="lower right", frameon=True)

  out_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_path, format="svg", bbox_inches="tight")
  plt.close(fig)


def main() -> None:
  """CLI entry point for conic benchmark plots."""
  parser = argparse.ArgumentParser()
  parser.add_argument("--json-in", type=Path, required=True)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_PLOTS_DIR)
  args = parser.parse_args()

  rows = json.loads(args.json_in.read_text(encoding="utf-8"))
  plot_conic_scaling(rows, args.out_dir / "conic_vs_clarabel_scs.svg")
  plot_dolan_more_profile(
      rows, args.out_dir / "conic_vs_clarabel_scs_profile.svg"
  )


if __name__ == "__main__":
  main()
