"""Generate ggplot SVG plots for Nonconvex-Concave MinMax benchmarks."""

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
    "PDHG": "#009E73",
    "ExtraGradient": "#0072B2",
    "ProxPointMinMax": "#E69F00",
    "Optax GDA (lr=0.05)": "#D55E00",
}
SOLVER_MARKERS = {
    "PDHG": "o",
    "ExtraGradient": "s",
    "ProxPointMinMax": "^",
    "Optax GDA (lr=0.05)": "X",
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


def plot_minmax_ncc(
    rows: list[dict[str, typing.Any]], out_path: Path
) -> None:
  """Generates a 2-panel plot: m_y sweep (n=32) and dimension scaling."""
  _apply_mocha_style()
  fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)

  by_key = {(r["problem_name"], r["solver_name"]): r for r in rows}
  solvers = [
      "PDHG",
      "ExtraGradient",
      "ProxPointMinMax",
      "Optax GDA (lr=0.05)",
  ]

  # Left panel: m_y sweep at n=32
  m_ys = [0.05, 0.2, 1.0]
  ax0 = axes[0]
  for s in solvers:
    xs = []
    ys = []
    for my in m_ys:
      row = by_key.get((f"NC-C QP (n=32, m_y={my})", s))
      if row is not None:
        xs.append(my)
        ys.append(float(row["grad_evals"]))
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
  ax0.set_xticks(m_ys)
  ax0.set_xticklabels([str(v) for v in m_ys])
  ax0.minorticks_off()
  ax0.set_title("Concavity Sweep (n=32)", fontweight="bold")
  ax0.set_xlabel("Dual Strong Concavity m_y")
  ax0.set_ylabel("Gradient Evaluations")
  ax0.grid(True, which="major")

  # Right panel: Dimension scaling at m_y=0.2
  dims = [32, 64, 256]
  ax1 = axes[1]
  for s in solvers:
    xs = []
    ys = []
    for d in dims:
      row = by_key.get((f"NC-C QP (n={d}, m_y=0.2)", s))
      if row is not None:
        xs.append(d)
        ys.append(1e3 * float(row["run_s"]))
    if xs:
      ax1.plot(
          xs,
          ys,
          label=s,
          color=SOLVER_COLORS[s],
          marker=SOLVER_MARKERS[s],
          linewidth=2.0,
          markersize=6.5,
      )
  ax1.set_xscale("log")
  ax1.set_yscale("log")
  ax1.set_xticks(dims)
  ax1.set_xticklabels([f"{d}x{d}" for d in dims])
  ax1.minorticks_off()
  ax1.set_title("Dimension Scaling (m_y=0.2)", fontweight="bold")
  ax1.set_xlabel("Dimension (n x m)")
  ax1.set_ylabel("Solve Time (ms)")
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
      "Nonconvex-Concave MinMax QP (Kong & Monteiro 2021, float64)",
      fontweight="bold",
      fontsize=12,
  )
  out_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_path, format="svg", bbox_inches="tight")
  plt.close(fig)


def plot_dolan_more_profiles(
    rows: list[dict[str, typing.Any]], out_path: Path
) -> None:
  """Generates 2-panel Dolan-More profiles (runtime and gradient evals)."""
  _apply_mocha_style()
  fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)

  solvers = [
      "PDHG",
      "ExtraGradient",
      "ProxPointMinMax",
      "Optax GDA (lr=0.05)",
  ]
  problems = sorted({str(r["problem_name"]) for r in rows})
  by_key = {(r["problem_name"], r["solver_name"]): r for r in rows}

  taus = [1.0, 1.5, 2.0, 4.0, 8.0, 16.0]
  panels = [
      ("Runtime Profile", "run_s", "Runtime Ratio tau", taus),
      (
          "Gradient Evaluations Profile",
          "grad_evals",
          "Grad-Eval Ratio tau",
          [1.0, 1.5, 2.0, 4.0, 8.0, 16.0],
      ),
  ]

  for ax, (title, metric, xlabel, taus) in zip(axes, panels, strict=True):
    best_val: dict[str, float] = {}
    for p in problems:
      vals = [
          float(by_key[(p, s)][metric])
          for s in solvers
          if (p, s) in by_key and by_key[(p, s)]["converged"]
      ]
      best_val[p] = min(vals) if vals else float("inf")

    for s in solvers:
      ratios = []
      for p in problems:
        row = by_key.get((p, s))
        if row is not None and row["converged"] and best_val[p] > 0:
          ratios.append(float(row[metric]) / best_val[p])
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
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Fraction of Problems Solved")
    ax.set_title(title, fontweight="bold")
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
      "Nonconvex-Concave MinMax Dolan-More Profiles (5 Problems)",
      fontweight="bold",
      fontsize=12,
  )
  out_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_path, format="svg", bbox_inches="tight")
  plt.close(fig)


def main() -> None:
  """CLI entry point for nonconvex-concave MinMax plots."""
  parser = argparse.ArgumentParser()
  parser.add_argument("--json-in", type=Path, required=True)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_PLOTS_DIR)
  args = parser.parse_args()

  rows = json.loads(args.json_in.read_text(encoding="utf-8"))
  plot_minmax_ncc(rows, args.out_dir / "minmax_nonconvex_concave.svg")
  plot_dolan_more_profiles(
      rows, args.out_dir / "minmax_nonconvex_concave_profile.svg"
  )


if __name__ == "__main__":
  main()
