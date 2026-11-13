"""Generate SVG plots for the 5 small-scale literature replication studies."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PLOTS_DIR = Path("/home/wkong/Projects/proxax/docs-experiments/_static/plots")
DATA_DIR = Path("/tmp/proxax_repl")

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
    "legend.facecolor": "#1e1e2e",
    "legend.edgecolor": "#45475a",
    "savefig.facecolor": "#1e1e2e",
    "savefig.edgecolor": "#1e1e2e",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.65,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

PALETTE = ["#89b4fa", "#f38ba8", "#a6e3a1", "#fab387", "#cba6f7", "#94e2d5"]
MARKERS = ["o", "s", "^", "D", "v", "P"]


def _plot_kong2024_spectrahedron() -> None:
  rows = json.loads((DATA_DIR / "kong2024_spectrahedron.json").read_text())
  families = ["Spectrahedron QP (10x10)", "Simplex QP (n=30)"]
  solvers = ["APD", "AdapAPG", "PANOC", "AIPP", "AG", "UPFAG"]

  fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
  for ax, fam in zip(axes, families):
    sub = [r for r in rows if r["family"] == fam]
    for idx, s in enumerate(solvers):
      pts = sorted([r for r in sub if r["solver"] == s], key=lambda r: r["kappa"])
      xs = [r["kappa"] for r in pts]
      ys = [r["oracle_evals"] for r in pts]
      ax.plot(
          xs, ys,
          marker=MARKERS[idx], color=PALETTE[idx], linewidth=1.8, markersize=5.5,
          label=s,
      )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([10, 40, 100, 250])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_title(fam)
    ax.set_xlabel(r"Condition parameter $\kappa$")
    ax.set_ylabel("Oracle evaluations")
  axes[0].legend(frameon=False, ncol=2, loc="upper left")
  fig.suptitle("Curvature-Free APD vs. Fixed-Curvature Solvers (Kong, SIOPT 2024)", fontsize=12)
  fig.savefig(PLOTS_DIR / "kong2024_spectrahedron.svg", format="svg")
  plt.close(fig)


def _plot_kong2023_alm_vs_qp() -> None:
  rows = json.loads((DATA_DIR / "kong2023_alm_vs_qp.json").read_text())
  problems = [
      "Linearly Constrained QP (n=24, m=4)",
      "Constrained k-Fantope SPCA (8x8, k=2)",
  ]
  solvers = ["IALM (APD)", "IALM (PANOC)", "SProxALM", "QP-APD", "QP-AIPP"]

  fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
  for ax, prob in zip(axes, problems):
    sub = [r for r in rows if r["problem"] == prob]
    for idx, s in enumerate(solvers):
      pts = sorted([r for r in sub if r["solver"] == s], key=lambda r: r["tol"], reverse=True)
      xs = [r["tol"] for r in pts]
      ys = [r["oracle_evals"] for r in pts]
      ax.plot(
          xs, ys,
          marker=MARKERS[idx], color=PALETTE[idx], linewidth=1.8, markersize=5.5,
          label=s,
      )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_title(prob)
    ax.set_xlabel(r"Tolerance $\varepsilon$ (tighter $\rightarrow$)")
    ax.set_ylabel("Oracle evaluations")
  axes[0].legend(frameon=False, loc="upper left")
  fig.suptitle("ALM vs. Quadratic Penalty Scaling (Kong et al., 2023)", fontsize=12)
  fig.savefig(PLOTS_DIR / "kong2023_alm_vs_qp.svg", format="svg")
  plt.close(fig)


def _plot_kong2024_dpadmm() -> None:
  rows = json.loads((DATA_DIR / "kong2024_dpadmm.json").read_text())
  solvers = ["DPADMM", "PAdaptADMM", "ADMM (rho=5)"]

  fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
  for idx, s in enumerate(solvers):
    pts = sorted([r for r in rows if r["solver"] == s], key=lambda r: r["rank"])
    xs = [r["rank"] for r in pts]
    opt_y = [max(r["opt_residual"], 1e-12) for r in pts]
    feas_y = [max(r["feas_residual"], 1e-12) for r in pts]
    axes[0].plot(xs, opt_y, marker=MARKERS[idx], color=PALETTE[idx], linewidth=1.8, markersize=5.5, label=s)
    axes[1].plot(xs, feas_y, marker=MARKERS[idx], color=PALETTE[idx], linewidth=1.8, markersize=5.5, label=s)

  for ax, title, ylabel in [
      (axes[0], "Stationarity Residual vs. Block-3 Rank", r"Stationarity residual $\|v\|$"),
      (axes[1], "Feasibility Residual vs. Block-3 Rank", r"Feasibility residual $\|Ax - b\|$"),
  ]:
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 4, 6])
    ax.axhline(1e-3, color="#bac2de", linestyle="--", linewidth=1.0, alpha=0.7, label=r"tol ($10^{-3}$)")
    ax.set_title(title)
    ax.set_xlabel(r"Rank of $A_3$ ($m=6$; $r < 6$ is rank-deficient)")
    ax.set_ylabel(ylabel)
  axes[0].legend(frameon=False, loc="best")
  fig.suptitle("Rank-Free Convergence of DPADMM (Kong & Monteiro, SIOPT 2024)", fontsize=12)
  fig.savefig(PLOTS_DIR / "kong2024_dpadmm.svg", format="svg")
  plt.close(fig)


def _plot_kong2022_spectral() -> None:
  rows = json.loads((DATA_DIR / "kong2022_spectral.json").read_text())
  solvers = ["DAICG (monotone)", "DAICG (nonmonotone)", "IAICG"]

  fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
  for idx, s in enumerate(solvers):
    pts = sorted([r for r in rows if r["solver"] == s], key=lambda r: r["obs_prob"])
    xs = [r["obs_prob"] * 100 for r in pts]
    evals = [r["oracle_evals"] for r in pts]
    res = [r["opt_residual"] for r in pts]
    axes[0].plot(xs, evals, marker=MARKERS[idx], color=PALETTE[idx], linewidth=1.8, markersize=5.5, label=s)
    axes[1].plot(xs, res, marker=MARKERS[idx], color=PALETTE[idx], linewidth=1.8, markersize=5.5, label=s)

  axes[0].set_title("Oracle Evaluations vs. Observation Density")
  axes[0].set_xlabel("Observed matrix entries (%)")
  axes[0].set_ylabel("Oracle evaluations")
  axes[1].set_yscale("log")
  axes[1].axhline(1e-3, color="#bac2de", linestyle="--", linewidth=1.0, alpha=0.7, label=r"tol ($10^{-3}$)")
  axes[1].set_title("Final Stationarity Residual (600 outer iters cap)")
  axes[1].set_xlabel("Observed matrix entries (%)")
  axes[1].set_ylabel("Stationarity residual")
  axes[1].legend(frameon=False, loc="best")
  fig.suptitle("Doubly vs. Inner Acceleration on Spectral Matrix Completion (Kong & Monteiro, COAP 2022)", fontsize=11.5)
  fig.savefig(PLOTS_DIR / "kong2022_spectral.svg", format="svg")
  plt.close(fig)


def _plot_kong2021_minmax() -> None:
  rows = json.loads((DATA_DIR / "kong2021_minmax.json").read_text())
  solvers = ["ProxPointMinMax", "ExtraGradient", "PDHG"]

  fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
  for idx, s in enumerate(solvers):
    pts = sorted([r for r in rows if r["solver"] == s], key=lambda r: r["m_y"])
    xs = [r["m_y"] for r in pts]
    evals = [r["oracle_evals"] for r in pts]
    objs = [r["obj_val"] for r in pts]
    axes[0].plot(xs, evals, marker=MARKERS[idx], color=PALETTE[idx], linewidth=1.8, markersize=5.5, label=s)
    axes[1].plot(xs, objs, marker=MARKERS[idx], color=PALETTE[idx], linewidth=1.8, markersize=5.5, label=s)

  axes[0].set_yscale("log")
  axes[0].set_title("Oracle Evaluations vs. Dual Curvature")
  axes[0].set_xlabel(r"Dual strong concavity $m_y$")
  axes[0].set_ylabel("Oracle evaluations")
  axes[1].set_title("Stationary Objective Value")
  axes[1].set_xlabel(r"Dual strong concavity $m_y$")
  axes[1].set_ylabel(r"Final coupling $\Phi(\hat{x}, \hat{y})$")
  axes[0].legend(frameon=False, loc="best")
  fig.suptitle("Nonconvex-Concave Min-Max QP (Kong & Monteiro, SIOPT 2021)", fontsize=12)
  fig.savefig(PLOTS_DIR / "kong2021_minmax.svg", format="svg")
  plt.close(fig)


def main() -> None:
  PLOTS_DIR.mkdir(parents=True, exist_ok=True)
  _plot_kong2024_spectrahedron()
  _plot_kong2023_alm_vs_qp()
  _plot_kong2024_dpadmm()
  _plot_kong2022_spectral()
  _plot_kong2021_minmax()
  print("Generated all 5 SVG plots in", PLOTS_DIR)


if __name__ == "__main__":
  main()
