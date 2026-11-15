"""Generate ggplot SVG plots for Matrix-Free vs. Clarabel benchmarks."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MD = REPO_ROOT / "docs-experiments" / "matrix_free_vs_clarabel.md"
DEFAULT_PLOTS_DIR = REPO_ROOT / "docs-experiments" / "_static" / "plots"

SOLVER_COLORS = {
    "Clarabel": "#1B9E77",
    "IALM": "#0072B2",
    "IAIPAL": "#009E73",
}
SOLVER_MARKERS = {
    "Clarabel": "P",
    "IALM": "o",
    "IAIPAL": "s",
}

_ROW_RE = re.compile(
    r"^\|\s*`(?P<problem>[^`]+)`\s*\|\s*`(?P<solver>[^`]+)`\s*"
    r"\|\s*`(?P<status>[^`]+)`\s*\|\s*(?P<iters>\d+)\s*"
    r"\|\s*`[^`]+`\s*\|\s*`[^`]+`\s*\|\s*(?:\*\*)?(?P<runtime_ms>[0-9.]+)(?:\*\*)?\s*\|"
)


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


def _parse_results(md_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
  rows: dict[tuple[str, str], dict[str, Any]] = {}
  for line in md_path.read_text(encoding="utf-8").splitlines():
    m = _ROW_RE.match(line.strip())
    if m:
      rows[(m.group("problem"), m.group("solver"))] = {
          "status": m.group("status"),
          "iters": int(m.group("iters")),
          "runtime_ms": float(m.group("runtime_ms")),
      }
  return rows


def plot_scaling(
    rows: dict[tuple[str, str], dict[str, Any]], out_path: Path
) -> None:
  """Generates a 3-panel log-scale runtime scaling plot."""
  _apply_mocha_style()
  fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), constrained_layout=True)

  panels = [
      (
          "Factor-Model QPs (r=20, m=20)",
          "Dimension n",
          [
              (1000, "FactorQP-1000 (n=1000,r=20,m=20)"),
              (2000, "FactorQP-2000 (n=2000,r=20,m=20)"),
              (3000, "FactorQP-3000 (n=3000,r=20,m=20)"),
          ],
      ),
      (
          "Simplex Logistic Regression",
          "Samples = Features (N = n)",
          [
              (300, "LogReg-Simplex-300 (N=300,n=300)"),
              (600, "LogReg-Simplex-600 (N=600,n=600)"),
              (1000, "LogReg-Simplex-1000 (N=1000,n=1000)"),
          ],
      ),
      (
          "Dense-P QPs (m=20, k=20)",
          "Dimension n",
          [
              (1000, "DenseQP-1000 (n=1000,m=20,k=20)"),
              (1500, "DenseQP-1500 (n=1500,m=20,k=20)"),
          ],
      ),
  ]

  for ax, (title, xlabel, probs) in zip(axes, panels, strict=True):
    xs = [p[0] for p in probs]
    for solver in ("Clarabel", "IALM", "IAIPAL"):
      ys = [float(rows[(p[1], solver)]["runtime_ms"]) for p in probs]
      statuses = [str(rows[(p[1], solver)]["status"]) for p in probs]
      ax.plot(
          xs,
          ys,
          color=SOLVER_COLORS[solver],
          marker=SOLVER_MARKERS[solver],
          linewidth=2.2,
          markersize=7.5,
          label=solver,
      )
      for x, y, st in zip(xs, ys, statuses, strict=True):
        if st != "CONVERGED":
          ax.scatter(
              [x],
              [y],
              s=130,
              facecolors="none",
              edgecolors="#D55E00",
              linewidths=2.0,
              zorder=5,
          )
          ax.annotate(
              "10s cap",
              (x, y),
              textcoords="offset points",
              xytext=(-6, -15),
              fontsize=8,
              color="#D55E00",
              ha="center",
          )

    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=10)
    ax.grid(True, which="both", alpha=0.45)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))

  axes[0].set_ylabel("Runtime (ms, log scale)", fontweight="bold")
  axes[0].legend(loc="upper left", frameon=True)
  fig.suptitle(
      "Single-Instance CPU Scaling: Matrix-Free proxax vs. Clarabel",
      fontsize=13,
      fontweight="bold",
  )
  out_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_path, format="svg")
  plt.close(fig)


def plot_speedup(
    rows: dict[tuple[str, str], dict[str, Any]], out_path: Path
) -> None:
  """Generates a horizontal log-scale speedup chart vs. Clarabel."""
  _apply_mocha_style()
  problems = [
      ("FactorQP-3000", "FactorQP-3000 (n=3000,r=20,m=20)"),
      ("FactorQP-2000", "FactorQP-2000 (n=2000,r=20,m=20)"),
      ("FactorQP-1000", "FactorQP-1000 (n=1000,r=20,m=20)"),
      ("DenseQP-1000", "DenseQP-1000 (n=1000,m=20,k=20)"),
      ("LogReg-Simplex-300", "LogReg-Simplex-300 (N=300,n=300)"),
      ("LogReg-Simplex-600", "LogReg-Simplex-600 (N=600,n=600)"),
      ("LogReg-Simplex-1000", "LogReg-Simplex-1000 (N=1000,n=1000)"),
      ("DenseQP-1500", "DenseQP-1500 (n=1500,m=20,k=20)"),
  ]

  labels: list[str] = []
  speedups: list[float] = []
  colors: list[str] = []
  notes: list[str] = []

  for short, full in reversed(problems):
    cb_ms = float(rows[(full, "Clarabel")]["runtime_ms"])
    conv_px = [
        (s, float(rows[(full, s)]["runtime_ms"]))
        for s in ("IALM", "IAIPAL")
        if rows[(full, s)]["status"] == "CONVERGED"
    ]
    labels.append(short)
    if conv_px:
      best_s, best_ms = min(conv_px, key=lambda kv: kv[1])
      sp = cb_ms / best_ms
      speedups.append(sp)
      colors.append(SOLVER_COLORS[best_s])
      notes.append(f"{sp:.1f}× ({best_s}: {best_ms:.1f} ms)")
    else:
      _, best_ms = min(
          (
              (s, float(rows[(full, s)]["runtime_ms"]))
              for s in ("IALM", "IAIPAL")
          ),
          key=lambda kv: kv[1],
      )
      sp = cb_ms / best_ms
      speedups.append(sp)
      colors.append("#D55E00")
      notes.append(f"Clarabel wins ({cb_ms:.0f} ms vs 10s cap)")

  fig, ax = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
  y_pos = list(range(len(labels)))
  ax.barh(y_pos, speedups, color=colors, height=0.62, edgecolor="#999999")
  ax.axvline(
      1.0,
      color="#1B9E77",
      linestyle="--",
      linewidth=1.6,
      label="Clarabel (1.0×)",
  )
  ax.set_xscale("log")
  ax.set_xlim(0.05, 2500.0)
  ax.set_yticks(y_pos)
  ax.set_yticklabels(labels, fontweight="bold")
  ax.set_xlabel(
      "Speedup over Clarabel (log scale, > 1× means proxax faster)",
      fontweight="bold",
  )
  ax.set_title(
      "Best Matrix-Free proxax Solver vs. Clarabel (Single-Instance CPU)",
      fontsize=12.5,
      fontweight="bold",
      pad=10,
  )
  ax.grid(True, axis="x", which="both", alpha=0.4)

  for y, sp, note in zip(y_pos, speedups, notes, strict=True):
    if sp >= 1.0:
      ax.text(
          sp * 1.15,
          y,
          note,
          va="center",
          ha="left",
          fontsize=9,
          color="#222222",
      )
    else:
      ax.text(
          1.15, y, note, va="center", ha="left", fontsize=9, color="#D55E00"
      )

  ax.legend(loc="lower right", frameon=True)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_path, format="svg")
  plt.close(fig)


def main() -> int:
  """Generates SVG plots from the matrix_free_vs_clarabel Markdown table."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--md", type=Path, default=DEFAULT_MD)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_PLOTS_DIR)
  args = parser.parse_args()

  rows = _parse_results(args.md)
  plot_scaling(rows, args.out_dir / "matrix_free_vs_clarabel_scaling.svg")
  plot_speedup(rows, args.out_dir / "matrix_free_vs_clarabel_speedup.svg")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
