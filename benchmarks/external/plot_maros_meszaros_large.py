"""Generate ggplot SVG plots for Large-Scale Maros-Meszaros QPs."""

from __future__ import annotations

import argparse
import json
import typing
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLOTS_DIR = REPO_ROOT / "docs-experiments" / "_static" / "plots"


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


def plot_maros_large(
    rows: list[dict[str, typing.Any]], out_path: Path
) -> None:
  """Generates a 2-panel runtime and feasibility residual comparison."""
  _apply_mocha_style()
  fig, (ax0, ax1) = plt.subplots(
      1, 2, figsize=(12.5, 4.6), constrained_layout=True
  )

  by_key = {(r["problem_name"], r["solver_name"]): r for r in rows}
  problems = [r["problem_name"] for r in rows if r["solver_name"] == "IALM"]
  short_names = [p.split(" ")[0] for p in problems]
  x = np.arange(len(problems))
  has_iaipal = any(r["solver_name"] == "IAIPAL" for r in rows)
  series = [
      ("IALM", "proxax-IALM (BCOO+Ruiz)", "#D55E00"),
      *(
          [("IAIPAL", "proxax-IAIPAL (BCOO+Ruiz)", "#E69F00")]
          if has_iaipal
          else []
      ),
      ("Clarabel", "Clarabel (IPM)", "#1B9E77"),
  ]
  width = 0.26 if has_iaipal else 0.36
  offsets = np.linspace(
      -(len(series) - 1) * width / 2,
      (len(series) - 1) * width / 2,
      len(series),
  )

  for offset, (sname, label, color) in zip(offsets, series, strict=True):
    vals_ms = [1e3 * float(by_key[(p, sname)]["run_s"]) for p in problems]
    ax0.bar(x + offset, vals_ms, width, label=label, color=color)
  ax0.set_yscale("log")
  ax0.set_xticks(x)
  ax0.set_xticklabels(short_names, rotation=28, ha="right")
  ax0.set_ylabel("Solve Time (ms)")
  ax0.set_title("Runtime (10s limit)", fontweight="bold")
  ax0.grid(True, axis="y", which="major")
  ax0.legend(loc="upper left", frameon=True)

  for offset, (sname, label, color) in zip(offsets, series, strict=True):
    feas = [
        max(1e-16, float(by_key[(p, sname)]["feas_residual"]))
        for p in problems
    ]
    ax1.bar(x + offset, feas, width, label=label, color=color)
  ax1.axhline(
      1e-6, color="#B8860B", linestyle="--", linewidth=1.2, label="tol = 1e-6"
  )
  ax1.set_yscale("log")
  ax1.set_xticks(x)
  ax1.set_xticklabels(short_names, rotation=28, ha="right")
  ax1.set_ylabel("Primal Feasibility Residual")
  ax1.set_title("Primal Feasibility at Termination", fontweight="bold")
  ax1.grid(True, axis="y", which="major")
  ax1.legend(loc="lower right", frameon=True)

  fig.suptitle(
      "Large-Scale Maros-Meszaros Sparse QPs (n = 1,000..10,197, float64)",
      fontweight="bold",
      fontsize=12,
  )
  out_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_path, format="svg", bbox_inches="tight")
  plt.close(fig)


def main() -> None:
  """CLI entry point for large Maros-Meszaros plots."""
  parser = argparse.ArgumentParser()
  parser.add_argument("--json-in", type=Path, required=True)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_PLOTS_DIR)
  args = parser.parse_args()

  data = json.loads(args.json_in.read_text(encoding="utf-8"))
  rows = (
      data
      if isinstance(data, list)
      else data.get("results", data.get("rows", []))
  )
  plot_maros_large(rows, args.out_dir / "maros_meszaros_large.svg")


if __name__ == "__main__":
  main()
