"""Generate Mocha-themed SVG plots for Large-Scale Maros-Meszaros QPs."""

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
  width = 0.36

  ialm_ms = [1e3 * float(by_key[(p, "IALM")]["run_s"]) for p in problems]
  clar_ms = [1e3 * float(by_key[(p, "Clarabel")]["run_s"]) for p in problems]

  ax0.bar(
      x - width / 2,
      ialm_ms,
      width,
      label="proxax-IALM (BCOO+Ruiz)",
      color="#f38ba8",
  )
  ax0.bar(
      x + width / 2, clar_ms, width, label="Clarabel (IPM)", color="#94e2d5"
  )
  ax0.set_yscale("log")
  ax0.set_xticks(x)
  ax0.set_xticklabels(short_names, rotation=28, ha="right")
  ax0.set_ylabel("Solve Time (ms)")
  ax0.set_title("Runtime (10s limit)", fontweight="bold")
  ax0.grid(True, axis="y", which="major")
  ax0.legend(loc="upper left", frameon=True)

  ialm_feas = [
      max(1e-16, float(by_key[(p, "IALM")]["feas_residual"])) for p in problems
  ]
  clar_feas = [
      max(1e-16, float(by_key[(p, "Clarabel")]["feas_residual"]))
      for p in problems
  ]
  ax1.bar(
      x - width / 2,
      ialm_feas,
      width,
      label="proxax-IALM (BCOO+Ruiz)",
      color="#f38ba8",
  )
  ax1.bar(
      x + width / 2, clar_feas, width, label="Clarabel (IPM)", color="#94e2d5"
  )
  ax1.axhline(
      1e-6, color="#f9e2af", linestyle="--", linewidth=1.2, label="tol = 1e-6"
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
