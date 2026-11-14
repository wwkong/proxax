"""Generate Mocha-themed SVG plots for RTX 3060 GPU benchmarks."""

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
      "font.size": 9.5,
  })


def _extract_batch_series(
    rows: list[dict[str, typing.Any]], prefix: str, solver_name: str
) -> tuple[list[int], list[float]]:
  pts: list[tuple[int, float]] = []
  for r in rows:
    pname = r["problem_name"]
    if pname.startswith(prefix) and r["solver_name"] == solver_name:
      b_str = pname.split("B=")[1].rstrip(")")
      pts.append((int(b_str), 1e3 * float(r["run_s"])))
  pts.sort(key=lambda t: t[0])
  return [p[0] for p in pts], [p[1] for p in pts]


def plot_gpu_benchmarks(
    gpu_batch: list[dict[str, typing.Any]],
    cpu_batch: list[dict[str, typing.Any]],
    gpu_minmax: list[dict[str, typing.Any]],
    out_path: Path,
) -> None:
  """Generates a 3-panel GPU vs CPU & Optax GPU comparison SVG."""
  _apply_mocha_style()
  fig, axes = plt.subplots(
      1, 3, figsize=(15.5, 4.6), constrained_layout=True
  )

  # Panel 1 & 2: Batched SOCP and Saddle QP scaling
  panels = [
      (
          axes[0],
          "Batched SOCP",
          "proxax-IALM (vmap)",
          "proxax-IALM (threads)",
          "Batched SOCP (n=64)",
      ),
      (
          axes[1],
          "Batched Saddle QP",
          "proxax-PDHG (vmap)",
          "proxax-PDHG (threads)",
          "Batched Saddle QP (n=64)",
      ),
  ]
  for ax, prefix, gpu_s, cpu_s, title in panels:
    bs_g, ms_g = _extract_batch_series(gpu_batch, prefix, gpu_s)
    bs_c, ms_c = _extract_batch_series(cpu_batch, prefix, cpu_s)
    bs_cl, ms_cl = _extract_batch_series(
        gpu_batch, prefix, "Clarabel (threads)"
    )
    bs_sc, ms_sc = _extract_batch_series(gpu_batch, prefix, "SCS (threads)")

    ax.plot(
        bs_g, ms_g, "o-", color="#a6e3a1", linewidth=2.0, label="Proxax GPU"
    )
    ax.plot(
        bs_c, ms_c, "s--", color="#89b4fa", linewidth=1.8, label="Proxax CPU"
    )
    ax.plot(
        bs_cl, ms_cl, "^:", color="#94e2d5", linewidth=1.8, label="Clarabel CPU"
    )
    ax.plot(
        bs_sc, ms_sc, "d:", color="#f9e2af", linewidth=1.8, label="SCS CPU"
    )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(bs_g)
    ax.set_xticklabels([str(b) for b in bs_g])
    ax.set_xlabel("Batch Size B")
    ax.set_ylabel("Solve Time (ms)")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, which="major")
    ax.legend(loc="upper left", frameon=True, fontsize=8.5)

  # Panel 3: Nonconvex-Concave Min-Max on GPU (Proxax vs Optax)
  ax2 = axes[2]
  probs = [
      "NC-C QP (n=32, m_y=0.2)",
      "NC-C QP (n=64, m_y=0.2)",
      "NC-C QP (n=256, m_y=0.2)",
  ]
  short = ["n=32", "n=64", "n=256"]
  by_k = {(r["problem_name"], r["solver_name"]): r for r in gpu_minmax}
  sols = [
      ("PDHG", "PDHG (GPU)", "#a6e3a1"),
      ("ExtraGradient", "ExtraGrad (GPU)", "#89b4fa"),
      ("Optax GDA (lr=0.05)", "Optax GDA (GPU)", "#f38ba8"),
  ]
  x = np.arange(len(probs))
  width = 0.25
  offsets = np.linspace(-width, width, len(sols))
  for off, (sname, label, color) in zip(offsets, sols, strict=True):
    vals = [1e3 * float(by_k[(p, sname)]["run_s"]) for p in probs]
    ax2.bar(x + off, vals, width, label=label, color=color)
  ax2.set_yscale("log")
  ax2.set_xticks(x)
  ax2.set_xticklabels(short)
  ax2.set_xlabel("Problem Dimension (m_y = 0.2)")
  ax2.set_ylabel("Solve Time (ms)")
  ax2.set_title("NC-C Min-Max on GPU (vs Optax)", fontweight="bold")
  ax2.grid(True, axis="y", which="major")
  ax2.legend(loc="upper left", frameon=True, fontsize=8.5)

  fig.suptitle(
      "Proxax on NVIDIA RTX 3060 GPU (float64) vs CPU Baselines & Optax GPU",
      fontweight="bold",
      fontsize=12,
  )
  out_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_path, format="svg", bbox_inches="tight")
  plt.close(fig)


def main() -> None:
  """CLI entry point for GPU benchmark plots."""
  parser = argparse.ArgumentParser()
  parser.add_argument("--gpu-batch", type=Path, required=True)
  parser.add_argument("--cpu-batch", type=Path, required=True)
  parser.add_argument("--gpu-minmax", type=Path, required=True)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_PLOTS_DIR)
  args = parser.parse_args()

  plot_gpu_benchmarks(
      json.loads(args.gpu_batch.read_text(encoding="utf-8")),
      json.loads(args.cpu_batch.read_text(encoding="utf-8")),
      json.loads(args.gpu_minmax.read_text(encoding="utf-8")),
      args.out_dir / "gpu_benchmarks.svg",
  )


if __name__ == "__main__":
  main()
