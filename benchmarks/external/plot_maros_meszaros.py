"""Generate Mocha-themed SVG plots for Maros-Meszaros QP benchmark results."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MD = REPO_ROOT / "docs-experiments" / "maros_meszaros.md"
DEFAULT_PLOTS_DIR = REPO_ROOT / "docs-experiments" / "_static" / "plots"

SOLVER_ORDER = ["Clarabel", "IALM", "IAIPAL", "QPAPD", "QPAIPP", "SProxALM"]
SOLVER_COLORS = {
    "Clarabel": "#94e2d5",
    "IALM": "#89b4fa",
    "IAIPAL": "#a6e3a1",
    "QPAPD": "#fab387",
    "QPAIPP": "#cba6f7",
    "SProxALM": "#f38ba8",
}
SOLVER_MARKERS = {
    "Clarabel": "P",
    "IALM": "o",
    "IAIPAL": "s",
    "QPAPD": "^",
    "QPAIPP": "D",
    "SProxALM": "v",
}

_ROW_RE = re.compile(
    r"^\|\s*`(?P<problem>[^`]+)`\s*\|\s*`(?P<solver>[^`]+)`\s*"
    r"\|\s*`(?P<status>[^`]+)`\s*"
    r"\|\s*(?P<outer>\d+)\s*\|\s*(?P<inner>\d+)\s*\|\s*(?P<grad_evals>\d+)\s*"
    r"\|\s*`(?P<opt_res>[^`]+)`\s*\|\s*`(?P<feas_res>[^`]+)`\s*"
    r"\|\s*(?P<runtime_ms>[0-9.]+)\s*\|$"
)


def _apply_mocha_style() -> None:
    plt.rcParams.update(
        {
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
        }
    )


def parse_markdown_results(md_path: Path) -> list[dict[str, Any]]:
    """Extract Full Results rows from docs-experiments/maros_meszaros.md."""
    rows: list[dict[str, Any]] = []
    in_full = False
    for raw_line in md_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line == "### Full Results":
            in_full = True
            continue
        if not in_full or not line.startswith("|"):
            continue
        m = _ROW_RE.match(line)
        if m:
            rows.append(
                {
                    "problem": m.group("problem"),
                    "solver": m.group("solver"),
                    "status": m.group("status"),
                    "outer": int(m.group("outer")),
                    "inner": int(m.group("inner")),
                    "grad_evals": int(m.group("grad_evals")),
                    "opt_res": float(m.group("opt_res")),
                    "feas_res": float(m.group("feas_res")),
                    "runtime_ms": float(m.group("runtime_ms")),
                }
            )
    if not rows:
        raise ValueError(f"No benchmark rows parsed from {md_path}")
    return rows


def load_json_results(json_path: Path) -> list[dict[str, Any]]:
    """Load results from a JSON dump."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"Expected non-empty JSON list in {json_path}")
    return data


def _active_solvers(
    rows: list[dict[str, Any]], include_clarabel: bool = True
) -> list[str]:
    present = {r["solver"] for r in rows}
    return [
        s
        for s in SOLVER_ORDER
        if s in present and (include_clarabel or s != "Clarabel")
    ]


def _dolan_more_curve(
    rows: list[dict[str, Any]],
    problems: list[str],
    solvers: list[str],
    metric_key: str,
    tau_max: float,
) -> dict[str, tuple[list[float], list[float]]]:
    """Compute Dolan-More step curves (taus, solved_fractions) per solver."""
    by_prob: dict[str, dict[str, dict[str, Any]]] = {p: {} for p in problems}
    for r in rows:
        if r["solver"] in solvers:
            by_prob[r["problem"]][r["solver"]] = r

    ratios: dict[str, list[float]] = {s: [] for s in solvers}
    for p in problems:
        converged_vals = [
            float(r[metric_key])
            for r in by_prob[p].values()
            if r["status"] == "CONVERGED"
        ]
        best = max(min(converged_vals), 1e-12) if converged_vals else 1.0
        for s in solvers:
            r_opt = by_prob[p].get(s)
            if r_opt is not None and r_opt["status"] == "CONVERGED":
                ratios[s].append(max(float(r_opt[metric_key]) / best, 1.0))
            else:
                ratios[s].append(math.inf)

    n_probs = len(problems)
    curves: dict[str, tuple[list[float], list[float]]] = {}
    for s in solvers:
        finite = sorted(v for v in ratios[s] if math.isfinite(v))
        xs = [1.0]
        ys = [sum(1 for v in finite if v <= 1.0) / n_probs]
        for v in finite:
            if v > xs[-1]:
                xs.append(v)
                ys.append(sum(1 for r in finite if r <= v) / n_probs)
        if xs[-1] < tau_max:
            xs.append(tau_max)
            ys.append(ys[-1])
        curves[s] = (xs, ys)
    return curves


def plot_performance_profiles(
    rows: list[dict[str, Any]],
    out_path: Path,
    title_prefix: str = "Maros–Meszaros",
) -> None:
    """Plot runtime and gradient-eval Dolan-More performance profiles."""
    problems = list(dict.fromkeys(r["problem"] for r in rows))
    tau_max = 1e4
    rt_solvers = _active_solvers(rows, include_clarabel=True)
    fo_solvers = _active_solvers(rows, include_clarabel=False)

    runtime_curves = _dolan_more_curve(
        rows, problems, rt_solvers, "runtime_ms", tau_max
    )
    grad_curves = _dolan_more_curve(
        rows, problems, fo_solvers, "grad_evals", tau_max
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    panels = [
        (
            axes[0],
            runtime_curves,
            rt_solvers,
            "Steady-State Runtime Profile",
            r"Runtime factor $\tau$",
        ),
        (
            axes[1],
            grad_curves,
            fo_solvers,
            "Gradient Evaluations Profile (First-Order)",
            r"Oracle eval factor $\tau$",
        ),
    ]

    for ax, curves, solvers, title, xlabel in panels:
        for s in solvers:
            xs, ys = curves[s]
            ax.step(
                xs,
                ys,
                where="post",
                color=SOLVER_COLORS[s],
                linewidth=2.0,
                label=s,
            )
            ax.scatter(
                [xs[0]],
                [ys[0]],
                color=SOLVER_COLORS[s],
                marker=SOLVER_MARKERS[s],
                s=28,
                zorder=5,
            )
        ax.set_xscale("log")
        ax.set_xlim(1.0, tau_max)
        ax.set_ylim(-0.02, 1.06)
        ax.set_xticks([1, 10, 100, 1000, 10000])
        ax.set_xticklabels(["1", "10", r"$10^2$", r"$10^3$", r"$10^4$"])
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Fraction of QPs solved")

    axes[0].legend(frameon=False, loc="lower right", ncol=2)
    fig.suptitle(
        f"{title_prefix} QP Profiles "
        f"({len(problems)} problems, tol = $10^{{-6}}$)",
        fontsize=12,
    )
    fig.savefig(out_path, format="svg")
    plt.close(fig)


def plot_breakdown(
    rows: list[dict[str, Any]],
    out_path: Path,
    title_prefix: str = "Maros–Meszaros",
) -> None:
    """Plot per-problem runtimes and gradient-eval vs runtime Pareto scatter."""
    problems = list(dict.fromkeys(r["problem"] for r in rows))
    short_labels = [p.split(" ")[0] for p in problems]
    y_positions = list(range(len(problems)))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.5, 5.4),
        gridspec_kw={"width_ratios": [1.35, 1.0]},
        constrained_layout=True,
    )

    ax_left = axes[0]
    rt_solvers = _active_solvers(rows, include_clarabel=True)
    mid = (len(rt_solvers) - 1) / 2.0
    for idx, s in enumerate(rt_solvers):
        offset = (idx - mid) * 0.095
        conv_x, conv_y = [], []
        fail_x, fail_y = [], []
        for p_idx, p in enumerate(problems):
            match = [r for r in rows if r["problem"] == p and r["solver"] == s]
            if not match:
                continue
            r = match[0]
            if r["status"] == "CONVERGED":
                conv_x.append(r["runtime_ms"])
                conv_y.append(p_idx + offset)
            else:
                fail_x.append(r["runtime_ms"])
                fail_y.append(p_idx + offset)

        ax_left.scatter(
            conv_x,
            conv_y,
            color=SOLVER_COLORS[s],
            marker=SOLVER_MARKERS[s],
            s=36,
            label=s,
            zorder=4,
        )
        if fail_x:
            ax_left.scatter(
                fail_x,
                fail_y,
                facecolors="none",
                edgecolors=SOLVER_COLORS[s],
                marker="X",
                s=46,
                linewidths=1.2,
                alpha=0.85,
                zorder=3,
            )

    ax_left.set_xscale("log")
    ax_left.set_yticks(y_positions)
    ax_left.set_yticklabels(short_labels)
    ax_left.invert_yaxis()
    ax_left.set_xlabel("Steady-state runtime (ms)")
    ax_left.set_title("Per-Problem Runtime (solid = converged, X = max iter)")
    ax_left.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax_left.set_xticks([0.01, 0.1, 1, 10, 100, 1000, 10000])
    ax_left.set_xticklabels(["0.01", "0.1", "1", "10", "100", "1k", "10k"])
    ax_left.legend(frameon=False, loc="lower right", ncol=2)

    ax_right = axes[1]
    for s in _active_solvers(rows, include_clarabel=False):
        pts = [
            r
            for r in rows
            if (
                r["solver"] == s
                and r["status"] == "CONVERGED"
                and r["grad_evals"] > 0
            )
        ]
        ax_right.scatter(
            [r["grad_evals"] for r in pts],
            [r["runtime_ms"] for r in pts],
            color=SOLVER_COLORS[s],
            marker=SOLVER_MARKERS[s],
            s=40,
            alpha=0.9,
            label=s,
            zorder=4,
        )

    ax_right.set_xscale("log")
    ax_right.set_yscale("log")
    ax_right.set_xlabel("Gradient evaluations")
    ax_right.set_ylabel("Steady-state runtime (ms)")
    ax_right.set_title("Oracle Cost vs. Runtime (Converged Runs)")

    fig.suptitle(
        f"{title_prefix} Breakdown "
        f"({len(problems)} QPs, float64 CPU, tol = $10^{{-6}}$)",
        fontsize=12,
    )
    fig.savefig(out_path, format="svg")
    plt.close(fig)


def plot_qafiro_trajectory(traj_path: Path, out_path: Path) -> None:
    """Plot optimality and feasibility trajectories on QAFIRO."""
    data = json.loads(traj_path.read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)

    panels = [
        (axes[0], "opt_residuals", "Optimality Residual Trajectory"),
        (axes[1], "feas_residuals", "Feasibility Residual Trajectory"),
    ]
    for ax, key, title in panels:
        for s in SOLVER_ORDER:
            entry = data.get(s)
            if entry is None:
                continue
            grad_list = entry["grad_evals"]
            val_list = entry[key]
            pts = [
                (grad_list[i], max(val_list[i], 1e-16))
                for i in range(min(len(grad_list), len(val_list)))
                if grad_list[i] > 0
            ]
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            tag = "conv" if entry["status"] == "CONVERGED" else "max iter"
            ax.plot(
                xs,
                ys,
                color=SOLVER_COLORS[s],
                linewidth=1.9,
                alpha=0.92,
                label=f"{s} ({tag})",
            )
            end_marker = (
                SOLVER_MARKERS[s] if entry["status"] == "CONVERGED" else "X"
            )
            ax.scatter(
                [xs[-1]],
                [ys[-1]],
                color=SOLVER_COLORS[s],
                marker=end_marker,
                s=42,
                zorder=5,
            )

        ax.axhline(
            1e-6, color="#bac2de", linestyle="--", linewidth=1.1, alpha=0.75
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Gradient evaluations")
        ax.set_ylabel("Residual")
        ax.set_title(title)

    axes[0].legend(frameon=False, loc="lower left", fontsize=8)
    fig.suptitle(
        r"Hardest Instance: `QAFIRO` ($n=32, m=27$, tol = $10^{-6}$)",
        fontsize=12,
    )
    fig.savefig(out_path, format="svg")
    plt.close(fig)


def main() -> None:
    """CLI entry point for generating Maros-Meszaros SVG plots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--markdown",
        type=Path,
        default=DEFAULT_MD,
        help="Markdown file containing the Full Results table.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional JSON results file (overrides --markdown).",
    )
    parser.add_argument(
        "--traj-json",
        type=Path,
        default=None,
        help="Optional QAFIRO trajectory JSON file.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=DEFAULT_PLOTS_DIR,
        help="Directory to write SVG plots.",
    )
    parser.add_argument(
        "--prefix",
        default="maros_meszaros",
        help="Output filename prefix for SVG plots.",
    )
    parser.add_argument(
        "--title-prefix",
        default="Maros–Meszaros",
        help="Title prefix for generated plots.",
    )
    args = parser.parse_args()

    _apply_mocha_style()
    args.plots_dir.mkdir(parents=True, exist_ok=True)
    rows = (
        load_json_results(args.json)
        if args.json
        else parse_markdown_results(args.markdown)
    )

    plot_performance_profiles(
        rows,
        args.plots_dir / f"{args.prefix}_profile.svg",
        title_prefix=args.title_prefix,
    )
    plot_breakdown(
        rows,
        args.plots_dir / f"{args.prefix}_breakdown.svg",
        title_prefix=args.title_prefix,
    )
    if args.traj_json is not None:
        plot_qafiro_trajectory(
            args.traj_json,
            args.plots_dir / "maros_meszaros_qafiro_trajectory.svg",
        )


if __name__ == "__main__":
    main()
