"""Optional Matplotlib visualization utilities for solver trajectories."""

from __future__ import annotations

import math
import typing
from collections import abc

import jax.numpy as jnp

if typing.TYPE_CHECKING:
  from proxax.solvers import base

XAxis = typing.Literal["iter", "time", "evals"]

_MOCHA_STYLE: dict[str, typing.Any] = {
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
    "axes.prop_cycle": (
        "cycler('color', ['#89b4fa', '#f38ba8', '#a6e3a1', "
        "'#fab387', '#cba6f7', '#94e2d5'])"
    ),
}


def _style_context(plt: typing.Any, style: str | None) -> typing.Any:
  """Returns a Matplotlib context manager for `style` (supports `'mocha'`)."""
  if style is None:
    return plt.rc_context({})
  avail = getattr(getattr(plt, "style", None), "available", ())
  if style == "mocha" and "mocha" not in avail:
    # Build cycler lazily without requiring `cycler` at import time.
    rc = dict(_MOCHA_STYLE)
    try:
      from cycler import cycler  # noqa: PLC0415

      rc["axes.prop_cycle"] = cycler(
          color=[
              "#89b4fa",
              "#f38ba8",
              "#a6e3a1",
              "#fab387",
              "#cba6f7",
              "#94e2d5",
          ]
      )
    except ImportError:
      rc.pop("axes.prop_cycle", None)
    return plt.rc_context(rc)
  return plt.style.context(style)


def _import_pyplot() -> typing.Any:
  """Lazily imports `matplotlib.pyplot` or raises a clear install hint."""
  try:
    import matplotlib.pyplot as plt  # noqa: PLC0415
  except ImportError as exc:
    raise ImportError(
        "Plotting requires `matplotlib`. "
        "Install with `pip install proxax[plot]`."
    ) from exc
  return plt


def _x_series(
    history: base.HistoryTracker,
    x_axis: XAxis,
) -> tuple[list[int] | list[float], str]:
  """Extracts the requested x-axis values and axis label from `history`."""
  if x_axis == "iter":
    return list(history.outer_iters), "Outer iteration"
  if x_axis == "time":
    return list(history.elapsed_times), "Elapsed time (s)"
  if x_axis == "evals":
    evals = [
        g + p
        for g, p in zip(history.grad_evals, history.prox_evals, strict=True)
    ]
    return evals, "Oracle evals (grad + prox)"
  raise ValueError(
      f"Unknown x_axis={x_axis!r}; expected 'iter', 'time', or 'evals'."
  )


def _has_feasibility(histories: abc.Iterable[base.HistoryTracker]) -> bool:
  """Returns True if any history has non-zero feasibility metrics."""
  return any(
      any(v > 0.0 for v in h.feas_residuals)
      or any(t > 0.0 for t in h.feas_tols)
      for h in histories
  )


def _normalize_axes(
    ax: typing.Any,
    n_panels: int,
    plt: typing.Any,
) -> tuple[typing.Any, list[typing.Any]]:
  """Creates or normalizes a list of `n_panels` Matplotlib axes."""
  if ax is None:
    fig, raw_axes = plt.subplots(
        n_panels,
        1,
        figsize=(7.0, 2.6 * n_panels),
        sharex=True,
        constrained_layout=True,
    )
    axes_list = list(raw_axes) if n_panels > 1 else [raw_axes]
    return fig, axes_list
  axes_list = list(ax) if isinstance(ax, (list, tuple)) else list(ax.flat)
  if len(axes_list) < n_panels:
    raise ValueError(
        f"Expected at least {n_panels} axes; got {len(axes_list)}."
    )
  return axes_list[0].figure, axes_list[:n_panels]


def _plot_residual_panel(
    axis: typing.Any,
    xs: list[int] | list[float],
    residuals: list[float],
    tols: list[float],
    *,
    ylabel: str,
    label: str | None,
    show_tol: bool,
    alpha: float,
) -> None:
  """Plots a residual curve on log scale with an optional tolerance line."""
  axis.plot(xs, residuals, label=label, alpha=alpha, linewidth=1.2)
  if any(r > 0.0 for r in residuals):
    axis.set_yscale("log")
  if show_tol and tols and tols[-1] > 0.0:
    axis.axhline(tols[-1], linestyle="--", linewidth=0.8, alpha=0.6)
  axis.set_ylabel(ylabel)
  axis.grid(True, which="both", alpha=0.3)


def plot_histories(
    histories: abc.Mapping[str, base.HistoryTracker],
    *,
    x_axis: XAxis = "iter",
    title: str | None = None,
    show_tol: bool = True,
    log_x: bool = False,
    alpha: float = 0.75,
    style: str | None = "mocha",
    ax: typing.Any = None,
) -> tuple[typing.Any, list[typing.Any]]:
  """Plots and compares multiple `HistoryTracker` trajectories.

  Args:
    histories: Mapping from solver/run label to populated `HistoryTracker`.
    x_axis: Horizontal axis (`'iter'`, `'time'`, or `'evals'`).
    title: Optional figure title.
    show_tol: Whether to draw dashed target tolerance lines.
    log_x: Whether to use a logarithmic x-axis (helpful for wide eval ranges).
    alpha: Line opacity in `[0, 1]`.
    style: Matplotlib style name (defaults to Catppuccin `'mocha'`).
    ax: Optional pre-existing sequence of 2 (or 3) Matplotlib `Axes`.

  Returns:
    `(fig, axes)` tuple from Matplotlib.
  """
  if not histories:
    raise ValueError("`histories` must be non-empty.")
  for name, hist in histories.items():
    if len(hist) == 0:
      raise ValueError(f"HistoryTracker {name!r} has no recorded steps.")

  plt = _import_pyplot()
  include_feas = _has_feasibility(histories.values())
  n_panels = 3 if include_feas else 2
  with _style_context(plt, style):
    fig, axes = _normalize_axes(ax, n_panels, plt)

    x_label = "Outer iteration"
    for name, hist in histories.items():
      xs, x_label = _x_series(hist, x_axis)
      axes[0].plot(xs, hist.obj_vals, label=name, alpha=alpha, linewidth=1.2)
      _plot_residual_panel(
          axes[1],
          xs,
          hist.opt_residuals,
          hist.opt_tols,
          ylabel="Opt. residual",
          label=name,
          show_tol=show_tol,
          alpha=alpha,
      )
      if include_feas:
        _plot_residual_panel(
            axes[2],
            xs,
            hist.feas_residuals,
            hist.feas_tols,
            ylabel="Feas. residual",
            label=name,
            show_tol=show_tol,
            alpha=alpha,
        )

    if all(v > 0.0 for h in histories.values() for v in h.obj_vals):
      axes[0].set_yscale("log")
    axes[0].set_ylabel("Objective")
    axes[0].grid(True, which="both", alpha=0.3)
    if len(histories) > 1:
      axes[0].legend()
    if log_x:
      axes[-1].set_xscale("log")
    axes[-1].set_xlabel(x_label)
    if title is not None:
      fig.suptitle(title)
  return fig, axes


def plot_history(
    history: base.HistoryTracker,
    *,
    x_axis: XAxis = "iter",
    title: str | None = None,
    show_tol: bool = True,
    log_x: bool = False,
    alpha: float = 0.75,
    style: str | None = "mocha",
    ax: typing.Any = None,
) -> tuple[typing.Any, list[typing.Any]]:
  """Plots objective and residual convergence for a single `HistoryTracker`.

  Args:
    history: Populated `HistoryTracker` instance.
    x_axis: Horizontal axis (`'iter'`, `'time'`, or `'evals'`).
    title: Optional figure title.
    show_tol: Whether to draw dashed target tolerance lines.
    log_x: Whether to use a logarithmic x-axis.
    alpha: Line opacity in `[0, 1]`.
    style: Matplotlib style name (defaults to Catppuccin `'mocha'`).
    ax: Optional pre-existing sequence of 2 (or 3) Matplotlib `Axes`.

  Returns:
    `(fig, axes)` tuple from Matplotlib.
  """
  if len(history) == 0:
    raise ValueError("Cannot plot an empty `HistoryTracker`.")
  return plot_histories(
      {"run": history},
      x_axis=x_axis,
      title=title,
      show_tol=show_tol,
      log_x=log_x,
      alpha=alpha,
      style=style,
      ax=ax,
  )


def plot_spectrum(
    target: typing.Any,
    *,
    title: str | None = None,
    log_y: bool = True,
    alpha: float = 0.75,
    style: str | None = "mocha",
    ax: typing.Any = None,
) -> tuple[typing.Any, typing.Any]:
  """Plots the singular-value spectrum of a 2D matrix or `Solution`.

  Args:
    target: A 2D array or a `Solution` whose `.x` is a 2D array.
    title: Optional axis title.
    log_y: Whether to use a logarithmic y-axis when values are positive.
    alpha: Line opacity in `[0, 1]`.
    style: Matplotlib style name (defaults to Catppuccin `'mocha'`).
    ax: Optional pre-existing Matplotlib `Axes`.

  Returns:
    `(fig, ax)` tuple from Matplotlib.
  """
  is_sol = hasattr(target, "x") and hasattr(target, "stats")
  mat = target.x if is_sol else target
  arr = jnp.asarray(mat)
  if arr.ndim != 2:
    raise ValueError(
        f"`plot_spectrum` expects a 2D matrix; got shape {arr.shape}."
    )
  plt = _import_pyplot()
  with _style_context(plt, style):
    if ax is None:
      fig, axis = plt.subplots(figsize=(6.5, 3.5), constrained_layout=True)
    else:
      axis = ax
      fig = axis.figure

    svals = [float(s) for s in jnp.linalg.svdvals(arr)]
    indices = list(range(1, len(svals) + 1))
    axis.plot(
        indices,
        svals,
        marker="o",
        markersize=3.8,
        alpha=alpha,
        linewidth=1.2,
    )
    if log_y and all(s > 0.0 for s in svals):
      axis.set_yscale("log")
    axis.set_xlabel("Singular value index")
    axis.set_ylabel("Singular value")
    axis.grid(True, which="both", alpha=0.3)
    if title is not None:
      axis.set_title(title)
  return fig, axis


def plot_performance_profile(
    results: abc.Mapping[str, abc.Sequence[float]],
    *,
    tau_max: float | None = None,
    title: str | None = None,
    alpha: float = 0.75,
    style: str | None = "mocha",
    ax: typing.Any = None,
) -> tuple[typing.Any, typing.Any]:
  """Plots a Dolan-Moré performance profile across benchmark problems.

  Args:
    results: Mapping from solver name to a sequence of per-problem costs (e.g.
      runtime or evals). Values that are `<= 0`, `nan`, or `inf` are treated as
      unsolved.
    tau_max: Optional upper bound on the performance ratio axis `tau >= 1`.
    title: Optional axis title.
    alpha: Line opacity in `[0, 1]`.
    style: Matplotlib style name (defaults to Catppuccin `'mocha'`).
    ax: Optional pre-existing Matplotlib `Axes`.

  Returns:
    `(fig, ax)` tuple from Matplotlib.
  """
  if not results:
    raise ValueError("`results` must be non-empty.")
  lengths = {len(seq) for seq in results.values()}
  if len(lengths) != 1 or 0 in lengths:
    raise ValueError(
        "All solvers in `results` must have the same non-zero problem count."
    )
  n_probs = next(iter(lengths))

  best_per_prob: list[float] = []
  for p_idx in range(n_probs):
    valid = [
        float(seq[p_idx])
        for seq in results.values()
        if math.isfinite(float(seq[p_idx])) and float(seq[p_idx]) > 0.0
    ]
    best_per_prob.append(min(valid) if valid else math.inf)

  ratios: dict[str, list[float]] = {}
  finite_ratios: list[float] = [1.0]
  for name, seq in results.items():
    r_list: list[float] = []
    for p_idx in range(n_probs):
      val = float(seq[p_idx])
      best = best_per_prob[p_idx]
      if math.isfinite(val) and val > 0.0 and math.isfinite(best):
        ratio = val / best
        r_list.append(ratio)
        finite_ratios.append(ratio)
      else:
        r_list.append(math.inf)
    ratios[name] = r_list

  max_tau = tau_max if tau_max is not None else max(finite_ratios) * 1.05
  plt = _import_pyplot()
  with _style_context(plt, style):
    if ax is None:
      fig, axis = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)
    else:
      axis = ax
      fig = axis.figure

    for name, r_list in ratios.items():
      solved = sorted(r for r in r_list if math.isfinite(r) and r <= max_tau)
      xs = [1.0, *solved, max_tau]
      ys = [0.0]
      for idx in range(len(solved)):
        ys.append((idx + 1) / n_probs)
      ys.append(len(solved) / n_probs)
      axis.step(xs, ys, where="post", label=name, alpha=alpha, linewidth=1.2)

    axis.set_xlim(1.0, max_tau)
    axis.set_ylim(0.0, 1.05)
    axis.set_xlabel(r"Performance ratio $\tau$")
    axis.set_ylabel(r"Fraction solved $\rho_s(\tau)$")
    axis.grid(True, alpha=0.3)
    axis.legend()
    if title is not None:
      axis.set_title(title)
  return fig, axis


def plot_eval_breakdown(
    solutions: abc.Mapping[str, base.Solution],
    *,
    runtimes: abc.Mapping[str, float] | None = None,
    title: str | None = None,
    alpha: float = 0.8,
    style: str | None = "mocha",
    ax: typing.Any = None,
) -> tuple[typing.Any, typing.Any]:
  """Plots `smooth`/`grad`/`prox` evals and optional warm-started runtimes.

  Args:
    solutions: Mapping from solver name to `Solution`.
    runtimes: Optional mapping from solver name to steady-state runtime (s).
    title: Optional figure/axis title.
    alpha: Bar opacity in `[0, 1]`.
    style: Matplotlib style name (defaults to Catppuccin `'mocha'`).
    ax: Optional pre-existing `Axes` (1 or 2 axes depending on `runtimes`).

  Returns:
    `(fig, axes)` tuple from Matplotlib.
  """
  if not solutions:
    raise ValueError("`solutions` must be non-empty.")
  names = list(solutions.keys())
  smooth_vals = [int(solutions[n].stats.smooth_evals) for n in names]
  grad_vals = [int(solutions[n].stats.grad_evals) for n in names]
  prox_vals = [int(solutions[n].stats.prox_evals) for n in names]

  n_panels = 2 if runtimes is not None else 1
  plt = _import_pyplot()
  with _style_context(plt, style):
    if ax is None:
      fig, raw_axes = plt.subplots(
          1,
          n_panels,
          figsize=(5.4 * n_panels, 3.6),
          constrained_layout=True,
      )
      axes_list = list(raw_axes) if n_panels > 1 else [raw_axes]
    else:
      axes_list = (
          [ax]
          if n_panels == 1 and not isinstance(ax, (list, tuple))
          else list(ax)
      )
      fig = axes_list[0].figure

    idx = list(range(len(names)))
    width = 0.25
    ax0 = axes_list[0]
    ax0.bar(
        [i - width for i in idx],
        smooth_vals,
        width=width,
        label="smooth",
        alpha=alpha,
    )
    ax0.bar(idx, grad_vals, width=width, label="grad", alpha=alpha)
    ax0.bar(
        [i + width for i in idx],
        prox_vals,
        width=width,
        label="prox",
        alpha=alpha,
    )
    ax0.set_xticks(idx)
    ax0.set_xticklabels(names)
    if all(v > 0 for v in (*smooth_vals, *grad_vals, *prox_vals)):
      ax0.set_yscale("log")
    ax0.set_ylabel("Evaluations")
    ax0.set_title("Oracle Evaluation Breakdown")
    ax0.grid(True, axis="y", which="both", alpha=0.3)
    ax0.legend()

    if runtimes is not None:
      ax1 = axes_list[1]
      rt_vals = [float(runtimes[n]) * 1e3 for n in names]
      ax1.bar(idx, rt_vals, width=0.5, alpha=alpha)
      ax1.set_xticks(idx)
      ax1.set_xticklabels(names)
      ax1.set_ylabel("Steady-state runtime (ms)")
      ax1.set_title("Warm-Started JIT Runtime")
      ax1.grid(True, axis="y", alpha=0.3)

    if title is not None:
      if n_panels > 1:
        fig.suptitle(title)
      else:
        ax0.set_title(title)
  return fig, (axes_list[0] if n_panels == 1 else axes_list)


def plot_tolerance_scaling(
    tols: abc.Sequence[float],
    evals_by_solver: abc.Mapping[str, abc.Sequence[float]],
    *,
    title: str | None = None,
    alpha: float = 0.75,
    style: str | None = "mocha",
    ax: typing.Any = None,
) -> tuple[typing.Any, typing.Any]:
  """Plots empirical complexity (oracle evals vs target tolerance) on log-log.

  Args:
    tols: Sequence of target tolerances (e.g. `[1e-2, 1e-3, 1e-4, 1e-5]`).
    evals_by_solver: Mapping from solver name to eval counts matching `tols`.
    title: Optional axis title.
    alpha: Line opacity in `[0, 1]`.
    style: Matplotlib style name (defaults to Catppuccin `'mocha'`).
    ax: Optional pre-existing Matplotlib `Axes`.

  Returns:
    `(fig, ax)` tuple from Matplotlib.
  """
  if not tols or not evals_by_solver:
    raise ValueError("`tols` and `evals_by_solver` must be non-empty.")
  plt = _import_pyplot()
  with _style_context(plt, style):
    if ax is None:
      fig, axis = plt.subplots(figsize=(6.5, 3.8), constrained_layout=True)
    else:
      axis = ax
      fig = axis.figure

    for name, vals in evals_by_solver.items():
      if len(vals) != len(tols):
        raise ValueError(
            f"Solver {name!r} has {len(vals)} points; expected {len(tols)}."
        )
      axis.plot(
          tols,
          vals,
          marker="o",
          markersize=3.8,
          label=name,
          alpha=alpha,
          linewidth=1.2,
      )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.invert_xaxis()
    axis.set_xlabel("Target tolerance")
    axis.set_ylabel("Oracle evals (grad + prox)")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    if title is not None:
      axis.set_title(title)
  return fig, axis


def plot_phase_trajectory(
    trajectories: abc.Mapping[
        str, tuple[abc.Sequence[float], abc.Sequence[float]]
    ],
    *,
    optimum: tuple[float, float] | None = (0.0, 0.0),
    title: str | None = None,
    alpha: float = 0.75,
    style: str | None = "mocha",
    ax: typing.Any = None,
) -> tuple[typing.Any, typing.Any]:
  """Plots 2D `(x_k, y_k)` saddle-point or iterate phase trajectories.

  Args:
    trajectories: Mapping from solver name to `(xs, ys)` coordinate sequences.
    optimum: Optional `(x*, y*)` reference point to mark with a star.
    title: Optional axis title.
    alpha: Line opacity in `[0, 1]`.
    style: Matplotlib style name (defaults to Catppuccin `'mocha'`).
    ax: Optional pre-existing Matplotlib `Axes`.

  Returns:
    `(fig, ax)` tuple from Matplotlib.
  """
  if not trajectories:
    raise ValueError("`trajectories` must be non-empty.")
  plt = _import_pyplot()
  with _style_context(plt, style):
    if ax is None:
      fig, axis = plt.subplots(figsize=(5.5, 4.8), constrained_layout=True)
    else:
      axis = ax
      fig = axis.figure

    for name, (xs, ys) in trajectories.items():
      axis.plot(
          xs,
          ys,
          marker=".",
          markersize=4.0,
          label=name,
          alpha=alpha,
          linewidth=1.2,
      )
    if optimum is not None:
      axis.scatter(
          [optimum[0]],
          [optimum[1]],
          marker="*",
          s=90,
          zorder=5,
          label="Optimum",
      )
    axis.set_xlabel(r"$x$")
    axis.set_ylabel(r"$y$")
    axis.grid(True, alpha=0.3)
    axis.legend()
    if title is not None:
      axis.set_title(title)
  return fig, axis


def animate_phase_trajectory(
    trajectories: abc.Mapping[
        str, tuple[abc.Sequence[float], abc.Sequence[float]]
    ],
    *,
    optimum: tuple[float, float] | None = (0.0, 0.0),
    title: str | None = None,
    alpha: float = 0.75,
    tail: int | None = None,
    interval_ms: int = 80,
    stride: int = 1,
    save_path: str | None = None,
    dpi: int = 150,
    style: str | None = "mocha",
    ax: typing.Any = None,
) -> tuple[typing.Any, typing.Any, typing.Any]:
  """Animates 2D `(x_k, y_k)` phase trajectories as a Matplotlib animation.

  Args:
    trajectories: Mapping from solver name to `(xs, ys)` coordinate sequences.
    optimum: Optional `(x*, y*)` reference point to mark with a star.
    title: Optional axis title.
    alpha: Line opacity in `[0, 1]`.
    tail: Optional trailing window length in steps (`None` keeps full path).
    interval_ms: Delay between frames in milliseconds.
    stride: Step stride between animation frames (`>= 1`).
    save_path: Optional output path (e.g. `.gif`) saved via `PillowWriter`.
    dpi: Output resolution when `save_path` is provided.
    style: Matplotlib style name (defaults to Catppuccin `'mocha'`).
    ax: Optional pre-existing Matplotlib `Axes`.

  Returns:
    `(fig, ax, anim)` tuple with the `FuncAnimation` instance.
  """
  if not trajectories:
    raise ValueError("`trajectories` must be non-empty.")
  if stride < 1:
    raise ValueError("`stride` must be >= 1.")
  plt = _import_pyplot()
  try:
    from matplotlib import animation as mpl_anim  # noqa: PLC0415
  except ImportError as exc:
    raise ImportError(
        "Animation requires `matplotlib.animation`. "
        "Install with `pip install proxax[plot]`."
    ) from exc

  series = {
      name: ([float(v) for v in xs], [float(v) for v in ys])
      for name, (xs, ys) in trajectories.items()
  }
  max_len = max(len(xs) for xs, _ in series.values())
  all_x = [x for xs, _ in series.values() for x in xs]
  all_y = [y for _, ys in series.values() for y in ys]
  if optimum is not None:
    all_x.append(float(optimum[0]))
    all_y.append(float(optimum[1]))
  x_span = max(max(all_x) - min(all_x), 1e-6)
  y_span = max(max(all_y) - min(all_y), 1e-6)
  frames = list(range(0, max_len, stride))
  if frames[-1] != max_len - 1:
    frames.append(max_len - 1)

  with _style_context(plt, style):
    if ax is None:
      fig, axis = plt.subplots(figsize=(5.5, 4.8), constrained_layout=True)
    else:
      axis = ax
      fig = axis.figure

    lines: dict[str, typing.Any] = {}
    heads: dict[str, typing.Any] = {}
    for name in series:
      (line,) = axis.plot([], [], label=name, alpha=alpha, linewidth=1.2)
      color = line.get_color()
      (head,) = axis.plot([], [], marker="o", markersize=5.0, color=color)
      lines[name] = line
      heads[name] = head
    if optimum is not None:
      axis.scatter(
          [optimum[0]],
          [optimum[1]],
          marker="*",
          s=90,
          zorder=5,
          label="Optimum",
      )
    axis.set_xlim(min(all_x) - 0.1 * x_span, max(all_x) + 0.1 * x_span)
    axis.set_ylim(min(all_y) - 0.1 * y_span, max(all_y) + 0.1 * y_span)
    axis.set_xlabel(r"$x$")
    axis.set_ylabel(r"$y$")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="upper right")
    if title is not None:
      axis.set_title(title)
    step_text = axis.text(
        0.03, 0.96, "", transform=axis.transAxes, va="top", fontsize=9
    )

    def _update(step_idx: int) -> list[typing.Any]:
      artists: list[typing.Any] = [step_text]
      for name, (xs, ys) in series.items():
        end = min(step_idx + 1, len(xs))
        start = 0 if tail is None else max(0, end - tail)
        lines[name].set_data(xs[start:end], ys[start:end])
        heads[name].set_data([xs[end - 1]], [ys[end - 1]])
        artists.extend([lines[name], heads[name]])
      step_text.set_text(f"step {step_idx}")
      return artists

    anim = mpl_anim.FuncAnimation(
        fig,
        _update,
        frames=frames,
        interval=interval_ms,
        blit=False,
    )
    if save_path is not None:
      fps = max(1, round(1000 / max(interval_ms, 1)))
      anim.save(
          save_path,
          writer=mpl_anim.PillowWriter(fps=fps),
          dpi=dpi,
          savefig_kwargs={"facecolor": fig.get_facecolor()},
      )
  return fig, axis, anim


def plot_sparse_recovery(
    estimates: abc.Mapping[str, typing.Any],
    *,
    truth: typing.Any = None,
    title: str | None = None,
    alpha: float = 0.75,
    style: str | None = "mocha",
    ax: typing.Any = None,
) -> tuple[typing.Any, typing.Any]:
  """Plots recovered sparse vectors side-by-side against optional ground truth.

  Args:
    estimates: Mapping from penalty/solver name to 1D recovered vector.
    truth: Optional 1D ground-truth sparse vector.
    title: Optional axis title.
    alpha: Marker/line opacity in `[0, 1]`.
    style: Matplotlib style name (defaults to Catppuccin `'mocha'`).
    ax: Optional pre-existing Matplotlib `Axes`.

  Returns:
    `(fig, ax)` tuple from Matplotlib.
  """
  if not estimates:
    raise ValueError("`estimates` must be non-empty.")
  plt = _import_pyplot()
  with _style_context(plt, style):
    if ax is None:
      fig, axis = plt.subplots(figsize=(6.8, 3.6), constrained_layout=True)
    else:
      axis = ax
      fig = axis.figure

    if truth is not None:
      truth_vals = [float(v) for v in jnp.asarray(truth).ravel()]
      axis.stem(
          list(range(len(truth_vals))),
          truth_vals,
          linefmt="--",
          markerfmt="o",
          basefmt=" ",
          label="Ground truth",
      )
    markers = ["s", "^", "D", "v", "x"]
    for idx, (name, vec) in enumerate(estimates.items()):
      vals = [float(v) for v in jnp.asarray(vec).ravel()]
      axis.plot(
          list(range(len(vals))),
          vals,
          linestyle="None",
          marker=markers[idx % len(markers)],
          markersize=5.0,
          alpha=alpha,
          label=name,
      )
    axis.axhline(0.0, linewidth=0.8, alpha=0.4)
    axis.set_xlabel("Coordinate index")
    axis.set_ylabel("Coefficient value")
    axis.grid(True, alpha=0.3)
    axis.legend()
    if title is not None:
      axis.set_title(title)
  return fig, axis
