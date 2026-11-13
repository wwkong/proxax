"""Optional Matplotlib visualization utilities for solver trajectories."""

from __future__ import annotations

import math
import typing
from collections import abc

import jax.numpy as jnp

if typing.TYPE_CHECKING:
  from proxax.solvers import base

XAxis = typing.Literal["iter", "time", "evals"]


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
) -> None:
  """Plots a residual curve on log scale with an optional tolerance line."""
  axis.plot(xs, residuals, label=label)
  if any(r > 0.0 for r in residuals):
    axis.set_yscale("log")
  if show_tol and tols and tols[-1] > 0.0:
    axis.axhline(tols[-1], linestyle="--", linewidth=1.0, alpha=0.7)
  axis.set_ylabel(ylabel)
  axis.grid(True, which="both", alpha=0.3)


def plot_histories(
    histories: abc.Mapping[str, base.HistoryTracker],
    *,
    x_axis: XAxis = "iter",
    title: str | None = None,
    show_tol: bool = True,
    ax: typing.Any = None,
) -> tuple[typing.Any, list[typing.Any]]:
  """Plots and compares multiple `HistoryTracker` trajectories.

  Args:
    histories: Mapping from solver/run label to populated `HistoryTracker`.
    x_axis: Horizontal axis (`'iter'`, `'time'`, or `'evals'`).
    title: Optional figure title.
    show_tol: Whether to draw dashed target tolerance lines.
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
  fig, axes = _normalize_axes(ax, n_panels, plt)

  x_label = "Outer iteration"
  for name, hist in histories.items():
    xs, x_label = _x_series(hist, x_axis)
    axes[0].plot(xs, hist.obj_vals, label=name)
    _plot_residual_panel(
        axes[1],
        xs,
        hist.opt_residuals,
        hist.opt_tols,
        ylabel="Opt. residual",
        label=name,
        show_tol=show_tol,
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
      )

  axes[0].set_ylabel("Objective")
  axes[0].grid(True, alpha=0.3)
  if len(histories) > 1:
    axes[0].legend()
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
    ax: typing.Any = None,
) -> tuple[typing.Any, list[typing.Any]]:
  """Plots objective and residual convergence for a single `HistoryTracker`.

  Args:
    history: Populated `HistoryTracker` instance.
    x_axis: Horizontal axis (`'iter'`, `'time'`, or `'evals'`).
    title: Optional figure title.
    show_tol: Whether to draw dashed target tolerance lines.
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
      ax=ax,
  )


def plot_spectrum(
    target: typing.Any,
    *,
    title: str | None = None,
    log_y: bool = True,
    ax: typing.Any = None,
) -> tuple[typing.Any, typing.Any]:
  """Plots the singular-value spectrum of a 2D matrix or `Solution`.

  Args:
    target: A 2D array or a `Solution` whose `.x` is a 2D array.
    title: Optional axis title.
    log_y: Whether to use a logarithmic y-axis when values are positive.
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
  if ax is None:
    fig, axis = plt.subplots(figsize=(6.5, 3.5), constrained_layout=True)
  else:
    axis = ax
    fig = axis.figure

  svals = [float(s) for s in jnp.linalg.svdvals(arr)]
  indices = list(range(1, len(svals) + 1))
  axis.plot(indices, svals, marker="o", markersize=4.0)
  if log_y and any(s > 0.0 for s in svals):
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
    ax: typing.Any = None,
) -> tuple[typing.Any, typing.Any]:
  """Plots a Dolan-Moré performance profile across benchmark problems.

  Args:
    results: Mapping from solver name to a sequence of per-problem costs (e.g.
      runtime or evals). Values that are `<= 0`, `nan`, or `inf` are treated as
      unsolved.
    tau_max: Optional upper bound on the performance ratio axis `tau >= 1`.
    title: Optional axis title.
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
    axis.step(xs, ys, where="post", label=name)

  axis.set_xlim(1.0, max_tau)
  axis.set_ylim(0.0, 1.05)
  axis.set_xlabel(r"Performance ratio $\tau$")
  axis.set_ylabel(r"Fraction solved $\rho_s(\tau)$")
  axis.grid(True, alpha=0.3)
  axis.legend()
  if title is not None:
    axis.set_title(title)
  return fig, axis
