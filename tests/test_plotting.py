"""Tests for `proxax.plotting`, `Solution.plot`, and `HistoryTracker.plot`."""

from __future__ import annotations

import math
import sys
import types
import typing

import jax
import jax.numpy as jnp
import pytest

from proxax import px


class _FakeAxis:
  """Minimal Matplotlib `Axes` stand-in recording plotting calls."""

  def __init__(self, figure: _FakeFigure) -> None:
    self.figure = figure
    self.lines: list[tuple[list[typing.Any], list[typing.Any], str | None]] = []
    self.steps: list[tuple[list[typing.Any], list[typing.Any], str | None]] = []
    self.hlines: list[float] = []
    self.yscale: str | None = None
    self.xlabel: str | None = None
    self.ylabel: str | None = None
    self.title: str | None = None
    self.legend_called = False

  def plot(
      self,
      xs: typing.Sequence[typing.Any],
      ys: typing.Sequence[typing.Any],
      *,
      label: str | None = None,
      **_: typing.Any,
  ) -> None:
    self.lines.append((list(xs), list(ys), label))

  def step(
      self,
      xs: typing.Sequence[typing.Any],
      ys: typing.Sequence[typing.Any],
      *,
      label: str | None = None,
      **_: typing.Any,
  ) -> None:
    self.steps.append((list(xs), list(ys), label))

  def axhline(self, y: float, **_: typing.Any) -> None:
    self.hlines.append(float(y))

  def set_yscale(self, value: str) -> None:
    self.yscale = value

  def set_xlabel(self, label: str) -> None:
    self.xlabel = label

  def set_ylabel(self, label: str) -> None:
    self.ylabel = label

  def set_title(self, title: str) -> None:
    self.title = title

  def set_xlim(self, *_: typing.Any) -> None:
    pass

  def set_ylim(self, *_: typing.Any) -> None:
    pass

  def grid(self, *_: typing.Any, **__: typing.Any) -> None:
    pass

  def legend(self) -> None:
    self.legend_called = True


class _FakeFigure:
  """Minimal Matplotlib `Figure` stand-in."""

  def __init__(self, n_axes: int) -> None:
    self.axes = [_FakeAxis(self) for _ in range(n_axes)]
    self.title: str | None = None

  def suptitle(self, title: str) -> None:
    self.title = title


def _install_fake_pyplot(monkeypatch: pytest.MonkeyPatch) -> None:
  """Installs a lightweight fake `matplotlib.pyplot` in `sys.modules`."""
  plt_mod = types.ModuleType("matplotlib.pyplot")

  def _subplots(
      nrows: int = 1,
      ncols: int = 1,
      **_: typing.Any,
  ) -> tuple[_FakeFigure, typing.Any]:
    total = nrows * ncols
    fig = _FakeFigure(total)
    return fig, (fig.axes[0] if total == 1 else fig.axes)

  plt_mod.subplots = _subplots  # type: ignore[attr-defined]
  mpl_mod = types.ModuleType("matplotlib")
  mpl_mod.pyplot = plt_mod  # type: ignore[attr-defined]
  monkeypatch.setitem(sys.modules, "matplotlib", mpl_mod)
  monkeypatch.setitem(sys.modules, "matplotlib.pyplot", plt_mod)


def test_solution_track_history_and_plot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """`px.solve(..., track_history=True)` populates `sol.history` & `.plot()`."""
  _install_fake_pyplot(monkeypatch)
  a_mat = jnp.array([[3.0, 1.0], [1.0, 2.0]])
  b_vec = jnp.array([1.0, -2.0])
  prob = px.CompositeProblem(
      smooth=lambda x: 0.5 * jnp.sum((a_mat @ x - b_vec) ** 2),
      proximal=px.prox.L1(0.1),
  )

  with pytest.raises(ValueError, match="jit=False"):
    px.solve(prob, jnp.array([5.0, -5.0]), track_history=True)

  sol_no_hist = px.solve(prob, jnp.array([5.0, -5.0]), method="apd")
  with pytest.raises(ValueError, match="no recorded history"):
    sol_no_hist.plot()

  sol = px.solve(
      prob,
      jnp.array([5.0, -5.0]),
      method="apd",
      opt_tol=1e-6,
      jit=False,
      track_history=True,
  )
  assert sol.history is not None
  fig, axes = sol.plot(x_axis="evals", title="APD")
  assert fig.title == "APD"
  assert len(axes) == 2
  assert axes[1].yscale == "log"
  assert len(axes[1].hlines) == 1
  assert axes[-1].xlabel == "Oracle evals (grad + prox)"


def test_plot_histories_constrained_and_time_axis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Constrained histories render 3 panels and multi-run legend."""
  _install_fake_pyplot(monkeypatch)
  a_mat = jnp.array([[2.0, 1.0], [1.0, 2.0]])
  b_vec = jnp.array([1.0, -1.0])
  c_mat = jnp.array([[1.0, 1.0]])
  d_vec = jnp.array([1.0])
  prob = px.CompositeProblem(
      smooth=lambda x: 0.5 * jnp.sum((a_mat @ x - b_vec) ** 2),
      constraints=px.constraints.equality(c_mat, d_vec),
  )

  sol1 = px.solve(
      prob,
      jnp.zeros(2),
      method="ialm",
      opt_tol=1e-4,
      feas_tol=1e-4,
      jit=False,
      track_history=True,
  )
  sol2 = px.solve(
      prob,
      jnp.zeros(2),
      method="iaipal",
      opt_tol=1e-4,
      feas_tol=1e-4,
      jit=False,
      track_history=True,
  )
  assert sol1.history is not None and sol2.history is not None

  fig, axes = px.plotting.plot_histories(
      {"IALM": sol1.history, "IAIPAL": sol2.history},
      x_axis="time",
  )
  assert len(axes) == 3
  assert axes[0].legend_called
  assert axes[-1].xlabel == "Elapsed time (s)"
  assert axes[2].ylabel == "Feas. residual"
  assert fig is axes[0].figure


def test_plot_spectrum_and_performance_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Singular-value spectrum and Dolan-Moré performance profile render."""
  _install_fake_pyplot(monkeypatch)
  mat = jnp.diag(jnp.array([5.0, 2.0, 0.1]))
  _, axis = px.plotting.plot_spectrum(mat, title="Spectrum")
  assert axis.yscale == "log"
  assert axis.title == "Spectrum"
  assert len(axis.lines[0][1]) == 3

  with pytest.raises(ValueError, match="2D matrix"):
    px.plotting.plot_spectrum(jnp.zeros(4))

  _, prof_ax = px.plotting.plot_performance_profile(
      {"A": [1.0, 4.0, math.inf], "B": [2.0, 2.0, 3.0]},
      title="Benchmark",
  )
  assert prof_ax.legend_called
  assert len(prof_ax.steps) == 2


def test_missing_matplotlib_error_and_jit_compat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Missing `matplotlib` gives an install hint; `Solution` stays jittable."""
  monkeypatch.setitem(sys.modules, "matplotlib", None)
  monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
  with pytest.raises(ImportError, match="proxax\\[plot\\]"):
    px.plotting.plot_spectrum(jnp.eye(2))

  @jax.jit
  def _roundtrip(x: jax.Array) -> px.Solution:
    return px.solvers.SolverTracker.from_init(
        x=x,
        subgrad=x,
        opt_res=jnp.asarray(0.0),
        obj=jnp.asarray(0.0),
        opt_tol=1e-5,
    ).to_solution()

  sol = _roundtrip(jnp.zeros(2))
  assert sol.history is None
