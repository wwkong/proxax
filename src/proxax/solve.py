"""Top-level `solve` dispatcher for unconstrained and constrained problems."""

from __future__ import annotations

import dataclasses
import time
import typing

import jax

from proxax import _typing
from proxax import problem as problem_mod
from proxax.oracle import block as block_mod
from proxax.oracle import minmax
from proxax.oracle import spectral
from proxax.solvers import base
from proxax.solvers.block import admm
from proxax.solvers.block import dp_admm
from proxax.solvers.block import p_adapt_admm
from proxax.solvers.constrained import iaipal
from proxax.solvers.constrained import ialm
from proxax.solvers.constrained import qp_aipp
from proxax.solvers.constrained import qp_apd
from proxax.solvers.constrained import s_prox_alm
from proxax.solvers.minmax import extra_gradient
from proxax.solvers.minmax import pdhg
from proxax.solvers.minmax import prox_point
from proxax.solvers.unconstrained import ac_acg
from proxax.solvers.unconstrained import adap_apg
from proxax.solvers.unconstrained import adap_fista
from proxax.solvers.unconstrained import ag
from proxax.solvers.unconstrained import aipp
from proxax.solvers.unconstrained import apd
from proxax.solvers.unconstrained import da_icg
from proxax.solvers.unconstrained import ecg
from proxax.solvers.unconstrained import ia_icg
from proxax.solvers.unconstrained import nc_fista
from proxax.solvers.unconstrained import panoc
from proxax.solvers.unconstrained import upfag

_SolverFactory = typing.Callable[[float, float], typing.Any]

_BLOCK_METHODS: dict[str, _SolverFactory] = {
    "admm": lambda ot, ft: admm.ADMM(opt_tol=ot, feas_tol=ft),
    "dp_admm": lambda ot, ft: dp_admm.DPADMM(opt_tol=ot, feas_tol=ft),
    "p_adapt_admm": lambda ot, ft: p_adapt_admm.PAdaptADMM(
        opt_tol=ot, feas_tol=ft
    ),
}

_MINMAX_METHODS: dict[str, _SolverFactory] = {
    "pdhg": lambda ot, _ft: pdhg.PDHG(opt_tol=ot),
    "extra_gradient": lambda ot, _ft: extra_gradient.ExtraGradient(opt_tol=ot),
    "extragradient": lambda ot, _ft: extra_gradient.ExtraGradient(opt_tol=ot),
    "eg": lambda ot, _ft: extra_gradient.ExtraGradient(opt_tol=ot),
    "mirror_prox": lambda ot, _ft: extra_gradient.ExtraGradient(opt_tol=ot),
    "prox_point_minmax": lambda ot, _ft: prox_point.ProxPointMinMax(opt_tol=ot),
    "prox_point": lambda ot, _ft: prox_point.ProxPointMinMax(opt_tol=ot),
    "ppm": lambda ot, _ft: prox_point.ProxPointMinMax(opt_tol=ot),
}

_SPECTRAL_METHODS: dict[str, _SolverFactory] = {
    "ia_icg": lambda ot, _ft: ia_icg.IAICG(opt_tol=ot),
    "iaicg": lambda ot, _ft: ia_icg.IAICG(opt_tol=ot),
    "aicg": lambda ot, _ft: ia_icg.IAICG(opt_tol=ot),
    "da_icg": lambda ot, _ft: da_icg.DAICG(opt_tol=ot),
    "daicg": lambda ot, _ft: da_icg.DAICG(opt_tol=ot),
    "d_aicg": lambda ot, _ft: da_icg.DAICG(opt_tol=ot),
}

_COMPOSITE_METHODS: dict[str, _SolverFactory] = {
    "aipp": lambda ot, _ft: aipp.AIPP(opt_tol=ot),
    "ag": lambda ot, _ft: ag.AG(opt_tol=ot),
    "apd": lambda ot, _ft: apd.APD(opt_tol=ot),
    "pf_apd": lambda ot, _ft: apd.APD(opt_tol=ot),
    "ecg": lambda ot, _ft: ecg.ECG(opt_tol=ot),
    "upfag": lambda ot, _ft: upfag.UPFAG(opt_tol=ot),
    "nc_fista": lambda ot, _ft: nc_fista.NCFISTA(opt_tol=ot),
    "ncfista": lambda ot, _ft: nc_fista.NCFISTA(opt_tol=ot),
    "adap_fista": lambda ot, _ft: adap_fista.AdapFISTA(opt_tol=ot),
    "adap_nc_fista": lambda ot, _ft: adap_fista.AdapFISTA(opt_tol=ot),
    "adapfista": lambda ot, _ft: adap_fista.AdapFISTA(opt_tol=ot),
    "ac_acg": lambda ot, _ft: ac_acg.ACACG(opt_tol=ot),
    "acacg": lambda ot, _ft: ac_acg.ACACG(opt_tol=ot),
    "adap_apg": lambda ot, _ft: adap_apg.AdapAPG(opt_tol=ot),
    "adapapg": lambda ot, _ft: adap_apg.AdapAPG(opt_tol=ot),
    "panoc": lambda ot, _ft: panoc.PANOC(opt_tol=ot),
    "qp_aipp": lambda ot, ft: qp_aipp.QPAIPP(opt_tol=ot, feas_tol=ft),
    "qp_apd": lambda ot, ft: qp_apd.QPAPD(opt_tol=ot, feas_tol=ft),
    "ialm": lambda ot, ft: ialm.IALM(opt_tol=ot, feas_tol=ft),
    "panoc_alm": lambda ot, ft: ialm.IALM(
        opt_tol=ot, feas_tol=ft, subproblem_solver="panoc"
    ),
    "nl_iapial": lambda ot, ft: ialm.IALM(
        opt_tol=ot,
        feas_tol=ft,
        subproblem_solver="aipp",
        dampen=0.0,
    ),
    "aidal": lambda ot, ft: ialm.IALM(
        opt_tol=ot,
        feas_tol=ft,
        subproblem_solver="aipp",
        dampen=0.25,
        dual_step=0.5,
    ),
    "s_prox_alm": lambda ot, ft: s_prox_alm.SProxALM(opt_tol=ot, feas_tol=ft),
    "sproxalm": lambda ot, ft: s_prox_alm.SProxALM(opt_tol=ot, feas_tol=ft),
    "iaipal": lambda ot, ft: iaipal.IAIPAL(opt_tol=ot, feas_tol=ft),
    "iapial": lambda ot, ft: iaipal.IAIPAL(opt_tol=ot, feas_tol=ft),
}


def _normalize_method(method: str) -> str:
  return method.strip().lower().replace("-", "_")


def _select_solver(
    problem: (
        problem_mod.CompositeProblem
        | block_mod.BlockProblem
        | minmax.MinMaxProblem
        | spectral.SpectralProblem
    ),
    solver: typing.Any,
    method: str | None,
    opt_tol: float,
    feas_tol: float,
) -> tuple[typing.Any, typing.Any]:
  """Resolves `(chosen_solver, target_problem)` from problem type & options."""
  if isinstance(problem, block_mod.BlockProblem):
    if solver is not None:
      return solver, problem
    key = "admm" if method is None else _normalize_method(method)
    factory = _BLOCK_METHODS.get(key)
    if factory is None:
      raise ValueError(
          f"Unknown BlockProblem solver method {method!r}. "
          "Expected 'admm', 'dp_admm', or 'p_adapt_admm'."
      )
    return factory(opt_tol, feas_tol), problem

  if isinstance(problem, minmax.MinMaxProblem):
    if solver is not None:
      return solver, problem
    if method is None:
      if problem.m_x is not None or problem.m_y is not None:
        return prox_point.ProxPointMinMax(opt_tol=opt_tol), problem
      return extra_gradient.ExtraGradient(opt_tol=opt_tol), problem
    factory = _MINMAX_METHODS.get(_normalize_method(method))
    if factory is None:
      raise ValueError(
          f"Unknown min-max solver `method` {method!r}; expected 'pdhg', "
          "'extra_gradient', or 'prox_point_minmax'."
      )
    return factory(opt_tol, feas_tol), problem

  if isinstance(solver, ia_icg.IAICG | da_icg.DAICG):
    return solver, problem

  if solver is not None:
    chosen = solver
  elif method is None:
    if isinstance(problem, spectral.SpectralProblem):
      return ia_icg.IAICG(opt_tol=opt_tol), problem
    chosen = (
        qp_aipp.QPAIPP(opt_tol=opt_tol, feas_tol=feas_tol)
        if problem.is_constrained
        else aipp.AIPP(opt_tol=opt_tol)
    )
  else:
    key = _normalize_method(method)
    spectral_factory = _SPECTRAL_METHODS.get(key)
    if spectral_factory is not None:
      return spectral_factory(opt_tol, feas_tol), problem
    composite_factory = _COMPOSITE_METHODS.get(key)
    if composite_factory is None:
      raise ValueError(
          f"Unknown solver `method` {method!r}; expected one of 'aipp', "
          "'apd', 'ecg', 'ag', 'upfag', 'nc_fista', 'adap_fista', 'ac_acg', "
          "'adap_apg', 'ia_icg', 'da_icg', 'qp_aipp', 'qp_apd', 'ialm', "
          "'nl_iapial', 'aidal', 's_prox_alm', or 'iaipal'."
      )
    chosen = composite_factory(opt_tol, feas_tol)

  target = (
      problem.as_composite_problem()
      if isinstance(problem, spectral.SpectralProblem)
      else problem
  )
  return chosen, target


def solve(
    problem: (
        problem_mod.CompositeProblem
        | block_mod.BlockProblem
        | minmax.MinMaxProblem
        | spectral.SpectralProblem
    ),
    x0: _typing.PyTree,
    *,
    solver: typing.Any = None,
    method: str | None = None,
    opt_tol: float = 1e-5,
    feas_tol: float = 1e-5,
    jit: bool = True,
    callback: typing.Any = None,
    time_limit: float | None = None,
    track_history: bool = False,
) -> base.Solution:
  """Solves a composite, block, minmax, or spectral optimization problem.

  Args:
    problem: Optimization problem instance.
    x0: Initial primal PyTree iterate.
    solver: Optional explicit solver instance (`AIPP`, `APD`, `QPAIPP`,
      `QPAPD`, `IALM`, etc.).
    method: Optional solver name (`'aipp'`, `'apd'`, `'ecg'`, `'ag'`,
      `'upfag'`, `'nc_fista'`, `'adap_fista'`, `'ac_acg'`, `'adap_apg'`,
      `'ia_icg'`, `'da_icg'`, `'qp_aipp'`, `'qp_apd'`, `'ialm'`,
      `'nl_iapial'`, `'aidal'`, `'s_prox_alm'`, or `'iaipal'`). Mutually
      exclusive with `solver`. When both `solver` and `method` are `None`,
      dispatches to `IAICG` for `SpectralProblem`, `QPAIPP` if constrained,
      and `AIPP` if unconstrained.
    opt_tol: Default stationarity tolerance used when `solver` is `None`.
    feas_tol: Default primal feasibility tolerance used when `solver` is `None`.
    jit: If True, executes inner/outer loops via `jax.lax.while_loop` (`jit` and
      `vmap` compatible). If False, runs eager Python loops supporting
      `callback` and `time_limit`.
    callback: Optional per-outer-iteration callback (`jit=False` only, except
      `HistoryTracker` which also supports `jit=True`).
    time_limit: Optional wall-clock limit in seconds (`jit=False` only).
    track_history: If True, records per-iteration objective and residual history
      into `solution.history` for `solution.plot()` (works with `jit=True`).

  Returns:
    A `Solution` PyTree with certified primal `x`, optional `dual`, exact
    stationarity residual `subgrad`, `status`, and `stats`.
  """
  if solver is not None and method is not None:
    raise ValueError("Specify at most one of `solver` or `method`.")
  if track_history and callback is not None and jit:
    raise ValueError(
        "Combining a custom `callback` with `track_history=True` "
        "requires `jit=False`."
    )
  chosen, target = _select_solver(problem, solver, method, opt_tol, feas_tol)
  tracker: base.HistoryTracker | None = None
  effective_cb = callback
  if track_history:
    tracker = base.HistoryTracker()
    if callback is None:
      effective_cb = tracker
    else:

      def _combined_cb(st: typing.Any) -> bool:
        stop_t = tracker(st)
        stop_u = bool(callback(st))
        return stop_t or stop_u

      effective_cb = _combined_cb

  t0 = time.perf_counter()
  sol = typing.cast(
      base.Solution,
      chosen.solve(
          target,
          x0,
          jit=jit,
          callback=effective_cb,
          time_limit=time_limit,
      ),
  )
  runtime_ms: float | None = None
  if not isinstance(sol.status, jax.core.Tracer):
    sol.status.block_until_ready()
    for leaf in jax.tree.leaves(sol.x):
      if hasattr(leaf, "block_until_ready"):
        leaf.block_until_ready()
    runtime_ms = (time.perf_counter() - t0) * 1000.0
  if tracker is not None or runtime_ms is not None:
    return dataclasses.replace(
        sol,
        history=tracker if tracker is not None else sol.history,
        runtime_ms=runtime_ms,
    )
  return sol
