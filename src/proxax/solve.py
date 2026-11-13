"""Top-level `solve` dispatcher for unconstrained and constrained problems."""

from __future__ import annotations

import typing

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
from proxax.solvers.unconstrained import upfag


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
) -> base.Solution:
  """Solves a `CompositeProblem` via `AIPP`, `APD`, or constrained solvers.

  Args:
    problem: Unconstrained or set/cone-constrained `CompositeProblem`.
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
    callback: Optional per-outer-iteration callback (`jit=False` only).
    time_limit: Optional wall-clock limit in seconds (`jit=False` only).

  Returns:
    A `Solution` PyTree with certified primal `x`, optional `dual`, exact
    stationarity residual `subgrad`, `status`, and `stats`.
  """
  if solver is not None and method is not None:
    raise ValueError("Specify at most one of `solver` or `method`.")

  if isinstance(problem, block_mod.BlockProblem):
    blk_x0 = typing.cast(tuple[_typing.PyTree, ...], x0)
    if solver is not None:
      return typing.cast(
          base.Solution,
          solver.solve(
              problem,
              blk_x0,
              jit=jit,
              callback=callback,
              time_limit=time_limit,
          ),
      )
    blk_method = (
        "admm" if method is None else method.strip().lower().replace("-", "_")
    )
    if blk_method == "admm":
      return admm.ADMM(opt_tol=opt_tol, feas_tol=feas_tol).solve(
          problem,
          blk_x0,
          jit=jit,
          callback=callback,
          time_limit=time_limit,
      )
    if blk_method == "dp_admm":
      return dp_admm.DPADMM(opt_tol=opt_tol, feas_tol=feas_tol).solve(
          problem,
          blk_x0,
          jit=jit,
          callback=callback,
          time_limit=time_limit,
      )
    if blk_method == "p_adapt_admm":
      return p_adapt_admm.PAdaptADMM(
          opt_tol=opt_tol, feas_tol=feas_tol
      ).solve(
          problem,
          blk_x0,
          jit=jit,
          callback=callback,
          time_limit=time_limit,
      )
    raise ValueError(
        f"Unknown BlockProblem solver method {method!r}. "
        "Expected 'admm', 'dp_admm', or 'p_adapt_admm'."
    )

  if isinstance(problem, minmax.MinMaxProblem):
    mm_x0 = typing.cast(tuple[_typing.PyTree, _typing.PyTree], x0)
    if solver is not None:
      return typing.cast(
          base.Solution,
          solver.solve(
              problem,
              mm_x0,
              jit=jit,
              callback=callback,
              time_limit=time_limit,
          ),
      )
    if method is None:
      if problem.m_x is not None or problem.m_y is not None:
        return prox_point.ProxPointMinMax(opt_tol=opt_tol).solve(
            problem,
            mm_x0,
            jit=jit,
            callback=callback,
            time_limit=time_limit,
        )
      return extra_gradient.ExtraGradient(opt_tol=opt_tol).solve(
          problem,
          mm_x0,
          jit=jit,
          callback=callback,
          time_limit=time_limit,
      )
    mm_key = method.strip().lower().replace("-", "_")
    if mm_key == "pdhg":
      return pdhg.PDHG(opt_tol=opt_tol).solve(
          problem, mm_x0, jit=jit, callback=callback, time_limit=time_limit
      )
    if mm_key in ("extra_gradient", "extragradient", "eg", "mirror_prox"):
      return extra_gradient.ExtraGradient(opt_tol=opt_tol).solve(
          problem, mm_x0, jit=jit, callback=callback, time_limit=time_limit
      )
    if mm_key in ("prox_point_minmax", "prox_point", "ppm"):
      return prox_point.ProxPointMinMax(opt_tol=opt_tol).solve(
          problem, mm_x0, jit=jit, callback=callback, time_limit=time_limit
      )
    raise ValueError(
        f"Unknown min-max solver `method` {method!r}; expected 'pdhg', "
        "'extra_gradient', or 'prox_point_minmax'."
    )

  if isinstance(solver, ia_icg.IAICG | da_icg.DAICG):
    return solver.solve(
        problem,
        x0,
        jit=jit,
        callback=callback,
        time_limit=time_limit,
    )

  chosen: base.Solver
  if solver is not None:
    chosen = solver
  elif method is None:
    if isinstance(problem, spectral.SpectralProblem):
      return ia_icg.IAICG(opt_tol=opt_tol).solve(
          problem,
          x0,
          jit=jit,
          callback=callback,
          time_limit=time_limit,
      )
    if problem.is_constrained:
      chosen = qp_aipp.QPAIPP(opt_tol=opt_tol, feas_tol=feas_tol)
    else:
      chosen = aipp.AIPP(opt_tol=opt_tol)
  else:
    key = method.strip().lower().replace("-", "_")
    if key == "aipp":
      chosen = aipp.AIPP(opt_tol=opt_tol)
    elif key == "ag":
      chosen = ag.AG(opt_tol=opt_tol)
    elif key == "apd" or key == "pf_apd":
      chosen = apd.APD(opt_tol=opt_tol)
    elif key == "ecg":
      chosen = ecg.ECG(opt_tol=opt_tol)
    elif key == "upfag":
      chosen = upfag.UPFAG(opt_tol=opt_tol)
    elif key == "nc_fista" or key == "ncfista":
      chosen = nc_fista.NCFISTA(opt_tol=opt_tol)
    elif key in ("adap_fista", "adap_nc_fista", "adapfista"):
      chosen = adap_fista.AdapFISTA(opt_tol=opt_tol)
    elif key in ("ac_acg", "acacg"):
      chosen = ac_acg.ACACG(opt_tol=opt_tol)
    elif key in ("adap_apg", "adapapg"):
      chosen = adap_apg.AdapAPG(opt_tol=opt_tol)
    elif key in ("ia_icg", "iaicg", "aicg"):
      return ia_icg.IAICG(opt_tol=opt_tol).solve(
          problem,
          x0,
          jit=jit,
          callback=callback,
          time_limit=time_limit,
      )
    elif key in ("da_icg", "daicg", "d_aicg"):
      return da_icg.DAICG(opt_tol=opt_tol).solve(
          problem,
          x0,
          jit=jit,
          callback=callback,
          time_limit=time_limit,
      )
    elif key == "qp_aipp":
      chosen = qp_aipp.QPAIPP(opt_tol=opt_tol, feas_tol=feas_tol)
    elif key == "qp_apd":
      chosen = qp_apd.QPAPD(opt_tol=opt_tol, feas_tol=feas_tol)
    elif key == "ialm":
      chosen = ialm.IALM(opt_tol=opt_tol, feas_tol=feas_tol)
    elif key == "nl_iapial":
      chosen = ialm.IALM(
          opt_tol=opt_tol,
          feas_tol=feas_tol,
          subproblem_solver="aipp",
          dampen=0.0,
      )
    elif key == "aidal":
      chosen = ialm.IALM(
          opt_tol=opt_tol,
          feas_tol=feas_tol,
          subproblem_solver="aipp",
          dampen=0.25,
          dual_step=0.5,
      )
    elif key in ("s_prox_alm", "sproxalm"):
      chosen = s_prox_alm.SProxALM(opt_tol=opt_tol, feas_tol=feas_tol)
    elif key in ("iaipal", "iapial"):
      chosen = iaipal.IAIPAL(opt_tol=opt_tol, feas_tol=feas_tol)
    else:
      raise ValueError(
          f"Unknown solver `method` {method!r}; expected one of 'aipp', "
          "'apd', 'ecg', 'ag', 'upfag', 'nc_fista', 'adap_fista', 'ac_acg', "
          "'adap_apg', 'ia_icg', 'da_icg', 'qp_aipp', 'qp_apd', 'ialm', "
          "'nl_iapial', 'aidal', 's_prox_alm', or 'iaipal'."
      )

  comp_prob = (
      problem.as_composite_problem()
      if isinstance(problem, spectral.SpectralProblem)
      else problem
  )
  return chosen.solve(
      comp_prob,
      x0,
      jit=jit,
      callback=callback,
      time_limit=time_limit,
  )
