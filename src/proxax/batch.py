"""Batched solving and implicit differentiation across CPU threads or vmap."""

from __future__ import annotations

import atexit
import collections.abc
import concurrent.futures
import os
import threading
import typing
import weakref

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse as jsparse

import proxax.solve as solve_mod
from proxax import _typing
from proxax import implicit as implicit_mod
from proxax import oracle
from proxax import precondition as precondition_mod
from proxax import problem as problem_mod
from proxax.solvers import base

AnyProblem: typing.TypeAlias = (
    problem_mod.CompositeProblem
    | oracle.BlockProblem
    | oracle.MinMaxProblem
    | oracle.SpectralProblem
)


def _infer_batch_size(params_batch: _typing.PyTree) -> int:
  """Returns the leading batch dimension from a PyTree of batched params."""
  leaves = jax.tree.leaves(params_batch)
  if not leaves:
    raise ValueError("`params_batch` must contain at least one array leaf.")
  batch_size = int(leaves[0].shape[0])
  if batch_size <= 0:
    raise ValueError("`params_batch` leading batch size must be positive.")
  return batch_size


def _resolve_backend(backend: str, batch_size: int) -> str:
  """Resolves `'auto'` into `'threads'` (CPU, B>1) or `'vmap'`."""
  if backend not in ("auto", "vmap", "threads"):
    raise ValueError(
        f"Unknown batch backend {backend!r}; expected 'auto', 'vmap', or "
        "'threads'."
    )
  if backend == "auto":
    if jax.default_backend() == "cpu" and batch_size > 1:
      return "threads"
    return "vmap"
  return backend


def _is_bcoo_or_leaf(val: typing.Any) -> bool:
  """Treats `BCOO` sparse arrays as atomic leaves when slicing along axis 0."""
  return isinstance(val, jsparse.BCOO)


def _slice_pytree(tree_val: _typing.PyTree, idx: int) -> _typing.PyTree:
  """Extracts element `idx` along axis 0 from every leaf of `tree_val`."""
  return jax.tree.map(
      lambda a: a[idx], tree_val, is_leaf=_is_bcoo_or_leaf
  )


def _slice_range_pytree(
    tree_val: _typing.PyTree, start: int, end: int
) -> _typing.PyTree:
  """Extracts slice `start:end` along axis 0 from every leaf of `tree_val`."""
  return jax.tree.map(
      lambda a: a[start:end], tree_val, is_leaf=_is_bcoo_or_leaf
  )


_BATCH_SOLVE_CACHE: weakref.WeakKeyDictionary[
    typing.Any, dict[tuple[typing.Any, ...], typing.Any]
] = weakref.WeakKeyDictionary()

_BATCH_IMPLICIT_CACHE: weakref.WeakKeyDictionary[
    typing.Any, dict[tuple[typing.Any, ...], typing.Any]
] = weakref.WeakKeyDictionary()

_THREAD_POOL_LOCK = threading.Lock()
_THREAD_POOL_CACHE: dict[int, concurrent.futures.ThreadPoolExecutor] = {}


def _shutdown_thread_pools() -> None:
  """Shuts down cached ThreadPoolExecutors cleanly at interpreter exit."""
  with _THREAD_POOL_LOCK:
    for pool in _THREAD_POOL_CACHE.values():
      pool.shutdown(wait=False)
    _THREAD_POOL_CACHE.clear()


atexit.register(_shutdown_thread_pools)


def _get_thread_pool(
    max_workers: int | None, batch_size: int
) -> concurrent.futures.ThreadPoolExecutor:
  """Returns a persistent ThreadPoolExecutor sized for `batch_size`."""
  workers = (
      max_workers
      if max_workers is not None
      else min(batch_size, os.cpu_count() or 8)
  )
  workers = max(1, workers)
  with _THREAD_POOL_LOCK:
    pool = _THREAD_POOL_CACHE.get(workers)
    if pool is None:
      pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
      _THREAD_POOL_CACHE[workers] = pool
    return pool


def _get_compiled_batch_runner(
    problem_fn: collections.abc.Callable[[_typing.PyTree], AnyProblem],
    *,
    solver: typing.Any,
    method: str | None,
    opt_tol: float,
    feas_tol: float,
    time_limit: float | None,
    mode: str,
    precondition: bool = False,
) -> typing.Any:
  """Returns a cached JIT-compiled single or vmap runner for `problem_fn`."""
  key = (
      id(solver) if solver is not None else None,
      method,
      opt_tol,
      feas_tol,
      time_limit,
      mode,
      precondition,
  )
  try:
    per_fn = _BATCH_SOLVE_CACHE.setdefault(problem_fn, {})
    if key in per_fn:
      return per_fn[key]
  except TypeError:
    per_fn = None

  def _solve_one(
      x_init: _typing.PyTree, theta: _typing.PyTree
  ) -> base.Solution:
    prob = problem_fn(theta)
    if precondition:
      return precondition_mod.solve_preconditioned(
          typing.cast(problem_mod.CompositeProblem, prob),
          typing.cast(jax.Array, x_init),
          solver=solver,
          method=method,
          opt_tol=opt_tol,
          feas_tol=feas_tol,
          jit=True,
          time_limit=time_limit,
      )
    return solve_mod.solve(
        prob,
        x_init,
        solver=solver,
        method=method,
        opt_tol=opt_tol,
        feas_tol=feas_tol,
        jit=True,
        time_limit=time_limit,
    )

  compiled = (
      jax.jit(jax.vmap(_solve_one, in_axes=(None, 0)))
      if mode == "vmap"
      else jax.jit(_solve_one)
  )
  if per_fn is not None:
    per_fn[key] = compiled
  return compiled


def batch_solve(
    problem_fn: collections.abc.Callable[[_typing.PyTree], AnyProblem],
    x0: _typing.PyTree,
    params_batch: _typing.PyTree,
    *,
    solver: typing.Any = None,
    method: str | None = None,
    opt_tol: float = 1e-5,
    feas_tol: float = 1e-5,
    time_limit: float | None = None,
    backend: str = "auto",
    max_workers: int | None = None,
    chunk_size: int = 1,
    precondition: bool = False,
) -> base.Solution:
  """Solves a batch of parameterized problems via `vmap` or CPU threads.

  Args:
    problem_fn: Callable mapping a single parameter PyTree `theta` to a Proxax
      problem instance.
    x0: Shared initial primal PyTree iterate.
    params_batch: PyTree of batched parameters with leading axis `B`.
    solver: Optional explicit solver instance.
    method: Optional solver method string.
    opt_tol: Stationarity tolerance used when `solver` is `None`.
    feas_tol: Feasibility tolerance used when `solver` is `None`.
    time_limit: Optional per-solve wall-clock limit in seconds.
    backend: Execution backend (`'auto'`, `'vmap'`, or `'threads'`). `'auto'`
      uses `'threads'` on CPU when `B > 1` (eliminating `vmap` lockstep tail
      latency) and `'vmap'` on GPU/TPU or `B == 1`.
    max_workers: Optional `ThreadPoolExecutor` worker count for `'threads'`.
    chunk_size: Number of batch items per worker task for `'threads'`. Values
      `> 1` run a `vmap` of size `chunk_size` on each thread, reducing host
      dispatch overhead on uniform problems at the cost of intra-chunk lockstep.
    precondition: If `True`, applies Modified Ruiz Equilibration
      (`px.solve_preconditioned`) to each `CompositeProblem` in the batch.

  Returns:
    A batched `Solution` PyTree with leading axis `B`.
  """
  if chunk_size <= 0:
    raise ValueError("`chunk_size` must be a positive integer.")
  batch_size = _infer_batch_size(params_batch)
  resolved = _resolve_backend(backend, batch_size)
  runner_mode = "vmap" if (resolved == "vmap" or chunk_size > 1) else "threads"
  runner = _get_compiled_batch_runner(
      problem_fn,
      solver=solver,
      method=method,
      opt_tol=opt_tol,
      feas_tol=feas_tol,
      time_limit=time_limit,
      mode=runner_mode,
      precondition=precondition,
  )

  if resolved == "vmap":
    return typing.cast(base.Solution, runner(x0, params_batch))

  if chunk_size > 1:
    chunks = [
        _slice_range_pytree(params_batch, i, min(i + chunk_size, batch_size))
        for i in range(0, batch_size, chunk_size)
    ]
    pool = _get_thread_pool(max_workers, len(chunks))
    sols = list(pool.map(lambda ch: runner(x0, ch), chunks))
    return typing.cast(
        base.Solution,
        jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *sols),
    )

  pool = _get_thread_pool(max_workers, batch_size)
  sols = list(
      pool.map(
          lambda i: runner(x0, _slice_pytree(params_batch, i)),
          range(batch_size),
      )
  )
  return typing.cast(
      base.Solution,
      jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *sols),
  )


def batch_implicit_solve(
    problem_fn: collections.abc.Callable[
        [_typing.PyTree], implicit_mod.DifferentiableProblem
    ],
    x0: _typing.PyTree,
    params_batch: _typing.PyTree,
    *,
    solver: typing.Any = None,
    method: str | None = None,
    opt_tol: float = 1e-5,
    feas_tol: float = 1e-5,
    step_size: float = 1.0,
    linear_solver: str = "gmres",
    linear_tol: float = 1e-5,
    linear_maxiter: int = 100,
    damping: float = 1e-6,
    return_dual: bool = False,
    backend: str = "auto",
    max_workers: int | None = None,
) -> _typing.PyTree:
  """Solves and differentiates a batch of parameterized problems.

  Args:
    problem_fn: Callable mapping a single parameter PyTree `theta` to a
      differentiable Proxax problem instance.
    x0: Shared initial primal PyTree iterate.
    params_batch: PyTree of batched parameters with leading axis `B`.
    solver: Optional explicit Proxax solver instance.
    method: Optional solver name string.
    opt_tol: Stationarity tolerance used when `solver` is `None`.
    feas_tol: Feasibility tolerance used when `solver` is `None`.
    step_size: Proximal step scale `tau > 0` used in the fixed-point map.
    linear_solver: Matrix-free adjoint solver (`'gmres'`, `'cg'`, `'bicgstab'`).
    linear_tol: Relative tolerance for the adjoint linear solver.
    linear_maxiter: Maximum iterations for the adjoint linear solver.
    damping: Small Tikhonov regularization added to the adjoint operator.
    return_dual: If True and constrained, returns `(x_hat, dual_hat)`.
    backend: Execution backend (`'auto'`, `'vmap'`, or `'threads'`). `'auto'`
      uses `'threads'` on CPU when `B > 1` and `'vmap'` on GPU/TPU or `B == 1`.
    max_workers: Optional `ThreadPoolExecutor` worker count for `'threads'`.

  Returns:
    Batched optimal primal PyTree `x_hat` (or `(x_hat, dual_hat)` if
    `return_dual`), differentiable with respect to `params_batch`.
  """
  batch_size = _infer_batch_size(params_batch)
  resolved = _resolve_backend(backend, batch_size)

  if resolved == "vmap":
    return jax.vmap(
        lambda p: implicit_mod.implicit_solve(
            problem_fn,
            x0,
            p,
            solver=solver,
            method=method,
            opt_tol=opt_tol,
            feas_tol=feas_tol,
            step_size=step_size,
            linear_solver=linear_solver,
            linear_tol=linear_tol,
            linear_maxiter=linear_maxiter,
            damping=damping,
            return_dual=return_dual,
        )
    )(params_batch)

  first_theta = _slice_pytree(params_batch, 0)
  cache_key = (
      id(solver) if solver is not None else None,
      method,
      opt_tol,
      feas_tol,
      step_size,
      linear_solver,
      linear_tol,
      linear_maxiter,
      damping,
      return_dual,
      batch_size,
      max_workers,
      tuple((tuple(a.shape), str(a.dtype)) for a in jax.tree.leaves(x0)),
      tuple(
          (tuple(a.shape), str(a.dtype)) for a in jax.tree.leaves(first_theta)
      ),
  )
  try:
    per_fn = _BATCH_IMPLICIT_CACHE.setdefault(problem_fn, {})
    if cache_key in per_fn:
      cached_fn = per_fn[cache_key]
      return cached_fn(x0, params_batch)
  except TypeError:
    per_fn = None

  probe_prob = problem_fn(first_theta)
  is_minmax = isinstance(probe_prob, oracle.MinMaxProblem)
  is_block = isinstance(probe_prob, oracle.BlockProblem)
  has_dual = is_block or (
      isinstance(probe_prob, problem_mod.CompositeProblem)
      and probe_prob.is_constrained
  )

  def _one_fwd(
      x_init: _typing.PyTree, theta: _typing.PyTree
  ) -> tuple[_typing.PyTree, _typing.PyTree]:
    sol = solve_mod.solve(
        problem_fn(theta),
        x_init,
        solver=solver,
        method=method,
        opt_tol=opt_tol,
        feas_tol=feas_tol,
        jit=True,
    )
    return sol.x, sol.dual

  compiled_fwd = jax.jit(_one_fwd).lower(x0, first_theta).compile()
  single_struct = jax.eval_shape(_one_fwd, x0, first_theta)
  batch_struct = jax.tree.map(
      lambda s: jax.ShapeDtypeStruct(  # type: ignore[no-untyped-call]
          (batch_size, *s.shape), s.dtype
      ),
      single_struct,
  )

  def _host_batch_fwd(
      x_init: _typing.PyTree, pb: _typing.PyTree
  ) -> tuple[_typing.PyTree, _typing.PyTree]:
    x_init_np = jax.tree.map(np.asarray, x_init)
    pb_np = jax.tree.map(np.asarray, pb)
    pool = _get_thread_pool(max_workers, batch_size)
    outs = list(
        pool.map(
            lambda i: compiled_fwd(x_init_np, _slice_pytree(pb_np, i)),
            range(batch_size),
        )
    )
    return typing.cast(
        tuple[_typing.PyTree, _typing.PyTree],
        jax.tree.map(
            lambda *xs: np.stack([np.asarray(x) for x in xs], 0), *outs
        ),
    )

  def _run_fwd_dispatch(
      x_init: _typing.PyTree, pb: _typing.PyTree
  ) -> tuple[_typing.PyTree, _typing.PyTree]:
    if any(
        isinstance(leaf, jax.core.Tracer)
        for leaf in jax.tree.leaves((x_init, pb))
    ):
      return typing.cast(
          tuple[_typing.PyTree, _typing.PyTree],
          jax.pure_callback(_host_batch_fwd, batch_struct, x_init, pb),
      )
    x_np, p_np = _host_batch_fwd(x_init, pb)
    return jax.tree.map(jnp.asarray, x_np), jax.tree.map(jnp.asarray, p_np)

  def _adj_one(
      x_init: _typing.PyTree,
      xh: _typing.PyTree,
      ph: _typing.PyTree,
      th: _typing.PyTree,
      go: _typing.PyTree,
  ) -> _typing.PyTree:
    d_th = implicit_mod.implicit_vjp_from_solution(
        problem_fn,
        x_init,
        xh,
        ph,
        th,
        go,
        is_minmax=is_minmax,
        is_block=is_block,
        has_dual=has_dual,
        step_size=step_size,
        linear_solver=linear_solver,
        linear_tol=linear_tol,
        linear_maxiter=linear_maxiter,
        damping=damping,
        return_dual=return_dual,
    )
    # Unwrap BCOO cotangents before vmap unbatching (float0 indices fail
    # BCOO.__init__).
    return jax.tree.map(
        lambda d: d.data if isinstance(d, jsparse.BCOO) else d,
        d_th,
        is_leaf=_is_bcoo_or_leaf,
    )

  p_axis = 0 if has_dual else None
  _vmap_bwd = jax.jit(jax.vmap(_adj_one, in_axes=(None, 0, p_axis, 0, 0)))

  @jax.custom_vjp
  def _batched_fn(
      x_init: _typing.PyTree, pb: _typing.PyTree
  ) -> _typing.PyTree:
    x_b, p_b = _run_fwd_dispatch(x_init, pb)
    return (x_b, p_b) if (return_dual and has_dual) else x_b

  def _fwd(
      x_init: _typing.PyTree, pb: _typing.PyTree
  ) -> tuple[
      _typing.PyTree,
      tuple[_typing.PyTree, _typing.PyTree, _typing.PyTree, _typing.PyTree],
  ]:
    x_b, p_b = _run_fwd_dispatch(x_init, pb)
    out = (x_b, p_b) if (return_dual and has_dual) else x_b
    return out, (x_init, x_b, p_b, pb)

  def _bwd(
      res: tuple[
          _typing.PyTree, _typing.PyTree, _typing.PyTree, _typing.PyTree
      ],
      g_out: _typing.PyTree,
  ) -> tuple[None, _typing.PyTree]:
    x_init, x_b, p_b, pb = res
    d_pb_raw = _vmap_bwd(x_init, x_b, p_b, pb, g_out)
    d_pb = jax.tree.map(
        lambda orig, d: (
            jax.tree.unflatten(
                jax.tree.structure(orig),
                [d, np.zeros(orig.indices.shape, dtype=jax.dtypes.float0)],
            )
            if isinstance(orig, jsparse.BCOO)
            else d
        ),
        pb,
        d_pb_raw,
        is_leaf=_is_bcoo_or_leaf,
    )
    return (None, d_pb)

  _batched_fn.defvjp(_fwd, _bwd)
  if per_fn is not None:
    per_fn[cache_key] = _batched_fn
  return _batched_fn(x0, params_batch)
