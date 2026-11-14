"""Batched solving and implicit differentiation across CPU threads or vmap."""

from __future__ import annotations

import collections.abc
import concurrent.futures
import typing

import jax
import jax.numpy as jnp
import numpy as np

import proxax.solve as solve_mod
from proxax import _typing
from proxax import implicit as implicit_mod
from proxax import oracle
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


def _slice_pytree(tree_val: _typing.PyTree, idx: int) -> _typing.PyTree:
  """Extracts element `idx` along axis 0 from every leaf of `tree_val`."""
  return jax.tree.map(lambda a: a[idx], tree_val)


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

  Returns:
    A batched `Solution` PyTree with leading axis `B`.
  """
  batch_size = _infer_batch_size(params_batch)
  resolved = _resolve_backend(backend, batch_size)

  def _solve_one(theta: _typing.PyTree) -> base.Solution:
    return solve_mod.solve(
        problem_fn(theta),
        x0,
        solver=solver,
        method=method,
        opt_tol=opt_tol,
        feas_tol=feas_tol,
        jit=True,
        time_limit=time_limit,
    )

  if resolved == "vmap":
    return typing.cast(
        base.Solution,
        jax.jit(jax.vmap(_solve_one))(params_batch),
    )

  jit_one = jax.jit(_solve_one)
  with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
    sols = list(
        pool.map(
            lambda i: jit_one(_slice_pytree(params_batch, i)),
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
  probe_prob = problem_fn(first_theta)
  is_minmax = isinstance(probe_prob, oracle.MinMaxProblem)
  is_block = isinstance(probe_prob, oracle.BlockProblem)
  has_dual = is_block or (
      isinstance(probe_prob, problem_mod.CompositeProblem)
      and probe_prob.is_constrained
  )

  @jax.jit
  def _one_fwd(theta: _typing.PyTree) -> tuple[_typing.PyTree, _typing.PyTree]:
    sol = solve_mod.solve(
        problem_fn(theta),
        x0,
        solver=solver,
        method=method,
        opt_tol=opt_tol,
        feas_tol=feas_tol,
        jit=True,
    )
    return sol.x, sol.dual

  single_struct = jax.eval_shape(_one_fwd, first_theta)
  batch_struct = jax.tree.map(
      lambda s: jax.ShapeDtypeStruct(  # type: ignore[no-untyped-call]
          (batch_size, *s.shape), s.dtype
      ),
      single_struct,
  )

  def _host_batch_fwd(
      pb: _typing.PyTree,
  ) -> tuple[_typing.PyTree, _typing.PyTree]:
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers
    ) as pool:
      outs = list(
          pool.map(
              lambda i: _one_fwd(_slice_pytree(pb, i)),
              range(batch_size),
          )
      )
    return typing.cast(
        tuple[_typing.PyTree, _typing.PyTree],
        jax.tree.map(
            lambda *xs: np.stack([np.asarray(x) for x in xs], 0), *outs
        ),
    )

  @jax.custom_vjp
  def _batched_fn(pb: _typing.PyTree) -> _typing.PyTree:
    x_b, p_b = jax.pure_callback(_host_batch_fwd, batch_struct, pb)
    return (x_b, p_b) if (return_dual and has_dual) else x_b

  def _fwd(
      pb: _typing.PyTree,
  ) -> tuple[
      _typing.PyTree, tuple[_typing.PyTree, _typing.PyTree, _typing.PyTree]
  ]:
    x_b, p_b = jax.pure_callback(_host_batch_fwd, batch_struct, pb)
    out = (x_b, p_b) if (return_dual and has_dual) else x_b
    return out, (x_b, p_b, pb)

  def _bwd(
      res: tuple[_typing.PyTree, _typing.PyTree, _typing.PyTree],
      g_out: _typing.PyTree,
  ) -> tuple[_typing.PyTree]:
    x_b, p_b, pb = res
    p_axis = None if p_b is None else 0

    def _adj_one(
        xh: _typing.PyTree,
        ph: _typing.PyTree,
        th: _typing.PyTree,
        go: _typing.PyTree,
    ) -> _typing.PyTree:
      return implicit_mod.implicit_vjp_from_solution(
          problem_fn,
          x0,
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

    d_pb = jax.vmap(_adj_one, in_axes=(0, p_axis, 0, 0))(x_b, p_b, pb, g_out)
    return (d_pb,)

  _batched_fn.defvjp(_fwd, _bwd)
  return _batched_fn(params_batch)
