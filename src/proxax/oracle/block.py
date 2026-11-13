"""Block-constrained separable composite optimization problem definition."""

from __future__ import annotations

import collections.abc as abc
import dataclasses
import typing

import jax
import jax.numpy as jnp

from proxax import _typing
from proxax import tree
from proxax.constraints import linear_op
from proxax.oracle import terms
from proxax.prox import base as prox_base


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class BlockTerm:
  """Single separable block ``f_i(x_i) + h_i(x_i)`` with linear map ``A_i``.

  Attributes:
    smooth: Smooth differentiable objective ``f_i`` (defaults to ``0``).
    proximal: Convex ``ProxOperator`` ``h_i`` (defaults to ``prox.Zero()``).
    linear_op: Linear operator ``A_i`` mapping block ``x_i`` into the shared
      coupling space (defaults to ``constraints.identity``).
  """

  smooth: terms.SmoothLike = dataclasses.field(default_factory=terms.ZeroSmooth)
  proximal: prox_base.ProxOperator = dataclasses.field(
      default_factory=prox_base.Zero
  )
  linear_op: linear_op.OperatorLike | None = None

  def __post_init__(self) -> None:
    """Normalizes ``smooth`` and ``linear_op`` fields on construction."""
    if self.smooth is None or (
        callable(self.smooth)
        and not isinstance(self.smooth, terms.SmoothObjective)
    ):
      object.__setattr__(self, "smooth", terms.as_smooth_term(self.smooth))
    if self.linear_op is None:
      object.__setattr__(self, "linear_op", linear_op.identity)
    elif not isinstance(self.linear_op, linear_op.LinearOperator):
      object.__setattr__(
          self, "linear_op", linear_op.as_linear_operator(self.linear_op)
      )

  @property
  def is_zero_smooth(self) -> bool:
    """Returns True if ``smooth`` is identically zero."""
    return isinstance(self.smooth, terms.ZeroSmooth)

  @property
  def is_identity_op(self) -> bool:
    """Returns True if ``linear_op`` is the identity operator."""
    return isinstance(self.linear_op, linear_op.IdentityLinearOperator)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class BlockProblem:
  """Separable multi-block problem ``min sum_i (f_i(x_i) + h_i(x_i))``.

  Subject to ``sum_i A_i(x_i) = rhs``.

  Attributes:
    blocks: Sequence/tuple of ``BlockTerm`` objects, one per block ``x_i``.
    rhs: Right-hand side ``b`` in the coupling constraint ``sum_i A_i(x_i) = b``
      (defaults to ``0.0``).
  """

  blocks: tuple[BlockTerm, ...] | abc.Sequence[BlockTerm]
  rhs: _typing.PyTree = 0.0

  def __post_init__(self) -> None:
    """Normalizes ``blocks`` to a tuple and validates non-emptiness."""
    blocks_tuple = tuple(self.blocks)
    if not blocks_tuple:
      raise ValueError("BlockProblem requires at least one BlockTerm.")
    object.__setattr__(self, "blocks", blocks_tuple)

  @property
  def num_blocks(self) -> int:
    """Number of separable blocks ``K``."""
    return len(self.blocks)

  def bind_operators(
      self,
      x_blocks: tuple[_typing.PyTree, ...],
  ) -> tuple[linear_op.BoundLinearOperator, ...]:
    """Binds each block's linear operator to its corresponding block PyTree."""
    if len(x_blocks) != len(self.blocks):
      raise ValueError(
          f"Expected {len(self.blocks)} block iterates, got {len(x_blocks)}."
      )
    return tuple(
        typing.cast(linear_op.LinearOperator, blk.linear_op).bind(x_i)
        for blk, x_i in zip(self.blocks, x_blocks, strict=True)
    )

  def coupling_residual(
      self,
      x_blocks: tuple[_typing.PyTree, ...],
      bound_ops: tuple[linear_op.BoundLinearOperator, ...] | None = None,
  ) -> _typing.PyTree:
    """Computes ``sum_i A_i(x_i) - rhs``."""
    ops = bound_ops if bound_ops is not None else self.bind_operators(x_blocks)
    total = ops[0].matvec(x_blocks[0])
    for op, x_i in zip(ops[1:], x_blocks[1:], strict=True):
      total = tree.add(total, op.matvec(x_i))
    return _sub_rhs(total, self.rhs)

  def objective_value(
      self,
      x_blocks: tuple[_typing.PyTree, ...],
  ) -> jax.Array:
    """Evaluates ``sum_i (f_i(x_i) + h_i(x_i))``."""
    total = jnp.asarray(0.0, dtype=tree.infer_dtype(x_blocks[0]))
    for blk, x_i in zip(self.blocks, x_blocks, strict=True):
      smooth = typing.cast(terms.SmoothObjective, blk.smooth)
      total = total + smooth.value(x_i) + blk.proximal.value(x_i)
    return total


def _sub_rhs(total: _typing.PyTree, rhs: _typing.PyTree) -> _typing.PyTree:
  """Subtracts scalar or matching-PyTree ``rhs`` from ``total``."""
  if not isinstance(rhs, dict | list | tuple) and jnp.ndim(rhs) == 0:
    return jax.tree.map(
        lambda leaf: leaf - jnp.asarray(rhs, dtype=leaf.dtype),
        total,
    )
  return tree.sub(total, rhs)
