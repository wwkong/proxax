"""PyTree combinators for applying distinct proximal operators per subtree."""

from __future__ import annotations

import dataclasses
import typing

import jax
import jax.numpy as jnp

from proxax import _typing
from proxax import tree
from proxax.prox import base


def _is_prox_op(node: typing.Any) -> bool:
  return isinstance(node, base.ProxOperator)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class MultiProx:
  """Applies a PyTree prefix of `ProxOperator`s to matching subtrees of `x`.

  Mirrors `optax.multi_transform` for separable composite objectives
  `h(x) = sum_i h_i(x_i)` over PyTree components.

  Attributes:
    operators: PyTree whose leaves (up to `ProxOperator` nodes) match a PyTree
      prefix of `x`.
  """

  operators: _typing.PyTree

  @property
  def strong_convexity(self) -> jax.Array | float:
    """Global strong convexity `min_i mu_i` across separable blocks."""
    ops: list[base.ProxOperator] = jax.tree.leaves(
        self.operators,
        is_leaf=_is_prox_op,
    )
    if not ops:
      return 0.0
    mus = [op.strong_convexity for op in ops]
    acc: jax.Array | float = mus[0]
    for mu in mus[1:]:
      acc = jnp.minimum(acc, mu)
    return acc

  def value(self, x: _typing.PyTree) -> jax.Array:
    """Evaluates `sum_i h_i(x_i)`."""
    vals_tree = jax.tree.map(
        lambda op, sub_x: op.value(sub_x),
        self.operators,
        x,
        is_leaf=_is_prox_op,
    )
    vals = jax.tree.leaves(vals_tree)
    if not vals:
      return jnp.asarray(0.0, dtype=tree.infer_dtype(x))
    return tree.functools_sum(vals)

  def prox(
      self,
      x: _typing.PyTree,
      step_size: _typing.Scalar,
  ) -> _typing.PyTree:
    """Applies `op_i.prox(x_i, step_size)` blockwise."""
    return jax.tree.map(
        lambda op, sub_x: op.prox(sub_x, step_size),
        self.operators,
        x,
        is_leaf=_is_prox_op,
    )

  def domain_project(self, x: _typing.PyTree) -> _typing.PyTree:
    """Projects each block `x_i` onto `dom(h_i)`."""
    return jax.tree.map(
        lambda op, sub_x: op.domain_project(sub_x),
        self.operators,
        x,
        is_leaf=_is_prox_op,
    )
