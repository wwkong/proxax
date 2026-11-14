"""PyTree linear operators and set-inclusion constraints `A(x) in C`."""

from __future__ import annotations

import dataclasses
import typing
from collections import abc

import jax
import jax.numpy as jnp
from jax.experimental import sparse as jsparse

from proxax import _typing
from proxax import tree
from proxax.constraints import sets


@typing.runtime_checkable
class BoundLinearOperator(typing.Protocol):
  """Linear map `A: X -> Y` bound to a concrete primal PyTree structure."""

  @property
  def operator_norm_sq(self) -> jax.Array | float | None:
    """Optional upper bound on `||A||_op^2`."""
    ...

  def matvec(self, x: _typing.PyTree) -> _typing.PyTree:
    """Evaluates `A(x)`."""
    ...

  def rmatvec(self, y: _typing.PyTree) -> _typing.PyTree:
    """Evaluates the adjoint `A^*(y)`."""
    ...


@typing.runtime_checkable
class LinearOperator(typing.Protocol):
  """Unbound linear map `A: X -> Y` capable of binding to a primal template."""

  @property
  def operator_norm_sq(self) -> jax.Array | float | None:
    """Optional upper bound on `||A||_op^2`."""
    ...

  def matvec(self, x: _typing.PyTree) -> _typing.PyTree:
    """Evaluates `A(x)`."""
    ...

  def bind(self, x_example: _typing.PyTree) -> BoundLinearOperator:
    """Binds `A` to `x_example`'s PyTree shape/dtype to enable `rmatvec`."""
    ...


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class IdentityLinearOperator:
  """Identity linear operator `x -> x` on arbitrary PyTrees."""

  @property
  def operator_norm_sq(self) -> float:
    """Exact squared operator norm `||I||_op^2 = 1`."""
    return 1.0

  def matvec(self, x: _typing.PyTree) -> _typing.PyTree:
    """Returns `x` unchanged."""
    return x

  def rmatvec(self, y: _typing.PyTree) -> _typing.PyTree:
    """Returns `y` unchanged (`I^* = I`)."""
    return y

  def bind(self, x_example: _typing.PyTree) -> IdentityLinearOperator:
    """Returns `self` without tracing."""
    del x_example
    return self


identity: typing.Final = IdentityLinearOperator()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class MatrixLinearOperator:
  """Dense or BCOO matrix operator `x -> matrix @ x`."""

  matrix: jax.Array | jsparse.BCOO
  norm_sq_bound: jax.Array | float | None = dataclasses.field(
      default=None,
      metadata={"static": False},
  )

  @property
  def operator_norm_sq(self) -> jax.Array | float | None:
    """Optional upper bound on `||matrix||_2^2`."""
    return self.norm_sq_bound

  def matvec(self, x: jax.Array) -> jax.Array:
    """Computes `matrix @ x`."""
    return typing.cast(jax.Array, self.matrix @ x)

  def rmatvec(self, y: jax.Array) -> jax.Array:
    """Computes `matrix.T @ y`."""
    return typing.cast(jax.Array, self.matrix.T @ y)

  def bind(self, x_example: _typing.PyTree) -> MatrixLinearOperator:
    """Returns `self` (`matrix.T` needs no primal shape metadata)."""
    del x_example
    return self


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class BoundCallableLinearOperator:
  """Bound callable linear operator synthesizing `A^*` automatically."""

  matvec_fn: abc.Callable[[_typing.PyTree], _typing.PyTree]
  rmatvec_fn: abc.Callable[[_typing.PyTree], _typing.PyTree] | None
  in_struct: _typing.PyTree
  norm_sq_bound: jax.Array | float | None = None

  @property
  def operator_norm_sq(self) -> jax.Array | float | None:
    """Optional upper bound on `||A||_op^2`."""
    return self.norm_sq_bound

  def matvec(self, x: _typing.PyTree) -> _typing.PyTree:
    """Evaluates `matvec_fn(x)`."""
    return self.matvec_fn(x)

  def rmatvec(self, y: _typing.PyTree) -> _typing.PyTree:
    """Evaluates `rmatvec_fn(y)` or synthesizes `A^*(y)` automatically."""
    if self.rmatvec_fn is not None:
      return self.rmatvec_fn(y)
    transpose_fn = jax.linear_transpose(self.matvec_fn, self.in_struct)
    (adj_x,) = transpose_fn(y)
    return adj_x


jax.tree_util.register_dataclass(
    BoundCallableLinearOperator,
    data_fields=["norm_sq_bound"],
    meta_fields=["matvec_fn", "rmatvec_fn", "in_struct"],
)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class CallableLinearOperator:
  """Matrix-free linear operator `x -> matvec_fn(x)` on arbitrary PyTrees."""

  matvec_fn: abc.Callable[[_typing.PyTree], _typing.PyTree]
  rmatvec_fn: abc.Callable[[_typing.PyTree], _typing.PyTree] | None = None
  norm_sq_bound: jax.Array | float | None = None

  @property
  def operator_norm_sq(self) -> jax.Array | float | None:
    """Optional upper bound on `||A||_op^2`."""
    return self.norm_sq_bound

  def matvec(self, x: _typing.PyTree) -> _typing.PyTree:
    """Evaluates `matvec_fn(x)`."""
    return self.matvec_fn(x)

  def bind(self, x_example: _typing.PyTree) -> BoundCallableLinearOperator:
    """Captures shape/dtype structure of `x_example` for `linear_transpose`."""
    in_struct = jax.eval_shape(lambda z: z, x_example)
    return BoundCallableLinearOperator(
        matvec_fn=self.matvec_fn,
        rmatvec_fn=self.rmatvec_fn,
        in_struct=in_struct,
        norm_sq_bound=self.norm_sq_bound,
    )


jax.tree_util.register_dataclass(
    CallableLinearOperator,
    data_fields=["norm_sq_bound"],
    meta_fields=["matvec_fn", "rmatvec_fn"],
)


OperatorLike: typing.TypeAlias = (
    LinearOperator
    | jax.Array
    | jsparse.BCOO
    | abc.Callable[[_typing.PyTree], _typing.PyTree]
)


def as_linear_operator(
    op: OperatorLike,
    *,
    rmatvec: abc.Callable[[_typing.PyTree], _typing.PyTree] | None = None,
    operator_norm_sq: jax.Array | float | None = None,
) -> LinearOperator:
  """Coerces a matrix, callable, or `LinearOperator` into a `LinearOperator`."""
  if isinstance(op, LinearOperator):
    return op
  if isinstance(op, (jax.Array, jsparse.BCOO)):
    return MatrixLinearOperator(matrix=op, norm_sq_bound=operator_norm_sq)
  if callable(op):
    return CallableLinearOperator(
        matvec_fn=op,
        rmatvec_fn=rmatvec,
        norm_sq_bound=operator_norm_sq,
    )
  raise TypeError(f"Unsupported linear operator type: {type(op)!r}.")


@typing.runtime_checkable
class BoundConstraint(typing.Protocol):
  """Protocol for a bound set-inclusion constraint `A(x) in C`."""

  @property
  def operator_norm_sq(self) -> jax.Array | float | None:
    """Optional upper bound on `||A||_op^2`."""
    ...

  def zero_dual(self, x_example: _typing.PyTree) -> _typing.PyTree:
    """Creates a zero dual variable matching the range space `Y`."""
    ...

  def apply(self, x: _typing.PyTree) -> _typing.PyTree:
    """Computes `A(x)`."""
    ...

  def adjoint(self, dual: _typing.PyTree) -> _typing.PyTree:
    """Computes `A^*(dual)`."""
    ...

  def residual(self, x: _typing.PyTree) -> _typing.PyTree:
    """Computes feasibility residual `A(x) - Pi_C(A(x))`."""
    ...

  def penalty_and_dual(
      self,
      x: _typing.PyTree,
      *,
      multiplier: _typing.PyTree | None,
      penalty: _typing.Scalar,
      dampen: _typing.Scalar = 0.0,
  ) -> tuple[jax.Array, _typing.PyTree]:
    """Computes `(psi_c(x; p_bar), p_hat(x))` for penalty / AL subproblems."""
    ...


@typing.runtime_checkable
class Constraint(typing.Protocol):
  """Unbound set-inclusion constraint `A(x) in C`."""

  def bind(self, x_example: _typing.PyTree) -> BoundConstraint:
    """Binds linear operator(s) to `x_example`'s PyTree shape/dtype."""
    ...


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class BoundSetConstraint:
  """Bound set-inclusion constraint `operator(x) in target_set`."""

  operator: BoundLinearOperator
  target_set: sets.ConvexSet

  @property
  def operator_norm_sq(self) -> jax.Array | float | None:
    """Optional upper bound on `||A||_op^2`."""
    return self.operator.operator_norm_sq

  def zero_dual(self, x_example: _typing.PyTree) -> _typing.PyTree:
    """Creates a zero PyTree in the dual/range space `Y`."""
    return tree.zeros_like(self.operator.matvec(tree.zeros_like(x_example)))

  def apply(self, x: _typing.PyTree) -> _typing.PyTree:
    """Computes `A(x)`."""
    return self.operator.matvec(x)

  def adjoint(self, dual: _typing.PyTree) -> _typing.PyTree:
    """Computes `A^*(dual)`."""
    return self.operator.rmatvec(dual)

  def residual(self, x: _typing.PyTree) -> _typing.PyTree:
    """Computes `A(x) - Pi_C(A(x))`."""
    ax = self.operator.matvec(x)
    return tree.sub(ax, self.target_set.project(ax))

  def penalty_and_dual(
      self,
      x: _typing.PyTree,
      *,
      multiplier: _typing.PyTree | None,
      penalty: _typing.Scalar,
      dampen: _typing.Scalar = 0.0,
  ) -> tuple[jax.Array, _typing.PyTree]:
    """Evaluates Moreau-envelope penalty `psi_c` and effective dual `p_hat`."""
    ax = self.operator.matvec(x)
    if multiplier is None:
      proj = self.target_set.project(ax)
      diff = tree.sub(ax, proj)
      p_hat = tree.scale(penalty, diff)
      psi = 0.5 * penalty * tree.norm_sq(diff)
      return psi, p_hat

    p_bar = tree.scale(1.0 - dampen, multiplier)
    w = tree.add_scaled(ax, 1.0 / penalty, p_bar)
    proj = self.target_set.project(w)
    diff = tree.sub(w, proj)
    p_hat = tree.scale(penalty, diff)
    psi = 0.5 * penalty * tree.norm_sq(diff) - (0.5 / penalty) * tree.norm_sq(
        p_bar
    )
    return psi, p_hat


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class SetConstraint:
  """Unbound set-inclusion constraint `operator(x) in target_set`."""

  operator: LinearOperator
  target_set: sets.ConvexSet

  def bind(self, x_example: _typing.PyTree) -> BoundSetConstraint:
    """Binds `operator` to `x_example`."""
    return BoundSetConstraint(
        operator=self.operator.bind(x_example),
        target_set=self.target_set,
    )


def _is_constraint(node: typing.Any) -> bool:
  return isinstance(node, Constraint | BoundConstraint)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class BoundProductConstraint:
  """Product of named/indexed `BoundConstraint` objects with PyTree duals."""

  constraints: _typing.PyTree

  @property
  def operator_norm_sq(self) -> jax.Array | float | None:
    """Upper bound `sum_i ||A_i||_op^2` if all components provide bounds."""
    leaves = jax.tree.leaves(self.constraints, is_leaf=_is_constraint)
    bounds = [c.operator_norm_sq for c in leaves]
    if not bounds or any(b is None for b in bounds):
      return None
    non_null = [jnp.asarray(b) for b in bounds if b is not None]
    return tree.functools_sum(non_null)

  def zero_dual(self, x_example: _typing.PyTree) -> _typing.PyTree:
    """Returns matching PyTree of zero dual variables."""
    return jax.tree.map(
        lambda c: c.zero_dual(x_example),
        self.constraints,
        is_leaf=_is_constraint,
    )

  def apply(self, x: _typing.PyTree) -> _typing.PyTree:
    """Evaluates `{k: A_k(x)}` across all constraints."""
    return jax.tree.map(
        lambda c: c.apply(x),
        self.constraints,
        is_leaf=_is_constraint,
    )

  def adjoint(self, dual: _typing.PyTree) -> _typing.PyTree:
    """Computes `sum_k A_k^*(dual_k)`."""
    leaves = jax.tree.leaves(self.constraints, is_leaf=_is_constraint)
    dual_subtrees = jax.tree.leaves(
        jax.tree.map(
            lambda c, d: c.adjoint(d),
            self.constraints,
            dual,
            is_leaf=_is_constraint,
        ),
        is_leaf=lambda node: False,
    )
    del dual_subtrees
    # Map each constraint + dual subtree into a primal PyTree, then sum them.
    primal_terms: list[_typing.PyTree] = []

    def _collect(c: BoundConstraint, d: _typing.PyTree) -> None:
      primal_terms.append(c.adjoint(d))

    jax.tree.map(_collect, self.constraints, dual, is_leaf=_is_constraint)
    if not leaves:
      raise ValueError(
          "ProductConstraint must contain at least one constraint."
      )
    acc = primal_terms[0]
    for term in primal_terms[1:]:
      acc = tree.add(acc, term)
    return acc

  def residual(self, x: _typing.PyTree) -> _typing.PyTree:
    """Computes `{k: A_k(x) - Pi_{C_k}(A_k(x))}`."""
    return jax.tree.map(
        lambda c: c.residual(x),
        self.constraints,
        is_leaf=_is_constraint,
    )

  def penalty_and_dual(
      self,
      x: _typing.PyTree,
      *,
      multiplier: _typing.PyTree | None,
      penalty: _typing.Scalar,
      dampen: _typing.Scalar = 0.0,
  ) -> tuple[jax.Array, _typing.PyTree]:
    """Sums scalar penalties and collects the PyTree of effective duals."""
    penalties: list[jax.Array] = []
    if multiplier is None:

      def _eval_no_mult(c: BoundConstraint) -> _typing.PyTree:
        psi_i, p_i = c.penalty_and_dual(
            x,
            multiplier=None,
            penalty=penalty,
            dampen=dampen,
        )
        penalties.append(psi_i)
        return p_i

      p_hat = jax.tree.map(
          _eval_no_mult,
          self.constraints,
          is_leaf=_is_constraint,
      )
    else:

      def _eval_with_mult(
          c: BoundConstraint,
          m_i: _typing.PyTree,
      ) -> _typing.PyTree:
        psi_i, p_i = c.penalty_and_dual(
            x,
            multiplier=m_i,
            penalty=penalty,
            dampen=dampen,
        )
        penalties.append(psi_i)
        return p_i

      p_hat = jax.tree.map(
          _eval_with_mult,
          self.constraints,
          multiplier,
          is_leaf=_is_constraint,
      )

    total_psi = (
        tree.functools_sum(penalties)
        if penalties
        else jnp.asarray(0.0, dtype=tree.infer_dtype(x))
    )
    return total_psi, p_hat


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class ProductConstraint:
  """Product of named/indexed `Constraint` objects arranged as a PyTree."""

  constraints: _typing.PyTree

  def bind(self, x_example: _typing.PyTree) -> BoundProductConstraint:
    """Binds every component constraint to `x_example`."""
    bound = jax.tree.map(
        lambda c: c.bind(x_example),
        self.constraints,
        is_leaf=_is_constraint,
    )
    return BoundProductConstraint(constraints=bound)


def equality(
    operator: OperatorLike,
    rhs: _typing.PyTree = 0.0,
    *,
    rmatvec: abc.Callable[[_typing.PyTree], _typing.PyTree] | None = None,
    operator_norm_sq: jax.Array | float | None = None,
) -> SetConstraint:
  """Creates an equality constraint `operator(x) == rhs`."""
  lin_op = as_linear_operator(
      operator,
      rmatvec=rmatvec,
      operator_norm_sq=operator_norm_sq,
  )
  return SetConstraint(operator=lin_op, target_set=sets.Singleton(rhs))


def less_equal(
    operator: OperatorLike,
    rhs: _typing.PyTree = 0.0,
    *,
    rmatvec: abc.Callable[[_typing.PyTree], _typing.PyTree] | None = None,
    operator_norm_sq: jax.Array | float | None = None,
) -> SetConstraint:
  """Creates an inequality constraint `operator(x) <= rhs`."""
  lin_op = as_linear_operator(
      operator,
      rmatvec=rmatvec,
      operator_norm_sq=operator_norm_sq,
  )
  return SetConstraint(operator=lin_op, target_set=sets.UpperBound(rhs))


def greater_equal(
    operator: OperatorLike,
    rhs: _typing.PyTree = 0.0,
    *,
    rmatvec: abc.Callable[[_typing.PyTree], _typing.PyTree] | None = None,
    operator_norm_sq: jax.Array | float | None = None,
) -> SetConstraint:
  """Creates an inequality constraint `operator(x) >= rhs`."""
  lin_op = as_linear_operator(
      operator,
      rmatvec=rmatvec,
      operator_norm_sq=operator_norm_sq,
  )
  return SetConstraint(operator=lin_op, target_set=sets.LowerBound(rhs))


def box(
    operator: OperatorLike,
    *,
    lower: _typing.PyTree = -jnp.inf,
    upper: _typing.PyTree = jnp.inf,
    rmatvec: abc.Callable[[_typing.PyTree], _typing.PyTree] | None = None,
    operator_norm_sq: jax.Array | float | None = None,
) -> SetConstraint:
  """Creates a two-sided box constraint `lower <= operator(x) <= upper`."""
  lin_op = as_linear_operator(
      operator,
      rmatvec=rmatvec,
      operator_norm_sq=operator_norm_sq,
  )
  return SetConstraint(
      operator=lin_op,
      target_set=sets.BoxSet(lower=lower, upper=upper),
  )


def in_set(
    operator: OperatorLike,
    target_set: sets.ConvexSet,
    *,
    rmatvec: abc.Callable[[_typing.PyTree], _typing.PyTree] | None = None,
    operator_norm_sq: jax.Array | float | None = None,
) -> SetConstraint:
  """Creates a general convex-set constraint `operator(x) in target_set`."""
  lin_op = as_linear_operator(
      operator,
      rmatvec=rmatvec,
      operator_norm_sq=operator_norm_sq,
  )
  return SetConstraint(operator=lin_op, target_set=target_set)


def normalize_constraints(
    spec: Constraint | _typing.PyTree | None,
) -> Constraint | None:
  """Normalizes `None`, a `Constraint`, or a PyTree of `Constraint` objects."""
  if spec is None:
    return None
  if isinstance(spec, Constraint):
    return spec
  leaves = jax.tree.leaves(spec, is_leaf=_is_constraint)
  if not leaves or not all(isinstance(leaf, Constraint) for leaf in leaves):
    raise TypeError(
        "`constraints` must be None, a Constraint, or a PyTree of Constraints."
    )
  return ProductConstraint(constraints=spec)
