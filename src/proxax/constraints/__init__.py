"""Linear operators, convex sets, and set-inclusion constraints."""

from __future__ import annotations

from proxax.constraints import linear_op
from proxax.constraints import sets

BoundCallableLinearOperator = linear_op.BoundCallableLinearOperator
BoundConstraint = linear_op.BoundConstraint
BoundLinearOperator = linear_op.BoundLinearOperator
BoundProductConstraint = linear_op.BoundProductConstraint
BoundSetConstraint = linear_op.BoundSetConstraint
BoxSet = sets.BoxSet
CallableLinearOperator = linear_op.CallableLinearOperator
Constraint = linear_op.Constraint
ConvexSet = sets.ConvexSet
LinearOperator = linear_op.LinearOperator
LowerBound = sets.LowerBound
MatrixLinearOperator = linear_op.MatrixLinearOperator
ProductConstraint = linear_op.ProductConstraint
ProductSet = sets.ProductSet
SecondOrderCone = sets.SecondOrderCone
SetConstraint = linear_op.SetConstraint
Singleton = sets.Singleton
UpperBound = sets.UpperBound
ZeroSet = sets.ZeroSet
as_linear_operator = linear_op.as_linear_operator
box = linear_op.box
equality = linear_op.equality
greater_equal = linear_op.greater_equal
in_set = linear_op.in_set
less_equal = linear_op.less_equal
normalize_constraints = linear_op.normalize_constraints

__all__ = [
    "BoundCallableLinearOperator",
    "BoundConstraint",
    "BoundLinearOperator",
    "BoundProductConstraint",
    "BoundSetConstraint",
    "BoxSet",
    "CallableLinearOperator",
    "Constraint",
    "ConvexSet",
    "LinearOperator",
    "LowerBound",
    "MatrixLinearOperator",
    "ProductConstraint",
    "ProductSet",
    "SecondOrderCone",
    "SetConstraint",
    "Singleton",
    "UpperBound",
    "ZeroSet",
    "as_linear_operator",
    "box",
    "equality",
    "greater_equal",
    "in_set",
    "less_equal",
    "normalize_constraints",
]
