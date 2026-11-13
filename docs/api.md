# API Reference

All public symbols in Proxax are exposed through the unified `px` facade
(`from proxax import px`, or `import proxax as px`). For conceptual
introductions and basic examples, see {doc}`guide/solving_problems` and
{doc}`guide/prox_and_constraints`.

---

## Problems, Solutions & Top-Level `px.solve`

Core problem containers (`px.CompositeProblem`, `px.SpectralProblem`), the
unified `px.solve` entry point, common return types (`px.Solution`,
`px.SolveStats`, `px.SolveStatus`, `px.StoppingConfig`, `px.KKTResiduals`),
composable solver state trackers (`px.solvers.SolverTracker`,
`px.solvers.ConstrainedTracker`, `px.solvers.OracleCounts`), and type aliases
(`px.typing`).

```{eval-rst}
.. automodule:: proxax.problem
   :members:

.. automodule:: proxax.solve
   :members:

.. automodule:: proxax.solvers.base
   :members:

.. automodule:: proxax.typing
   :members:
```

---

## Proximal Operators (`px.prox`)

Convex norm regularizers (`L1`, `L2`, `L2Squared`, `ElasticNet`, `NuclearNorm`),
set indicators (`NonNegative`, `Box`, `BoxL1`, `Simplex`, `L2Ball`, `L1Ball`),
and PyTree combinators (`MultiProx`, `Shifted`, `Zero`).

```{eval-rst}
.. automodule:: proxax.prox.base
   :members:

.. automodule:: proxax.prox.norms
   :members:

.. automodule:: proxax.prox.indicators
   :members:

.. automodule:: proxax.prox.combinators
   :members:
```

---

## Constraints & Convex Sets (`px.constraints`)

Convenience builders (`equality`, `less_equal`, `greater_equal`, `box`,
`in_set`), matrix, identity (`identity`), and callable PyTree linear operators,
and Euclidean-projected `ConvexSet` implementations (`Singleton`, `UpperBound`,
`LowerBound`, `BoxSet`, `SecondOrderCone`, `ProductSet`).

```{eval-rst}
.. automodule:: proxax.constraints.linear_op
   :members:

.. automodule:: proxax.constraints.sets
   :members:
```

---

## Oracles & Stationarity Certificates (`px.oracle`, `px.certificates`)

First-order smooth, composite, penalized, and spectral oracles, together with
exact subdifferential refinement (`refine`, `refine_icg`).

```{eval-rst}
.. automodule:: proxax.oracle.terms
   :members:

.. automodule:: proxax.oracle.composite
   :members:

.. automodule:: proxax.oracle.spectral
   :members:

.. automodule:: proxax.certificates.stationarity
   :members:
```

---

## Unconstrained Composite Solvers (`px.solvers`)

Certified proximal-point and accelerated composite gradient solvers for
$\min_x f(x) + h(x)$.

```{eval-rst}
.. automodule:: proxax.solvers.unconstrained.apd
   :members:

.. automodule:: proxax.solvers.unconstrained.aipp
   :members:

.. automodule:: proxax.solvers.unconstrained.ac_acg
   :members:

.. automodule:: proxax.solvers.unconstrained.adap_fista
   :members:

.. automodule:: proxax.solvers.unconstrained.nc_fista
   :members:

.. automodule:: proxax.solvers.unconstrained.adap_apg
   :members:

.. automodule:: proxax.solvers.unconstrained.upfag
   :members:

.. automodule:: proxax.solvers.unconstrained.ag
   :members:

.. automodule:: proxax.solvers.unconstrained.ecg
   :members:
```

---

## Set- and Cone-Constrained Solvers (`px.solvers`)

Quadratic penalty and inexact/smoothed proximal augmented Lagrangian solvers for
$\min_x f(x) + h(x)$ subject to $\mathcal{A}(x) \in \mathcal{C}$.

```{eval-rst}
.. automodule:: proxax.solvers.constrained.ialm
   :members:

.. automodule:: proxax.solvers.constrained.qp_apd
   :members:

.. automodule:: proxax.solvers.constrained.qp_aipp
   :members:

.. automodule:: proxax.solvers.constrained.iaipal
   :members:

.. automodule:: proxax.solvers.constrained.s_prox_alm
   :members:
```

---

## Matrix Spectral Solvers (`px.solvers`)

Singular-value composite solvers requiring only a single thin SVD per outer
iteration on `SpectralProblem` or `NuclearNorm` objectives.

```{eval-rst}
.. automodule:: proxax.solvers.unconstrained.da_icg
   :members:

.. automodule:: proxax.solvers.unconstrained.ia_icg
   :members:
```

---

## Optax Compatibility (`px.optax_compat`)

Single-step `optax.GradientTransformation` and
`optax.GradientTransformationExtraArgs` state machines, together with composable
outer trackers (`OptaxTracker`, `ConstrainedOptaxTracker`) and inner subproblem
carries (`ACGSubproblemCarry`, `PFAPDSubproblemCarry`), for integrating Proxax
into custom Optax loops.

```{eval-rst}
.. automodule:: proxax.optax_compat.transforms
   :members:
```

---

## Internal Subproblem Solvers (`proxax.solvers.inner`)

Strongly convex inner accelerated composite gradient subroutines (`ACG` and
`PF.ACG`) used inside `AIPP`, `APD`, `IAICG`, `DAICG`, `IALM`, and `IAIPAL`.

```{eval-rst}
.. automodule:: proxax.solvers.inner.acg
   :members:

.. automodule:: proxax.solvers.inner.pf_acg
   :members:
```
