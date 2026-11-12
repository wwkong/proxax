# `proxax` Documentation

Certified first-order proximal point methods for unconstrained and constrained
nonconvex composite optimization in **JAX** and **Optax**.

## Mathematical Scope

1. **Unconstrained Nonconvex Composite Optimization (`AIPP`):**

   $$\min_{x \in \mathcal{X}} \; f(x) + h(x)$$

2. **Set / Cone-Constrained Nonconvex Composite Optimization (`QPAIPP`):**

   $$\min_{x \in \mathcal{X}} \; f(x) + h(x) \quad \text{s.t.} \quad \mathcal{A}(x) \in \mathcal{C}$$

```{toctree}
:maxdepth: 2
:caption: Contents

nc_opt_reference
api
```
