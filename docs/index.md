# `proxax` Documentation

Certified first-order proximal point methods for unconstrained and constrained
nonconvex composite optimization in **JAX** and **Optax**.

## Mathematical Scope

1. **Unconstrained Nonconvex Composite & Spectral Optimization (`AIPP`, `APD`, `ECG`, `AG`, `UPFAG`, `NCFISTA`, `AdapFISTA`, `ACACG`, `AdapAPG`, `IAICG`, `DAICG`):**

   $$\min_{x \in \mathcal{X}} \; f(x) + h(x)$$

2. **Set / Cone-Constrained Nonconvex Composite Optimization (`QPAIPP`, `QPAPD`, `IALM`, `SProxALM`, `IAIPAL`):**

   $$\min_{x \in \mathcal{X}} \; f(x) + h(x) \quad \text{s.t.} \quad \mathcal{A}(x) \in \mathcal{C}$$

```{toctree}
:maxdepth: 2
:caption: Contents

nc_opt_reference
api
```
