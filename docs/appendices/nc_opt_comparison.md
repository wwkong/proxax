# Appendix B: Comparison & Mapping to `nc_opt`

Proxax ports and generalizes the certified nonconvex composite optimization
suite of **[`wwkong/nc_opt`](https://github.com/wwkong/nc_opt)** (MATLAB) into
an idiomatic **JAX** and **Optax** library. This appendix summarizes the key
architectural differences and provides complete lookup tables for users
familiar with `nc_opt`.

---

## 1. Architectural Differences: MATLAB `nc_opt` vs. JAX Proxax

| Design Dimension | `wwkong/nc_opt` (MATLAB) | Proxax (Python / JAX / Optax) |
| :--- | :--- | :--- |
| **Primal & Dual Variable Spaces** | Flat column vectors `R^n` or 2D matrices `R^{p x q}`; multi-block variables (e.g., Sparse PCA) must be concatenated and sliced manually. | Arbitrary **JAX PyTrees** (`jax.Array`, nested `dict`, `tuple`, registered dataclasses) with `prox.MultiProx` and `ProductConstraint` preserving exact structure. |
| **Gradients & Linear Adjoints** | User must code both `grad_f(x)` and explicit matrix transposes `A'(p)` by hand. | Smooth functions default to `jax.value_and_grad(fn)`, and callable linear operators synthesize $\mathcal{A}^*$ automatically via `jax.linear_transpose`. |
| **Constraint Domain** | Primarily affine equality constraints $A x = b$ (with auxiliary slack variables required for inequalities or cones). | Unified convex-set inclusion $\mathcal{A}(x) \in \mathcal{C}$ supporting equalities, inequalities, boxes, second-order cones, and PyTree Cartesian products natively. |
| **Execution & Parallelism** | Mutable object-oriented MATLAB `handle` classes (`Oracle`, `AlgFramework`) running sequentially on CPU. | Immutable PyTree state machines compiled end-to-end with `jax.lax.while_loop` (`jax.jit`) and parallelizable across batches on CPU/GPU/TPU via `jax.vmap`. |
| **Step-by-Step Integration** | Monolithic `solve` calls only. | Both full `proxax.solve(...)` and single-inner-step `optax.GradientTransformation` state machines (`proxax.optax_compat`). |
| **Floating-Point Precision** | IEEE `float64` only. | Works in both default JAX `float32` (with cancellation-free algebra and `Precision.HIGHEST` SVD guards against GPU TF32 truncation) and `float64`. |

---

## 2. Module & Solver Mapping (`nc_opt` $\to$ Proxax)

| `nc_opt` File / Class | Proxax Equivalent | Notes |
| :--- | :--- | :--- |
| `src/oracles/Oracle.m` | `proxax.oracle.CompositeOracle`, `proxax.oracle.terms.SmoothTerm`, `proxax.prox.ProxOperator` | Separates smooth curvature `(m, M)` from nonsmooth `ProxOperator` and `domain_project`. |
| `src/oracles/SpectralOracle.m` + `refine_ICG.m` | `proxax.SpectralOracle`, `proxax.SpectralProblem`, `proxax.oracle.spectral.refine_icg` | Curvature redistribution $\xi$, 1D `VectorSpectralSubOracle`, and exact matrix subgradient refinement. |
| `Oracle.proxify` | `proxax.oracle.ProxRegularizedOracle`, `proxax.oracle.RescaledProxOracle` | Zero-retracing PyTree subproblem wrappers for `AIPP` and `APD`. |
| `refine.m` | `proxax.certificates.refine` | Single adaptive composite gradient step returning exact $(\hat{x}, \hat{v})$. |
| `ACG.m`, `PF_ACG.m` | `proxax.solvers.inner.acg`, `proxax.solvers.inner.pf_acg` | Inner strongly convex accelerated composite gradient engines. |
| `R_AIPP.m` / `AIPP.m` | `proxax.solvers.AIPP` (`method="aipp"`) | Known-$m$ or adaptive-$\lambda_k$ accelerated inexact proximal point solver. |
| `PF_APD.m` (`tests/papers/apd`) | `proxax.solvers.APD` (`method="apd"`) | Parameter-free accelerated proximal descent with dynamic curvature decay. |
| `ECG.m` | `proxax.solvers.ECG` (`method="ecg"`) | Adaptive backtracking composite proximal gradient. |
| `AG.m`, `UPFAG.m` | `proxax.solvers.AG` (`"ag"`), `proxax.solvers.UPFAG` (`"upfag"`) | Ghadimi–Lan and unified parameter-free accelerated gradient methods. |
| `NC_FISTA.m`, `ADAP_FISTA.m` | `proxax.solvers.NCFISTA` (`"nc_fista"`), `proxax.solvers.AdapFISTA` (`"adap_fista"`) | Nonconvex and adaptive parameter-free FISTA. |
| `AC_ACG.m` | `proxax.solvers.ACACG` (`method="ac_acg"`) | Average-curvature accelerated composite gradient. |
| `AdapAPG.m` | `proxax.solvers.AdapAPG` (`method="adap_apg"`) | Lin–Xiao adaptive accelerated proximal gradient with relaxation. |
| `IA_ICG.m`, `DA_ICG.m` | `proxax.solvers.IAICG` (`"ia_icg"`), `proxax.solvers.DAICG` (`"da_icg"`) | Inner-accelerated and doubly-accelerated spectral composite solvers. |
| `QP_AIPP.m` / `penalty.m` | `proxax.solvers.QPAIPP` (`"qp_aipp"`), `proxax.solvers.QPAPD` (`"qp_apd"`) | Quadratic penalty frameworks driven by `AIPP` or parameter-free `APD`. |
| `NL_IAPIAL.m`, `AIDAL.m` | `proxax.solvers.IALM` (`"ialm"`, `"nl_iapial"`, `"aidal"`) | Proximal and dampened ($\theta \in [0, 1)$) inexact augmented Lagrangian. |
| `IAIPAL.m` | `proxax.solvers.IAIPAL` (`method="iaipal"`) | Classical-Lagrangian single-`ACG`-step proximal ALM. |
| `sProxALM.m` | `proxax.solvers.SProxALM` (`method="s_prox_alm"`) | Single-loop smoothed proximal ALM. |

---

## 3. Benchmark & Example Mapping

Every benchmark script in `nc_opt/examples/` has a runnable JAX counterpart in
`proxax/examples/` and a corresponding tutorial page:

| `nc_opt` MATLAB Script & Generator | Proxax Script | Documentation Section |
| :--- | :--- | :--- |
| `examples/unconstrained/basic_convex_qp.m` | `examples/unconstrained_nc_opt.py` | {doc}`../examples/unconstrained` (Section 1) |
| `examples/unconstrained/nonconvex_qp.m` (`test_fn_unconstr_01.m`) | `examples/unconstrained_nc_opt.py` | {doc}`../examples/unconstrained` (Section 2) |
| `examples/unconstrained/nonconvex_svm.m` (`test_fn_svm_01.m`) | `examples/unconstrained_nc_opt.py` | {doc}`../examples/unconstrained` (Section 3) |
| `examples/spectral/nonconvex_spectral_mc.m` (`test_fn_spectral_mc_01.m`) | `examples/spectral_matrix_completion.py` | {doc}`../examples/spectral` |
| `examples/constrained/nonconvex_lin_constr_qp.m` (`test_fn_lin_constr_01.m`) | `examples/constrained_nc_opt.py` | {doc}`../examples/constrained` (Section 1) |
| `examples/constrained/nonconvex_sparse_pca.m` (`test_fn_spca_01.m`) | `examples/constrained_nc_opt.py` | {doc}`../examples/constrained` (Section 2) |
| `examples/matrix_completion/nonconvex_bounded_mc.m` | `examples/constrained_nc_opt.py` | {doc}`../examples/constrained` (Section 3) |
| *(JAX / Optax native extension)* | `examples/optax_and_vmap.py` | {doc}`../guide/optax_and_jax` |
