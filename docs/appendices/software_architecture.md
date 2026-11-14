# Appendix E: Software Architecture

This appendix provides a visual overview of Proxax's layered package structure,
solve-time dispatch pipeline, inner–outer solver composition, and batched /
differentiable execution paths.

> **Tip:** All diagrams below support interactive **scroll-to-zoom, drag-to-pan**,
> and the **fullscreen** button in the corner.

---

## 1. Layered Package Architecture

Proxax separates user-facing entry points, problem/operator modeling,
certified solver engines, and the underlying `jax` / `optax` runtime substrate
into four layers:

```{mermaid}
%%{init: {'flowchart': {'curve': 'basis', 'nodeSpacing': 24, 'rankSpacing': 40}}}%%
flowchart TB
    classDef user stroke:#3b82f6,stroke-width:2px
    classDef model stroke:#10b981,stroke-width:2px
    classDef solver stroke:#8b5cf6,stroke-width:2px
    classDef substrate stroke:#f59e0b,stroke-width:2px

    subgraph L1["1. User & Ecosystem Layer"]
        direction TB
        PX(["proxax.px (curated facade)"]):::user
        SOLVE["solve / batch_solve"]:::user
        DIFF["implicit_solve / layers"]:::user
        OPTAX["optax_compat"]:::user
        PLOT["plotting"]:::user
        PX --> SOLVE & DIFF & OPTAX & PLOT
    end

    subgraph L2["2. Problem & Operator Modeling Layer"]
        direction TB
        PROB["problem<br/>CompositeProblem,<br/>LinearConstraint"]:::model
        ORA["oracle<br/>composite, spectral,<br/>minmax, block, terms"]:::model
        PREC["precondition<br/>ruiz_equilibrate"]:::model
        PROX["prox<br/>norms, indicators,<br/>folded_concave, exotic"]:::model
        CONS["constraints<br/>sets, linear_op"]:::model
        PROB --> PROX & CONS
        ORA --> PROX & CONS
        PREC ~~~ CONS
    end

    subgraph L3["3. Solver & Certificate Engine Layer"]
        direction TB
        CON["solvers.constrained<br/>IALM, IAIPAL, SProxALM,<br/>QPAIPP, QPAPD"]:::solver
        UNC["solvers.unconstrained<br/>APD, PANOC, AIPP, ..."]:::solver
        INN["solvers.inner<br/>ACG, PFACG"]:::solver
        MM["solvers.minmax<br/>ExtraGradient, PDHG, ProxPoint"]:::solver
        BLK["solvers.block<br/>ADMM, DPADMM, PAdaptADMM"]:::solver
        CERT["certificates<br/>stationarity, minmax"]:::solver
        LOOP["_loop & tree<br/>while_loop & PyTree linalg"]:::solver
        CON --> INN & UNC
        UNC & MM & BLK --> CERT & LOOP
    end

    subgraph L4["4. JAX & Optax Runtime Substrate"]
        direction LR
        JAX[("jax<br/>jit, vmap, grad, custom_vjp,<br/>linear_transpose, while_loop")]:::substrate
        OX[("optax<br/>optax.tree, projections,<br/>GradientTransformationExtraArgs")]:::substrate
    end

    L1 --> L2 --> L3 --> L4

    style L1 fill:transparent,stroke:#3b82f6,stroke-width:1.5px
    style L2 fill:transparent,stroke:#10b981,stroke-width:1.5px
    style L3 fill:transparent,stroke:#8b5cf6,stroke-width:1.5px
    style L4 fill:transparent,stroke:#f59e0b,stroke-width:1.5px
```

---

## 2. `px.solve` Execution & Dispatch Pipeline

When `px.solve(problem, x0, solver=...)` is called, Proxax normalizes the
problem, projects the initial iterate onto $\operatorname{dom}(h)$, dispatches
either to a compiled `jax.lax.while_loop` (`jit=True`) or a Python loop with
wall-clock `time_limit` support (`jit=False`), and emits a certified `Solution`:

```{mermaid}
%%{init: {'flowchart': {'curve': 'basis', 'nodeSpacing': 30, 'rankSpacing': 36}}}%%
flowchart TB
    classDef io stroke:#3b82f6,stroke-width:2px
    classDef decision stroke:#f43f5e,stroke-width:2px
    classDef engine stroke:#8b5cf6,stroke-width:2px
    classDef cert stroke:#10b981,stroke-width:2px

    IN(["px.solve(problem, x0,<br/>solver=..., jit=True)"]):::io
    NORM["Normalize problem &<br/>bind LinearOperator to x0"]
    DOM["Project x0 onto dom(h)<br/>via proximal.domain_project"]
    DISP{"solver is None?"}:::decision
    AUTO["Auto-select default solver<br/>by problem class & constraints"]
    EXEC["Initialize SolverState"]
    JIT{"jit=True?"}:::decision
    XLA["jax.lax.while_loop<br/>(zero host-dispatch overhead)"]:::engine
    PY["Python loop via _loop.py<br/>(time_limit & callbacks)"]:::engine
    CERT["Compute stationarity / KKT<br/>certificate (x_hat, p_hat, v_hat)"]:::cert
    OUT(["Solution(status, x, dual,<br/>opt_res, feas_res, stats)"]):::cert

    IN --> NORM --> DOM --> DISP
    DISP -->|"Yes (auto)"| AUTO --> EXEC
    DISP -->|"No"| EXEC
    EXEC --> JIT
    JIT -->|"Yes"| XLA --> CERT
    JIT -->|"No"| PY --> CERT
    CERT --> OUT
```

---

## 3. Constrained Inner–Outer Solver Hierarchy

Constrained solvers (`IALM`, `IAIPAL`, `SProxALM`, `QPAIPP`, `QPAPD`) transform
set- and cone-constrained problems into smooth penalized or proximal-regularized
subproblems without re-compiling XLA graphs across outer iterations:

```{mermaid}
%%{init: {'flowchart': {'curve': 'basis', 'nodeSpacing': 30, 'rankSpacing': 36}}}%%
flowchart TB
    classDef prob stroke:#3b82f6,stroke-width:2px
    classDef oracle stroke:#10b981,stroke-width:2px
    classDef decision stroke:#f43f5e,stroke-width:2px
    classDef inner stroke:#8b5cf6,stroke-width:2px
    classDef outer stroke:#f59e0b,stroke-width:2px
    classDef done stroke:#10b981,stroke-width:2px

    CP(["CompositeProblem<br/>min f(x) + h(x) s.t. A(x) in C"]):::prob
    PEN["PenalizedOracle<br/>(dynamic multiplier p_bar, penalty c)"]:::oracle
    REG["ProxRegularizedOracle<br/>(dynamic center x_k, smooth_reg, prox_reg)"]:::oracle

    SUB{"Inner subproblem solver"}:::decision
    ACG["inner.ACG / inner.PFACG<br/>accelerated convex"]:::inner
    APD["unconstrained.APD<br/>parameter-free"]:::inner
    PANOC["unconstrained.PANOC<br/>quasi-Newton FBE"]:::inner

    REF["Stationarity & KKT refinement<br/>(certificates.stationarity)"]:::outer
    UPD{"opt_res <= opt_tol &<br/>feas_res <= feas_tol?"}:::decision
    STEP["Outer dual / penalty update<br/>p_{k+1}, c_{k+1}"]:::outer
    DONE(["Certified KKT Solution<br/>(x_hat, p_hat, v_hat)"]):::done

    CP --> PEN --> REG --> SUB
    SUB --> ACG & APD & PANOC
    ACG & APD & PANOC --> REF --> UPD
    UPD -->|"No"| STEP --> PEN
    UPD -->|"Yes"| DONE
```

* **Zero re-compilation:** `PenalizedOracle` and `ProxRegularizedOracle` are
  registered as JAX PyTrees (`jax.tree_util.register_dataclass`) with callables in
  `meta_fields` and scalar/array parameters (`center`, `multiplier`, `penalty`)
  in `data_fields`.

---

## 4. Batched Solving & Implicit Differentiation

For batched workloads (`px.batch_solve`, `px.batch_implicit_solve`, and
`proxax.layers`), Proxax chooses between multi-threaded precompiled execution on
CPU and vectorized `jax.vmap` execution on GPU, while sharing a `custom_vjp`
implicit-differentiation backward pass:

```{mermaid}
%%{init: {'flowchart': {'curve': 'basis', 'nodeSpacing': 30, 'rankSpacing': 36}}}%%
flowchart TB
    classDef io stroke:#3b82f6,stroke-width:2px
    classDef decision stroke:#f43f5e,stroke-width:2px
    classDef cpu stroke:#10b981,stroke-width:2px
    classDef gpu stroke:#8b5cf6,stroke-width:2px
    classDef bwd stroke:#f59e0b,stroke-width:2px

    B_IN(["px.batch_solve / batch_implicit_solve<br/>(x0_batch, params_batch, backend='auto')"]):::io
    DEV{"Device & batch size B?"}:::decision
    THR["Persistent ThreadPoolExecutor +<br/>precompiled jax.pure_callback"]:::cpu
    VMAP["jax.vmap over<br/>jit-compiled px.solve"]:::gpu
    FWD(["Batched Solution: x_star_batch"]):::io
    BWD["Backward pass (custom_vjp)<br/>implicit linear solve on KKT /<br/>fixed-point residual (CG / GMRES)"]:::bwd

    B_IN --> DEV
    DEV -->|"CPU, B > 1 (threads)"| THR --> FWD
    DEV -->|"GPU or B = 1 (vmap)"| VMAP --> FWD
    FWD --> BWD
```

See [Appendix D: Implementation Optimizations](implementation_optimizations.md)
for details on thread-pool caching, NumPy pre-slicing, and `vmap` memory layout.
