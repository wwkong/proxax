# Appendix E: Software Architecture

This appendix provides a visual overview of Proxax's layered package structure,
solve-time dispatch pipeline, inner–outer solver composition, and batched /
differentiable execution paths.

---

## 1. Layered Package Architecture

Proxax separates user-facing entry points, problem/operator modeling,
certified solver engines, and the underlying `jax` / `optax` runtime substrate
into four layers:

```{mermaid}
%%{init: {'theme': 'base', 'flowchart': {'curve': 'basis', 'padding': 16, 'nodeSpacing': 30, 'rankSpacing': 46}, 'themeVariables': {'fontFamily': 'Inter, system-ui, -apple-system, sans-serif', 'fontSize': '13.5px', 'lineColor': '#64748b', 'primaryTextColor': '#0f172a'}}}%%
flowchart TB
    classDef user fill:#dbeafe,stroke:#2563eb,color:#1e3a8a,stroke-width:1.5px
    classDef model fill:#d1fae5,stroke:#059669,color:#064e3b,stroke-width:1.5px
    classDef solver fill:#ede9fe,stroke:#7c3aed,color:#3b0764,stroke-width:1.5px
    classDef substrate fill:#fef3c7,stroke:#d97706,color:#78350f,stroke-width:1.5px

    subgraph L1["User & Ecosystem Layer"]
        direction LR
        PX(["<b>proxax.px</b><br/>curated facade"]):::user
        SOLVE["<b>solve / batch_solve</b><br/>single & batched entry"]:::user
        DIFF["<b>implicit_solve / layers</b><br/>differentiable optimization"]:::user
        OPTAX["<b>optax_compat</b><br/>GradientTransformationExtraArgs"]:::user
        PLOT["<b>plotting</b><br/>convergence & spectrum"]:::user
    end

    subgraph L2["Problem & Operator Modeling Layer"]
        direction LR
        PROB["<b>problem</b><br/>CompositeProblem, LinearConstraint"]:::model
        PROX["<b>prox</b><br/>norms, indicators, folded_concave,<br/>exotic, combinators"]:::model
        CONS["<b>constraints</b><br/>sets, linear_op"]:::model
        ORA["<b>oracle</b><br/>composite, spectral, spectral_problem,<br/>minmax, block, terms"]:::model
        PREC["<b>precondition</b><br/>ruiz_equilibrate"]:::model
    end

    subgraph L3["Solver & Certificate Engine Layer"]
        direction LR
        UNC["<b>solvers.unconstrained</b><br/>APD, PANOC, AIPP, AdapFISTA, ..."]:::solver
        CON["<b>solvers.constrained</b><br/>IALM, IAIPAL, SProxALM, QPAIPP, QPAPD"]:::solver
        MM["<b>solvers.minmax</b><br/>ExtraGradient, PDHG, ProxPoint"]:::solver
        BLK["<b>solvers.block</b><br/>ADMM, DPADMM, PAdaptADMM"]:::solver
        INN["<b>solvers.inner</b><br/>ACG, PFACG"]:::solver
        CERT["<b>certificates</b><br/>stationarity, minmax"]:::solver
        LOOP["<b>_loop & tree</b><br/>while_loop / PyTree linalg"]:::solver
    end

    subgraph L4["JAX & Optax Runtime Substrate"]
        direction LR
        JAX[("<b>jax</b><br/>jit, vmap, grad, linear_transpose,<br/>custom_vjp, lax.while_loop")]:::substrate
        OX[("<b>optax</b><br/>optax.tree, optax.projections,<br/>GradientTransformationExtraArgs")]:::substrate
    end

    PX --> SOLVE & DIFF & OPTAX & PLOT
    SOLVE & DIFF & OPTAX --> PROB & ORA & PREC
    PROB --> PROX & CONS
    ORA --> PROX & CONS
    SOLVE --> UNC & CON & MM & BLK
    CON --> INN & UNC
    UNC & CON & MM & BLK --> CERT & LOOP
    L2 --> JAX & OX
    L3 --> JAX & OX

    style L1 fill:#eff6ff,stroke:#93c5fd,stroke-width:1.5px,color:#1e3a8a
    style L2 fill:#ecfdf5,stroke:#6ee7b7,stroke-width:1.5px,color:#064e3b
    style L3 fill:#f5f3ff,stroke:#c4b5fd,stroke-width:1.5px,color:#3b0764
    style L4 fill:#fffbeb,stroke:#fcd34d,stroke-width:1.5px,color:#78350f
```

---

## 2. `px.solve` Execution & Dispatch Pipeline

When `px.solve(problem, x0, solver=...)` is called, Proxax normalizes the
problem, projects the initial iterate onto $\operatorname{dom}(h)$, dispatches
either to a compiled `jax.lax.while_loop` (`jit=True`) or a Python loop with
wall-clock `time_limit` support (`jit=False`), and emits a certified `Solution`:

```{mermaid}
%%{init: {'theme': 'base', 'flowchart': {'curve': 'basis', 'padding': 14, 'nodeSpacing': 34, 'rankSpacing': 44}, 'themeVariables': {'fontFamily': 'Inter, system-ui, -apple-system, sans-serif', 'fontSize': '13.5px', 'lineColor': '#64748b', 'primaryTextColor': '#0f172a'}}}%%
flowchart LR
    classDef io fill:#dbeafe,stroke:#2563eb,color:#1e3a8a,stroke-width:1.5px
    classDef step fill:#f1f5f9,stroke:#475569,color:#0f172a,stroke-width:1.5px
    classDef decision fill:#ffe4e6,stroke:#e11d48,color:#881337,stroke-width:1.5px
    classDef engine fill:#ede9fe,stroke:#7c3aed,color:#3b0764,stroke-width:1.5px
    classDef cert fill:#dcfce7,stroke:#16a34a,color:#14532d,stroke-width:2px

    IN(["<b>px.solve</b>(problem, x0,<br/>solver=..., jit=True)"]):::io
    NORM["Normalize problem &<br/>bind LinearOperator to x0"]:::step
    DOM["Project x0 onto dom(h)<br/>via proximal.domain_project"]:::step
    DISP{"solver is None?"}:::decision
    AUTO["Auto-select default solver<br/>by problem class & constraints"]:::step
    EXEC["Initialize SolverState"]:::step
    JIT{"jit=True?"}:::decision
    XLA["<b>jax.lax.while_loop</b><br/>(zero host-dispatch overhead)"]:::engine
    PY["<b>Python loop</b> via _loop.py<br/>(supports time_limit & callbacks)"]:::engine
    CERT["Compute stationarity / KKT<br/>certificate (x_hat, p_hat, v_hat)"]:::cert
    OUT(["<b>Solution</b>(status, x, dual,<br/>opt_res, feas_res, stats)"]):::cert

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
%%{init: {'theme': 'base', 'flowchart': {'curve': 'basis', 'padding': 14, 'nodeSpacing': 32, 'rankSpacing': 42}, 'themeVariables': {'fontFamily': 'Inter, system-ui, -apple-system, sans-serif', 'fontSize': '13.5px', 'lineColor': '#64748b', 'primaryTextColor': '#0f172a'}}}%%
flowchart TB
    classDef prob fill:#dbeafe,stroke:#2563eb,color:#1e3a8a,stroke-width:1.5px
    classDef oracle fill:#d1fae5,stroke:#059669,color:#064e3b,stroke-width:1.5px
    classDef decision fill:#ffe4e6,stroke:#e11d48,color:#881337,stroke-width:1.5px
    classDef inner fill:#ede9fe,stroke:#7c3aed,color:#3b0764,stroke-width:1.5px
    classDef outer fill:#fef3c7,stroke:#d97706,color:#78350f,stroke-width:1.5px
    classDef done fill:#dcfce7,stroke:#16a34a,color:#14532d,stroke-width:2px

    CP(["<b>CompositeProblem</b><br/>min f(x) + h(x) s.t. A(x) in C"]):::prob
    PEN["<b>PenalizedOracle</b><br/>(dynamic multiplier p_bar, penalty c)"]:::oracle
    REG["<b>ProxRegularizedOracle</b><br/>(dynamic center x_k, smooth_reg, prox_reg)"]:::oracle

    SUB{"Inner subproblem solver"}:::decision
    ACG["<b>inner.ACG / inner.PFACG</b><br/>accelerated convex"]:::inner
    APD["<b>unconstrained.APD</b><br/>parameter-free"]:::inner
    PANOC["<b>unconstrained.PANOC</b><br/>quasi-Newton FBE"]:::inner

    REF["<b>Stationarity & KKT refinement</b><br/>(certificates.stationarity)"]:::outer
    UPD{"opt_res <= opt_tol &<br/>feas_res <= feas_tol?"}:::decision
    STEP["Outer dual / penalty update<br/>p_{k+1}, c_{k+1}"]:::outer
    DONE(["<b>Certified KKT Solution</b><br/>(x_hat, p_hat, v_hat)"]):::done

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
%%{init: {'theme': 'base', 'flowchart': {'curve': 'basis', 'padding': 14, 'nodeSpacing': 34, 'rankSpacing': 44}, 'themeVariables': {'fontFamily': 'Inter, system-ui, -apple-system, sans-serif', 'fontSize': '13.5px', 'lineColor': '#64748b', 'primaryTextColor': '#0f172a'}}}%%
flowchart LR
    classDef io fill:#dbeafe,stroke:#2563eb,color:#1e3a8a,stroke-width:1.5px
    classDef decision fill:#ffe4e6,stroke:#e11d48,color:#881337,stroke-width:1.5px
    classDef cpu fill:#d1fae5,stroke:#059669,color:#064e3b,stroke-width:1.5px
    classDef gpu fill:#ede9fe,stroke:#7c3aed,color:#3b0764,stroke-width:1.5px
    classDef bwd fill:#fef3c7,stroke:#d97706,color:#78350f,stroke-width:1.5px

    B_IN(["<b>px.batch_solve / batch_implicit_solve</b><br/>(x0_batch, params_batch, backend='auto')"]):::io
    DEV{"Device &<br/>batch size B?"}:::decision
    THR["<b>Persistent ThreadPoolExecutor</b> +<br/>precompiled jax.pure_callback"]:::cpu
    VMAP["<b>jax.vmap</b> over<br/>jit-compiled px.solve"]:::gpu
    FWD(["Batched Solution<br/><b>x_star_batch</b>"]):::io
    BWD["<b>Backward pass (custom_vjp)</b><br/>implicit linear solve on KKT /<br/>fixed-point residual (CG / GMRES)"]:::bwd

    B_IN --> DEV
    DEV -->|"CPU, B > 1<br/>(backend='threads')"| THR --> FWD
    DEV -->|"GPU or B = 1<br/>(backend='vmap')"| VMAP --> FWD
    FWD --> BWD
```

See [Appendix D: Implementation Optimizations](implementation_optimizations.md)
for details on thread-pool caching, NumPy pre-slicing, and `vmap` memory layout.
