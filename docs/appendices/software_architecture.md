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
%%{init: {'theme': 'base', 'flowchart': {'curve': 'basis', 'nodeSpacing': 18, 'rankSpacing': 28, 'padding': 12}, 'themeVariables': {'darkMode': true, 'background': 'transparent', 'primaryColor': '#1e293b', 'primaryTextColor': '#f8fafc', 'primaryBorderColor': '#60a5fa', 'lineColor': '#94a3b8', 'secondaryColor': '#0f172a', 'tertiaryColor': '#1e293b', 'textColor': '#f8fafc', 'clusterBkg': 'rgba(15, 23, 42, 0.45)', 'clusterBorder': '#475569', 'edgeLabelBackground': '#0f172a'}}}%%
flowchart TB
    classDef user fill:#1e3a5f,stroke:#60a5fa,color:#f8fafc,stroke-width:2px
    classDef model fill:#064e3b,stroke:#34d399,color:#f8fafc,stroke-width:2px
    classDef solver fill:#3b0764,stroke:#a78bfa,color:#f8fafc,stroke-width:2px
    classDef substrate fill:#451a03,stroke:#fbbf24,color:#f8fafc,stroke-width:2px

    subgraph L1["1. User & Ecosystem"]
        direction TB
        PX(["proxax.px (facade)"]):::user
        SOLVE["solve / batch_solve"]:::user
        DIFF["implicit_solve / layers"]:::user
        OPTAX["optax_compat"]:::user
        PLOT["plotting"]:::user
        PX --> SOLVE & DIFF & OPTAX & PLOT
    end

    subgraph L2["2. Problem & Modeling"]
        direction TB
        PROB["problem<br/>CompositeProblem, LinearConstraint"]:::model
        ORA["oracle<br/>composite, spectral, minmax, block"]:::model
        PREC["precondition<br/>ruiz_equilibrate"]:::model
        PROX["prox<br/>norms, indicators, folded_concave"]:::model
        CONS["constraints<br/>sets, linear_op"]:::model
        PROB --> PROX & CONS
        ORA --> PROX & CONS
        PREC ~~~ CONS
    end

    subgraph L3["3. Solvers & Certificates"]
        direction TB
        CON["solvers.constrained<br/>IALM, IAIPAL, SProxALM, QPAIPP"]:::solver
        UNC["solvers.unconstrained<br/>APD, PANOC, AIPP, ..."]:::solver
        INN["solvers.inner<br/>ACG, PFACG"]:::solver
        MM["solvers.minmax<br/>ExtraGradient, PDHG, ProxPoint"]:::solver
        BLK["solvers.block<br/>ADMM, DPADMM, PAdaptADMM"]:::solver
        CERT["certificates<br/>stationarity, minmax"]:::solver
        LOOP["_loop & tree<br/>while_loop, PyTree linalg"]:::solver
        CON --> INN & UNC
        UNC & MM & BLK --> CERT & LOOP
    end

    subgraph L4["4. JAX & Optax"]
        direction LR
        JAX[("jax: jit, vmap, grad, custom_vjp, while_loop")]:::substrate
        OX[("optax: tree, projections, ExtraArgs")]:::substrate
    end

    L1 --> L2 --> L3 --> L4

    style L1 fill:rgba(30,58,95,0.2),stroke:#60a5fa,stroke-width:1.5px,color:#f8fafc
    style L2 fill:rgba(6,78,59,0.2),stroke:#34d399,stroke-width:1.5px,color:#f8fafc
    style L3 fill:rgba(59,7,100,0.2),stroke:#a78bfa,stroke-width:1.5px,color:#f8fafc
    style L4 fill:rgba(69,26,3,0.2),stroke:#fbbf24,stroke-width:1.5px,color:#f8fafc
```

---

## 2. `px.solve` Execution & Dispatch Pipeline

When `px.solve(problem, x0, solver=...)` is called, Proxax normalizes the
problem, projects the initial iterate onto $\operatorname{dom}(h)$, dispatches
either to a compiled `jax.lax.while_loop` (`jit=True`) or a Python loop with
wall-clock `time_limit` support (`jit=False`), and emits a certified `Solution`:

```{mermaid}
%%{init: {'theme': 'base', 'flowchart': {'curve': 'basis', 'nodeSpacing': 24, 'rankSpacing': 30, 'padding': 12}, 'themeVariables': {'darkMode': true, 'background': 'transparent', 'primaryColor': '#1e293b', 'primaryTextColor': '#f8fafc', 'primaryBorderColor': '#60a5fa', 'lineColor': '#94a3b8', 'textColor': '#f8fafc', 'edgeLabelBackground': '#0f172a'}}}%%
flowchart TB
    classDef io fill:#1e3a5f,stroke:#60a5fa,color:#f8fafc,stroke-width:2px
    classDef step fill:#1e293b,stroke:#94a3b8,color:#f8fafc,stroke-width:1.5px
    classDef decision fill:#4c0519,stroke:#fb7185,color:#f8fafc,stroke-width:2px
    classDef engine fill:#3b0764,stroke:#a78bfa,color:#f8fafc,stroke-width:2px
    classDef cert fill:#064e3b,stroke:#34d399,color:#f8fafc,stroke-width:2px

    IN(["px.solve(problem, x0,<br/>solver=..., jit=True)"]):::io
    NORM["Normalize problem &<br/>bind LinearOperator to x0"]:::step
    DOM["Project x0 onto dom(h)<br/>via proximal.domain_project"]:::step
    DISP{"solver is None?"}:::decision
    AUTO["Auto-select default solver<br/>by problem class & constraints"]:::step
    EXEC["Initialize SolverState"]:::step
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
%%{init: {'theme': 'base', 'flowchart': {'curve': 'basis', 'nodeSpacing': 24, 'rankSpacing': 30, 'padding': 12}, 'themeVariables': {'darkMode': true, 'background': 'transparent', 'primaryColor': '#1e293b', 'primaryTextColor': '#f8fafc', 'primaryBorderColor': '#60a5fa', 'lineColor': '#94a3b8', 'textColor': '#f8fafc', 'edgeLabelBackground': '#0f172a'}}}%%
flowchart TB
    classDef prob fill:#1e3a5f,stroke:#60a5fa,color:#f8fafc,stroke-width:2px
    classDef oracle fill:#064e3b,stroke:#34d399,color:#f8fafc,stroke-width:2px
    classDef decision fill:#4c0519,stroke:#fb7185,color:#f8fafc,stroke-width:2px
    classDef inner fill:#3b0764,stroke:#a78bfa,color:#f8fafc,stroke-width:2px
    classDef outer fill:#451a03,stroke:#fbbf24,color:#f8fafc,stroke-width:2px
    classDef done fill:#064e3b,stroke:#34d399,color:#f8fafc,stroke-width:2px

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
%%{init: {'theme': 'base', 'flowchart': {'curve': 'basis', 'nodeSpacing': 24, 'rankSpacing': 30, 'padding': 12}, 'themeVariables': {'darkMode': true, 'background': 'transparent', 'primaryColor': '#1e293b', 'primaryTextColor': '#f8fafc', 'primaryBorderColor': '#60a5fa', 'lineColor': '#94a3b8', 'textColor': '#f8fafc', 'edgeLabelBackground': '#0f172a'}}}%%
flowchart TB
    classDef io fill:#1e3a5f,stroke:#60a5fa,color:#f8fafc,stroke-width:2px
    classDef decision fill:#4c0519,stroke:#fb7185,color:#f8fafc,stroke-width:2px
    classDef cpu fill:#064e3b,stroke:#34d399,color:#f8fafc,stroke-width:2px
    classDef gpu fill:#3b0764,stroke:#a78bfa,color:#f8fafc,stroke-width:2px
    classDef bwd fill:#451a03,stroke:#fbbf24,color:#f8fafc,stroke-width:2px

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
