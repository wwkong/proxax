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
flowchart TB
    subgraph L1["User & Ecosystem Layer"]
        PX["proxax.px (curated facade)"]
        SOLVE["solve / batch_solve"]
        DIFF["implicit_solve / layers"]
        OPTAX["optax_compat"]
        PLOT["plotting"]
    end

    subgraph L2["Problem & Operator Modeling Layer"]
        PROB["problem (CompositeProblem, LinearConstraint)"]
        PROX["prox (norms, indicators, folded_concave, exotic, combinators)"]
        CONS["constraints (sets, linear_op)"]
        ORA["oracle (composite, spectral, spectral_problem, minmax, block, terms)"]
        PREC["precondition (ruiz_equilibrate)"]
    end

    subgraph L3["Solver & Certificate Engine Layer"]
        UNC["solvers.unconstrained (APD, PANOC, AIPP, AdapFISTA, ...)"]
        CON["solvers.constrained (IALM, IAIPAL, SProxALM, QPAIPP, QPAPD)"]
        MM["solvers.minmax (ExtraGradient, PDHG, ProxPoint)"]
        BLK["solvers.block (ADMM, DPADMM, PAdaptADMM)"]
        INN["solvers.inner (ACG, PFACG)"]
        CERT["certificates (stationarity, minmax)"]
        LOOP["_loop & tree (while_loop / PyTree linear algebra)"]
    end

    subgraph L4["JAX & Optax Runtime Substrate"]
        JAX["jax (jit, vmap, grad, linear_transpose, custom_vjp, lax.while_loop)"]
        OX["optax (optax.tree, optax.projections, GradientTransformationExtraArgs)"]
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
```

---

## 2. `px.solve` Execution & Dispatch Pipeline

When `px.solve(problem, x0, solver=...)` is called, Proxax normalizes the
problem, projects the initial iterate onto $\operatorname{dom}(h)$, dispatches
either to a compiled `jax.lax.while_loop` (`jit=True`) or a Python loop with
wall-clock `time_limit` support (`jit=False`), and emits a certified `Solution`:

```{mermaid}
flowchart LR
    IN["px.solve(problem, x0, solver=..., jit=True)"] --> NORM["Normalize problem &<br/>bind LinearOperator to x0"]
    NORM --> DOM["Project x0 onto dom(h)<br/>via proximal.domain_project"]
    DOM --> DISP{"solver is None?"}

    DISP -->|"Yes (auto)"| AUTO["Select default solver by<br/>problem class & constraints"]
    DISP -->|"No"| EXEC["Initialize SolverState"]
    AUTO --> EXEC

    EXEC --> JIT{"jit=True?"}
    JIT -->|"Yes"| XLA["jax.lax.while_loop<br/>(zero host-dispatch overhead)"]
    JIT -->|"No"| PY["Python loop via _loop.py<br/>(supports time_limit & callbacks)"]

    XLA & PY --> CERT["Compute stationarity / KKT<br/>certificate (x_hat, p_hat, v_hat)"]
    CERT --> OUT["Solution(status, x, dual,<br/>opt_res, feas_res, stats)"]
```

---

## 3. Constrained Inner–Outer Solver Hierarchy

Constrained solvers (`IALM`, `IAIPAL`, `SProxALM`, `QPAIPP`, `QPAPD`) transform
set- and cone-constrained problems into smooth penalized or proximal-regularized
subproblems without re-compiling XLA graphs across outer iterations:

```{mermaid}
flowchart TB
    CP["CompositeProblem<br/>min f(x) + h(x) s.t. A(x) in C"] --> PEN["PenalizedOracle<br/>(dynamic multiplier p_bar, penalty c)"]
    PEN --> REG["ProxRegularizedOracle<br/>(dynamic center x_k, smooth_reg, prox_reg)"]

    REG --> SUB{"Inner subproblem solver"}
    SUB -->|"Accelerated convex"| ACG["inner.ACG / inner.PFACG"]
    SUB -->|"Parameter-free"| APD["unconstrained.APD"]
    SUB -->|"Quasi-Newton FBE"| PANOC["unconstrained.PANOC"]

    ACG & APD & PANOC --> REF["Stationarity & KKT refinement<br/>(certificates.stationarity)"]
    REF --> UPD["Outer dual / penalty update<br/>p_{k+1}, c_{k+1}"]
    UPD -->|"Not converged"| PEN
    UPD -->|"opt_res <= opt_tol &<br/>feas_res <= feas_tol"| DONE["Certified KKT Solution"]
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
flowchart LR
    B_IN["px.batch_solve / px.batch_implicit_solve<br/>(x0_batch, params_batch, backend='auto')"] --> DEV{"Device &<br/>batch size B"}

    DEV -->|"CPU, B > 1<br/>(backend='threads')"| THR["Persistent ThreadPoolExecutor +<br/>precompiled jax.pure_callback"]
    DEV -->|"GPU or B = 1<br/>(backend='vmap')"| VMAP["jax.vmap over<br/>jit-compiled px.solve"]

    THR & VMAP --> FWD["Batched Solution<br/>x_star_batch"]
    FWD --> BWD["Backward pass (custom_vjp):<br/>implicit linear solve on KKT /<br/>fixed-point residual (CG / GMRES)"]
```

See [Appendix D: Implementation Optimizations](implementation_optimizations.md)
for details on thread-pool caching, NumPy pre-slicing, and `vmap` memory layout.
