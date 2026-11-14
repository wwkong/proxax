# Appendix D: Implementation Optimizations

This appendix documents the key engineering and JAX/XLA optimizations used inside Proxax to minimize host-dispatch overhead, avoid CPU `vmap` lockstep, and keep both forward solving and implicit differentiation fast.

---

## 1. Batched Execution & Thread-Pool Caching (`proxax.batch`)

### Persistent Thread Pool & Compiled-Runner Caching

When `backend="threads"` is selected (the `backend="auto"` default on CPU for batch size $B > 1$):

* **Persistent executor (`_THREAD_POOL_CACHE`):** `_get_thread_pool` reuses a `ThreadPoolExecutor` sized to `min(batch_size, os.cpu_count() or 8)` (or `max_workers`), avoiding per-call thread creation and teardown.
* **Weak-keyed compiled caches (`_BATCH_SOLVE_CACHE`, `_BATCH_IMPLICIT_CACHE`):** Compiled per-instance/per-chunk runners and `custom_vjp` wrappers are cached per `problem_fn` in `weakref.WeakKeyDictionary`s keyed by solver, tolerances, and leaf shapes/dtypes, eliminating repeated XLA tracing across training steps.
* **Pre-compiled forward executable:** In `px.batch_implicit_solve`, `compiled_fwd = jax.jit(_one_fwd).lower(x0, first_theta).compile()` is compiled **once** before entering `jax.pure_callback`, preventing nested JIT compilation contention across worker threads.

### Host-Side NumPy Pre-Slicing

Inside `_host_batch_fwd`, `x_init` and `params_batch` are converted to NumPy arrays (`np.asarray`) **before** `pool.map`:

* Worker threads slice `params_batch` using cheap NumPy indexing (`_slice_pytree(pb_np, i)`) rather than calling `jax.Array.__getitem__` (`jax.lax.slice` / `apply_primitive`) concurrently from inside `jax.pure_callback`.
* Each worker directly invokes the pre-compiled XLA executable `compiled_fwd` on NumPy inputs and stacks the resulting leaves once on return.

### Tracer-Aware Dispatch in `px.batch_implicit_solve`

`px.batch_implicit_solve` supports both **eager** `jax.value_and_grad` loops and **outer-`jax.jit`** compiled steps via `_run_fwd_dispatch`:

* **Eager path (no `jax.core.Tracer` inputs):** When called outside an outer `jax.jit` (including `jax.value_and_grad(loss_fn)` in a standard Python optimizer loop), `_run_fwd_dispatch` invokes `_host_batch_fwd` directly in Python, completely bypassing `jax.pure_callback`. Combined with a pre-compiled adjoint kernel (`_vmap_bwd = jax.jit(jax.vmap(_adj_one, ...))`), eager `jax.value_and_grad` runs without retracing `vmap` per step and without needing `block_until_ready()`.
* **Traced path (`jax.core.Tracer` inputs):** Inside an outer `jax.jit` (e.g., `jit(value_and_grad(loss_fn))`), `_run_fwd_dispatch` routes through `jax.pure_callback`. If called in a tight Python loop with async XLA CPU dispatch, call `grad.block_until_ready()` before queuing the next step so downstream kernels do not overlap with `pure_callback` workers re-entering the XLA CPU runtime.

### `chunk_size` Partitioning (`px.batch_solve`)

`px.batch_solve(..., backend="threads", chunk_size=c)` trades off early-exit granularity against host-dispatch overhead:

* **`chunk_size=1`:** Minimizes tail latency on heterogeneous active-set problems (e.g., linearly constrained QPs or state-constrained MPC), because each instance exits as soon as it reaches `opt_tol` / `feas_tol` without waiting for stragglers.
* **`chunk_size > 1` (e.g., `4` or `8`):** Dispatches `jit(jax.vmap(...))` over `c`-sized chunks, amortizing Python-to-XLA dispatch overhead and enabling intra-chunk SIMD vectorization on uniform workloads (e.g., box-constrained QPs).

---

## 2. XLA-Traceable Loops & Zero-Overhead `time_limit` (`proxax._loop`)

* **Pure XLA control flow:** Setting `jit=True` lowers solver iterations to `jax.lax.while_loop` (`proxax._loop.while_loop`), removing Python-to-XLA per-iteration dispatch overhead and allowing whole-loop fusion with `jax.lax.scan` (e.g., closed-loop MPC rollouts and multi-step training loops).
* **Zero-cost `time_limit=None`:** Wall-clock budgeting uses `jax.experimental.io_callback(_read_clock_host, ..., ordered=False)`. When `time_limit is None`, `remaining_time` short-circuits immediately so no `io_callback` nodes are emitted into the XLA graph.

---

## 3. Matrix-Free Implicit Differentiation (`proxax.implicit`)

* **Krylov adjoint solve:** Rather than materializing or factorizing $O(n^3)$ KKT matrices, `px.implicit_solve` differentiates the fixed-point residual map at `(x_hat, dual_hat)` using `jax.linear_transpose` / JVP linear operators solved via `gmres`, `cg`, or `bicgstab`.
* **No forward re-solve on backward:** `px.batch_implicit_solve` calls `implicit_vjp_from_solution` in `_bwd`, reusing the primal and dual solutions `(x_b, p_b)` already computed in `_fwd`.

---

## 4. Modified Ruiz Preconditioning

For ill-conditioned QPs (such as `Maros–Meszaros`), optional Modified Ruiz Equilibration diagonally rescales primal and dual coordinates while scaling stopping tolerances conservatively so that `opt_tol` and `feas_tol` remain certified in the **original** unscaled problem space.
