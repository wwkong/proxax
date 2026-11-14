# Experiment Reports

[← Back to Proxax Documentation](https://proxax.readthedocs.io/en/latest/)

This section collects writeups, tables, and replication notes for external
comparisons, internal solver suites, scaling studies, and paper-replication
studies run under `benchmarks/` and `experiments/` in the repository.

Each report documents:

- **Problem setup & citations** (exact formulation, dimensions, tolerances,
  and seeds)
- **Reproduction command** (`benchmarks/<study>.py` or `experiments/<study>/run.py`)
- **Summary tables & observations** (iteration counts, inner ACG/FISTA calls,
  wall-clock times, and stationarity/feasibility residuals)

## External Benchmarks

Comparisons against external optimization and differentiable-layer libraries
(`Clarabel`, `alpaqa`, and `cvxpylayers`) on CPU (`float64`, `tol = 1e-6`).

### Executive Summary

| Benchmark | Baseline | Problem Regime | Headline Result (`proxax` vs. Baseline) | Notes / Limits |
| :--- | :--- | :--- | :--- | :--- |
| [Large-Scale Matrix-Free Operators](large_matrix_free.md) | `Clarabel` | 1D Circulant LASSO & 2D FFT Deblur ($n \le 65{,}536$) | **20.9×–2,388× faster** (`352 ms` at $n=65{,}536$) | `Clarabel` hits KKT fill-in wall ($>225\text{ s}$) at $n \ge 16{,}384$ on 2D deblur |
| [Matrix-Free CPU Benchmarks](matrix_free_vs_clarabel.md) | `Clarabel` | Factor QPs ($n \le 3{,}000$), LogReg-Simplex, Dense QP | **75.9×–326.7×** (`FactorQP`), **1.2×–4.0×** (`LogReg`), **4.1×** (`DenseQP-1000`) | `DenseQP-1500` (`18 MB`) spills CPU L3 cache ($>10\text{ s}$ vs. `Clarabel` `1.10 s`) |
| [Batched Differentiable Layers](vmap_implicit_vs_cvxpylayers.md) | `cvxpylayers` | Forward + implicit VJP ($B \in \{1, 16, 64, 256\}$) | **5.16×–79.03× faster** (wins **12/12**) | Rel. $\ell_2$ grad diff `1.08e-6`–`1.07e-5` |
| [End-to-End Differentiable Training Loop](diff_training_loop.md) | `cvxpylayers` | 50-step Adam loop ($B \in \{16, 64\}$) | **15.19×–22.46× faster** with `threads` (`168–691 ms`) | `0.57–1.14 s` compile; `jit-scan` **1.8×–15.1× faster** |
| [Batched Convex Solves](batch_vs_clarabel.md) | `Clarabel(threads)` | Factor QP & LogReg-Simplex ($B \in \{1, 16, 64\}$) | **30×–337× faster** | `threads` eliminates CPU `vmap` lockstep (**3.4×–12.7×** over `vmap`) |
| [Batched Nonconvex Solves](batch_vs_alpaqa.md) | `alpaqa(procs)` | Box QP, Linear QP, Nonlinear-Ball QP ($B \in \{1, 16, 64\}$) | **62×–292×** (`box_qp`), **1.39×–6.5×** (`lin_qp`) | On `nl_ball_qp` ($n=50$), `proxax` wins $B=1$ (**6.5×**); `alpaqa` edges $B=16,64$ (`4.29 / 11.84 ms` vs. `5.20 / 15.26 ms`) |
| [Closed-Loop Warm-Started MPC Rollout](mpc_rollout.md) | `Clarabel` & `alpaqa` | 20-step closed-loop MPC ($N \in \{10, 20\}$, $B \in \{1, 16\}$) | **3.3×–40.7× faster** (`box`), **3.5×–10.5×** (`state_con`) | `warm,threads` eliminates CPU `vmap` lockstep at $B=16$ |
| [Nonconvex CPU Benchmarks](panoc_vs_alpaqa.md) | `alpaqa` | Single-solve Box QP, Linear QP, Nonlinear-Ball QP | **1.31×–4.21× faster** on constrained (`lin_qp`, `nl_ball_qp`) | C++ `alpaqa-PANOC` is ~1.8× faster on larger single-solve `box_qp` ($n=200, 500$) |
| [Maros–Meszaros QP Benchmarks](maros_meszaros.md) | `Clarabel` | 15 small classic sparse QPs | **15/15 converged** (`2.38 ms` avg vs. `Clarabel` `0.61 ms` avg) | Sparse direct IPM is ~4× faster on small stiff QPs |
| [Maros–Meszaros Medium-Scale QP Benchmarks](maros_meszaros_medium.md) | `Clarabel` | 23 medium sparse QPs ($n \le 300$) | `IALM` (Ruiz + orig-tol): **17/23** in 2 s (**18/23** in 10 s) | First-order ALM trails sparse IPM on ill-conditioned QPs |

### Reports

* [Large-Scale Matrix-Free Operators (`proxax` vs. `Clarabel`)](large_matrix_free.md)
* [Matrix-Free CPU Benchmarks (`proxax` vs. `Clarabel`)](matrix_free_vs_clarabel.md)
* [Batched Differentiable Layers (`proxax` vs. `cvxpylayers`)](vmap_implicit_vs_cvxpylayers.md)
* [End-to-End Differentiable Training Loop (`proxax` vs. `cvxpylayers`)](diff_training_loop.md)
* [Batched Convex Solves (`proxax` vs. `Clarabel`)](batch_vs_clarabel.md)
* [Batched Nonconvex Solves (`proxax` vs. `alpaqa`)](batch_vs_alpaqa.md)
* [Closed-Loop Warm-Started MPC Rollout (`proxax` vs. `Clarabel` & `alpaqa`)](mpc_rollout.md)
* [Nonconvex CPU Benchmarks (`proxax` vs. `alpaqa`)](panoc_vs_alpaqa.md)
* [Maros–Meszaros QP Benchmarks](maros_meszaros.md)
* [Maros–Meszaros Medium-Scale QP Benchmarks](maros_meszaros_medium.md)

## Internal Benchmarks

Solver comparisons and Dolan–Moré performance profiles across `proxax` algorithms:

### Executive Summary

| Benchmark | Solvers Compared | Problem Regime | Headline Result | Notes |
| :--- | :--- | :--- | :--- | :--- |
| [`PANOC` Ill-Conditioned & Exotic-Prox Benchmarks](panoc_benchmarks.md) | `PANOC`, `APD`, `AdapFISTA`, `ECG` | Lasso (`cond=100,1000`), Sparse Logistic, Spectrahedron, CappedSimplex, OSCAR | **6/6 converged**; lowest oracle evals on `Lasso (cond=1000)`, `Sparse Logistic`, `OSCAR` | `66` evals on `Lasso (cond=1000)` vs. `135` (`APD`) and `153` (`ECG`) |
| [`PANOC-ALM` Constrained Solver Benchmarks](panoc_alm_benchmarks.md) | `PANOC-ALM`, `IALM (APD)`, `IAIPAL`, `QP-APD`, `QP-AIPP` | Constrained Lasso (`cond=100,1000`), Sector Portfolio, Constrained Sparse Logistic | **4/4 converged**; **4.8× fewer oracle evals** than `IALM (APD)` on `Lasso (cond=1000)` | `6,064` evals vs. `29,272` (`IALM (APD)`) and `83,083` (`QP-APD`) |
| [Conic Optimization Benchmarks](conic_benchmarks.md) | `PANOC-ALM`, `IALM (APD)`, `QP-APD`, `QP-AIPP` | `ExpCone`, `Robust SOCP (cond=50)`, `Spectrahedron SDP`, `SOC Portfolio` | Only solver to converge on **4/4** conic problems | Solves `Robust SOCP` in **508** evals (`IALM (APD)` hits max iters) |

### Reports

* [`PANOC` Ill-Conditioned & Exotic-Prox Benchmarks](panoc_benchmarks.md)
* [`PANOC-ALM` Constrained Solver Benchmarks](panoc_alm_benchmarks.md)
* [Conic Optimization Benchmarks (`ExpCone`, `SOC`, `Spectrahedron`)](conic_benchmarks.md)

## Scaling Studies

Accelerator (`vmap` / `jit` / GPU) throughput and batch-scaling studies:

* [`PAdaptADMM` CPU/GPU & `jax.vmap` Scaling Study](padapt_vmap_gpu.md)
* [Batched Differentiable Layers (`jax.vmap` + `implicit_solve`)](vmap_implicit_benchmarks.md)

## Literature Replications

Reproductions of published benchmark tables using `proxax`:

* [`PAdaptADMM` CPU Benchmarks (Maia et al., 2026 — Table 1 DQP)](maia2026_table1_dqp.md)
* [Curvature-Free Nonconvex QPs (Kong, SIOPT 2024)](kong2024_spectrahedron.md)
* [ALM vs. Quadratic Penalty on Constrained NCO (Kong et al., 2023)](kong2023_alm_vs_qp.md)
* [Multi-Block Nonconvex Splitting: `DPADMM` vs. `PAdaptADMM` (Kong & Monteiro, SIOPT 2024)](kong2024_dpadmm.md)
* [Nonconvex Spectral Composite Optimization (Kong & Monteiro, COAP 2022)](kong2022_spectral.md)
* [Nonconvex-Concave Min-Max Optimization (Kong & Monteiro, SIOPT 2021)](kong2021_minmax.md)

```{toctree}
:maxdepth: 1
:hidden:
:caption: External Benchmarks

large_matrix_free
matrix_free_vs_clarabel
vmap_implicit_vs_cvxpylayers
diff_training_loop
batch_vs_clarabel
batch_vs_alpaqa
mpc_rollout
panoc_vs_alpaqa
maros_meszaros
maros_meszaros_medium
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Internal Benchmarks

panoc_benchmarks
panoc_alm_benchmarks
conic_benchmarks
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Scaling Studies

padapt_vmap_gpu
vmap_implicit_benchmarks
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Literature Replications

maia2026_table1_dqp
kong2024_spectrahedron
kong2023_alm_vs_qp
kong2024_dpadmm
kong2022_spectral
kong2021_minmax
```
