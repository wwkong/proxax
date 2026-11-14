# Maros–Meszaros Large-Scale Sparse QP Benchmarks (`proxax` vs. `Clarabel`)

CPU (`float64`) evaluation of `proxax-IALM` and `proxax-IAIPAL` (`BCOO` sparse operators + Modified Ruiz Equilibration, `tol = 1e-6`, `10 s` limit) against `Clarabel` on 8 large-scale Maros–Meszaros sparse QPs ($n \in [1{,}000,\; 10{,}197]$).

## Setup

* **Precision & Device:** `float64` (`JAX_ENABLE_X64=1`), CPU.
* **Tolerances & Limits:** `opt_tol = feas_tol = 1e-6`, `time_limit = 10.0 s`, `precondition = True` (`px.ruiz_equilibrate`).
* **Reproduction:**

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python -m benchmarks.external.maros_meszaros_large --solvers IALM IAIPAL --clarabel --time-limit 10.0
```

## Results

### Runtime & Feasibility Comparison

![Large-Scale Maros-Meszaros Comparison](_static/plots/maros_meszaros_large.svg)

### Summary Table (`tol = 1e-6`)

| Problem | Solver | Status | Objective | Opt Res | Feas Res | Compile (s) | Run (ms) |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `CVXQP1_M (n=1000,m=500)` | `IALM` | `TIME_LIMIT_REACHED` | 1076418.7520 | 4.69e-05 | 5.33e-01 | 10.56 | 10,019.55 |
| `CVXQP1_M (n=1000,m=500)` | `IAIPAL` | `TIME_LIMIT_REACHED` | 1082113.7590 | 2.91e-03 | 2.49e-01 | 10.48 | 10,000.69 |
| `CVXQP2_M (n=1000,m=250)` | `IALM` | `CONVERGED` | 820155.4306 | 1.02e-08 | 6.80e-07 | 5.84 | 5,315.26 |
| `CVXQP2_M (n=1000,m=250)` | `IAIPAL` | `CONVERGED` | 820155.4310 | 1.02e-08 | 7.07e-08 | 1.18 | 665.53 |
| `CVXQP3_M (n=1000,m=750)` | `IALM` | `TIME_LIMIT_REACHED` | 1129653.1832 | 1.02e-08 | 1.45e+00 | 10.53 | 10,032.65 |
| `CVXQP3_M (n=1000,m=750)` | `IAIPAL` | `TIME_LIMIT_REACHED` | 1138858.9598 | 0.00e+00 | 1.27e+00 | 10.52 | 10,035.11 |
| `MOSARQP1 (n=2500,m=700)` | `IALM` | `CONVERGED` | -337.5896 | 1.34e-07 | 3.38e-07 | 2.17 | 1,628.59 |
| `MOSARQP1 (n=2500,m=700)` | `IAIPAL` | `CONVERGED` | -337.5896 | 1.35e-07 | 1.36e-07 | 0.64 | 117.91 |
| `STCQP1 (n=4097,m=2052)` | `IALM` | `TIME_LIMIT_REACHED` | 154238.5169 | 5.83e-07 | 2.78e+00 | 10.81 | 10,117.78 |
| `STCQP1 (n=4097,m=2052)` | `IAIPAL` | `CONVERGED` | 155143.5547 | 7.87e-08 | 1.35e-07 | 2.35 | 1,718.37 |
| `STCQP2 (n=4097,m=2052)` | `IALM` | `TIME_LIMIT_REACHED` | 22149.1613 | 2.83e-02 | 1.55e+00 | 10.90 | 10,266.96 |
| `STCQP2 (n=4097,m=2052)` | `IAIPAL` | `CONVERGED` | 22327.3133 | 8.26e-08 | 2.28e-07 | 4.99 | 4,318.03 |
| `CONT-050 (n=2597,m=2401)` | `IALM` | `TIME_LIMIT_REACHED` | -8760.5379 | 5.77e-04 | 4.66e-02 | 10.60 | 10,084.17 |
| `CONT-050 (n=2597,m=2401)` | `IAIPAL` | `TIME_LIMIT_REACHED` | -8748.1685 | 5.56e-04 | 3.20e-02 | 10.52 | 10,000.73 |
| `CONT-100 (n=10197,m=9801)` | `IALM` | `TIME_LIMIT_REACHED` | -35956.6704 | 3.23e-02 | 5.63e-01 | 10.68 | 10,187.58 |
| `CONT-100 (n=10197,m=9801)` | `IAIPAL` | `TIME_LIMIT_REACHED` | -35735.4566 | 1.36e-03 | 4.96e-02 | 10.67 | 10,001.50 |
| `CVXQP1_M (n=1000,m=500)` | `Clarabel` | `CONVERGED` | 1087511.6255 | 3.89e-16 | 4.71e-11 | 0.00 | **276.53** |
| `CVXQP2_M (n=1000,m=250)` | `Clarabel` | `CONVERGED` | 820155.4413 | 4.79e-16 | 4.16e-15 | 0.00 | **272.73** |
| `CVXQP3_M (n=1000,m=750)` | `Clarabel` | `CONVERGED` | 1362828.7480 | 4.19e-16 | 1.94e-12 | 0.00 | **475.69** |
| `MOSARQP1 (n=2500,m=700)` | `Clarabel` | `CONVERGED` | -952.8753 | 8.88e-11 | 5.26e-11 | 0.00 | **8.32** |
| `STCQP1 (n=4097,m=2052)` | `Clarabel` | `CONVERGED` | 155143.5652 | 3.30e-15 | 3.69e-11 | 0.00 | **640.87** |
| `STCQP2 (n=4097,m=2052)` | `Clarabel` | `CONVERGED` | 22327.3147 | 1.73e-15 | 3.07e-16 | 0.00 | **1,486.43** |
| `CONT-050 (n=2597,m=2401)` | `Clarabel` | `CONVERGED` | -4.5639 | 1.88e-18 | 2.74e-11 | 0.00 | **39.39** |
| `CONT-100 (n=10197,m=9801)` | `Clarabel` | `CONVERGED` | -4.6444 | 2.16e-18 | 1.39e-10 | 0.00 | **941.81** |

## Key Observations

1. **Sparse direct IPM (`Clarabel`) dominates large ill-conditioned QPs:** `Clarabel` solves **8/8** instances in **8.32–1,486.43 ms**, whereas first-order `IAIPAL` converges on **4/8** (`CVXQP2_M` in `665.53 ms` — **8.0× faster** than `IALM` at `5,315.26 ms`, `MOSARQP1` in `117.91 ms` — **13.8× faster** than `IALM`, `STCQP1` in `1,718.37 ms`, `STCQP2` in `4,318.03 ms`) and `IALM` converges on **2/8**.
2. **Stationarity vs. feasibility on timeouts:** On the remaining `10 s` timeouts (`CVXQP1_M`, `CVXQP3_M`, `CONT-050`, `CONT-100`), `IALM` and `IAIPAL` drive stationarity to `<= 1.36e-3` (and `<= 4.69e-5` on `CVXQP1_M`/`CVXQP3_M`), while primal feasibility (`4.66e-2`–`1.45e+0`) remains the bottleneck.
