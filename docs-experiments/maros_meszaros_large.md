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
| `CVXQP1_M (n=1000,m=500)` | `IALM` | `TIME_LIMIT_REACHED` | 1076418.7528 | 3.85e-08 | 5.33e-01 | 11.09 | 10,359.88 |
| `CVXQP1_M (n=1000,m=500)` | `IAIPAL` | `TIME_LIMIT_REACHED` | 1087573.3597 | 7.97e+00 | 7.40e-13 | 10.86 | 10,404.04 |
| `CVXQP2_M (n=1000,m=250)` | `IALM` | `CONVERGED` | 820155.4306 | 1.02e-08 | 6.80e-07 | 5.22 | 4,659.53 |
| `CVXQP2_M (n=1000,m=250)` | `IAIPAL` | `TIME_LIMIT_REACHED` | 820155.6243 | 3.76e-01 | 2.95e-13 | 10.66 | 10,102.55 |
| `CVXQP3_M (n=1000,m=750)` | `IALM` | `TIME_LIMIT_REACHED` | 1137570.3106 | 6.52e-04 | 1.28e+00 | 11.01 | 10,453.48 |
| `CVXQP3_M (n=1000,m=750)` | `IAIPAL` | `TIME_LIMIT_REACHED` | 1140875.2186 | 1.02e+01 | 1.26e+00 | 10.97 | 10,426.37 |
| `MOSARQP1 (n=2500,m=700)` | `IALM` | `CONVERGED` | -337.5896 | 1.34e-07 | 3.38e-07 | 2.09 | 1,514.84 |
| `MOSARQP1 (n=2500,m=700)` | `IAIPAL` | `CONVERGED` | -337.5896 | 1.35e-07 | 1.36e-07 | 0.67 | 90.13 |
| `STCQP1 (n=4097,m=2052)` | `IALM` | `TIME_LIMIT_REACHED` | 154238.5169 | 8.24e-08 | 2.78e+00 | 10.83 | 10,275.81 |
| `STCQP1 (n=4097,m=2052)` | `IAIPAL` | `TIME_LIMIT_REACHED` | 155143.6204 | 2.35e-01 | 1.18e-12 | 12.99 | 12,319.61 |
| `STCQP2 (n=4097,m=2052)` | `IALM` | `TIME_LIMIT_REACHED` | 22149.2638 | 8.14e-08 | 1.55e+00 | 13.49 | 12,590.68 |
| `STCQP2 (n=4097,m=2052)` | `IAIPAL` | `TIME_LIMIT_REACHED` | 22327.3602 | 1.69e-01 | 9.68e-13 | 10.77 | 13,569.43 |
| `CONT-050 (n=2597,m=2401)` | `IALM` | `TIME_LIMIT_REACHED` | -8757.1507 | 5.17e-02 | 4.09e-02 | 11.20 | 10,589.79 |
| `CONT-050 (n=2597,m=2401)` | `IAIPAL` | `TIME_LIMIT_REACHED` | -8679.3507 | 8.34e-02 | 2.54e-07 | 11.25 | 10,735.01 |
| `CONT-100 (n=10197,m=9801)` | `IALM` | `TIME_LIMIT_REACHED` | -35895.9504 | 9.95e-07 | 3.58e-01 | 12.66 | 11,810.52 |
| `CONT-100 (n=10197,m=9801)` | `IAIPAL` | `TIME_LIMIT_REACHED` | -35930.3437 | 6.69e-01 | 5.06e-01 | 10.84 | 10,194.06 |
| `CVXQP1_M (n=1000,m=500)` | `Clarabel` | `CONVERGED` | 1087511.6255 | 3.89e-16 | 4.71e-11 | 0.00 | **279.23** |
| `CVXQP2_M (n=1000,m=250)` | `Clarabel` | `CONVERGED` | 820155.4413 | 4.79e-16 | 4.16e-15 | 0.00 | **277.43** |
| `CVXQP3_M (n=1000,m=750)` | `Clarabel` | `CONVERGED` | 1362828.7480 | 4.19e-16 | 1.94e-12 | 0.00 | **487.49** |
| `MOSARQP1 (n=2500,m=700)` | `Clarabel` | `CONVERGED` | -952.8753 | 8.88e-11 | 5.26e-11 | 0.00 | **8.10** |
| `STCQP1 (n=4097,m=2052)` | `Clarabel` | `CONVERGED` | 155143.5652 | 3.30e-15 | 3.69e-11 | 0.00 | **658.79** |
| `STCQP2 (n=4097,m=2052)` | `Clarabel` | `CONVERGED` | 22327.3147 | 1.73e-15 | 3.07e-16 | 0.00 | **1,526.56** |
| `CONT-050 (n=2597,m=2401)` | `Clarabel` | `CONVERGED` | -4.5639 | 1.88e-18 | 2.74e-11 | 0.00 | **39.25** |
| `CONT-100 (n=10197,m=9801)` | `Clarabel` | `CONVERGED` | -4.6444 | 2.16e-18 | 1.39e-10 | 0.00 | **923.81** |

## Key Observations

1. **Sparse direct IPM (`Clarabel`) dominates large ill-conditioned QPs:** `Clarabel` solves **8/8** instances in **8.10–1,526.56 ms**, whereas first-order `IALM` converges on **2/8** (`CVXQP2_M` in `4,659.53 ms`, `MOSARQP1` in `1,514.84 ms`) and `IAIPAL` converges on **1/8** (`MOSARQP1` in `90.13 ms` — **16.8× faster** than `IALM`).
2. **Complementary residual profiles (`IALM` vs. `IAIPAL`):** `IALM` drives stationarity to `<= 1e-6` on **6/8** problems (`3.85e-8`–`9.95e-7`) while feasibility lags (`4.09e-2`–`2.78e+0`); conversely, `IAIPAL` drives primal feasibility to `<= 2.54e-7` on **6/8** problems (`2.95e-13`–`2.54e-7`) while stationarity lags (`8.34e-2`–`1.02e+1`) within the `10 s` CPU budget.
