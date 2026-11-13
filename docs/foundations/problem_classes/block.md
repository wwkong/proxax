# Class 3: Block-Constrained Composite Optimization (`BlockProblem`)

## Formulation

Given $K$ separable `BlockTerm` blocks $\{(f_i, h_i, \mathcal{A}_i)\}_{i=1}^K$
and right-hand side $b$, a `BlockProblem` solves:

$$\min_{x_1, \dots, x_K} \; \sum_{i=1}^K \big(f_i(x_i) + h_i(x_i)\big) \qquad \text{s.t.} \qquad \sum_{i=1}^K \mathcal{A}_i(x_i) = b.$$

## Certified Block KKT $(\rho, \eta)$-Stationarity

Given tolerances $\rho, \eta > 0$ (`opt_tol`, `feas_tol`), block ADMM solvers
(`ADMM`, `DPADMM`, `PAdaptADMM`) return blocks
$\hat{x} = (\hat{x}_1, \dots, \hat{x}_K)$, dual multiplier $\hat{y}$, and
subgradients $\hat{v} = (\hat{v}_1, \dots, \hat{v}_K)$ satisfying:

$$\hat{v}_i \in \nabla f_i(\hat{x}_i) + \mathcal{A}_i^*(\hat{y}) + \partial h_i(\hat{x}_i), \qquad \|\hat{v}\|_2 \le \rho, \qquad \left\|\sum_{i=1}^K \mathcal{A}_i(\hat{x}_i) - b\right\|_2 \le \eta.$$
