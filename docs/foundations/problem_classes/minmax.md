# Class 4: Min-Max & Saddle-Point Optimization (`MinMaxProblem`)

## Formulation

Let $\mathcal{X}$ and $\mathcal{Y}$ be Euclidean PyTree spaces. A
`MinMaxProblem` solves:

$$\min_{x \in \mathcal{X}} \max_{y \in \mathcal{Y}} \; \Phi(x, y) + h(x) - g(y)$$

where $\Phi : \mathcal{X} \times \mathcal{Y} \to \mathbb{R}$ is differentiable
(possibly nonconvex-concave or weakly-convex–weakly-concave) and $h$, $g$ are
proper, closed, convex, and proximable regularizers or set indicators.

## Certified Primal-Dual $\rho$-Stationarity

Given tolerance $\rho > 0$ (`opt_tol`), min-max solvers (`ExtraGradient`,
`PDHG`, `ProxPointMinMax`) return a certified primal-dual pair
$(\hat{x}, \hat{y})$ and subgradient pair $(\hat{v}_x, \hat{v}_y)$ satisfying:

$$\hat{v}_x \in \nabla_x \Phi(\hat{x}, \hat{y}) + \partial h(\hat{x}), \qquad \hat{v}_y \in -\nabla_y \Phi(\hat{x}, \hat{y}) + \partial g(\hat{y}), \qquad \sqrt{\|\hat{v}_x\|_2^2 + \|\hat{v}_y\|_2^2} \le \rho.$$
