# Class 1: Unconstrained Composite Optimization (`CompositeProblem`)

## Formulation

$$\min_{x \in \mathcal{X}} \; \phi(x) := f(x) + h(x)$$

where $f$ is differentiable and $(m, M)$-smooth/weakly convex on
$\operatorname{dom} h$, and $h$ is proper, closed, $\mu$-strongly convex, and
admits a tractable proximal operator:

$$\operatorname{prox}_{\tau h}(u) := \operatorname*{argmin}_{z \in \mathcal{X}} \left\{ h(z) + \frac{1}{2\tau}\|z - u\|_2^2 \right\}, \qquad \tau > 0.$$

Note that "unconstrained" here means **no coupling linear operator
$\mathcal{A}(x) \in \mathcal{C}$ is present**: simple geometric constraints
$x \in K$ (boxes, balls, non-negative orthants, or simplices) are handled
directly and exactly by setting $h = \delta_K$.

## Certified $\rho$-Stationarity

A point $x^\star \in \operatorname{dom} h$ is **first-order stationary** if
$0 \in \nabla f(x^\star) + \partial h(x^\star)$. Given tolerance $\rho > 0$, an
unconstrained solver in Proxax returns a **certified $\rho$-stationary pair**
$(\hat{x}, \hat{v}) \in \operatorname{dom} h \times \mathcal{X}$ satisfying:

$$\hat{v} \in \nabla f(\hat{x}) + \partial h(\hat{x}), \qquad \|\hat{v}\| \le \rho.$$

In `Solution`, `sol.x` holds $\hat{x}$, `sol.subgrad` holds $\hat{v}$, and
`sol.stats.opt_residual` equals $\|\hat{v}\|$.
