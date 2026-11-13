# Class 5: Matrix Spectral Composite Optimization (`SpectralProblem`)

## Formulation

Let $\mathcal{X} = \mathbb{R}^{p \times q}$ with $r := \min(p, q)$, and let
$\sigma : \mathbb{R}^{p \times q} \to \mathbb{R}_+^r$ map a matrix $X$ to its
singular values $\sigma_1(X) \ge \dots \ge \sigma_r(X) \ge 0$. A
`SpectralProblem` solves:

$$\min_{X \in \mathbb{R}^{p \times q}} \; \Phi(X) := f_1(X) + (f_2^{\mathcal{V}} \circ \sigma)(X) + (h^{\mathcal{V}} \circ \sigma)(X)$$

where:

1. $f_1 : \mathbb{R}^{p \times q} \to \mathbb{R}$ is a general differentiable
   matrix objective with curvature constants $(m_1, M_1)$ (for example,
   $\frac{1}{2}\|P_\Omega(X - M_{\text{obs}})\|_F^2$).
2. $f_2^{\mathcal{V}} : \mathbb{R}^r \to \mathbb{R}$ is a differentiable
   **absolutely symmetric** vector function with curvature constants
   $(m_2, M_2)$, meaning $f_2^{\mathcal{V}}(\epsilon \odot \Pi s) =
   f_2^{\mathcal{V}}(s)$ for every sign vector $\epsilon \in \{\pm 1\}^r$ and
   permutation matrix $\Pi$ (for example, smooth concave log-sum, SCAD, or MCP
   singular-value components).
3. $h^{\mathcal{V}} : \mathbb{R}^r \to (-\infty, +\infty]$ is a proper, closed,
   convex, **absolutely symmetric** proximable vector function (for example,
   $\mu \|s\|_1$ for the nuclear norm $\mu \|X\|_*$, optionally restricted to a
   Frobenius ball $\|s\|_2 \le R$).

## Spectral Calculus & Curvature Redistribution

By **Lewis's Theorem** for orthogonally invariant functions, if
$X = P \operatorname{diag}(s) Q^\top$ is a thin SVD of $X$
($P \in \mathbb{R}^{p \times r}$, $Q \in \mathbb{R}^{q \times r}$ with
orthonormal columns and $s = \sigma(X)$), then:

$$\nabla (f_2^{\mathcal{V}} \circ \sigma)(X) = P \operatorname{diag}\!\big(\nabla f_2^{\mathcal{V}}(s)\big) Q^\top, \qquad P \operatorname{diag}\!\big(\partial h^{\mathcal{V}}(s)\big) Q^\top \subseteq \partial (h^{\mathcal{V}} \circ \sigma)(X),$$

and the matrix proximal operator reduces to a 1D vector proximal step on $s$:

$$\operatorname{prox}_{\tau (h^{\mathcal{V}} \circ \sigma)}(X) = P \operatorname{diag}\!\big(\operatorname{prox}_{\tau h^{\mathcal{V}}}(s)\big) Q^\top.$$

Moreover, because $\|X\|_F^2 = \|\sigma(X)\|_2^2$, `IAICG` and `DAICG` can shift
any scalar $\xi \ge 0$ of quadratic curvature between the matrix term and the
singular-value term without changing the total objective $\Phi(X)$:

$$f_1^\xi(X) := f_1(X) - \frac{\xi}{2}\|X\|_F^2, \qquad f_2^{\mathcal{V},\xi}(s) := f_2^{\mathcal{V}}(s) + \frac{\xi}{2}\|s\|_2^2.$$

Choosing $\xi = m_2$ convexifies $f_2^{\mathcal{V},\xi}$ completely! Each outer
iteration linearizes $f_1^\xi$ at $Z_0$, computes **a single thin SVD** of the
matrix gradient step, solves a cheap $r$-dimensional vector subproblem for the
new singular values, and constructs an exact matrix certificate:

$$\hat{V} \in \nabla f_1(\hat{Z}) + \nabla (f_2^{\mathcal{V}} \circ \sigma)(\hat{Z}) + \partial (h^{\mathcal{V}} \circ \sigma)(\hat{Z}), \qquad \|\hat{V}\|_F \le \rho.$$
