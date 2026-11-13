# Class 2: Set- and Cone-Constrained Optimization (`CompositeProblem`)

## Formulation

$$\min_{x \in \mathcal{X}} \; f(x) + h(x) \quad \text{s.t.} \quad \mathcal{A}(x) \in \mathcal{C}$$

where:

- $\mathcal{A} : \mathcal{X} \to \mathcal{Y}$ is a linear map between Euclidean
  PyTree spaces $(\mathcal{X}, \langle \cdot, \cdot \rangle)$ and
  $(\mathcal{Y}, \langle \cdot, \cdot \rangle)$, with **adjoint**
  $\mathcal{A}^* : \mathcal{Y} \to \mathcal{X}$ uniquely defined by:

  $$\langle \mathcal{A}(x), p \rangle = \langle x, \mathcal{A}^*(p) \rangle \qquad \forall x \in \mathcal{X},\; p \in \mathcal{Y}.$$

- $\mathcal{C} \subseteq \mathcal{Y}$ is a nonempty closed convex set (or a
  Cartesian product $\mathcal{C}_1 \times \dots \times \mathcal{C}_k$ of convex
  sets) equipped with its exact **Euclidean projector**:

  $$\Pi_{\mathcal{C}}(w) := \operatorname*{argmin}_{z \in \mathcal{C}} \; \frac{1}{2}\|z - w\|_2^2.$$

This single set-inclusion abstraction $\mathcal{A}(x) \in \mathcal{C}$ unifies:

| Constraint Type | Set $\mathcal{C}$ | Projector $\Pi_{\mathcal{C}}(w)$ | Normal Cone $N_{\mathcal{C}}(z)$ |
| :--- | :--- | :--- | :--- |
| Linear equality $\mathcal{A}(x) = b$ | $\{b\}$ (`Singleton(b)`) | $b$ | $\mathcal{Y}$ (free multipliers) |
| Inequality $\mathcal{A}(x) \le u$ | $(-\infty, u]$ (`UpperBound(u)`) | $\min(w, u)$ | $\{p \ge 0 : p_i (z_i - u_i) = 0\}$ |
| Inequality $\mathcal{A}(x) \ge l$ | $[l, +\infty)$ (`LowerBound(l)`) | $\max(w, l)$ | $\{p \le 0 : p_i (z_i - l_i) = 0\}$ |
| Two-sided bounds $l \le \mathcal{A}(x) \le u$ | $[l, u]$ (`BoxSet`) | $\operatorname{clip}(w, l, u)$ | Sign-complementary multipliers |
| Second-order cone $\|(u_1, \dots, u_d)\|_2 \le t$ | $\mathcal{K}_{\text{SOC}}$ (`SecondOrderCone`) | Closed-form Lorentz cone projection | $-\mathcal{K}_{\text{SOC}}$ with complementarity |

## Certified $(\rho, \eta)$-KKT Conditions

A primal-dual pair $(x^\star, p^\star) \in \operatorname{dom} h \times \mathcal{Y}$
is a **KKT point** if:

$$0 \in \nabla f(x^\star) + \partial h(x^\star) + \mathcal{A}^*(p^\star), \qquad \mathcal{A}(x^\star) \in \mathcal{C}, \qquad p^\star \in N_{\mathcal{C}}\!\big(\mathcal{A}(x^\star)\big).$$

Given optimality tolerance $\rho > 0$ (`opt_tol`) and feasibility tolerance
$\eta > 0$ (`feas_tol`), constrained solvers in Proxax return a **certified
$(\rho, \eta)$-KKT triplet** $(\hat{x}, \hat{p}, \hat{v})$ satisfying:

1. **Domain membership:** $\hat{x} \in \operatorname{dom} h$.
2. **Approximate primal feasibility:**

   $$\big\|\mathcal{A}(\hat{x}) - \Pi_{\mathcal{C}}\!\big(\mathcal{A}(\hat{x})\big)\big\| \le \eta.$$

3. **Normal-cone dual inclusion & Lagrangian stationarity:** there exists a
   projected point $\hat{z} = \Pi_{\mathcal{C}}(w) \in \mathcal{C}$ such that
   $\hat{p} \in N_{\mathcal{C}}(\hat{z})$ holds **exactly** and:

   $$\hat{v} \in \nabla f(\hat{x}) + \partial h(\hat{x}) + \mathcal{A}^*(\hat{p}), \qquad \|\hat{v}\| \le \rho.$$

In `Solution`, `sol.x` holds $\hat{x}$, `sol.dual` holds $\hat{p}$,
`sol.subgrad` holds $\hat{v}$, `sol.stats.opt_residual` equals $\|\hat{v}\|$,
and `sol.stats.feas_residual` equals
$\|\mathcal{A}(\hat{x}) - \Pi_{\mathcal{C}}(\mathcal{A}(\hat{x}))\|$.
