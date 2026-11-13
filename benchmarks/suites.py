"""Standardized benchmark problem suites across Proxax problem classes."""

from __future__ import annotations

import dataclasses
import time
import typing

import jax
import jax.numpy as jnp
import optax

from proxax import constraints
from proxax import optax_compat
from proxax import oracle
from proxax import prox
from proxax import px
from proxax import solvers
from proxax import tree


@dataclasses.dataclass(frozen=True, slots=True)
class BenchmarkProblem:
  """Container for a benchmark problem instance and initial point."""

  suite: str
  name: str
  prob: px.CompositeProblem | px.SpectralProblem
  x0: typing.Any


class _FantopeProx(prox.ProxOperator):
  """Indicator proximal operator for the k-Fantope."""

  def __init__(self, rank_k: int) -> None:
    self.rank_k = rank_k

  @property
  def strong_convexity(self) -> float:
    return 0.0

  def domain_project(self, x: typing.Any) -> jax.Array:
    return self.prox(x, jnp.asarray(1.0))

  def value(self, x: typing.Any) -> jax.Array:
    del x
    return jnp.asarray(0.0)

  def prox(self, x: typing.Any, step_size: typing.Any) -> jax.Array:
    del step_size
    mat = typing.cast(jax.Array, x)
    sym = 0.5 * (mat + mat.T)
    eigvals, eigvecs = jnp.linalg.eigh(sym)
    hi = jnp.max(eigvals) + 1.0
    lo = jnp.min(eigvals) - 1.0

    def _cond(state: tuple[jax.Array, jax.Array, jax.Array]) -> jax.Array:
      _, _, it = state
      return it < 40

    def _body(
        state: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
      low, high, it = state
      mid = 0.5 * (low + high)
      clipped = jnp.clip(eigvals - mid, 0.0, 1.0)
      low = jnp.where(jnp.sum(clipped) > self.rank_k, mid, low)
      high = jnp.where(jnp.sum(clipped) <= self.rank_k, mid, high)
      return low, high, it + 1

    lo_f, hi_f, _ = jax.lax.while_loop(
        _cond, _body, (lo, hi, jnp.asarray(0))
    )
    theta = 0.5 * (lo_f + hi_f)
    eigs_proj = jnp.clip(eigvals - theta, 0.0, 1.0)
    return typing.cast(
        jax.Array, (eigvecs * eigs_proj[None, :]) @ eigvecs.T
    )


def indefinite_box_qp(
    *,
    dim: int = 100,
    m_ratio: float = 0.25,
    seed: int = 0,
) -> BenchmarkProblem:
  """Generates an indefinite box-constrained QP with curvature ratio m/M."""
  key = jax.random.key(seed)
  k_mat, k_vec, k_x0 = jax.random.split(key, 3)
  q_mat, _ = jnp.linalg.qr(jax.random.normal(k_mat, (dim, dim)))
  n_neg = max(1, dim // 5)
  pos_eigs = jnp.linspace(0.5, 10.0, dim - n_neg)
  neg_eigs = -m_ratio * 10.0 * jnp.linspace(0.5, 1.0, n_neg)
  eigs = jnp.concatenate([neg_eigs, pos_eigs])
  h_mat = q_mat @ jnp.diag(eigs) @ q_mat.T
  h_mat = 0.5 * (h_mat + h_mat.T)
  c_vec = jax.random.normal(k_vec, (dim,))
  x0 = jax.random.uniform(k_x0, (dim,), minval=-0.5, maxval=0.5)

  def smooth_fn(x: jax.Array) -> jax.Array:
    return 0.5 * jnp.dot(x, h_mat @ x) + jnp.dot(c_vec, x)

  prob = px.CompositeProblem(
      smooth=oracle.SmoothTerm(
          smooth_fn,
          upper_curvature=10.0,
          lower_curvature=max(0.1, float(m_ratio * 10.0)),
      ),
      proximal=prox.Box(lower=-1.0, upper=1.0),
  )
  return BenchmarkProblem(
      suite="unconstrained",
      name=f"indef_box_qp(n={dim},r={m_ratio})",
      prob=prob,
      x0=x0,
  )


def sparse_nonconvex_margin(
    *,
    n_samples: int = 120,
    n_features: int = 40,
    l1_weight: float = 0.05,
    seed: int = 1,
) -> BenchmarkProblem:
  """Generates an L1-regularized nonconvex tanh-margin classification task."""
  key = jax.random.key(seed)
  k_a, k_w, k_noise = jax.random.split(key, 3)
  a_mat = jax.random.normal(k_a, (n_samples, n_features)) / jnp.sqrt(n_features)
  w_true = jax.random.normal(k_w, (n_features,))
  logits = a_mat @ w_true + 0.2 * jax.random.normal(k_noise, (n_samples,))
  labels = jnp.where(logits >= 0.0, 1.0, -1.0)

  def smooth_fn(x: jax.Array) -> jax.Array:
    margins = labels * (a_mat @ x)
    return jnp.mean(1.0 - jnp.tanh(margins))

  prob = px.CompositeProblem(
      smooth=smooth_fn,
      proximal=prox.L1(weight=l1_weight),
  )
  return BenchmarkProblem(
      suite="unconstrained",
      name=f"tanh_l1(n={n_samples},d={n_features})",
      prob=prob,
      x0=jnp.zeros(n_features),
  )


def simplex_linear_qp(
    *,
    n_eq: int = 4,
    dim: int = 24,
    seed: int = 777,
) -> BenchmarkProblem:
  """Generates a linearly-constrained nonconvex QP over the simplex."""
  key = jax.random.key(seed)
  k1, k2, k3, k4, k5 = jax.random.split(key, 5)
  d_diag = jax.random.randint(k1, (dim,), 1, 10).astype(jnp.ones(()).dtype)
  c_mat = jax.random.uniform(k2, (n_eq, dim))
  b_mat = jax.random.uniform(k3, (dim, dim))
  a_eq = jax.random.uniform(k4, (n_eq, dim))
  d_vec = jax.random.uniform(k5, (n_eq,))

  db_mat = d_diag[:, None] * b_mat
  hp = c_mat.T @ c_mat
  hn = db_mat.T @ db_mat
  xi = 2.0 / float(jnp.max(jnp.linalg.eigvalsh(hn)))
  tau = 20.0 / float(jnp.max(jnp.linalg.eigvalsh(hp)))
  sym = 0.5 * (tau * hp - xi * hn + (tau * hp - xi * hn).T)
  eigs = jnp.linalg.eigvalsh(sym)
  m_upper = float(jnp.maximum(0.1, jnp.max(eigs)))
  m_lower = float(jnp.maximum(0.1, -jnp.min(eigs)))
  rhs_b = a_eq @ (jnp.ones(dim) / dim)

  def smooth_fn(x: jax.Array) -> jax.Array:
    res = c_mat @ x - d_vec
    dbx = db_mat @ x
    return 0.5 * tau * jnp.dot(res, res) - 0.5 * xi * jnp.dot(dbx, dbx)

  prob = px.CompositeProblem(
      smooth=oracle.SmoothTerm(
          smooth_fn,
          upper_curvature=m_upper,
          lower_curvature=m_lower,
      ),
      proximal=prox.Simplex(scale=1.0),
      constraints=constraints.equality(a_eq, rhs=rhs_b),
  )
  return BenchmarkProblem(
      suite="constrained",
      name=f"simplex_qp(m={n_eq},n={dim})",
      prob=prob,
      x0=jnp.ones(dim) / dim,
  )


def fantope_sparse_pca(
    *,
    dim: int = 10,
    k_rank: int = 2,
    nu: float = 0.05,
    b_mcp: float = 2.0,
    seed: int = 42,
) -> BenchmarkProblem:
  """Generates a consensus k-Fantope Sparse PCA problem."""
  key = jax.random.key(seed)
  data = jax.random.normal(key, (40, dim))
  cov = (data.T @ data) / 40.0
  cov = 0.5 * (cov + cov.T)

  def mcp_concave_part(phi: jax.Array) -> jax.Array:
    abs_p = jnp.abs(phi)
    thresh = b_mcp * nu
    quad = -(phi**2) / (2.0 * b_mcp)
    lin = 0.5 * b_mcp * (nu**2) - nu * abs_p
    return jnp.sum(jnp.where(abs_p <= thresh, quad, lin))

  def smooth_fn(z: dict[str, jax.Array]) -> jax.Array:
    return -jnp.sum(cov * z["pi"]) + mcp_concave_part(z["phi"])

  prob = px.CompositeProblem(
      smooth=oracle.SmoothTerm(
          smooth_fn,
          upper_curvature=1.0 / b_mcp,
          lower_curvature=1.0 / b_mcp,
      ),
      proximal=prox.MultiProx({
          "pi": _FantopeProx(k_rank),
          "phi": prox.L1(weight=nu),
      }),
      constraints=constraints.equality(
          lambda z: z["pi"] - z["phi"],
          rhs=jnp.zeros((dim, dim)),
      ),
  )
  init = (float(k_rank) / float(dim)) * jnp.eye(dim)
  return BenchmarkProblem(
      suite="constrained",
      name=f"fantope_spca(p={dim},k={k_rank})",
      prob=prob,
      x0={"pi": init, "phi": init},
  )


def spectral_matrix_completion(
    *,
    rows: int = 8,
    cols: int = 8,
    rank: int = 2,
    obs_frac: float = 0.45,
    seed: int = 13,
) -> BenchmarkProblem:
  """Generates a nonconvex spectral-regularized matrix completion task."""
  key = jax.random.key(seed)
  k_u, k_v, k_mask, k_noise = jax.random.split(key, 4)
  u_mat = jax.random.normal(k_u, (rows, rank))
  v_mat = jax.random.normal(k_v, (cols, rank))
  m_true = u_mat @ v_mat.T
  mask = (jax.random.uniform(k_mask, (rows, cols)) < obs_frac).astype(
      m_true.dtype
  )
  obs = mask * (m_true + 0.05 * jax.random.normal(k_noise, (rows, cols)))

  prob = px.SpectralProblem(
      f1=lambda x: 0.5 * jnp.sum(mask * (x - obs) ** 2),
      f2_vec=lambda s: -0.05 * jnp.sum((s**2) / (1.0 + s**2)),
      prox_vec=prox.L1(weight=0.1),
      domain_proj=lambda x: jnp.clip(x, -3.0, 3.0),
  )
  return BenchmarkProblem(
      suite="spectral",
      name=f"spec_mc({rows}x{cols},r={rank})",
      prob=prob,
      x0=jnp.zeros((rows, cols)),
  )


def run_optax_prox_baseline(
    bp: BenchmarkProblem,
    *,
    steps: int = 500,
    lr: float = 0.05,
    use_adam: bool = False,
) -> dict[str, float | int | str]:
  """Runs an Optax proximal-gradient / Adam baseline."""
  assert isinstance(bp.prob, px.CompositeProblem)
  comp = bp.prob.oracle
  prox_op = bp.prob.proximal or prox.Zero()
  tx = (
      optax.chain(
          optax.adam(lr),
          optax_compat.proximal(prox_op, step_size=lr),
      )
      if use_adam
      else optax.chain(
          optax.scale(-lr),
          optax_compat.proximal(prox_op, step_size=lr),
      )
  )

  @jax.jit
  def _run(x_init: typing.Any) -> tuple[typing.Any, jax.Array]:
    def _step(
        _: int, carry: tuple[typing.Any, optax.OptState]
    ) -> tuple[typing.Any, optax.OptState]:
      x, opt_st = carry
      _val, grad = comp.smooth_value_and_grad(x)
      updates, opt_st = tx.update(grad, opt_st, x)
      return optax.apply_updates(x, updates), opt_st

    x_fin, _ = jax.lax.fori_loop(0, steps, _step, (x_init, tx.init(x_init)))
    _, grad_fin = comp.smooth_value_and_grad(x_fin)
    prox_step = prox_op.prox(tree.sub(x_fin, grad_fin), jnp.asarray(1.0))
    res = tree.norm(tree.sub(x_fin, prox_step))
    return x_fin, res

  t0 = time.perf_counter()
  x_fin, res = _run(bp.x0)
  res.block_until_ready()
  compile_s = time.perf_counter() - t0

  t1 = time.perf_counter()
  x_fin, res = _run(bp.x0)
  res.block_until_ready()
  run_s = time.perf_counter() - t1

  return {
      "solver": "optax_prox_adam" if use_adam else "optax_prox_gd",
      "obj_val": float(comp.value(x_fin)),
      "pg_residual": float(res),
      "grad_evals": steps,
      "compile_s": compile_s,
      "run_s": run_s,
  }


def run_vmap_throughput(
    *,
    batch_sizes: tuple[int, ...] = (1, 16, 64),
    dim: int = 16,
    opt_tol: float = 1e-3,
    seed: int = 0,
) -> list[dict[str, float | int]]:
  """Measures batched solve throughput under jax.jit(jax.vmap(...))."""
  key = jax.random.key(seed)
  q_mat, _ = jnp.linalg.qr(jax.random.normal(key, (dim, dim)))
  eigs = jnp.concatenate([
      -1.0 * jnp.ones(dim // 4),
      jnp.linspace(0.5, 5.0, dim - dim // 4),
  ])
  h_mat = 0.5 * (
      q_mat @ jnp.diag(eigs) @ q_mat.T + (q_mat @ jnp.diag(eigs) @ q_mat.T).T
  )
  solver = solvers.AdapFISTA(opt_tol=opt_tol)
  prox_op = prox.Box(lower=-1.0, upper=1.0)
  rows: list[dict[str, float | int]] = []

  for bsz in batch_sizes:
    k_c, k_x = jax.random.split(jax.random.fold_in(key, bsz))
    c_batch = jax.random.normal(k_c, (bsz, dim))
    x0_batch = jax.random.uniform(k_x, (bsz, dim), minval=-0.5, maxval=0.5)

    def _solve_one(c_vec: jax.Array, x_init: jax.Array) -> jax.Array:
      prob = px.CompositeProblem(
          smooth=oracle.SmoothTerm(
              lambda x: 0.5 * jnp.dot(x, h_mat @ x) + jnp.dot(c_vec, x),
              upper_curvature=5.0,
              lower_curvature=1.0,
          ),
          proximal=prox_op,
      )
      return typing.cast(
          jax.Array,
          px.solve(prob, x_init, solver=solver, opt_tol=opt_tol).x,
      )

    batched_fn = jax.jit(jax.vmap(_solve_one))
    t0 = time.perf_counter()
    out = batched_fn(c_batch, x0_batch)
    out.block_until_ready()
    compile_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    out = batched_fn(c_batch, x0_batch)
    out.block_until_ready()
    run_s = time.perf_counter() - t1

    rows.append({
        "batch_size": bsz,
        "compile_s": compile_s,
        "run_s": run_s,
        "throughput_per_s": float(bsz / max(run_s, 1e-12)),
    })

  return rows
