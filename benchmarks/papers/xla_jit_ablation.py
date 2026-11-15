"""Benchmark comparing eager vs. XLA-compiled proxax and CVXPY.

Measures single-solve latency (``jit=False`` vs. ``jit=True``), ``lax.scan``
parameterized sweeps, and implicit-layer VJP timings on CPU.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import importlib  # noqa: E402
import time  # noqa: E402
import typing  # noqa: E402
from collections.abc import Callable  # noqa: E402

import cvxpy as cp  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import numpy.typing as npt  # noqa: E402

import proxax as px  # noqa: E402

_FloatArray = npt.NDArray[np.float64]
_solve_mod: typing.Any = importlib.import_module("proxax.solve")


def _ready(tree: typing.Any) -> None:
    """Blocks until all JAX array leaves in ``tree`` finish executing."""
    jax.tree_util.tree_map(lambda leaf: leaf.block_until_ready(), tree)


def _min_ms(
    fn: Callable[[], typing.Any],
    *,
    warmups: int = 1,
    repeats: int = 5,
) -> float:
    """Returns the minimum execution time of ``fn()`` in milliseconds."""
    for _ in range(warmups):
        _ready(fn())
    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ready(fn())
        times.append((time.perf_counter() - t0) * 1000.0)
    return min(times)


def _cvxpy_min_ms(
    fn: Callable[[], typing.Any],
    *,
    warmups: int = 1,
    repeats: int = 5,
) -> float:
    """Returns the minimum execution time of a CVXPY call in milliseconds."""
    for _ in range(warmups):
        fn()
    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return min(times)


def _cvxpy_solve_min_ms(
    prob: cp.Problem,
    solver_name: str,
    *,
    warmups: int = 1,
    repeats: int = 5,
) -> float:
    """Returns the minimum cached CVXPY solve time in milliseconds."""
    return _cvxpy_min_ms(
        lambda: prob.solve(solver=solver_name),
        warmups=warmups,
        repeats=repeats,
    )


def _time_proxax_jit_vs_eager(
    prob: px.CompositeProblem,
    x0: jax.Array,
    solver: typing.Any,
    *,
    jit_repeats: int,
    eager_warmups: int,
    eager_repeats: int,
) -> tuple[px.Solution, px.Solution, float, float, float]:
    """Times cold JIT, steady-state JIT, and eager solves in milliseconds."""

    def _run_jit() -> px.Solution:
        sol = px.solve(prob, x0, solver=solver, jit=True)
        _ready(sol.x)
        return sol

    def _run_eager() -> px.Solution:
        sol = px.solve(prob, x0, solver=solver, jit=False)
        _ready(sol.x)
        return sol

    t0 = time.perf_counter()
    sol_jit = _run_jit()
    cold_ms = (time.perf_counter() - t0) * 1000.0
    steady_ms = _min_ms(_run_jit, warmups=1, repeats=jit_repeats)
    comp_ms = max(0.0, cold_ms - steady_ms)

    t1 = time.perf_counter()
    sol_nojit = _run_eager()
    first_eager_ms = (time.perf_counter() - t1) * 1000.0
    if eager_repeats > 0:
        eager_ms = _min_ms(
            _run_eager, warmups=eager_warmups, repeats=eager_repeats
        )
    else:
        eager_ms = first_eager_ms
    return sol_jit, sol_nojit, comp_ms, steady_ms, eager_ms


def make_box_qp(
    n: int, seed: int = 0
) -> tuple[px.CompositeProblem, jax.Array, _FloatArray, _FloatArray]:
    """Constructs a strictly convex box-constrained quadratic program."""
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((n, max(8, n // 2)))
    Q_np = (B @ B.T) / n + 0.25 * np.eye(n)
    c_np = rng.standard_normal(n)
    Q = jnp.asarray(Q_np)
    c = jnp.asarray(c_np)
    x0 = jnp.zeros(n)

    prob = px.CompositeProblem(
        smooth=lambda x: 0.5 * jnp.dot(x, Q @ x) + jnp.dot(c, x),
        proximal=px.prox.Box(lower=-1.0, upper=1.0),
    )
    return prob, x0, Q_np, c_np


def make_eq_box_qp(
    n: int, m: int, seed: int = 1
) -> tuple[
    px.CompositeProblem,
    jax.Array,
    _FloatArray,
    _FloatArray,
    _FloatArray,
    _FloatArray,
]:
    """Constructs a feasible equality- and box-constrained QP."""
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((n, max(8, n // 2)))
    Q_np = (F @ F.T) / n + 0.5 * np.eye(n)
    c_np = rng.standard_normal(n)
    A_np = rng.standard_normal((m, n)) / np.sqrt(n)
    x_feas = rng.uniform(-0.5, 0.5, size=n)
    b_np = A_np @ x_feas

    Q = jnp.asarray(Q_np)
    c = jnp.asarray(c_np)
    A = jnp.asarray(A_np)
    b = jnp.asarray(b_np)
    x0 = jnp.zeros(n)

    prob = px.CompositeProblem(
        smooth=lambda x: 0.5 * jnp.dot(x, Q @ x) + jnp.dot(c, x),
        proximal=px.prox.Box(lower=-1.0, upper=1.0),
        constraints=px.constraints.equality(A, b),
    )
    return prob, x0, Q_np, c_np, A_np, b_np


def make_fft_deblur(
    side: int, seed: int = 2
) -> tuple[
    px.CompositeProblem,
    jax.Array,
    int,
    _FloatArray,
    _FloatArray,
    float,
]:
    """Constructs a matrix-free 2D FFT L1 deblurring composite problem."""
    n = side * side
    rng = np.random.default_rng(seed)
    x_img = np.zeros((side, side))
    x_img[side // 4 : side // 2, side // 4 : side // 2] = 1.0
    x_true = x_img.ravel()
    k1 = np.zeros(side)
    k1[:3] = [0.5, 0.2, 0.05]
    k1[-2:] = [0.05, 0.2]
    ker = np.outer(k1, k1)
    ker_fft = np.fft.rfft2(ker)
    y_np = np.fft.irfft2(
        ker_fft * np.fft.rfft2(x_true.reshape(side, side)), s=(side, side)
    ).ravel()
    y_np += 0.01 * rng.standard_normal(n)

    ker_fft_j = jnp.asarray(ker_fft)
    y_j = jnp.asarray(y_np)
    lam = 0.02

    def blur(x: jax.Array) -> jax.Array:
        img = x.reshape((side, side))
        convolved = jnp.fft.irfft2(
            ker_fft_j * jnp.fft.rfft2(img), s=(side, side)
        )
        return convolved.ravel()

    prob = px.CompositeProblem(
        smooth=lambda x: 0.5 * jnp.sum((blur(x) - y_j) ** 2),
        proximal=px.prox.L1(lam),
    )
    x0 = jnp.zeros(n)
    return prob, x0, side, ker, y_np, lam


def _run_single_solve_benchmarks() -> None:
    """Runs single-solve JIT vs. eager and CVXPY benchmarks."""
    print("=== Single-Solve XLA JIT vs. Eager (jit=False) vs. CVXPY ===")
    for n in [64, 256, 512]:
        _solve_mod._COMPILED_SOLVE_CACHE.clear()
        prob, x0, Q_np, c_np = make_box_qp(n)
        panoc = px.solvers.PANOC(opt_tol=1e-6, max_iters=2000)
        sol_jit, sol_nojit, comp_ms, steady_jit_ms, eager_ms = (
            _time_proxax_jit_vs_eager(
                prob,
                x0,
                panoc,
                jit_repeats=7,
                eager_warmups=1,
                eager_repeats=3,
            )
        )

        def run_cvx_fresh(
            solver_name: str,
            dim: int = n,
            q_mat: _FloatArray = Q_np,
            c_vec: _FloatArray = c_np,
        ) -> float | None:
            xv = cp.Variable(dim)
            obj = cp.Minimize(
                0.5 * cp.quad_form(xv, cp.psd_wrap(q_mat)) + c_vec @ xv
            )
            p = cp.Problem(obj, [xv >= -1.0, xv <= 1.0])
            val = p.solve(solver=solver_name)
            return None if val is None else float(val)

        xv = cp.Variable(n)
        p_clar = cp.Problem(
            cp.Minimize(0.5 * cp.quad_form(xv, cp.psd_wrap(Q_np)) + c_np @ xv),
            [xv >= -1.0, xv <= 1.0],
        )
        p_clar.solve(solver=cp.CLARABEL)
        cvx_fresh_ms = _cvxpy_min_ms(
            lambda: run_cvx_fresh(cp.CLARABEL), warmups=1, repeats=5
        )
        cvx_clar_ms = _cvxpy_solve_min_ms(p_clar, cp.CLARABEL)
        cvx_scs_ms = _cvxpy_solve_min_ms(p_clar, cp.SCS)
        assert p_clar.value is not None
        obj_diff = abs(float(sol_jit.stats.obj_val) - float(p_clar.value))
        print(
            f"BoxQP n={n} | status={sol_jit.status} "
            f"g_jit={int(sol_jit.stats.grad_evals)} "
            f"g_eager={int(sol_nojit.stats.grad_evals)} | "
            f"jit=False: {eager_ms:.2f} ms | jit=True: {steady_jit_ms:.3f} ms "
            f"(comp {comp_ms:.0f} ms, "
            f"jit_boost={eager_ms / steady_jit_ms:.1f}x) | "
            f"CVXPY-Clar fresh/cached: "
            f"{cvx_fresh_ms:.2f}/{cvx_clar_ms:.2f} ms "
            f"({cvx_clar_ms / steady_jit_ms:.1f}x) | "
            f"CVXPY-SCS cached: {cvx_scs_ms:.2f} ms | obj_diff={obj_diff:.2e}",
            flush=True,
        )

    for n, m in [(64, 16), (128, 32)]:
        _solve_mod._COMPILED_SOLVE_CACHE.clear()
        prob, x0, Q_np, c_np, A_np, b_np = make_eq_box_qp(n, m)
        ialm = px.solvers.IALM(opt_tol=1e-5, feas_tol=1e-5)
        sol_jit, _, comp_ms, steady_jit_ms, eager_ms = (
            _time_proxax_jit_vs_eager(
                prob,
                x0,
                ialm,
                jit_repeats=5,
                eager_warmups=0,
                eager_repeats=0,
            )
        )

        xv = cp.Variable(n)
        p_clar = cp.Problem(
            cp.Minimize(0.5 * cp.quad_form(xv, cp.psd_wrap(Q_np)) + c_np @ xv),
            [A_np @ xv == b_np, xv >= -1.0, xv <= 1.0],
        )
        p_clar.solve(solver=cp.CLARABEL)
        cvx_clar_ms = _cvxpy_solve_min_ms(p_clar, cp.CLARABEL)
        cvx_scs_ms = _cvxpy_solve_min_ms(p_clar, cp.SCS)
        assert p_clar.value is not None
        obj_diff = abs(float(sol_jit.stats.obj_val) - float(p_clar.value))
        print(
            f"EqBoxQP (n={n},m={m}) | status={sol_jit.status} "
            f"g_jit={int(sol_jit.stats.grad_evals)} | "
            f"jit=False: {eager_ms:.2f} ms | jit=True: {steady_jit_ms:.2f} ms "
            f"(comp {comp_ms:.0f} ms, "
            f"jit_boost={eager_ms / steady_jit_ms:.1f}x) | "
            f"CVXPY-Clar(c): {cvx_clar_ms:.2f} ms | "
            f"CVXPY-SCS(c): {cvx_scs_ms:.2f} ms | obj_diff={obj_diff:.2e}",
            flush=True,
        )

    for side in [32, 64]:
        _solve_mod._COMPILED_SOLVE_CACHE.clear()
        prob, x0, _, ker, y_np, lam = make_fft_deblur(side)
        panoc = px.solvers.PANOC(opt_tol=1e-5, max_iters=2000)
        sol_jit, _, comp_ms, steady_jit_ms, eager_ms = (
            _time_proxax_jit_vs_eager(
                prob,
                x0,
                panoc,
                jit_repeats=5,
                eager_warmups=0,
                eager_repeats=2,
            )
        )

        if side <= 32:
            n = side * side
            fft_k = np.fft.fft2(ker)
            eye = np.eye(n).reshape(n, side, side)
            a_mat = (
                np.fft.ifft2(fft_k * np.fft.fft2(eye, axes=(1, 2)), axes=(1, 2))
                .real.reshape(n, n)
                .T
            )
            xv = cp.Variable(n)
            p_clar = cp.Problem(
                cp.Minimize(
                    0.5 * cp.sum_squares(a_mat @ xv - y_np) + lam * cp.norm1(xv)
                )
            )
            p_clar.solve(solver=cp.CLARABEL)
            cvx_clar_ms = _cvxpy_solve_min_ms(
                p_clar, cp.CLARABEL, warmups=0, repeats=1
            )
            cvx_str = (
                f"CVXPY-Clar(c): {cvx_clar_ms:.2f} ms "
                f"({cvx_clar_ms / steady_jit_ms:.1f}x)"
            )
        else:
            cvx_str = "CVXPY-Clar: infeasible dense matrix (n=4096)"

        print(
            f"FFTDeblur n={side * side} | status={sol_jit.status} "
            f"g_jit={int(sol_jit.stats.grad_evals)} | "
            f"jit=False: {eager_ms:.2f} ms | jit=True: {steady_jit_ms:.2f} ms "
            f"(comp {comp_ms:.0f} ms, "
            f"jit_boost={eager_ms / steady_jit_ms:.1f}x) | {cvx_str}"
        )


def _run_sweep_benchmark() -> None:
    """Runs 50-step closed-loop sweep with lax.scan, eager, and CVXPY DPP."""
    print(
        "\n=== Closed-Loop 50-Step Parameterized Sweep "
        "(lax.scan vs. Python eager vs. CVXPY DPP) ==="
    )
    n = 64
    num_steps = 50
    _, _, Q_np, c_np = make_box_qp(n, seed=7)
    rng = np.random.default_rng(7)
    C_traj_np = c_np[None, :] + 0.1 * rng.standard_normal((num_steps, n))
    Q_j = jnp.asarray(Q_np)
    C_traj_j = jnp.asarray(C_traj_np)
    solver = px.solvers.PANOC(opt_tol=1e-6, max_iters=500)

    def scan_solve(c_seq: jax.Array) -> jax.Array:
        def step(
            x_prev: jax.Array, c_k: jax.Array
        ) -> tuple[jax.Array, jax.Array]:
            p_k = px.CompositeProblem(
                smooth=lambda x: 0.5 * jnp.dot(x, Q_j @ x) + jnp.dot(c_k, x),
                proximal=px.prox.Box(lower=-1.0, upper=1.0),
            )
            s_k = px.solve(p_k, x_prev, solver=solver, jit=True)
            return s_k.x, s_k.stats.obj_val

        _, objs = jax.lax.scan(step, jnp.zeros(n), c_seq)
        return objs

    jit_scan = jax.jit(scan_solve)
    t0 = time.perf_counter()
    _ready(jit_scan(C_traj_j))
    scan_cold_ms = (time.perf_counter() - t0) * 1000.0
    scan_ms = _min_ms(lambda: jit_scan(C_traj_j), warmups=1, repeats=7)
    scan_comp_ms = max(0.0, scan_cold_ms - scan_ms)

    def py_loop_eager() -> jax.Array:
        x_cur = jnp.zeros(n)
        for k in range(num_steps):
            c_k = C_traj_j[k]
            p_k = px.CompositeProblem(
                smooth=lambda x, ck=c_k: (
                    0.5 * jnp.dot(x, Q_j @ x) + jnp.dot(ck, x)
                ),
                proximal=px.prox.Box(lower=-1.0, upper=1.0),
            )
            s_k = px.solve(p_k, x_cur, solver=solver, jit=False)
            x_cur = s_k.x
        return x_cur

    t0 = time.perf_counter()
    _ready(py_loop_eager())
    eager_loop_ms = (time.perf_counter() - t0) * 1000.0

    xv = cp.Variable(n)
    c_param = cp.Parameter(n)
    p_dpp = cp.Problem(
        cp.Minimize(0.5 * cp.quad_form(xv, cp.psd_wrap(Q_np)) + c_param @ xv),
        [xv >= -1.0, xv <= 1.0],
    )
    c_param.value = C_traj_np[0]
    p_dpp.solve(solver=cp.CLARABEL)

    def run_dpp(solver_name: str) -> None:
        for k in range(num_steps):
            c_param.value = C_traj_np[k]
            p_dpp.solve(solver=solver_name, warm_start=True)

    dpp_clar_ms = _cvxpy_min_ms(
        lambda: run_dpp(cp.CLARABEL), warmups=1, repeats=5
    )
    dpp_scs_ms = _cvxpy_min_ms(lambda: run_dpp(cp.SCS), warmups=1, repeats=5)

    print(
        f"50-step BoxQP Sweep (n={n}): px lax.scan jit=True={scan_ms:.2f} ms "
        f"(comp {scan_comp_ms:.0f} ms) | "
        f"px jit=False={eager_loop_ms:.2f} ms "
        f"({eager_loop_ms / scan_ms:.1f}x) | "
        f"CVXPY-Clar(DPP)={dpp_clar_ms:.2f} ms "
        f"({dpp_clar_ms / scan_ms:.1f}x) | "
        f"CVXPY-SCS(DPP)={dpp_scs_ms:.2f} ms ({dpp_scs_ms / scan_ms:.1f}x)",
        flush=True,
    )


def _run_implicit_vjp_benchmark() -> None:
    """Runs full jax.jit vs. eager outer AD benchmark on QPLayer."""
    print(
        "\n=== Differentiable Implicit Layer VJP "
        "(full jax.jit vs. eager outer AD) ==="
    )
    n_ad, m_ad = 50, 16
    _, _, Q_ad_np, c_ad_np, A_ad_np, b_ad_np = make_eq_box_qp(
        n_ad, m_ad, seed=11
    )
    Q_ad = jnp.asarray(Q_ad_np)
    c_ad = jnp.asarray(c_ad_np)
    A_ad = jnp.asarray(A_ad_np)
    b_ad = jnp.asarray(b_ad_np)
    qp_layer = px.layers.QPLayer(
        Q=Q_ad,
        proximal=px.prox.Box(lower=-1.0, upper=1.0),
        opt_tol=1e-5,
        feas_tol=1e-5,
        linear_tol=1e-5,
    )

    def qp_loss(c_p: jax.Array) -> jax.Array:
        x_star = qp_layer(c_p, A_ad, b_ad)
        return 0.5 * jnp.sum((x_star - 0.25) ** 2)

    vag_eager = jax.value_and_grad(qp_loss)
    vag_jit = jax.jit(vag_eager)

    t0 = time.perf_counter()
    v_j, g_j = vag_jit(c_ad)
    _ready((v_j, g_j))
    cold_ad_ms = (time.perf_counter() - t0) * 1000.0
    jit_ad_ms = _min_ms(lambda: vag_jit(c_ad), warmups=1, repeats=7)
    comp_ad_ms = max(0.0, cold_ad_ms - jit_ad_ms)

    v_e, g_e = vag_eager(c_ad)
    _ready((v_e, g_e))
    eager_ad_ms = _min_ms(lambda: vag_eager(c_ad), warmups=1, repeats=5)
    rel_err = float(
        jnp.linalg.norm(g_j - g_e) / jnp.maximum(jnp.linalg.norm(g_j), 1e-12)
    )
    print(
        f"QPLayer Fwd+VJP (n={n_ad},m={m_ad}): outer_eager: "
        f"{eager_ad_ms:.2f} ms | full_jit: {jit_ad_ms:.2f} ms "
        f"(comp {comp_ad_ms:.0f} ms, boost={eager_ad_ms / jit_ad_ms:.1f}x) | "
        f"grad_rel_err={rel_err:.2e}",
        flush=True,
    )


def run_all() -> None:
    """Runs all XLA JIT ablation benchmarks."""
    _run_single_solve_benchmarks()
    _run_sweep_benchmark()
    _run_implicit_vjp_benchmark()


if __name__ == "__main__":
    run_all()
