"""Benchmark comparing eager (jit=False) vs. XLA-compiled (jit=True) proxax
against CVXPY (Clarabel / SCS) across single solves and scan-fused sweeps.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import importlib
import time
import cvxpy as cp
import jax
import jax.numpy as jnp
import numpy as np
import proxax as px

solve_mod = importlib.import_module("proxax.solve")


def _min_ms(fn, *, warmups: int = 1, repeats: int = 5) -> float:
    for _ in range(warmups):
        res = fn()
        jax.block_until_ready(res)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        res = fn()
        jax.block_until_ready(res)
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(min(times))


def _cvxpy_min_ms(fn, *, warmups: int = 1, repeats: int = 5) -> float:
    for _ in range(warmups):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(min(times))


def make_box_qp(n: int, seed: int = 0):
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


def make_eq_box_qp(n: int, m: int, seed: int = 1):
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


def make_fft_deblur(side: int, seed: int = 2):
    n = side * side
    rng = np.random.default_rng(seed)
    x_true = np.zeros((side, side))
    x_true[side // 4 : side // 2, side // 4 : side // 2] = 1.0
    x_true = x_true.ravel()
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

    def blur(x):
        img = x.reshape((side, side))
        return jnp.fft.irfft2(ker_fft_j * jnp.fft.rfft2(img), s=(side, side)).ravel()

    prob = px.CompositeProblem(
        smooth=lambda x: 0.5 * jnp.sum((blur(x) - y_j) ** 2),
        proximal=px.prox.L1(lam),
    )
    x0 = jnp.zeros(n)
    return prob, x0, side, ker, y_np, lam


def run_all():
    print("=== Single-Solve XLA JIT vs. Eager (jit=False) vs. CVXPY ===")
    for n in [64, 256, 512]:
        solve_mod._COMPILED_SOLVE_CACHE.clear()
        prob, x0, Q_np, c_np = make_box_qp(n)
        solver = px.solvers.PANOC(opt_tol=1e-6, max_iters=2000)

        t0 = time.perf_counter()
        sol_jit = px.solve(prob, x0, solver=solver, jit=True)
        jax.block_until_ready(sol_jit.x)
        cold_jit_ms = (time.perf_counter() - t0) * 1000.0
        steady_jit_ms = _min_ms(
            lambda: px.solve(prob, x0, solver=solver, jit=True).x,
            warmups=1,
            repeats=7,
        )
        comp_ms = max(0.0, cold_jit_ms - steady_jit_ms)

        sol_nojit = px.solve(prob, x0, solver=solver, jit=False)
        jax.block_until_ready(sol_nojit.x)
        eager_ms = _min_ms(
            lambda: px.solve(prob, x0, solver=solver, jit=False).x,
            warmups=1,
            repeats=3,
        )

        def run_cvx_fresh(solver_name):
            xv = cp.Variable(n)
            obj = cp.Minimize(0.5 * cp.quad_form(xv, cp.psd_wrap(Q_np)) + c_np @ xv)
            p = cp.Problem(obj, [xv >= -1.0, xv <= 1.0])
            p.solve(solver=solver_name)
            return p.value

        xv = cp.Variable(n)
        p_clar = cp.Problem(
            cp.Minimize(0.5 * cp.quad_form(xv, cp.psd_wrap(Q_np)) + c_np @ xv),
            [xv >= -1.0, xv <= 1.0],
        )
        p_clar.solve(solver=cp.CLARABEL)
        cvx_clar_fresh_ms = _cvxpy_min_ms(lambda: run_cvx_fresh(cp.CLARABEL), warmups=1, repeats=5)
        cvx_clar_cached_ms = _cvxpy_min_ms(lambda: p_clar.solve(solver=cp.CLARABEL), warmups=1, repeats=5)
        cvx_scs_cached_ms = _cvxpy_min_ms(lambda: p_clar.solve(solver=cp.SCS), warmups=1, repeats=5)

        print(
            f"BoxQP n={n} | status={sol_jit.status} g_jit={int(sol_jit.stats.grad_evals)} g_eager={int(sol_nojit.stats.grad_evals)} | "
            f"jit=False: {eager_ms:.2f} ms | jit=True: {steady_jit_ms:.3f} ms "
            f"(comp {comp_ms:.0f} ms, jit_boost={eager_ms/steady_jit_ms:.1f}x) | "
            f"CVXPY-Clar fresh/cached: {cvx_clar_fresh_ms:.2f}/{cvx_clar_cached_ms:.2f} ms ({cvx_clar_cached_ms/steady_jit_ms:.1f}x) | "
            f"CVXPY-SCS cached: {cvx_scs_cached_ms:.2f} ms | obj_diff={abs(float(sol_jit.stats.obj_val) - p_clar.value):.2e}",
            flush=True,
        )

    for (n, m) in [(64, 16), (128, 32)]:
        solve_mod._COMPILED_SOLVE_CACHE.clear()
        prob, x0, Q_np, c_np, A_np, b_np = make_eq_box_qp(n, m)
        solver = px.solvers.IALM(opt_tol=1e-5, feas_tol=1e-5)

        t0 = time.perf_counter()
        sol_jit = px.solve(prob, x0, solver=solver, jit=True)
        jax.block_until_ready(sol_jit.x)
        cold_ms = (time.perf_counter() - t0) * 1000.0
        steady_jit_ms = _min_ms(
            lambda: px.solve(prob, x0, solver=solver, jit=True).x,
            warmups=1,
            repeats=5,
        )
        comp_ms = max(0.0, cold_ms - steady_jit_ms)

        t0 = time.perf_counter()
        sol_nojit = px.solve(prob, x0, solver=solver, jit=False)
        jax.block_until_ready(sol_nojit.x)
        eager_ms = (time.perf_counter() - t0) * 1000.0

        xv = cp.Variable(n)
        p_clar = cp.Problem(
            cp.Minimize(0.5 * cp.quad_form(xv, cp.psd_wrap(Q_np)) + c_np @ xv),
            [A_np @ xv == b_np, xv >= -1.0, xv <= 1.0],
        )
        p_clar.solve(solver=cp.CLARABEL)
        cvx_clar_cached_ms = _cvxpy_min_ms(lambda: p_clar.solve(solver=cp.CLARABEL), warmups=1, repeats=5)
        cvx_scs_cached_ms = _cvxpy_min_ms(lambda: p_clar.solve(solver=cp.SCS), warmups=1, repeats=5)

        print(
            f"EqBoxQP (n={n},m={m}) | status={sol_jit.status} g_jit={int(sol_jit.stats.grad_evals)} | "
            f"jit=False: {eager_ms:.2f} ms | jit=True: {steady_jit_ms:.2f} ms "
            f"(comp {comp_ms:.0f} ms, jit_boost={eager_ms/steady_jit_ms:.1f}x) | "
            f"CVXPY-Clar(c): {cvx_clar_cached_ms:.2f} ms | CVXPY-SCS(c): {cvx_scs_cached_ms:.2f} ms | "
            f"obj_diff={abs(float(sol_jit.stats.obj_val) - p_clar.value):.2e}",
            flush=True,
        )

    for side in [32, 64]:
        solve_mod._COMPILED_SOLVE_CACHE.clear()
        prob, x0, s, ker, y_np, lam = make_fft_deblur(side)
        solver = px.solvers.PANOC(opt_tol=1e-5, max_iters=2000)

        t0 = time.perf_counter()
        sol_jit = px.solve(prob, x0, solver=solver, jit=True)
        jax.block_until_ready(sol_jit.x)
        cold_ms = (time.perf_counter() - t0) * 1000.0
        steady_jit_ms = _min_ms(
            lambda: px.solve(prob, x0, solver=solver, jit=True).x,
            warmups=1,
            repeats=5,
        )
        comp_ms = max(0.0, cold_ms - steady_jit_ms)

        t0 = time.perf_counter()
        sol_nojit = px.solve(prob, x0, solver=solver, jit=False)
        jax.block_until_ready(sol_nojit.x)
        eager_ms = _min_ms(
            lambda: px.solve(prob, x0, solver=solver, jit=False).x,
            warmups=0,
            repeats=2,
        )

        if side <= 32:
            n = side * side
            fft_k = np.fft.fft2(ker)
            eye = np.eye(n).reshape(n, side, side)
            A_mat = np.fft.ifft2(fft_k * np.fft.fft2(eye, axes=(1, 2)), axes=(1, 2)).real.reshape(n, n).T
            xv = cp.Variable(n)
            p_clar = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(A_mat @ xv - y_np) + lam * cp.norm1(xv)))
            p_clar.solve(solver=cp.CLARABEL)
            cvx_clar_ms = _cvxpy_min_ms(lambda: p_clar.solve(solver=cp.CLARABEL), warmups=0, repeats=1)
            cvx_str = f"CVXPY-Clar(c): {cvx_clar_ms:.2f} ms ({cvx_clar_ms/steady_jit_ms:.1f}x)"
        else:
            cvx_str = "CVXPY-Clar: infeasible dense matrix (n=4096)"

        print(
            f"FFTDeblur n={side*side} | status={sol_jit.status} g_jit={int(sol_jit.stats.grad_evals)} | "
            f"jit=False: {eager_ms:.2f} ms | jit=True: {steady_jit_ms:.2f} ms "
            f"(comp {comp_ms:.0f} ms, jit_boost={eager_ms/steady_jit_ms:.1f}x) | {cvx_str}"
        )

    print("\n=== Closed-Loop 50-Step Parameterized Sweep (lax.scan vs. Python eager vs. CVXPY DPP) ===")
    n = 64
    K = 50
    _, _, Q_np, c_np = make_box_qp(n, seed=7)
    rng = np.random.default_rng(7)
    C_traj_np = c_np[None, :] + 0.1 * rng.standard_normal((K, n))
    Q_j = jnp.asarray(Q_np)
    C_traj_j = jnp.asarray(C_traj_np)
    solver = px.solvers.PANOC(opt_tol=1e-6, max_iters=500)

    def scan_solve(c_seq):
        def step(x_prev, c_k):
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
    objs_scan = jit_scan(C_traj_j)
    jax.block_until_ready(objs_scan)
    scan_cold_ms = (time.perf_counter() - t0) * 1000.0
    scan_ms = _min_ms(lambda: jit_scan(C_traj_j), warmups=1, repeats=7)
    scan_comp_ms = max(0.0, scan_cold_ms - scan_ms)

    def py_loop_eager():
        x_cur = jnp.zeros(n)
        objs = []
        for k in range(K):
            c_k = C_traj_j[k]
            p_k = px.CompositeProblem(
                smooth=lambda x, ck=c_k: 0.5 * jnp.dot(x, Q_j @ x) + jnp.dot(ck, x),
                proximal=px.prox.Box(lower=-1.0, upper=1.0),
            )
            s_k = px.solve(p_k, x_cur, solver=solver, jit=False)
            x_cur = s_k.x
            objs.append(s_k.stats.obj_val)
        return x_cur

    t0 = time.perf_counter()
    jax.block_until_ready(py_loop_eager())
    eager_loop_ms = (time.perf_counter() - t0) * 1000.0

    # CVXPY DPP parameter sweep (reuses canonicalization cache across all 50 steps)
    xv = cp.Variable(n)
    c_param = cp.Parameter(n)
    p_dpp = cp.Problem(
        cp.Minimize(0.5 * cp.quad_form(xv, cp.psd_wrap(Q_np)) + c_param @ xv),
        [xv >= -1.0, xv <= 1.0],
    )
    c_param.value = C_traj_np[0]
    p_dpp.solve(solver=cp.CLARABEL)

    def run_dpp(solver_name):
        for k in range(K):
            c_param.value = C_traj_np[k]
            p_dpp.solve(solver=solver_name, warm_start=True)

    dpp_clar_ms = _cvxpy_min_ms(lambda: run_dpp(cp.CLARABEL), warmups=1, repeats=5)
    dpp_scs_ms = _cvxpy_min_ms(lambda: run_dpp(cp.SCS), warmups=1, repeats=5)

    print(
        f"50-step BoxQP Sweep (n={n}): px lax.scan jit=True={scan_ms:.2f} ms "
        f"(comp {scan_comp_ms:.0f} ms) | px jit=False={eager_loop_ms:.2f} ms ({eager_loop_ms/scan_ms:.1f}x) | "
        f"CVXPY-Clar(DPP)={dpp_clar_ms:.2f} ms ({dpp_clar_ms/scan_ms:.1f}x) | "
        f"CVXPY-SCS(DPP)={dpp_scs_ms:.2f} ms ({dpp_scs_ms/scan_ms:.1f}x)",
        flush=True,
    )

    print("\n=== Differentiable Implicit Layer VJP (full jax.jit vs. eager outer AD) ===")
    n_ad, m_ad = 50, 16
    _, _, Q_ad_np, c_ad_np, A_ad_np, b_ad_np = make_eq_box_qp(n_ad, m_ad, seed=11)
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

    def qp_loss(c_p):
        x_star = qp_layer(c_p, A_ad, b_ad)
        return 0.5 * jnp.sum((x_star - 0.25) ** 2)

    vag_eager = jax.value_and_grad(qp_loss)
    vag_jit = jax.jit(vag_eager)

    t0 = time.perf_counter()
    v_j, g_j = vag_jit(c_ad)
    jax.block_until_ready((v_j, g_j))
    cold_ad_ms = (time.perf_counter() - t0) * 1000.0
    jit_ad_ms = _min_ms(lambda: vag_jit(c_ad), warmups=1, repeats=7)
    comp_ad_ms = max(0.0, cold_ad_ms - jit_ad_ms)

    # Eager outer value_and_grad runs forward + Krylov KKT adjoint without outer XLA fusion
    v_e, g_e = vag_eager(c_ad)
    jax.block_until_ready((v_e, g_e))
    eager_ad_ms = _min_ms(lambda: vag_eager(c_ad), warmups=1, repeats=5)
    rel_err = float(jnp.linalg.norm(g_j - g_e) / jnp.maximum(jnp.linalg.norm(g_j), 1e-12))

    print(
        f"QPLayer Fwd+VJP (n={n_ad},m={m_ad}): outer_eager: {eager_ad_ms:.2f} ms | "
        f"full_jit: {jit_ad_ms:.2f} ms (comp {comp_ad_ms:.0f} ms, boost={eager_ad_ms/jit_ad_ms:.1f}x) | "
        f"grad_rel_err={rel_err:.2e}",
        flush=True,
    )


if __name__ == "__main__":
    run_all()
