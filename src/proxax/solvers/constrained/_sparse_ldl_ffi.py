"""XLA FFI sparse quasi-definite LDL^T (`QDLDL`-style) KKT solver."""

# ruff: noqa: E501

from __future__ import annotations

import ctypes
import hashlib
import pathlib
import subprocess

import jax
import jax.numpy as jnp

_CPP_SOURCE = r"""
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {

// Minimal-degree / RCM-free natural or greedy-degree ordering is omitted for
// simplicity when n+m <= 2000; we run exact QDLDL symbolic + numeric LDL^T
// on the upper-triangular CSC matrix K (size N = n + m).
struct QdldlWorkspace {
  std::vector<int32_t> etree;
  std::vector<int32_t> Lnz;
  std::vector<int32_t> work;
  std::vector<int32_t> Lp;
  std::vector<int32_t> Li;
  std::vector<double> Lx;
  std::vector<double> D;
  std::vector<double> Dinv;
  std::vector<int32_t> yIdx;
  std::vector<double> yVals;
  std::vector<uint8_t> elim;
  std::vector<int32_t> LNext;
  std::vector<double> res;
  std::vector<double> dx;
  std::vector<double> Kx;
  std::vector<double> pb;
  std::vector<double> px;
  std::vector<int32_t> Ap;
  std::vector<int32_t> Ai;
  std::vector<double> Ax;
};

struct SparseKKTSolver {
  static void ComputeSymbolic(
      int32_t N,
      const int32_t* Ap,
      const int32_t* Ai,
      std::vector<int32_t>& etree,
      std::vector<int32_t>& Lp,
      std::vector<int32_t>& Li,
      std::vector<int32_t>& Lnz,
      std::vector<int32_t>& work) {
    etree.assign(N, -1);
    Lnz.assign(N, 0);
    work.assign(N, 0);

    for (int32_t j = 0; j < N; ++j) {
      work[j] = j;
      Lnz[j] = 0;
      for (int32_t p = Ap[j]; p < Ap[j + 1]; ++p) {
        int32_t i = Ai[p];
        if (i >= j) continue;
        while (work[i] != j) {
          if (etree[i] == -1) etree[i] = j;
          Lnz[i]++;
          work[i] = j;
          i = etree[i];
        }
      }
    }

    Lp.assign(N + 1, 0);
    for (int32_t j = 0; j < N; ++j) Lp[j + 1] = Lp[j] + Lnz[j];
    int32_t nnzL = Lp[N];
    Li.resize(nnzL);

    std::vector<int32_t> yIdx(N);
    std::vector<uint8_t> elim(N, 0);
    std::vector<int32_t> LNext(N);
    for (int32_t j = 0; j < N; ++j) LNext[j] = Lp[j];

    for (int32_t k = 0; k < N; ++k) {
      int32_t top = N;
      elim[k] = 1;
      for (int32_t p = Ap[k]; p < Ap[k + 1]; ++p) {
        int32_t i = Ai[p];
        if (i < k) {
          int32_t len = 0;
          int32_t c = i;
          while (c != -1 && c < k && !elim[c]) {
            yIdx[len++] = c;
            elim[c] = 1;
            c = etree[c];
          }
          while (len > 0) yIdx[--top] = yIdx[--len];
        }
      }
      for (int32_t idx = top; idx < N; ++idx) {
        int32_t j = yIdx[idx];
        elim[j] = 0;
        Li[LNext[j]++] = k;
      }
      elim[k] = 0;
    }
  }

  static void SolveNumericWithSymbolic(
      int32_t N,
      const int32_t* Ap,
      const int32_t* Ai,
      const double* Ax,
      const int32_t* etree,
      const int32_t* Lp,
      const int32_t* Li,
      const double* b,
      double* x,
      QdldlWorkspace& ws,
      int32_t refine_steps = 0) {
    int32_t nnzL = Lp[N];
    if (static_cast<int32_t>(ws.Lx.size()) < nnzL) ws.Lx.resize(nnzL);
    if (static_cast<int32_t>(ws.Dinv.size()) < N) {
      ws.Dinv.resize(N);
      ws.yIdx.resize(N);
      ws.yVals.assign(N, 0.0);
      ws.elim.assign(N, 0);
      ws.LNext.resize(N);
    }

    double* Lx = ws.Lx.data();
    double* Dinv = ws.Dinv.data();
    int32_t* yIdx = ws.yIdx.data();
    double* yVals = ws.yVals.data();
    uint8_t* elim = ws.elim.data();
    int32_t* LNext = ws.LNext.data();

    for (int32_t j = 0; j < N; ++j) LNext[j] = Lp[j];

    for (int32_t k = 0; k < N; ++k) {
      yVals[k] = 0.0;
      int32_t top = N;
      elim[k] = 1;

      for (int32_t p = Ap[k]; p < Ap[k + 1]; ++p) {
        int32_t i = Ai[p];
        if (i == k) {
          yVals[k] = Ax[p];
        } else if (i < k) {
          int32_t len = 0;
          int32_t c = i;
          while (c != -1 && c < k && !elim[c]) {
            yIdx[len++] = c;
            elim[c] = 1;
            c = etree[c];
          }
          while (len > 0) {
            yIdx[--top] = yIdx[--len];
          }
          yVals[i] += Ax[p];
        }
      }

      double d_k = yVals[k];
      yVals[k] = 0.0;
      for (int32_t idx = top; idx < N; ++idx) {
        int32_t j = yIdx[idx];
        double yj = yVals[j];
        yVals[j] = 0.0;
        elim[j] = 0;

        double l_kj = yj * Dinv[j];
        d_k -= l_kj * yj;

        int32_t pos = LNext[j]++;
        Lx[pos] = l_kj;

        for (int32_t p = Lp[j]; p < pos; ++p) {
          yVals[Li[p]] -= Lx[p] * yj;
        }
      }
      elim[k] = 0;

      if (std::abs(d_k) < 1e-14) {
        d_k = (d_k >= 0.0) ? 1e-14 : -1e-14;
      }
      Dinv[k] = 1.0 / d_k;
    }

    auto apply_solve = [&](const double* rhs_vec, double* sol_vec) {
      for (int32_t i = 0; i < N; ++i) sol_vec[i] = rhs_vec[i];
      for (int32_t j = 0; j < N; ++j) {
        double xj = sol_vec[j];
        for (int32_t p = Lp[j]; p < Lp[j + 1]; ++p) {
          sol_vec[Li[p]] -= Lx[p] * xj;
        }
      }
      for (int32_t j = 0; j < N; ++j) sol_vec[j] *= Dinv[j];
      for (int32_t j = N - 1; j >= 0; --j) {
        double xj = sol_vec[j];
        for (int32_t p = Lp[j]; p < Lp[j + 1]; ++p) {
          xj -= Lx[p] * sol_vec[Li[p]];
        }
        sol_vec[j] = xj;
      }
    };

    apply_solve(b, x);

    if (refine_steps > 0) {
      if (static_cast<int32_t>(ws.res.size()) < N) {
        ws.res.resize(N);
        ws.dx.resize(N);
      }
      double* res = ws.res.data();
      double* dx = ws.dx.data();
      for (int32_t step = 0; step < refine_steps; ++step) {
        for (int32_t i = 0; i < N; ++i) res[i] = b[i];
        for (int32_t j = 0; j < N; ++j) {
          double xj = x[j];
          for (int32_t p = Ap[j]; p < Ap[j + 1]; ++p) {
            int32_t i = Ai[p];
            double v = Ax[p];
            res[i] -= v * xj;
            if (i < j) res[j] -= v * x[i];
          }
        }
        apply_solve(res, dx);
        for (int32_t i = 0; i < N; ++i) x[i] += dx[i];
      }
    }
  }

  static void SolveUpperCSC(
      int32_t N,
      const int32_t* Ap,
      const int32_t* Ai,
      const double* Ax,
      const double* b,
      double* x,
      QdldlWorkspace& ws,
      int32_t refine_steps = 0) {
    ComputeSymbolic(N, Ap, Ai, ws.etree, ws.Lp, ws.Li, ws.Lnz, ws.work);
    SolveNumericWithSymbolic(
        N, Ap, Ai, Ax, ws.etree.data(), ws.Lp.data(), ws.Li.data(),
        b, x, ws, refine_steps);
  }
};

struct CachedProblemData {
  int32_t n = 0;
  int32_t m = 0;
  uint64_t hash = 0;
  std::vector<int32_t> order;
  std::vector<int32_t> inv_perm;
  std::vector<std::vector<std::pair<int32_t, double>>> pA_rows;
  std::vector<uint8_t> row_is_sparse;

  // Static union CSC of K_eff (0..n-1) + precomputed symbolic factorization
  std::vector<int32_t> Kp;
  std::vector<int32_t> Ki;
  std::vector<double> Kx_base;
  std::vector<int32_t> diag_pos;
  std::vector<std::vector<std::pair<int32_t, double>>> sparse_scatter;
  std::vector<int32_t> etree;
  std::vector<int32_t> Lp;
  std::vector<int32_t> Li;
};

static inline uint64_t SpotHashBuffers(
    int32_t n, int32_t m, const double* hf, const double* a) {
  uint64_t h = 14695981039346656037ULL;
  auto mix = [&](uint64_t x) {
    h ^= x;
    h *= 1099511628211ULL;
  };
  mix(static_cast<uint64_t>(n));
  mix(static_cast<uint64_t>(m));
  const uint64_t* hf_u = reinterpret_cast<const uint64_t*>(hf);
  for (int32_t i = 0; i < n; ++i) {
    mix(hf_u[static_cast<size_t>(i) * n + i]);
  }
  const uint64_t* a_u = reinterpret_cast<const uint64_t*>(a);
  if (n > 0) {
    for (int32_t k = 0; k < m; ++k) {
      size_t base = static_cast<size_t>(k) * n;
      mix(a_u[base]);
      mix(a_u[base + n - 1]);
    }
  }
  return h;
}

static inline uint64_t HashProblemBuffers(
    int32_t n, int32_t m, const double* hf, const double* a) {
  uint64_t h = 14695981039346656037ULL;
  auto mix = [&](uint64_t x) {
    h ^= x;
    h *= 1099511628211ULL;
  };
  mix(static_cast<uint64_t>(n));
  mix(static_cast<uint64_t>(m));
  const uint64_t* hf_u = reinterpret_cast<const uint64_t*>(hf);
  size_t nn = static_cast<size_t>(n) * static_cast<size_t>(n);
  for (size_t i = 0; i < nn; ++i) mix(hf_u[i]);
  const uint64_t* a_u = reinterpret_cast<const uint64_t*>(a);
  size_t mn = static_cast<size_t>(m) * static_cast<size_t>(n);
  for (size_t i = 0; i < mn; ++i) mix(a_u[i]);
  return h;
}

static std::shared_ptr<CachedProblemData> BuildCachedFromRawA(
    int32_t n,
    int32_t m,
    const double* hf,
    std::vector<std::vector<std::pair<int32_t, double>>>& raw_A,
    uint64_t h) {
  constexpr int32_t kSparseRowLimit = 32;
  auto data = std::make_shared<CachedProblemData>();
  data->n = n;
  data->m = m;
  data->hash = h;

  data->row_is_sparse.resize(m, 0);
  for (int32_t k = 0; k < m; ++k) {
    data->row_is_sparse[k] =
        (static_cast<int32_t>(raw_A[k].size()) <= kSparseRowLimit) ? 1 : 0;
  }

  // Build RCM ordering on 0..n-1 using union sparsity of H_f and sparse A rows
  data->order.resize(n);
  data->inv_perm.resize(n);
  if (n > 64) {
    std::vector<std::vector<int32_t>> adj(n);
    for (int32_t j = 0; j < n; ++j) {
      for (int32_t i = 0; i < j; ++i) {
        if (hf[static_cast<size_t>(i) * n + j] != 0.0) {
          adj[i].push_back(j);
          adj[j].push_back(i);
        }
      }
    }
    for (int32_t k = 0; k < m; ++k) {
      if (!data->row_is_sparse[k]) continue;
      const auto& row = raw_A[k];
      int32_t rlen = static_cast<int32_t>(row.size());
      for (int32_t p = 0; p < rlen; ++p) {
        int32_t u = row[p].first;
        for (int32_t q = p + 1; q < rlen; ++q) {
          int32_t v = row[q].first;
          adj[u].push_back(v);
          adj[v].push_back(u);
        }
      }
    }
    for (int32_t i = 0; i < n; ++i) {
      std::sort(adj[i].begin(), adj[i].end());
      adj[i].erase(std::unique(adj[i].begin(), adj[i].end()), adj[i].end());
    }
    std::vector<int32_t> deg(n);
    for (int32_t i = 0; i < n; ++i) deg[i] = static_cast<int32_t>(adj[i].size());

    std::vector<int32_t> verts(n);
    for (int32_t i = 0; i < n; ++i) verts[i] = i;
    std::sort(verts.begin(), verts.end(), [&](int32_t u, int32_t v) {
      return deg[u] < deg[v];
    });

    // Exact Greedy Minimum Degree ordering on 0..n-1
    std::vector<int32_t> order;
    order.reserve(n);
    std::vector<uint8_t> elim(n, 0);
    std::vector<uint8_t> mark(n, 0);
    std::vector<int32_t> nbrs;
    nbrs.reserve(64);

    for (int32_t step = 0; step < n; ++step) {
      int32_t best = -1;
      int32_t best_deg = n + 1;
      for (int32_t i = 0; i < n; ++i) {
        if (!elim[i] && deg[i] < best_deg) {
          best_deg = deg[i];
          best = i;
          if (best_deg <= 1) break;
        }
      }
      elim[best] = 1;
      order.push_back(best);

      nbrs.clear();
      for (int32_t u : adj[best]) {
        if (!elim[u]) {
          nbrs.push_back(u);
          mark[u] = 1;
        }
      }
      for (int32_t u : nbrs) {
        auto& au = adj[u];
        // Remove eliminated nodes from adj[u] and add missing nbrs
        size_t w_idx = 0;
        for (size_t idx = 0; idx < au.size(); ++idx) {
          int32_t v = au[idx];
          if (!elim[v]) au[w_idx++] = v;
        }
        au.resize(w_idx);
        for (int32_t v : au) mark[v] = 2;
        for (int32_t v : nbrs) {
          if (v != u && mark[v] != 2) {
            au.push_back(v);
          }
        }
        for (int32_t v : au) {
          if (mark[v] == 2) mark[v] = 1;
        }
        deg[u] = static_cast<int32_t>(au.size());
      }
      for (int32_t u : nbrs) mark[u] = 0;
    }
    data->order = std::move(order);
    for (int32_t new_i = 0; new_i < n; ++new_i) {
      data->inv_perm[data->order[new_i]] = new_i;
    }
  } else {
    for (int32_t i = 0; i < n; ++i) {
      data->order[i] = i;
      data->inv_perm[i] = i;
    }
  }

  // Store permuted A rows sorted by permuted column index
  data->pA_rows.resize(m);
  for (int32_t k = 0; k < m; ++k) {
    data->pA_rows[k].reserve(raw_A[k].size());
    for (const auto& kv : raw_A[k]) {
      data->pA_rows[k].emplace_back(data->inv_perm[kv.first], kv.second);
    }
    std::sort(data->pA_rows[k].begin(), data->pA_rows[k].end(),
              [](const auto& a_p, const auto& b_p) { return a_p.first < b_p.first; });
  }

  // Build static union sparsity pattern (Kp, Ki, Kx_base) of P H_f P^T + sum sparse pA_rows
  std::vector<std::vector<std::pair<int32_t, double>>> union_cols(n);
  for (int32_t j = 0; j < n; ++j) {
    int32_t pj = data->inv_perm[j];
    for (int32_t i = 0; i < j; ++i) {
      double v = hf[static_cast<size_t>(i) * n + j];
      if (v != 0.0) {
        int32_t pi = data->inv_perm[i];
        if (pi < pj) union_cols[pj].emplace_back(pi, v);
        else union_cols[pi].emplace_back(pj, v);
      }
    }
    union_cols[pj].emplace_back(pj, hf[static_cast<size_t>(j) * n + j]);
  }
  for (int32_t k = 0; k < m; ++k) {
    if (!data->row_is_sparse[k]) continue;
    const auto& row = data->pA_rows[k];
    int32_t rlen = static_cast<int32_t>(row.size());
    for (int32_t p = 0; p < rlen; ++p) {
      int32_t pi = row[p].first;
      for (int32_t q = p; q < rlen; ++q) {
        union_cols[row[q].first].emplace_back(pi, 0.0);
      }
    }
  }

  data->Kp.assign(n + 1, 0);
  data->diag_pos.resize(n);
  std::vector<double> col_acc(n, 0.0);
  std::vector<uint8_t> col_mark(n, 0);
  std::vector<int32_t> col_rows;
  col_rows.reserve(64);

  for (int32_t pj = 0; pj < n; ++pj) {
    col_rows.clear();
    for (const auto& kv : union_cols[pj]) {
      int32_t pi = kv.first;
      if (!col_mark[pi]) {
        col_mark[pi] = 1;
        col_rows.push_back(pi);
      }
      col_acc[pi] += kv.second;
    }
    std::sort(col_rows.begin(), col_rows.end());
    for (int32_t pi : col_rows) {
      int32_t pos = static_cast<int32_t>(data->Ki.size());
      if (pi == pj) data->diag_pos[pj] = pos;
      data->Ki.push_back(pi);
      data->Kx_base.push_back(col_acc[pi]);
      col_acc[pi] = 0.0;
      col_mark[pi] = 0;
    }
    data->Kp[pj + 1] = static_cast<int32_t>(data->Ki.size());
  }

  // Precompute scatter indices into Kx for every sparse A row
  data->sparse_scatter.resize(m);
  for (int32_t k = 0; k < m; ++k) {
    if (!data->row_is_sparse[k]) continue;
    const auto& row = data->pA_rows[k];
    int32_t rlen = static_cast<int32_t>(row.size());
    data->sparse_scatter[k].reserve((rlen * (rlen + 1)) / 2);
    for (int32_t p = 0; p < rlen; ++p) {
      int32_t pi = row[p].first;
      double vi = row[p].second;
      for (int32_t q = p; q < rlen; ++q) {
        int32_t pj = row[q].first;
        double val = vi * row[q].second;
        const int32_t* begin = data->Ki.data() + data->Kp[pj];
        const int32_t* end = data->Ki.data() + data->Kp[pj + 1];
        const int32_t* it = std::lower_bound(begin, end, pi);
        int32_t kx_pos = static_cast<int32_t>(it - data->Ki.data());
        data->sparse_scatter[k].emplace_back(kx_pos, val);
      }
    }
  }

  // Precompute symbolic factorization of (Kp, Ki)
  std::vector<int32_t> tmp_Lnz, tmp_work;
  SparseKKTSolver::ComputeSymbolic(
      n, data->Kp.data(), data->Ki.data(),
      data->etree, data->Lp, data->Li, tmp_Lnz, tmp_work);

  return data;
}

static std::mutex g_cache_mu;
static std::unordered_map<uint64_t, std::shared_ptr<const CachedProblemData>> g_cache;

static std::shared_ptr<const CachedProblemData> GetOrBuildCachedProblem(
    int32_t n, int32_t m, const double* hf, const double* a) {
  uint64_t spot = SpotHashBuffers(n, m, hf, a);

  thread_local int32_t tl_n = -1, tl_m = -1;
  thread_local const double* tl_hf = nullptr;
  thread_local const double* tl_a = nullptr;
  thread_local uint64_t tl_spot = 0;
  thread_local std::shared_ptr<const CachedProblemData> tl_data;

  if (tl_data && tl_n == n && tl_m == m && tl_hf == hf && tl_a == a &&
      tl_spot == spot) {
    return tl_data;
  }

  uint64_t h = HashProblemBuffers(n, m, hf, a);
  {
    std::lock_guard<std::mutex> lock(g_cache_mu);
    auto it = g_cache.find(h);
    if (it != g_cache.end() && it->second->n == n && it->second->m == m) {
      tl_n = n; tl_m = m; tl_hf = hf; tl_a = a; tl_spot = spot; tl_data = it->second;
      return it->second;
    }
  }

  std::vector<std::vector<std::pair<int32_t, double>>> raw_A(m);
  for (int32_t k = 0; k < m; ++k) {
    const double* row_k = a + static_cast<size_t>(k) * n;
    for (int32_t i = 0; i < n; ++i) {
      double v = row_k[i];
      if (v != 0.0) raw_A[k].emplace_back(i, v);
    }
  }
  auto data = BuildCachedFromRawA(n, m, hf, raw_A, h);
  {
    std::lock_guard<std::mutex> lock(g_cache_mu);
    if (g_cache.size() >= 64) g_cache.clear();
    g_cache[h] = data;
  }
  tl_n = n; tl_m = m; tl_hf = hf; tl_a = a; tl_spot = spot; tl_data = data;
  return data;
}

static std::shared_ptr<const CachedProblemData> GetOrBuildCachedProblemBcoo(
    int32_t n, int32_t m, int32_t nnzA,
    const double* hf, const double* a_data, const int32_t* a_idx) {
  // Fast O(n + min(64, nnzA)) spot hash
  uint64_t spot = 14695981039346656037ULL;
  auto mix = [&](uint64_t x) { spot ^= x; spot *= 1099511628211ULL; };
  mix(static_cast<uint64_t>(n));
  mix(static_cast<uint64_t>(m));
  mix(static_cast<uint64_t>(nnzA));
  const uint64_t* hf_u = reinterpret_cast<const uint64_t*>(hf);
  for (int32_t i = 0; i < n; ++i) mix(hf_u[static_cast<size_t>(i) * n + i]);
  const uint64_t* ad_u = reinterpret_cast<const uint64_t*>(a_data);
  if (nnzA > 0) {
    mix(ad_u[0]);
    mix(ad_u[nnzA - 1]);
  }

  thread_local int32_t tl_n = -1, tl_m = -1, tl_nnz = -1;
  thread_local const double* tl_hf = nullptr;
  thread_local const double* tl_ad = nullptr;
  thread_local const int32_t* tl_ai = nullptr;
  thread_local uint64_t tl_spot = 0;
  thread_local std::shared_ptr<const CachedProblemData> tl_data;

  if (tl_data && tl_n == n && tl_m == m && tl_nnz == nnzA &&
      tl_hf == hf && tl_ad == a_data && tl_ai == a_idx && tl_spot == spot) {
    return tl_data;
  }

  uint64_t h = 14695981039346656037ULL;
  auto mix_h = [&](uint64_t x) { h ^= x; h *= 1099511628211ULL; };
  mix_h(static_cast<uint64_t>(n));
  mix_h(static_cast<uint64_t>(m));
  mix_h(static_cast<uint64_t>(nnzA));
  size_t nn = static_cast<size_t>(n) * static_cast<size_t>(n);
  for (size_t i = 0; i < nn; ++i) mix_h(hf_u[i]);
  for (int32_t p = 0; p < nnzA; ++p) {
    mix_h(ad_u[p]);
    mix_h(static_cast<uint64_t>(a_idx[2 * p]));
    mix_h(static_cast<uint64_t>(a_idx[2 * p + 1]));
  }

  {
    std::lock_guard<std::mutex> lock(g_cache_mu);
    auto it = g_cache.find(h);
    if (it != g_cache.end() && it->second->n == n && it->second->m == m) {
      tl_n = n; tl_m = m; tl_nnz = nnzA;
      tl_hf = hf; tl_ad = a_data; tl_ai = a_idx; tl_spot = spot; tl_data = it->second;
      return it->second;
    }
  }

  std::vector<std::vector<std::pair<int32_t, double>>> raw_A(m);
  for (int32_t p = 0; p < nnzA; ++p) {
    int32_t r_k = a_idx[2 * p];
    int32_t c_i = a_idx[2 * p + 1];
    double v = a_data[p];
    if (v != 0.0 && r_k >= 0 && r_k < m && c_i >= 0 && c_i < n) {
      raw_A[r_k].emplace_back(c_i, v);
    }
  }
  // Sort & deduplicate each row in case BCOO is unsorted or has duplicate indices
  for (int32_t k = 0; k < m; ++k) {
    if (raw_A[k].size() > 1) {
      std::sort(raw_A[k].begin(), raw_A[k].end(),
                [](const auto& a_p, const auto& b_p) { return a_p.first < b_p.first; });
      size_t out_idx = 0;
      for (size_t idx = 0; idx < raw_A[k].size(); ++idx) {
        if (out_idx > 0 && raw_A[k][out_idx - 1].first == raw_A[k][idx].first) {
          raw_A[k][out_idx - 1].second += raw_A[k][idx].second;
        } else {
          raw_A[k][out_idx++] = raw_A[k][idx];
        }
      }
      raw_A[k].resize(out_idx);
    }
  }

  auto data = BuildCachedFromRawA(n, m, hf, raw_A, h);
  {
    std::lock_guard<std::mutex> lock(g_cache_mu);
    if (g_cache.size() >= 64) g_cache.clear();
    g_cache[h] = data;
  }
  tl_n = n; tl_m = m; tl_nnz = nnzA;
  tl_hf = hf; tl_ad = a_data; tl_ai = a_idx; tl_spot = spot; tl_data = data;
  return data;
}

static void SolveFromCached(
    const CachedProblemData* cached,
    const double* wc,
    const double* da,
    const double* r,
    double* out) {
  int32_t n = cached->n;
  int32_t m = cached->m;

  thread_local QdldlWorkspace ws;
  int32_t nnzK = static_cast<int32_t>(cached->Kx_base.size());
  if (static_cast<int32_t>(ws.Kx.size()) < nnzK) ws.Kx.resize(nnzK);
  std::memcpy(ws.Kx.data(), cached->Kx_base.data(),
              static_cast<size_t>(nnzK) * sizeof(double));
  double* Kx = ws.Kx.data();

  for (int32_t pj = 0; pj < n; ++pj) {
    Kx[cached->diag_pos[pj]] += da[cached->order[pj]];
  }

  bool has_dense = false;
  for (int32_t k = 0; k < m; ++k) {
    double w = wc[k];
    if (w <= 0.0) continue;
    if (cached->row_is_sparse[k]) {
      for (const auto& sc : cached->sparse_scatter[k]) {
        Kx[sc.first] += w * sc.second;
      }
    } else {
      has_dense = true;
    }
  }

  if (!has_dense) {
    if (static_cast<int32_t>(ws.pb.size()) < n) {
      ws.pb.resize(n);
      ws.px.resize(n);
    }
    for (int32_t i = 0; i < n; ++i) ws.pb[cached->inv_perm[i]] = r[i];
    SparseKKTSolver::SolveNumericWithSymbolic(
        n, cached->Kp.data(), cached->Ki.data(), Kx,
        cached->etree.data(), cached->Lp.data(), cached->Li.data(),
        ws.pb.data(), ws.px.data(), ws, 0);
    for (int32_t new_i = 0; new_i < n; ++new_i) {
      out[cached->order[new_i]] = ws.px[new_i];
    }
    return;
  }

  // Hybrid path: append active dense rows as 2x2 KKT border columns
  int32_t m_dense = 0;
  for (int32_t k = 0; k < m; ++k) {
    if (wc[k] > 0.0 && !cached->row_is_sparse[k]) ++m_dense;
  }
  int32_t N = n + m_dense;

  ws.Ap.resize(N + 1);
  std::memcpy(ws.Ap.data(), cached->Kp.data(),
              static_cast<size_t>(n + 1) * sizeof(int32_t));
  ws.Ai.resize(nnzK);
  ws.Ax.resize(nnzK);
  std::memcpy(ws.Ai.data(), cached->Ki.data(),
              static_cast<size_t>(nnzK) * sizeof(int32_t));
  std::memcpy(ws.Ax.data(), Kx, static_cast<size_t>(nnzK) * sizeof(double));

  uint64_t dense_hash = 14695981039346656037ULL;
  int32_t d_idx = 0;
  for (int32_t k = 0; k < m; ++k) {
    double w = wc[k];
    if (w <= 0.0 || cached->row_is_sparse[k]) continue;
    dense_hash ^= static_cast<uint64_t>(k + 1);
    dense_hash *= 1099511628211ULL;
    for (const auto& kv : cached->pA_rows[k]) {
      ws.Ai.push_back(kv.first);
      ws.Ax.push_back(kv.second);
    }
    ws.Ai.push_back(n + d_idx);
    ws.Ax.push_back(-1.0 / w);
    ws.Ap[n + d_idx + 1] = static_cast<int32_t>(ws.Ai.size());
    ++d_idx;
  }

  if (static_cast<int32_t>(ws.pb.size()) < N) {
    ws.pb.resize(N);
    ws.px.resize(N);
  }
  std::fill(ws.pb.begin(), ws.pb.begin() + N, 0.0);
  for (int32_t i = 0; i < n; ++i) ws.pb[cached->inv_perm[i]] = r[i];

  thread_local uint64_t tl_prob_hash = 0;
  thread_local uint64_t tl_dense_hash = 0;
  thread_local int32_t tl_dense_N = -1;
  thread_local std::vector<int32_t> tl_etree;
  thread_local std::vector<int32_t> tl_Lp;
  thread_local std::vector<int32_t> tl_Li;

  if (tl_dense_N != N || tl_prob_hash != cached->hash ||
      tl_dense_hash != dense_hash) {
    SparseKKTSolver::ComputeSymbolic(
        N, ws.Ap.data(), ws.Ai.data(),
        tl_etree, tl_Lp, tl_Li, ws.Lnz, ws.work);
    tl_prob_hash = cached->hash;
    tl_dense_hash = dense_hash;
    tl_dense_N = N;
  }

  SparseKKTSolver::SolveNumericWithSymbolic(
      N, ws.Ap.data(), ws.Ai.data(), ws.Ax.data(),
      tl_etree.data(), tl_Lp.data(), tl_Li.data(),
      ws.pb.data(), ws.px.data(), ws, 2);

  for (int32_t new_i = 0; new_i < n; ++new_i) {
    out[cached->order[new_i]] = ws.px[new_i];
  }
}

ffi::Error QdldlKktSolveImpl(
    ffi::Buffer<ffi::F64> H_f,
    ffi::Buffer<ffi::F64> A_mat,
    ffi::Buffer<ffi::F64> w_c,
    ffi::Buffer<ffi::F64> diag_add,
    ffi::Buffer<ffi::F64> rhs,
    ffi::ResultBuffer<ffi::F64> out_d) {
  auto dims_A = A_mat.dimensions();
  int32_t m = static_cast<int32_t>(dims_A[0]);
  int32_t n = static_cast<int32_t>(dims_A[1]);
  auto cached = GetOrBuildCachedProblem(n, m, H_f.typed_data(), A_mat.typed_data());
  SolveFromCached(cached.get(), w_c.typed_data(), diag_add.typed_data(),
                  rhs.typed_data(), out_d->typed_data());
  return ffi::Error::Success();
}

ffi::Error QdldlKktSolveBcooImpl(
    ffi::Buffer<ffi::F64> H_f,
    ffi::Buffer<ffi::F64> A_data,
    ffi::Buffer<ffi::S32> A_indices,
    ffi::Buffer<ffi::F64> w_c,
    ffi::Buffer<ffi::F64> diag_add,
    ffi::Buffer<ffi::F64> rhs,
    ffi::ResultBuffer<ffi::F64> out_d) {
  int32_t m = static_cast<int32_t>(w_c.dimensions()[0]);
  int32_t n = static_cast<int32_t>(rhs.dimensions()[0]);
  int32_t nnzA = static_cast<int32_t>(A_data.dimensions()[0]);
  auto cached = GetOrBuildCachedProblemBcoo(
      n, m, nnzA, H_f.typed_data(), A_data.typed_data(), A_indices.typed_data());
  SolveFromCached(cached.get(), w_c.typed_data(), diag_add.typed_data(),
                  rhs.typed_data(), out_d->typed_data());
  return ffi::Error::Success();
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ProxaxQdldlKktSolve, QdldlKktSolveImpl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ProxaxQdldlKktSolveBcoo, QdldlKktSolveBcooImpl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>());
"""

_REGISTERED = False


def _ensure_registered() -> tuple[str, str]:
  global _REGISTERED
  target_dense = "proxax_qdldl_kkt_solve"
  target_bcoo = "proxax_qdldl_kkt_solve_bcoo"
  if _REGISTERED:
    return target_dense, target_bcoo

  cache_dir = pathlib.Path.home() / ".cache" / "proxax" / "ffi"
  cache_dir.mkdir(parents=True, exist_ok=True)
  src_hash = hashlib.sha256(_CPP_SOURCE.encode("utf-8")).hexdigest()[:12]
  so_path = cache_dir / f"libproxax_qdldl_{src_hash}.so"
  if not so_path.exists():
    cpp_path = cache_dir / f"proxax_qdldl_{src_hash}.cc"
    cpp_path.write_text(_CPP_SOURCE, encoding="utf-8")
    inc_dir = jax.ffi.include_dir()
    subprocess.run(
        [
            "g++",
            "-O3",
            "-march=native",
            "-std=c++17",
            "-shared",
            "-fPIC",
            f"-I{inc_dir}",
            str(cpp_path),
            "-o",
            str(so_path),
        ],
        check=True,
    )

  lib = ctypes.CDLL(str(so_path))
  jax.ffi.register_ffi_target(
      target_dense,
      jax.ffi.pycapsule(lib.ProxaxQdldlKktSolve),
      platform="cpu",
  )
  jax.ffi.register_ffi_target(
      target_bcoo,
      jax.ffi.pycapsule(lib.ProxaxQdldlKktSolveBcoo),
      platform="cpu",
  )
  _REGISTERED = True
  return target_dense, target_bcoo


def qdldl_kkt_solve(
    h_f: jax.Array,
    a_mat: jax.Array,
    w_c: jax.Array,
    diag_add: jax.Array,
    rhs: jax.Array,
) -> jax.Array:
  """Calls the XLA FFI C++ QDLDL KKT solver for `(H_f, A, w_c, diag_add, rhs)`."""
  target_dense, _ = _ensure_registered()
  n = int(rhs.shape[0])
  return jax.ffi.ffi_call(
      target_dense,
      jax.ShapeDtypeStruct((n,), jnp.float64),
  )(h_f, a_mat, w_c, diag_add, rhs)


def qdldl_kkt_solve_bcoo(
    h_f: jax.Array,
    a_data: jax.Array,
    a_indices: jax.Array,
    w_c: jax.Array,
    diag_add: jax.Array,
    rhs: jax.Array,
) -> jax.Array:
  """Calls the XLA FFI C++ QDLDL KKT solver for sparse BCOO `A`."""
  _, target_bcoo = _ensure_registered()
  n = int(rhs.shape[0])
  return jax.ffi.ffi_call(
      target_bcoo,
      jax.ShapeDtypeStruct((n,), jnp.float64),
  )(h_f, a_data, a_indices.astype(jnp.int32), w_c, diag_add, rhs)
