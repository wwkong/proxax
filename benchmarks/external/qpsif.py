"""Minimal fixed-format QPSIF / MPS parser for quadratic programs."""

from __future__ import annotations

import dataclasses
import pathlib
import typing

import numpy as np
import scipy.sparse as sp


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class QPSIFProblem:
  """Dense representation of a QPSIF quadratic program.

  Represents:
    ``min 0.5 * x^T P x + q^T x + r``
    ``s.t. l_con <= A x <= u_con,  lb <= x <= ub``.
  """

  name: str
  P: np.ndarray
  q: np.ndarray
  r: float
  A: np.ndarray
  l_con: np.ndarray
  u_con: np.ndarray
  lb: np.ndarray
  ub: np.ndarray


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class QPSIFSparseProblem:
  """Sparse ``scipy.sparse.csc_matrix`` representation of a QPSIF QP."""

  name: str
  P: typing.Any
  q: np.ndarray
  r: float
  A: typing.Any
  l_con: np.ndarray
  u_con: np.ndarray
  lb: np.ndarray
  ub: np.ndarray


def ruiz_equilibrate(
    qp: QPSIFProblem,
    num_iters: int = 10,
) -> tuple[QPSIFProblem, np.ndarray, np.ndarray, float]:
  """Applies Modified Ruiz Equilibration to a ``QPSIFProblem``.

  Returns ``(scaled_qp, d_x, d_y, cost_scale)`` where ``x = d_x * x_scaled``
  and dual ``y = y_scaled / (cost_scale * d_y)``.
  """
  p_mat = qp.P.copy()
  q_vec = qp.q.copy()
  a_mat = qp.A.copy()
  n = p_mat.shape[0]
  m = a_mat.shape[0]
  d_x = np.ones(n, dtype=np.float64)
  d_y = np.ones(m, dtype=np.float64)
  c = 1.0
  for _ in range(num_iters):
    norm_p = np.max(np.abs(p_mat), axis=1) if n > 0 else np.zeros(0)
    norm_at = np.max(np.abs(a_mat), axis=0) if m > 0 else np.zeros(n)
    rx = np.maximum(norm_p, norm_at)
    sx = np.where(rx > 1e-12, 1.0 / np.sqrt(rx), 1.0)
    sx = np.clip(sx, 1e-4, 1e4)
    p_mat = (sx[:, None] * p_mat) * sx[None, :]
    q_vec = sx * q_vec
    d_x = d_x * sx
    if m > 0:
      a_mat = a_mat * sx[None, :]
      ry = np.max(np.abs(a_mat), axis=1)
      sy = np.where(ry > 1e-12, 1.0 / np.sqrt(ry), 1.0)
      sy = np.clip(sy, 1e-4, 1e4)
      a_mat = sy[:, None] * a_mat
      d_y = d_y * sy
    mean_p = float(np.mean(np.max(np.abs(p_mat), axis=1))) if n > 0 else 0.0
    norm_q = float(np.max(np.abs(q_vec))) if n > 0 else 0.0
    cost_scale = max(mean_p, norm_q, 1e-12)
    gamma = float(np.clip(1.0 / cost_scale, 1e-4, 1e4))
    p_mat = gamma * p_mat
    q_vec = gamma * q_vec
    c *= gamma

  scaled_qp = QPSIFProblem(
      name=qp.name,
      P=p_mat,
      q=q_vec,
      r=qp.r * c,
      A=a_mat,
      l_con=qp.l_con * d_y,
      u_con=qp.u_con * d_y,
      lb=qp.lb / d_x,
      ub=qp.ub / d_x,
  )
  return scaled_qp, d_x, d_y, c


def ruiz_equilibrate_sparse(
    qp: QPSIFSparseProblem,
    num_iters: int = 10,
) -> tuple[QPSIFSparseProblem, np.ndarray, np.ndarray, float]:
  """Applies Modified Ruiz Equilibration to a ``QPSIFSparseProblem``."""
  p_mat = qp.P.tocsc(copy=True)
  q_vec = qp.q.copy()
  a_mat = qp.A.tocsc(copy=True)
  n = p_mat.shape[0]
  m = a_mat.shape[0]
  d_x = np.ones(n, dtype=np.float64)
  d_y = np.ones(m, dtype=np.float64)
  c = 1.0
  for _ in range(num_iters):
    norm_p = (
        np.abs(p_mat).max(axis=1).toarray().ravel()
        if n > 0
        else np.zeros(0)
    )
    norm_at = (
        np.abs(a_mat).max(axis=0).toarray().ravel()
        if m > 0
        else np.zeros(n)
    )
    rx = np.maximum(norm_p, norm_at)
    sx = np.where(rx > 1e-12, 1.0 / np.sqrt(rx), 1.0)
    sx = np.clip(sx, 1e-4, 1e4)
    dx_op = sp.diags(sx)
    p_mat = (dx_op @ p_mat @ dx_op).tocsc()
    q_vec = sx * q_vec
    d_x = d_x * sx
    if m > 0:
      a_mat = (a_mat @ dx_op).tocsc()
      ry = np.abs(a_mat).max(axis=1).toarray().ravel()
      sy = np.where(ry > 1e-12, 1.0 / np.sqrt(ry), 1.0)
      sy = np.clip(sy, 1e-4, 1e4)
      a_mat = (sp.diags(sy) @ a_mat).tocsc()
      d_y = d_y * sy
    mean_p = (
        float(np.mean(np.abs(p_mat).max(axis=1).toarray().ravel()))
        if n > 0
        else 0.0
    )
    norm_q = float(np.max(np.abs(q_vec))) if n > 0 else 0.0
    cost_scale = max(mean_p, norm_q, 1e-12)
    gamma = float(np.clip(1.0 / cost_scale, 1e-4, 1e4))
    p_mat = gamma * p_mat
    q_vec = gamma * q_vec
    c *= gamma

  scaled_qp = QPSIFSparseProblem(
      name=qp.name,
      P=p_mat,
      q=q_vec,
      r=qp.r * c,
      A=a_mat,
      l_con=qp.l_con * d_y,
      u_con=qp.u_con * d_y,
      lb=qp.lb / d_x,
      ub=qp.ub / d_x,
  )
  return scaled_qp, d_x, d_y, c


def _parse_sif_core(
    path: pathlib.Path,
) -> tuple[
    str,
    int,
    int,
    np.ndarray,
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[int],
    list[int],
    list[float],
    list[int],
    list[int],
    list[float],
]:
  """Parses a ``.SIF`` file into sparse COO index/value lists and vectors."""
  name = path.stem
  section = ""
  obj_row: str | None = None
  row_types: dict[str, str] = {}
  con_names: list[str] = []
  con_idx: dict[str, int] = {}

  var_names: list[str] = []
  var_idx: dict[str, int] = {}
  col_entries: list[tuple[int, str, float]] = []
  rhs_vals: dict[str, float] = {}
  rhs_name: str | None = None
  range_vals: dict[str, float] = {}
  range_name: str | None = None
  bound_records: list[tuple[str, str, float | None]] = []
  bound_name: str | None = None
  quad_entries: list[tuple[int, int, float]] = []

  for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
    if not raw.strip() or raw.startswith("*"):
      continue
    if not raw.startswith(" "):
      parts = raw.split()
      header = parts[0].upper()
      if header == "NAME":
        if len(parts) > 1:
          name = parts[1]
        continue
      if header == "ENDATA":
        break
      section = header
      continue

    if section == "ROWS":
      parts = raw.split()
      rtype, rname = parts[0].upper(), parts[1]
      row_types[rname] = rtype
      if rtype == "N" and obj_row is None:
        obj_row = rname
      elif rtype in {"E", "L", "G"}:
        con_idx[rname] = len(con_names)
        con_names.append(rname)
      continue

    if section == "COLUMNS":
      tokens = raw.split()
      if len(tokens) >= 4 and tokens[2] == "'MARKER'":
        continue
      col_name = tokens[0]
      if col_name not in var_idx:
        var_idx[col_name] = len(var_names)
        var_names.append(col_name)
      j = var_idx[col_name]
      for k in range(1, len(tokens) - 1, 2):
        col_entries.append((j, tokens[k], float(tokens[k + 1])))
      continue

    if section == "RHS":
      tokens = raw.split()
      # In fixed MPS, RHS name can be blank (omitted), leaving even count.
      if len(tokens) % 2 == 1:
        r_tag = tokens[0]
        start = 1
      else:
        r_tag = ""
        start = 0
      if rhs_name is None:
        rhs_name = r_tag
      if r_tag != rhs_name:
        continue
      for k in range(start, len(tokens) - 1, 2):
        rhs_vals[tokens[k]] = float(tokens[k + 1])
      continue

    if section == "RANGES":
      tokens = raw.split()
      if len(tokens) % 2 == 1:
        r_tag = tokens[0]
        start = 1
      else:
        r_tag = ""
        start = 0
      if range_name is None:
        range_name = r_tag
      if r_tag != range_name:
        continue
      for k in range(start, len(tokens) - 1, 2):
        range_vals[tokens[k]] = float(tokens[k + 1])
      continue

    if section == "BOUNDS":
      tokens = raw.split()
      btype = tokens[0].upper()
      if btype in {"FR", "MI", "PL", "BV"}:
        # Format: BTYPE [BND_NAME] VAR_NAME
        b_tag = tokens[1] if len(tokens) >= 3 else ""
        vname = tokens[2] if len(tokens) >= 3 else tokens[1]
        val = None
      else:
        # Format: BTYPE [BND_NAME] VAR_NAME VAL
        b_tag = tokens[1] if len(tokens) >= 4 else ""
        vname = tokens[2] if len(tokens) >= 4 else tokens[1]
        val = float(tokens[3] if len(tokens) >= 4 else tokens[2])
      if bound_name is None:
        bound_name = b_tag
      if b_tag == bound_name:
        bound_records.append((btype, vname, val))
      continue

    if section == "QUADOBJ":
      tokens = raw.split()
      i = var_idx[tokens[0]]
      j = var_idx[tokens[1]]
      val = float(tokens[2])
      quad_entries.append((i, j, val))

  n = len(var_names)
  m = len(con_names)
  q = np.zeros((n,), dtype=np.float64)
  a_rows: list[int] = []
  a_cols: list[int] = []
  a_vals: list[float] = []
  p_rows: list[int] = []
  p_cols: list[int] = []
  p_vals: list[float] = []

  for j, rname, val in col_entries:
    if rname == obj_row:
      q[j] += val
    elif rname in con_idx:
      a_rows.append(con_idx[rname])
      a_cols.append(j)
      a_vals.append(val)

  for i, j, val in quad_entries:
    p_rows.append(i)
    p_cols.append(j)
    p_vals.append(val)
    if i != j:
      p_rows.append(j)
      p_cols.append(i)
      p_vals.append(val)

  r_const = -rhs_vals.get(obj_row, 0.0) if obj_row is not None else 0.0

  l_con = np.full((m,), -np.inf, dtype=np.float64)
  u_con = np.full((m,), np.inf, dtype=np.float64)
  for rname, idx in con_idx.items():
    rtype = row_types[rname]
    b = rhs_vals.get(rname, 0.0)
    rng = range_vals.get(rname)
    if rtype == "E":
      if rng is None:
        l_con[idx] = b
        u_con[idx] = b
      elif rng >= 0.0:
        l_con[idx] = b
        u_con[idx] = b + rng
      else:
        l_con[idx] = b + rng
        u_con[idx] = b
    elif rtype == "L":
      u_con[idx] = b
      if rng is not None:
        l_con[idx] = b - abs(rng)
    elif rtype == "G":
      l_con[idx] = b
      if rng is not None:
        u_con[idx] = b + abs(rng)

  lb = np.zeros((n,), dtype=np.float64)
  ub = np.full((n,), np.inf, dtype=np.float64)
  for btype, vname, val in bound_records:
    if vname not in var_idx:
      continue
    j = var_idx[vname]
    if btype == "LO" and val is not None:
      lb[j] = val
    elif btype == "UP" and val is not None:
      ub[j] = val
    elif btype == "FX" and val is not None:
      lb[j] = val
      ub[j] = val
    elif btype == "FR":
      lb[j] = -np.inf
      ub[j] = np.inf
    elif btype == "MI":
      lb[j] = -np.inf
    elif btype == "PL":
      ub[j] = np.inf

  return (
      name,
      n,
      m,
      q,
      r_const,
      l_con,
      u_con,
      lb,
      ub,
      a_rows,
      a_cols,
      a_vals,
      p_rows,
      p_cols,
      p_vals,
  )


def load_qpsif(path: pathlib.Path) -> QPSIFProblem:
  """Parses a ``.SIF`` file in QPSIF format into dense float64 arrays."""
  (
      name,
      n,
      m,
      q,
      r_const,
      l_con,
      u_con,
      lb,
      ub,
      a_rows,
      a_cols,
      a_vals,
      p_rows,
      p_cols,
      p_vals,
  ) = _parse_sif_core(path)
  P = np.zeros((n, n), dtype=np.float64)
  A = np.zeros((m, n), dtype=np.float64)
  if a_rows:
    np.add.at(A, (a_rows, a_cols), a_vals)
  if p_rows:
    np.add.at(P, (p_rows, p_cols), p_vals)
  P = 0.5 * (P + P.T)
  return QPSIFProblem(
      name=name,
      P=P,
      q=q,
      r=r_const,
      A=A,
      l_con=l_con,
      u_con=u_con,
      lb=lb,
      ub=ub,
  )


def load_qpsif_sparse(path: pathlib.Path) -> QPSIFSparseProblem:
  """Parses a ``.SIF`` file into sparse ``scipy.sparse.csc_matrix`` arrays."""
  (
      name,
      n,
      m,
      q,
      r_const,
      l_con,
      u_con,
      lb,
      ub,
      a_rows,
      a_cols,
      a_vals,
      p_rows,
      p_cols,
      p_vals,
  ) = _parse_sif_core(path)
  P = sp.coo_matrix((p_vals, (p_rows, p_cols)), shape=(n, n), dtype=np.float64)
  P = (0.5 * (P + P.T)).tocsc()
  A = sp.coo_matrix((a_vals, (a_rows, a_cols)), shape=(m, n), dtype=np.float64)
  A = A.tocsc()
  return QPSIFSparseProblem(
      name=name,
      P=P,
      q=q,
      r=r_const,
      A=A,
      l_con=l_con,
      u_con=u_con,
      lb=lb,
      ub=ub,
  )
