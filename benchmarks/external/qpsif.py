"""Minimal fixed-format QPSIF / MPS parser for quadratic programs."""

from __future__ import annotations

import dataclasses
import pathlib

import numpy as np


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


def load_qpsif(path: pathlib.Path) -> QPSIFProblem:
  """Parses a ``.SIF`` file in QPSIF format into dense float64 arrays."""
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
  P = np.zeros((n, n), dtype=np.float64)
  q = np.zeros((n,), dtype=np.float64)
  A = np.zeros((m, n), dtype=np.float64)

  for j, rname, val in col_entries:
    if rname == obj_row:
      q[j] += val
    elif rname in con_idx:
      A[con_idx[rname], j] += val

  for i, j, val in quad_entries:
    P[i, j] += val
    if i != j:
      P[j, i] += val
  P = 0.5 * (P + P.T)

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
