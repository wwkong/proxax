"""Primal-dual stationarity refinement certificates for min-max problems."""

from __future__ import annotations

import dataclasses
import typing

import jax
import jax.numpy as jnp

from proxax import _typing
from proxax import tree
from proxax.oracle import minmax


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class MinMaxRefinementResult:
  """Certified primal-dual point `(x_hat, y_hat)` and subgradients."""

  x_hat: _typing.PyTree
  y_hat: _typing.PyTree
  v_x: _typing.PyTree
  v_y: _typing.PyTree
  obj_val: jax.Array
  residual: jax.Array
  smooth_evals: jax.Array
  grad_evals: jax.Array
  prox_evals: jax.Array


def refine_minmax(
    oracle: minmax.MinMaxOracle,
    x: _typing.PyTree,
    y: _typing.PyTree,
    *,
    tau: jax.Array | float = 1.0,
    sigma: jax.Array | float = 1.0,
    norm_kind: str = "l2",
) -> MinMaxRefinementResult:
  """Computes exact primal-dual subdifferential membership at `(x_hat, y_hat)`.

  Given `(x, y)` and step sizes `(tau, sigma)`, sets:
    - `x_hat = prox_{tau h}(x - tau * grad_x Phi(x, y))`
    - `y_hat = prox_{sigma g}(y + sigma * grad_y Phi(x, y))`
  so that:
    - `v_x = grad_x Phi(x_hat, y_hat) + (x - x_hat) / tau - grad_x Phi(x, y)`
      lies in `grad_x Phi(x_hat, y_hat) + d h(x_hat)`, and
    - `v_y = -grad_y Phi(x_hat, y_hat) + (y - y_hat) / sigma + grad_y Phi(x, y)`
      lies in `-grad_y Phi(x_hat, y_hat) + d g(y_hat)`.
  """
  dtype = tree.infer_dtype(x)
  tau_arr = jnp.asarray(tau, dtype=dtype)
  sigma_arr = jnp.asarray(sigma, dtype=dtype)

  _, gx, gy = oracle.phi_val_grad(x, y)
  x_hat = oracle.prox_x.prox(
      tree.sub(x, tree.scale(tau_arr, gx)),
      tau_arr,
  )
  y_hat = oracle.prox_y.prox(
      tree.add(y, tree.scale(sigma_arr, gy)),
      sigma_arr,
  )
  val_hat, gx_hat, gy_hat = oracle.phi_val_grad(x_hat, y_hat)

  v_x = tree.add(
      gx_hat,
      tree.sub(tree.scale(1.0 / tau_arr, tree.sub(x, x_hat)), gx),
  )
  v_y = tree.add(
      tree.scale(-1.0, gy_hat),
      tree.add(tree.scale(1.0 / sigma_arr, tree.sub(y, y_hat)), gy),
  )
  obj_hat = val_hat + oracle.prox_x.value(x_hat) - oracle.prox_y.value(y_hat)
  res = tree.norm((v_x, v_y), kind=typing.cast(typing.Any, norm_kind))
  return MinMaxRefinementResult(
      x_hat=x_hat,
      y_hat=y_hat,
      v_x=v_x,
      v_y=v_y,
      obj_val=obj_hat,
      residual=res,
      smooth_evals=jnp.asarray(2, dtype=jnp.int32),
      grad_evals=jnp.asarray(2, dtype=jnp.int32),
      prox_evals=jnp.asarray(2, dtype=jnp.int32),
  )
