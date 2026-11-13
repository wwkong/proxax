# Examples

End-to-end worked examples covering each problem class in Proxax, with runnable scripts in `examples/`:

- {doc}`unconstrained` — simplex QPs, nonconvex QPs, and nonconvex SVMs (`examples/unconstrained.py`)
- {doc}`constrained` — linearly constrained QPs, $k$-Fantope sparse PCA, and bounded matrix completion (`examples/constrained.py`)
- {doc}`minmax` — bilinear matrix games and nonconvex-nonconcave saddle-point problems (`examples/minmax.py`)
- {doc}`block` — multi-block consensus and ADMM splitting (`examples/block.py`)
- {doc}`spectral` — nuclear-norm and nonconvex spectral matrix completion (`examples/spectral_matrix_completion.py`)
- {doc}`layers` — end-to-end Optax training through a differentiable `SimplexLayer` (`examples/layers_training.py`)

```{toctree}
:maxdepth: 2
:hidden:

unconstrained
constrained
minmax
block
spectral
layers
```
