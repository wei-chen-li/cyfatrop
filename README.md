# cyfatrop

Python wrapper for [Fatrop](https://github.com/meco-group/fatrop)

## Installation

```sh
pip install .
```

By default, a source install downloads checksum-verified Fatrop **v1.2.0** and BLASFEO **0.1.4.3**
archives, compiles both, and links them statically into the extension. Installing an
already-built wheel does not download or compile native dependencies.

Linux and standard CPython 3.10–3.14 are the initial targets. BLASFEO defaults
to its portable `GENERIC` CPU target. To select an optimized target explicitly:

```sh
pip install . -Ccmake.define.CYFATROP_BLASFEO_TARGET=X64_INTEL_HASWELL
```

Override either source archive using
`-Ccmake.define.CYFATROP_FATROP_URL=/path/to/fatrop.tar.gz` and
`-Ccmake.define.CYFATROP_BLASFEO_URL=/path/to/blasfeo.tar.gz`.

## Usage

```python
from cyfatrop import Solver

solver = Solver(problem, options={"tolerance": 1e-8, "max_iter": 100})
result = solver.solve()
print(result.status, result.objective, result.stats)
print(result.x, result.u)
```

See [examples/lqr.py](examples/lqr.py) for a double integrator with bounded controls. Run it with
`python examples/lqr.py`.

`problem` is a python object that contains the following methods, similar to
Fatrop's [OcpAbstract](https://github.com/meco-group/fatrop/blob/main/include/fatrop/ocp/ocp_abstract.hpp).
Methods are cached when constructing the solver. Rebuild it after changing
dimensions or replacing methods.

### Horizon length

| Getter | Method |
|---|---|
| Horizon length | `get_horizon_length()` |

`get_horizon_length()` returns the number of stages **K**. There are K−1
dynamics transitions: 0→1, 1→2, …, K-2→K-1.

### Dimensions

The signatures of the dimension methods choose the interface automatically:

| Getter | Fixed | Stage-dependent |
|---|---|---|
| State size | `get_nx()` | `get_nx(k)` |
| Control size | `get_nu()` | `get_nu(k)` |
| Equality count | `get_ng()` | `get_ng(k)` |
| Inequality count | `get_ng_ineq()` | `get_ng_ineq(k)` |
| State tangent size | `get_nx_tangent()` | `get_nx_tangent(k)` |
| Control tangent size | `get_nu_tangent()` | `get_nu_tangent(k)` |

The last two getters are optional and default to the corresponding Euclidean
dimension. All implemented dimension getters must consistently use `()` or
`(k)`.

In fixed mode, `get_ng_terminal()` and `get_ng_ineq_terminal()` optionally reduce
the final equality and inequality counts. Their defaults are `get_ng()` and
`get_ng_ineq()`. These getters always take no arguments and are ignored in
stage-dependent mode.

### Problem data

Every callback below is required, including those returning empty arrays, except
the two retractions. In fixed mode a corresponding batch implementation can
replace any required numerical callback.

Use variable ordering **[u, x]**. Let `tx` and `tu` be tangent dimensions, and
`tx_next = get_nx_tangent(k+1)`. In fixed mode all callback shapes use nominal
dimensions, including the terminal stage.

| Method | Arguments | Output shape |
|---|---|---|
| `eval_BAt` | `(x_next, u, x, k)` | `(tu+tx, tx_next)` |
| `eval_RSQt` | `(scale, u, x, lam_dyn, lam_eq, lam_ineq, k)` | `(tu+tx, tu+tx)` |
| `eval_Gt` | `(u, x, k)` | `(tu+tx, ng)` |
| `eval_Gt_ineq` | `(u, x, k)` | `(tu+tx, ng_ineq)` |
| `eval_b` | `(x_next, u, x, k)` | `(tx_next,)` |
| `eval_g` | `(u, x, k)` | `(ng,)` |
| `eval_gineq` | `(u, x, k)` | `(ng_ineq,)` |
| `eval_rq` | `(scale, u, x, k)` | `(tu+tx,)` |
| `eval_L` | `(scale, u, x, k)` | scalar |
| `get_bounds` | `(k)` | pair of `(ng_ineq,)` arrays |
| `get_initial_xk` | `(k)` | `(nx,)` |
| `get_initial_uk` | `(k)` | `(nu,)` |
| `apply_retraction_xk` | `(x, delta_x, alpha, k)` | `(nx,)` |
| `apply_retraction_uk` | `(u, delta_u, alpha, k)` | `(nu,)` |

### Batch callbacks and terminal cropping

Only fixed mode supports batching. Prefix any problem data callback with **`batched_`**.
Its signature omits `k`. Every array input/output gains a leading batch axis; `scale`
and `alpha` stay scalars. `batched_eval_L` returns `(batch_size,)` and
`batched_get_bounds` returns two `(batch_size, ng_ineq)` arrays.

Each available batch callback takes precedence independently; other callbacks
fall back to their stage implementation. Rows follow ascending stage order:
K rows for stage functions, K−1 rows for dynamics.

Python sees nominal-sized terminal arrays even when Fatrop stores smaller blocks.
This applies to both batch callbacks and unbatched fallbacks in fixed mode:

* The terminal control and control step are zero placeholders; Fatrop has no
  terminal control variables. The terminal dynamics multiplier is zero too.
* Terminal equality/inequality multipliers contain the active leading entries
  followed by zeros, up to the nominal dimension.
* Terminal residuals and bounds retain only the leading terminal-count entries.
  Transposed constraint Jacobians retain the corresponding leading columns
  and discard control rows.
* Terminal Hessians retain their state–state block; gradients retain state
  entries. Terminal control initialization/retraction outputs are discarded.

Unused entries can contain arbitrary values, but callbacks must still return
the complete nominal shape. In stage-dependent mode, provide actual dimensions,
including `get_nu(K-1) == 0` if desired; no padding or automatic truncation occurs.

## Development

```sh
pip install '.[test]'
python -m pytest
python -m build
```
