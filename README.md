# cyfatrop

Python wrapper for [Fatrop](https://github.com/meco-group/fatrop)

## Installation

```sh
pip install .
```

By default, a source install downloads checksum-verified Fatrop **v1.2.0** and BLASFEO **0.1.4.3**
archives, compiles both, and links them statically into the extension. Installing an
already-built wheel does not download or compile native dependencies.

BLASFEO defaults to its portable `GENERIC` CPU target. To select an optimized target explicitly:

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

`problem` must inherit one of the public Python abstract classes in [abstract.py](src/cyfatrop/abstract.py):

```python
from cyfatrop import OcpAbstract, TrajOptAbstract

class StagewiseProblem(OcpAbstract):
    # Implement getters and per-stage callbacks.
    ...

class TrajOptProblem(TrajOptAbstract):
    # Implement getters, batched callbacks, and initial constraints.
    ...
```

See [examples/lqr.py](examples/lqr.py) for both interfaces to a double integrator with bounded controls.
Run it with `python examples/lqr.py`.

## Development

```sh
pip install '.[test]'
python -m pytest
python -m build
```
