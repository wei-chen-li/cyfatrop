from __future__ import annotations

import operator
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import _native
from .abstract import OcpAbstract, TrajOptAbstract

_DIMS = ("nx", "nu", "ng", "ng_ineq", "nx_tangent", "nu_tangent")
_CALLBACKS = (
    "eval_BAt", "eval_RSQt", "eval_Gt", "eval_Gt_ineq", "eval_b", "eval_g",
    "eval_gineq", "eval_rq", "eval_L", "get_bounds", "get_initial_xk",
    "get_initial_uk", "apply_retraction_xk", "apply_retraction_uk",
)


def _index(value, name, minimum=0):
    try:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must return an integer") from exc
    if not minimum <= result <= np.iinfo(np.int32).max:
        raise ValueError(f"{name} must be between {minimum} and {np.iinfo(np.int32).max}")
    return result


def _readonly(array):
    array.setflags(write=False)
    return array


def _array(value, shape, context):
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{context}: expected np.ndarray with shape {shape}")
    if value.shape != shape:
        raise ValueError(f"{context}: expected shape {shape}, got {value.shape}")
    if value.dtype.kind not in "biuf":
        raise TypeError(f"{context}: expected a real numeric ndarray, got {value.dtype}")
    return np.array(value, dtype=np.float64, order="C", copy=True)


def _callback_shape(name, n, nx_next=0):
    """Shapes for a stage's dimensions or a fixed adapter's nominal dimensions."""
    m = n["nu_tangent"] + n["nx_tangent"]
    return {
        "eval_BAt": (m, nx_next), "eval_RSQt": (m, m),
        "eval_Gt": (m, n["ng"]), "eval_Gt_ineq": (m, n["ng_ineq"]),
        "eval_b": (nx_next,), "eval_g": (n["ng"],), "eval_gineq": (n["ng_ineq"],),
        "eval_rq": (m,), "eval_L": (), "get_bounds": (n["ng_ineq"],),
        "get_initial_xk": (n["nx"],), "get_initial_uk": (n["nu"],),
        "apply_retraction_xk": (n["nx"],), "apply_retraction_uk": (n["nu"],),
    }[name]


class _Problem(ABC):
    """Native adapter contract: dimensions, batched families, and dispatch hooks.

    Subclasses populate dimensions and batched before native construction.
    Shared helpers operate on normalized native dimensions only.
    """

    def __init__(self, problem: OcpAbstract | TrajOptAbstract):
        self.problem = problem
        self.K = _index(problem.get_horizon_length(), "get_horizon_length", minimum=1)
        self.dimensions: dict[str, list[int]] = {}
        self.batched: dict[str, Any] = {}

    @abstractmethod
    def _evaluate(self, name, k, args):
        """Return one normalized stage result when no native cache entry exists."""

    @abstractmethod
    def _prepare(self, names, primal, lam, scale, delta, alpha):
        """Return normalized stage results to cache for one horizon operation."""

    @abstractmethod
    def trajectories(self, primal):
        """Extract owned state and control trajectories for the public Result."""

    def _intermediate(self, iteration, primal, stats):
        """Convert the native iterate to immutable trajectories and notify the user."""
        x, u = self.trajectories(primal)
        for trajectory in (x, u):
            if isinstance(trajectory, np.ndarray):
                _readonly(trajectory)
            else:
                for stage in trajectory:
                    _readonly(stage)
        self.problem.intermediate(iteration, x, u, stats)

    def _build_layout(self):
        d = self.dimensions
        for k in range(self.K):
            if d["ng"][k] > d["nx_tangent"][k] + d["nu_tangent"][k]:
                raise ValueError(f"Stage {k}: equality count exceeds tangent variable count")
        self.u_offsets, self.x_offsets, self.du_offsets, self.dx_offsets = [], [], [], []
        p = t = 0
        for k in range(self.K):
            self.u_offsets.append(p)
            self.x_offsets.append(p + d["nu"][k])
            self.du_offsets.append(t)
            self.dx_offsets.append(t + d["nu_tangent"][k])
            p += d["nu"][k] + d["nx"][k]
            t += d["nu_tangent"][k] + d["nx_tangent"][k]
        self.eq_offsets = self._offsets(d["ng"])
        self.dyn_offsets = self._offsets(d["nx_tangent"][1:], sum(d["ng"]))
        self.ineq_offsets = self._offsets(
            d["ng_ineq"], sum(d["ng"]) + sum(d["nx_tangent"][1:])
        )
        total = sum(d["ng"]) + sum(d["ng_ineq"]) + sum(d["nx_tangent"][1:])
        if max(p, t, total) > np.iinfo(np.int32).max:
            raise ValueError("Total problem dimensions exceed Fatrop's index range")

    @staticmethod
    def _offsets(sizes, start=0):
        result = []
        for n in sizes:
            result.append(start)
            start += n
        return result

    def _normalize(self, name, value, shape, context):
        if name == "get_bounds":
            if not isinstance(value, (tuple, list)) or len(value) != 2:
                raise TypeError(f"{context}: expected a (lower, upper) pair of ndarrays")
            return tuple(_array(v, shape, context + f" {side}")
                         for v, side in zip(value, ("lower", "upper")))
        if name == "eval_L" and shape == ():
            value = np.asarray(value)
        return _array(value, shape, context)

    @staticmethod
    def _checked_output(name, k, value):
        if name == "get_bounds" and (np.any(np.isnan(value)) or np.any(value[0] > value[1])):
            raise ValueError(f"get_bounds at stage {k}: lower bounds must not exceed upper bounds or contain NaN")
        return np.array(value, dtype=np.float64, order="C", copy=True)

    @staticmethod
    def _block(vector, offset, size):
        return vector[offset:offset + size]

    def _primal_blocks(self, primal):
        d = self.dimensions
        xs = tuple(self._block(primal, o, n).copy() for o, n in zip(self.x_offsets, d["nx"]))
        us = tuple(self._block(primal, o, n).copy() for o, n in zip(self.u_offsets, d["nu"]))
        return xs, us


class _OcpAbstractProblem(_Problem):
    """Per-stage evaluation using actual dimensions at every stage."""

    def __init__(self, problem: OcpAbstract):
        super().__init__(problem)
        for dim in _DIMS:
            getter = getattr(problem, "get_" + dim)
            self.dimensions[dim] = [
                _index(getter(k), f"get_{dim}({k})") for k in range(self.K)
            ]
        self.methods = {name: getattr(problem, name) for name in _CALLBACKS}
        self._build_layout()

    def _shape(self, name, k):
        n = {dim: values[k] for dim, values in self.dimensions.items()}
        nx_next = self.dimensions["nx_tangent"][k + 1] if name in ("eval_BAt", "eval_b") else 0
        return _callback_shape(name, n, nx_next)

    def _evaluate(self, name, k, args):
        value = self.methods[name](*args, k)
        value = self._normalize(name, value, self._shape(name, k), f"{name} at stage {k}")
        if name == "get_bounds":
            value = np.stack(value)
        return self._checked_output(name, k, value)

    def _prepare(self, names, primal, lam, scale, delta, alpha):
        return {}

    def trajectories(self, primal):
        return self._primal_blocks(primal)


class _TrajOptAbstractProblem(_Problem):
    """Batched evaluation, terminal cropping, and initial-constraint assembly."""

    def __init__(self, problem: TrajOptAbstract):
        super().__init__(problem)
        self.nominal = {}
        for dim in _DIMS:
            n = _index(getattr(problem, "get_" + dim)(), "get_" + dim)
            self.nominal[dim] = n
            self.dimensions[dim] = [n] * self.K
        for dim in ("ng", "ng_ineq"):
            name = f"get_{dim}_terminal"
            n = _index(getattr(problem, name)(), name)
            if n > self.nominal[dim]:
                raise ValueError(f"{name} cannot exceed get_{dim}()")
            self.dimensions[dim][-1] = n
        self.dimensions["nu"][-1] = 0
        self.dimensions["nu_tangent"][-1] = 0
        # Keep ordinary counts separate from the appended initial equalities.
        self.path_ng = self.dimensions["ng"].copy()
        self.dimensions["ng"][0] += self.nominal["nx_tangent"]
        self.batched = {name: getattr(problem, "batched_" + name) for name in _CALLBACKS}
        self.initial = {name: getattr(problem, name)
                        for name in ("eval_g_initial", "eval_Gt_initial")}
        self._build_layout()

    def _shape(self, name):
        return _callback_shape(name, self.nominal, self.nominal["nx_tangent"])

    def _evaluate(self, name, k, args):
        raise RuntimeError(f"Internal error: fixed-mode {name} at stage {k} missing from native batch cache")

    def _crop(self, name, k, value):
        if name == "get_bounds":
            value = np.stack(value)
        if k == self.K - 1:
            u = self.nominal["nu_tangent"]
            if name == "eval_RSQt":
                value = value[u:, u:]
            elif name == "eval_rq":
                value = value[u:]
            elif name in ("eval_Gt", "eval_Gt_ineq"):
                dim = "ng" if name == "eval_Gt" else "ng_ineq"
                count = self.path_ng[-1] if dim == "ng" else self.dimensions[dim][-1]
                value = value[u:, :count]
            elif name in ("eval_g", "eval_gineq"):
                dim = "ng" if name == "eval_g" else "ng_ineq"
                count = self.path_ng[-1] if dim == "ng" else self.dimensions[dim][-1]
                value = value[:count]
            elif name == "get_bounds":
                value = value[:, :self.dimensions["ng_ineq"][-1]]
        return self._checked_output(name, k, value)

    def _batch_blocks(self, vector, offsets, sizes, width):
        batch = np.zeros((len(offsets), width))
        for k, (offset, size) in enumerate(zip(offsets, sizes)):
            batch[k, :size] = self._block(vector, offset, size)
        return _readonly(batch)

    def _batch_args(self, name, primal, lam, scale, delta, alpha):
        if name.startswith("get_"):
            return ()
        d, n = self.dimensions, self.nominal
        if name == "apply_retraction_uk":
            u = self._batch_blocks(primal, self.u_offsets[:-1], d["nu"][:-1], n["nu"])
            du = self._batch_blocks(delta, self.du_offsets[:-1], d["nu_tangent"][:-1],
                                    n["nu_tangent"])
            return u, du, alpha
        x = self._batch_blocks(primal, self.x_offsets, d["nx"], n["nx"])
        if name == "apply_retraction_xk":
            dx = self._batch_blocks(delta, self.dx_offsets, d["nx_tangent"], n["nx_tangent"])
            return x, dx, alpha
        u = self._batch_blocks(primal, self.u_offsets[:-1], d["nu"][:-1], n["nu"])
        if name in ("eval_BAt", "eval_b"):
            return x[1:], u, x[:-1]
        if name == "eval_RSQt":
            ld = self._batch_blocks(lam, self.dyn_offsets,
                                    d["nx_tangent"][1:], n["nx_tangent"])
            le = self._batch_blocks(lam, self.eq_offsets, self.path_ng, n["ng"])
            li = self._batch_blocks(lam, self.ineq_offsets, d["ng_ineq"], n["ng_ineq"])
            offset = self.eq_offsets[0] + self.path_ng[0]
            initial = _readonly(lam[offset:offset + n["nx_tangent"]].copy())
            return scale, u, x, ld, le, li, initial
        if name in ("eval_rq", "eval_L"):
            return scale, u, x
        return u, x

    def _prepare(self, names, primal, lam, scale, delta, alpha):
        """One invocation per native horizon operation; no per-stage Python dispatch."""
        result = {}
        for name in names:
            dynamics = name in ("eval_BAt", "eval_b")
            controls = name in ("get_initial_uk", "apply_retraction_uk")
            count = self.K - 1 if dynamics or controls else self.K
            if dynamics and not count:
                continue
            args = self._batch_args(name, primal, lam, scale, delta, alpha)
            value = self.batched[name](*args)
            shape = (count,) + self._shape(name)
            value = self._normalize(name, value, shape, f"batched_{name} for {count} stages")
            result[name] = tuple(self._crop(
                name, k, tuple(v[k] for v in value) if name == "get_bounds" else value[k]
            ) for k in range(count))
            if controls:
                # Native horizon dispatch still requests an empty terminal stage.
                result[name] = (*result[name], np.empty(0))
            if name in ("eval_g", "eval_Gt"):
                initial_name = name + "_initial"
                x0 = primal[self.x_offsets[0]:self.x_offsets[0] + self.nominal["nx"]]
                tx = self.nominal["nx_tangent"]
                shape = (tx,) if name == "eval_g" else (tx, tx)
                initial = _array(self.initial[initial_name](x0), shape, initial_name)
                if name == "eval_Gt":
                    initial = np.vstack([np.zeros((self.dimensions["nu_tangent"][0], tx)), initial])
                first = np.concatenate([result[name][0], initial], axis=-1)
                result[name] = (first, *result[name][1:])
        return result

    def trajectories(self, primal):
        xs, us = self._primal_blocks(primal)
        controls = np.stack(us[:-1]) if self.K > 1 else np.empty((0, self.nominal["nu"]))
        return np.stack(xs), controls


@dataclass(frozen=True)
class Result:
    status: _native.Status
    success: bool
    objective: float
    x: np.ndarray | tuple[np.ndarray, ...]
    u: np.ndarray | tuple[np.ndarray, ...]
    stats: dict[str, Any]


class Solver:
    """Solve an OcpAbstract or TrajOptAbstract problem.

    Callbacks and dimensions are cached once. Construct a new solver after
    changing dimensions or replacing methods. Callback-owned problem data may
    change between solves. See README.md for the complete callback contract.
    """

    def __init__(self, problem: OcpAbstract | TrajOptAbstract, options=None):
        state_dependent = isinstance(problem, OcpAbstract)
        fixed = isinstance(problem, TrajOptAbstract)
        if state_dependent == fixed:
            raise TypeError("problem must inherit exactly one of OcpAbstract or TrajOptAbstract")
        adapter = _OcpAbstractProblem if state_dependent else _TrajOptAbstractProblem
        self._problem = adapter(problem)
        self._options = dict(options) if options is not None else {}
        self._solving = False
        self._solver = _native.NativeSolver(self._problem, self._options)

    @property
    def dimensions(self):
        return {name: tuple(values) for name, values in self._problem.dimensions.items()}

    def solve(self):
        if self._solving:
            raise RuntimeError("A Solver cannot be called recursively")
        self._solving = True
        try:
            raw = self._solver.solve()
            x, u = self._problem.trajectories(raw["primal"])
            status = raw["status"]
            return Result(status, status in (_native.Status.Success, _native.Status.StopAtAcceptablePoint),
                          raw["objective"], x, u, raw["stats"])
        finally:
            self._solving = False
