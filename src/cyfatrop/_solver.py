from __future__ import annotations

import inspect
import operator
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import _native

_DIMS = ("nx", "nu", "ng", "ng_ineq", "nx_tangent", "nu_tangent")
_CALLBACKS = (
    "eval_BAt", "eval_RSQt", "eval_Gt", "eval_Gt_ineq", "eval_b", "eval_g",
    "eval_gineq", "eval_rq", "eval_L", "get_bounds", "get_initial_xk",
    "get_initial_uk", "apply_retraction_xk", "apply_retraction_uk",
)
_RETRACTIONS = ("apply_retraction_xk", "apply_retraction_uk")


def _lookup(obj, name):
    method = getattr(obj, name, None)
    if method is not None and not callable(method):
        raise TypeError(f"{name} must be callable")
    return method


def _dimension_signature(method, name):
    try:
        params = list(inspect.signature(method).parameters.values())
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Cannot inspect {name}; provide a Python wrapper with () or (k)") from exc
    positional = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    if len(params) <= 1 and all(
        p.kind in positional and p.default is inspect.Parameter.empty for p in params
    ):
        return "fixed" if not params else "stagewise"
    raise TypeError(f"{name} must have an unambiguous () or (k) signature")


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


class _Problem:
    """Cached duck-typed protocol, also used by the native horizon evaluator."""

    def __init__(self, problem):
        self.problem = problem
        getters = {}
        modes = set()
        for dim in _DIMS:
            name = "get_" + dim
            method = _lookup(problem, name)
            if method is None:
                if "tangent" in dim:
                    continue
                raise TypeError(f"Missing required method {name}")
            modes.add(_dimension_signature(method, name))
            getters[dim] = method
        if len(modes) != 1:
            raise TypeError("Dimension getters must consistently use () or (k) signatures")
        self.mode = modes.pop()
        horizon = _lookup(problem, "get_horizon_length")
        if horizon is None:
            raise TypeError("Missing required method get_horizon_length")
        if _dimension_signature(horizon, "get_horizon_length") != "fixed":
            raise TypeError("get_horizon_length must have signature ()")
        self.K = _index(horizon(), "get_horizon_length", minimum=1)
        self.nominal = {}
        self.dimensions = {}
        for dim in _DIMS:
            method = getters.get(dim)
            if method is None:
                primal = dim.removesuffix("_tangent")
                self.dimensions[dim] = self.dimensions[primal].copy()
                if self.mode == "fixed":
                    self.nominal[dim] = self.nominal[primal]
            elif self.mode == "fixed":
                n = _index(method(), "get_" + dim)
                self.nominal[dim] = n
                self.dimensions[dim] = [n] * self.K
            else:
                self.dimensions[dim] = [
                    _index(method(k), f"get_{dim}({k})") for k in range(self.K)
                ]
        if self.mode == "fixed":
            for dim in ("ng", "ng_ineq"):
                name = f"get_{dim}_terminal"
                method = _lookup(problem, name)
                if method is not None and _dimension_signature(method, name) != "fixed":
                    raise TypeError(f"{name} must have signature ()")
                n = self.nominal[dim] if method is None else _index(method(), name)
                if n > self.nominal[dim]:
                    raise ValueError(f"{name} cannot exceed get_{dim}()")
                self.dimensions[dim][-1] = n
            self.dimensions["nu"][-1] = 0
            self.dimensions["nu_tangent"][-1] = 0

        self.methods = {}
        self.batched = {}
        for name in _CALLBACKS:
            batch = _lookup(problem, "batched_" + name) if self.mode == "fixed" else None
            method = _lookup(problem, name)
            if batch is not None:
                self.batched[name] = batch
            elif method is not None:
                self.methods[name] = method
            elif name not in _RETRACTIONS:
                alternative = f" or batched_{name}" if self.mode == "fixed" else ""
                raise TypeError(f"Missing required callback {name}{alternative}")
        for variable in ("x", "u"):
            name = f"apply_retraction_{variable}k"
            if name not in self.methods and name not in self.batched:
                primal = self.dimensions[f"n{variable}"]
                tangent = self.dimensions[f"n{variable}_tangent"]
                unequal = primal != tangent
                if self.mode == "fixed":
                    unequal = self.nominal[f"n{variable}"] != self.nominal[f"n{variable}_tangent"]
                if unequal:
                    raise ValueError(f"{name} is required when primal and tangent dimensions differ")

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

    def _shape(self, name, k):
        n = self.nominal if self.mode == "fixed" else {
            dim: values[k] for dim, values in self.dimensions.items()
        }
        m = n["nu_tangent"] + n["nx_tangent"]
        nxt = (n["nx_tangent"] if self.mode == "fixed"
               else self.dimensions["nx_tangent"][k + 1]) if name in ("eval_BAt", "eval_b") else 0
        return {
            "eval_BAt": (m, nxt), "eval_RSQt": (m, m),
            "eval_Gt": (m, n["ng"]), "eval_Gt_ineq": (m, n["ng_ineq"]),
            "eval_b": (nxt,), "eval_g": (n["ng"],), "eval_gineq": (n["ng_ineq"],),
            "eval_rq": (m,), "eval_L": (), "get_bounds": (n["ng_ineq"],),
            "get_initial_xk": (n["nx"],), "get_initial_uk": (n["nu"],),
            "apply_retraction_xk": (n["nx"],), "apply_retraction_uk": (n["nu"],),
        }[name]

    def _normalize(self, name, value, shape, context):
        if name == "get_bounds":
            if not isinstance(value, (tuple, list)) or len(value) != 2:
                raise TypeError(f"{context}: expected a (lower, upper) pair of ndarrays")
            return tuple(_array(v, shape, context + f" {side}")
                         for v, side in zip(value, ("lower", "upper")))
        if name == "eval_L" and shape == ():
            value = np.asarray(value)
        return _array(value, shape, context)

    def _crop(self, name, k, value):
        if name == "get_bounds":
            value = np.stack(value)
        if self.mode == "fixed" and k == self.K - 1:
            u = self.nominal["nu_tangent"]
            if name == "eval_RSQt":
                value = value[u:, u:]
            elif name == "eval_rq":
                value = value[u:]
            elif name in ("eval_Gt", "eval_Gt_ineq"):
                dim = "ng" if name == "eval_Gt" else "ng_ineq"
                value = value[u:, :self.dimensions[dim][-1]]
            elif name in ("eval_g", "eval_gineq"):
                dim = "ng" if name == "eval_g" else "ng_ineq"
                value = value[:self.dimensions[dim][-1]]
            elif name == "get_bounds":
                value = value[:, :self.dimensions["ng_ineq"][-1]]
            elif name in ("get_initial_uk", "apply_retraction_uk"):
                value = value[:0]
        if name == "get_bounds" and (np.any(np.isnan(value)) or np.any(value[0] > value[1])):
            raise ValueError(f"get_bounds at stage {k}: lower bounds must not exceed upper bounds or contain NaN")
        return np.array(value, dtype=np.float64, order="C", copy=True)

    def _pad_args(self, name, k, args):
        if self.mode != "fixed" or k != self.K - 1:
            return args
        args = list(args)
        n = self.nominal
        sizes = {}
        if name in ("eval_BAt", "eval_b", "eval_RSQt", "eval_rq", "eval_L"):
            sizes[1] = n["nu"]
        elif name in ("eval_Gt", "eval_Gt_ineq", "eval_g", "eval_gineq"):
            sizes[0] = n["nu"]
        elif name == "apply_retraction_uk":
            sizes = {0: n["nu"], 1: n["nu_tangent"]}
        if name == "eval_RSQt":
            sizes.update({3: n["nx_tangent"], 4: n["ng"], 5: n["ng_ineq"]})
        for position, length in sizes.items():
            padded = np.zeros(length)
            padded[:len(args[position])] = args[position]
            args[position] = _readonly(padded)
        return tuple(args)

    def _evaluate(self, name, k, args):
        args = self._pad_args(name, k, args)
        method = self.methods.get(name)
        if method is None and name in _RETRACTIONS:
            value = args[0] + args[2] * args[1]
        else:
            value = method(*args, k)
        value = self._normalize(name, value, self._shape(name, k), f"{name} at stage {k}")
        return self._crop(name, k, value)

    def _stage_args(self, name, k, primal, lam, scale, delta, alpha):
        d = self.dimensions

        def block(vector, offset, size):
            return vector[offset:offset + size]

        if name.startswith("get_"):
            return ()
        u = block(primal, self.u_offsets[k], d["nu"][k])
        x = block(primal, self.x_offsets[k], d["nx"][k])
        if name in ("eval_BAt", "eval_b"):
            return block(primal, self.x_offsets[k + 1], d["nx"][k + 1]), u, x
        if name == "eval_RSQt":
            ld = (block(lam, self.dyn_offsets[k], d["nx_tangent"][k + 1])
                  if k < self.K - 1 else _readonly(np.empty(0)))
            return (scale, u, x, ld,
                    block(lam, self.eq_offsets[k], d["ng"][k]),
                    block(lam, self.ineq_offsets[k], d["ng_ineq"][k]))
        if name in ("eval_rq", "eval_L"):
            return scale, u, x
        if name == "apply_retraction_xk":
            return x, block(delta, self.dx_offsets[k], d["nx_tangent"][k]), alpha
        if name == "apply_retraction_uk":
            return u, block(delta, self.du_offsets[k], d["nu_tangent"][k]), alpha
        return u, x

    def _prepare(self, names, primal, lam, scale, delta, alpha):
        """One invocation per native horizon operation; no per-stage Python dispatch."""
        result = {}
        for name in names:
            count = self.K - 1 if name in ("eval_BAt", "eval_b") else self.K
            if not count:
                continue
            rows = [self._pad_args(name, k, self._stage_args(
                name, k, primal, lam, scale, delta, alpha
            )) for k in range(count)]
            args = []
            for column in zip(*rows):
                args.append(_readonly(np.stack(column)) if isinstance(column[0], np.ndarray)
                            else column[0])
            value = self.batched[name](*args)
            shape = (count,) + self._shape(name, 0)
            value = self._normalize(name, value, shape, f"batched_{name} for {count} stages")
            result[name] = tuple(self._crop(
                name, k, tuple(v[k] for v in value) if name == "get_bounds" else value[k]
            ) for k in range(count))
        return result

    def trajectories(self, primal):
        d = self.dimensions
        xs = tuple(primal[o:o + n].copy() for o, n in zip(self.x_offsets, d["nx"]))
        us = tuple(primal[o:o + n].copy() for o, n in zip(self.u_offsets, d["nu"]))
        if self.mode == "fixed":
            us = np.stack((*us[:-1], np.zeros(self.nominal["nu"])))
            return np.stack(xs), us
        return xs, us


@dataclass(frozen=True)
class Result:
    status: _native.Status
    success: bool
    objective: float
    x: np.ndarray | tuple[np.ndarray, ...]
    u: np.ndarray | tuple[np.ndarray, ...]
    stats: dict[str, Any]


class Solver:
    """Solve a duck-typed fixed-dimension or stage-dependent OCP.

    Callbacks and dimensions are discovered once. Construct a new solver after
    changing dimensions or replacing methods. Callback-owned problem data may
    change between solves. See README.md for the complete callback contract.
    """

    def __init__(self, problem, options=None):
        self._problem = _Problem(problem)
        self._options = dict(options) if options is not None else {}
        self._solving = False
        self._solver = _native.NativeSolver(self._problem, self._options)

    @property
    def mode(self):
        return self._problem.mode

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
