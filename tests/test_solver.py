import gc
import inspect
from collections import Counter

import numpy as np
import pytest

from cyfatrop import (
    FATROP_VERSION, OcpAbstract, Solver, Status, TrajOptAbstract,
)
from cyfatrop._solver import (
    _CALLBACKS, _OcpAbstractProblem, _Problem, _TrajOptAbstractProblem,
)
from examples.lqr import DoubleIntegrator, DoubleIntegrator2


OPTIONS = {"tolerance": 1e-9, "max_iter": 100, "print_level": 0, "suppress_banner": True}


class StageIntegrator(OcpAbstract):
    """Scalar reference with a fixed initial state and bounded controls."""
    target = np.array([0., 0., 3.])
    initial_state = np.array([.75])

    def get_horizon_length(self): return len(self.target)
    def get_nx(self, k): return 1
    def get_nu(self, k): return int(k < len(self.target) - 1)
    def get_ng(self, k): return int(k == 0)
    def get_ng_ineq(self, k): return self.get_nu(k)
    def eval_BAt(self, xp, u, x, k): return np.ones((2, 1))
    def eval_b(self, xp, u, x, k): return x + u - xp
    def eval_RSQt(self, scale, u, x, ld, le, li, k): return scale * np.eye(len(u) + len(x))
    def eval_Gt(self, u, x, k):
        return np.vstack([np.zeros((len(u), 1)), np.ones((1, 1))]) if k == 0 else np.empty((len(u) + 1, 0))
    def eval_g(self, u, x, k): return x - self.initial_state if k == 0 else np.empty(0)
    def eval_Gt_ineq(self, u, x, k):
        return np.array([[1.], [0.]]) if len(u) else np.empty((1, 0))
    def eval_gineq(self, u, x, k): return u.copy()
    def eval_L(self, scale, u, x, k):
        return scale * .5 * (u @ u + (x[0] - self.target[k]) ** 2)
    def eval_rq(self, scale, u, x, k):
        return scale * np.concatenate([u, x - self.target[k]])
    def get_bounds(self, k):
        return np.full(self.get_ng_ineq(k), -.25), np.full(self.get_ng_ineq(k), .25)
    def get_initial_xk(self, k): return np.zeros(1)
    def get_initial_uk(self, k): return np.zeros(self.get_nu(k))


class FixedIntegrator(TrajOptAbstract):
    target = StageIntegrator.target
    initial_state = StageIntegrator.initial_state

    def get_horizon_length(self): return len(self.target)
    def get_nx(self): return 1
    def get_nu(self): return 1
    def get_ng(self): return 0
    def get_ng_ineq(self): return 1
    def get_ng_ineq_terminal(self): return 0
    def batched_eval_BAt(self, xp, u, x): return np.ones((len(x), 2, 1))
    def batched_eval_b(self, xp, u, x): return x + u - xp
    def batched_eval_RSQt(self, scale, u, x, ld, le, li, lam_initial):
        assert np.all(u[-1] == 0) and np.all(ld[-1] == 0)
        assert le.shape == (len(x), 0) and lam_initial.shape == (1,)
        return np.broadcast_to(scale * np.eye(2), (len(x), 2, 2))
    def batched_eval_Gt(self, u, x): return np.empty((len(x), 2, 0))
    def batched_eval_g(self, u, x): return np.empty((len(x), 0))
    def batched_eval_Gt_ineq(self, u, x):
        return np.broadcast_to(np.array([[1.], [0.]]), (len(x), 2, 1))
    def batched_eval_gineq(self, u, x): return u.copy()
    def batched_eval_L(self, scale, u, x):
        return scale * .5 * (u[:, 0] ** 2 + (x[:, 0] - self.target) ** 2)
    def batched_eval_rq(self, scale, u, x):
        return scale * np.concatenate([u, x - self.target[:, None]], axis=1)
    def batched_get_bounds(self):
        return np.full((len(self.target), 1), -.25), np.full((len(self.target), 1), .25)
    def batched_get_initial_xk(self): return np.zeros((len(self.target), 1))
    def batched_get_initial_uk(self): return np.zeros((len(self.target), 1))
    def eval_g_initial(self, x0): return x0 - self.initial_state
    def eval_Gt_initial(self, x0): return np.eye(1)


def record_callbacks(problem):
    problem.counts = Counter()
    problem.saved_inputs = []
    fixed = isinstance(problem, TrajOptAbstract)
    names = ["batched_" + n if fixed else n for n in _CALLBACKS]
    if fixed:
        names += ["eval_g_initial", "eval_Gt_initial"]
    for name in names:
        original = getattr(problem, name)

        def callback(*args, name=name, original=original):
            problem.counts[name] += 1
            for value in args:
                if isinstance(value, np.ndarray):
                    assert not value.flags.writeable
                    problem.saved_inputs.append((value, value.copy()))
            return original(*args)

        setattr(problem, name, callback)
    return problem


@pytest.mark.parametrize("base", [OcpAbstract, TrajOptAbstract, _Problem])
def test_abstract_classes_cannot_instantiate(base):
    with pytest.raises(TypeError, match="abstract"):
        base()


@pytest.mark.parametrize("base,complete,name", [
    (base, complete, name)
    for base, complete in ((OcpAbstract, StageIntegrator), (TrajOptAbstract, FixedIntegrator))
    for name in sorted(base.__abstractmethods__)
])
def test_missing_abstract_implementation(base, complete, name):
    incomplete = type("Incomplete", (complete,), {name: getattr(base, name)})
    with pytest.raises(TypeError, match=name):
        incomplete()


def test_solver_requires_exactly_one_mode_class():
    class Both(FixedIntegrator, StageIntegrator): pass

    for problem in (object(), Both()):
        with pytest.raises(TypeError, match="exactly one"):
            Solver(problem)


@pytest.mark.parametrize("cls,adapter", [
    (StageIntegrator, _OcpAbstractProblem),
    (FixedIntegrator, _TrajOptAbstractProblem),
])
def test_adapter_selection_and_no_signature_inspection(monkeypatch, cls, adapter):
    monkeypatch.setattr(inspect, "signature", lambda *a, **kw: pytest.fail("Signature inspected"))
    problem = cls()
    # Ambiguous-to-inspection signatures remain ordinary valid Python calls.
    problem.get_nx = lambda *args: 1
    solver = Solver(problem, OPTIONS)
    assert isinstance(solver._problem, adapter)
    assert solver.solve().success


def test_unused_adapter_dispatch_hooks():
    stage = _OcpAbstractProblem(StageIntegrator())
    assert stage.batched == {}
    assert stage._prepare([], None, None, 1., None, 0.) == {}
    fixed = _TrajOptAbstractProblem(FixedIntegrator())
    with pytest.raises(RuntimeError, match="eval_g at stage 0 missing from native batch cache"):
        fixed._evaluate("eval_g", 0, ())


@pytest.mark.parametrize("cls", [StageIntegrator, FixedIntegrator])
def test_extra_callbacks_are_ignored(cls):
    problem = cls()
    if cls is FixedIntegrator:
        for name in _CALLBACKS:
            setattr(problem, name, None)
    else:
        for name in _CALLBACKS:
            setattr(problem, "batched_" + name, None)
        problem.eval_g_initial = problem.eval_Gt_initial = None
        problem.get_ng_terminal = problem.get_ng_ineq_terminal = None
    assert Solver(problem, OPTIONS).solve().success


@pytest.mark.parametrize("cls", [StageIntegrator, FixedIntegrator])
@pytest.mark.parametrize("callback", [None, lambda: np.zeros(1)])
def test_invalid_callback_implementations_fail_only_when_called(cls, callback):
    problem = cls()
    name = "get_initial_xk" if cls is StageIntegrator else "batched_eval_b"
    setattr(problem, name, callback)
    solver = Solver(problem, OPTIONS)
    with pytest.raises(TypeError):
        solver.solve()


@pytest.mark.parametrize("cls", [StageIntegrator, FixedIntegrator])
def test_constructor_does_not_execute_numerical_callbacks(cls):
    problem = cls()
    prefix = "batched_" if cls is FixedIntegrator else ""
    for name in _CALLBACKS:
        setattr(problem, prefix + name, lambda *a: pytest.fail("Premature numerical callback"))
    if cls is FixedIntegrator:
        problem.eval_g_initial = problem.eval_Gt_initial = lambda *a: pytest.fail("Premature initial callback")
    Solver(problem)


@pytest.mark.parametrize("cls", [StageIntegrator, FixedIntegrator])
def test_analytic_bounded_ocp(cls):
    solver = Solver(cls(), OPTIONS)
    result = solver.solve()
    assert result.success and result.status == Status.Success
    np.testing.assert_allclose(np.stack(result.x).ravel(), [.75, 1., 1.25], atol=2e-6)
    if cls is StageIntegrator:
        assert isinstance(result.x, tuple) and result.u[-1].shape == (0,)
        controls = np.concatenate(result.u)
    else:
        assert result.x.flags.owndata and result.u.flags.owndata
        assert result.u.shape == (solver._problem.K - 1, 1)
        controls = result.u.ravel()
    np.testing.assert_allclose(controls, [.25, .25], atol=2e-6)
    assert result.objective == pytest.approx(2.375, abs=2e-6)
    assert result.stats["iterations"] > 0 and result.stats["solve_time"] > 0
    assert result.stats["eval_jacobian_time"] >= 0
    assert solver.dimensions["ng"] == (1, 0, 0)


@pytest.mark.parametrize("cls", [StageIntegrator, FixedIntegrator])
def test_dispatch_and_retained_inputs(monkeypatch, cls):
    problem = record_callbacks(cls())

    def forbidden(*args):
        pytest.fail("Native code used the wrong adapter dispatch hook")

    monkeypatch.setattr(_TrajOptAbstractProblem, "_evaluate", forbidden)
    monkeypatch.setattr(_OcpAbstractProblem, "_prepare", forbidden)
    assert Solver(problem, OPTIONS).solve().success
    counts = problem.counts
    prefix = "batched_" if cls is FixedIntegrator else ""
    if prefix:
        assert counts["batched_eval_BAt"] == counts["batched_eval_Gt"] == counts["batched_eval_Gt_ineq"] > 0
        assert counts["batched_eval_b"] == counts["batched_eval_g"] == counts["batched_eval_gineq"] > 0
    else:
        assert counts["eval_BAt"] > 0
    assert counts[prefix + "get_initial_xk"] == counts[prefix + "get_initial_uk"] == (1 if prefix else 3)
    if prefix:
        assert counts["eval_g_initial"] == counts["batched_eval_g"]
        assert counts["eval_Gt_initial"] == counts["batched_eval_Gt"]
    gc.collect()
    for retained, saved in problem.saved_inputs:
        np.testing.assert_array_equal(retained, saved)


@pytest.mark.parametrize("cls", [StageIntegrator, FixedIntegrator])
def test_repeated_solves_refresh_problem_data(cls):
    problem = record_callbacks(cls())
    solver = Solver(problem, OPTIONS)
    first = solver.solve()
    old_x = np.stack(first.x).copy()
    problem.target = problem.target + 2.
    problem.initial_state = problem.initial_state + 2.
    second = solver.solve()
    np.testing.assert_allclose(np.stack(second.x), old_x + 2., atol=2e-6)
    np.testing.assert_array_equal(np.stack(first.x), old_x)


@pytest.mark.parametrize("acceleration_limit", [1., 100.])
def test_fixed_and_state_dependent_examples_agree(acceleration_limit):
    stage = DoubleIntegrator()
    fixed = DoubleIntegrator2()
    stage.max_acceleration = fixed.model.max_acceleration = acceleration_limit
    a = Solver(stage, OPTIONS).solve()
    b = Solver(fixed, OPTIONS).solve()
    assert a.success and b.success
    np.testing.assert_array_equal(b.x, np.stack(a.x))
    np.testing.assert_array_equal(b.u, np.stack(a.u[:-1]))
    assert b.objective == a.objective


class TerminalOcp(TrajOptAbstract):
    """Initial state, ordinary path constraints, and terminal constraints coexist."""
    def get_horizon_length(self): return 2
    def get_nx(self): return 2
    def get_nu(self): return 3
    def get_ng(self): return 2
    def get_ng_ineq(self): return 2
    def get_ng_terminal(self): return 1
    def get_ng_ineq_terminal(self): return 1
    def batched_get_initial_xk(self): return np.zeros((2, 2))
    def batched_get_initial_uk(self): return np.array([[0., 0., 0.], [1e20, 1e20, 1e20]])
    def batched_eval_BAt(self, xp, u, x):
        return np.array([[[1., 0.], [0., 1.], [0., 0.], [1., 0.], [0., 1.]]])
    def batched_eval_b(self, xp, u, x): return x + u[:, :2] - xp
    def batched_eval_g(self, u, x):
        assert np.all(u[-1] == 0)
        return np.array([[u[0, 1], u[0, 2]], [x[1, 0] - 3., 1e20]])
    def batched_eval_Gt(self, u, x):
        result = np.zeros((2, 5, 2))
        result[0, 1, 0] = result[0, 2, 1] = 1.
        result[1, :3, :] = 1e20
        result[1, 3, 0] = 1.
        result[1, 3:, 1] = 1e20
        return result
    def batched_eval_gineq(self, u, x):
        return np.array([[u[0, 0], x[0, 1]], [x[1, 1], 1e20]])
    def batched_eval_Gt_ineq(self, u, x):
        result = np.zeros((2, 5, 2))
        result[0, 0, 0] = result[0, 4, 1] = result[1, 4, 0] = 1.
        result[1, :3, :] = result[1, 3:, 1] = 1e20
        return result
    def batched_get_bounds(self):
        # Inconsistent discarded bounds must have no effect.
        return np.array([[-5., 1.], [1., 999.]]), np.array([[5., 5.], [5., -999.]])
    def batched_eval_L(self, scale, u, x):
        return scale * .5 * (np.sum(u * u, axis=1) + np.sum(x * x, axis=1))
    def batched_eval_rq(self, scale, u, x):
        result = scale * np.concatenate([u, x], axis=1)
        result[1, :3] = 1e20
        return result
    def batched_eval_RSQt(self, scale, u, x, ld, le, li, lam_initial):
        assert ld.shape == le.shape == li.shape == (2, 2)
        assert np.all(ld[-1] == 0) and le[1, 1] == li[1, 1] == 0
        assert lam_initial.shape == (2,)
        result = np.broadcast_to(scale * np.eye(5), (2, 5, 5)).copy()
        result[1, :3, :] = result[1, :, :3] = 1e20
        return result
    def batched_apply_retraction_uk(self, u, du, alpha):
        assert np.all(u[-1] == 0) and np.all(du[-1] == 0)
        result = u + alpha * du
        result[-1] = 1e20
        return result
    def eval_g_initial(self, x0): return x0 - [1., 2.]
    def eval_Gt_initial(self, x0): return np.eye(2)


def test_terminal_counts_crop_and_discard_control_blocks():
    solver = Solver(TerminalOcp(), OPTIONS)
    assert solver.dimensions["nu"] == (3, 0)
    assert solver.dimensions["ng"] == (4, 1)
    assert solver.dimensions["ng_ineq"] == (2, 1)
    result = solver.solve()
    assert result.success
    np.testing.assert_allclose(result.x, [[1., 2.], [3., 2.]], atol=2e-6)
    np.testing.assert_allclose(result.u, [[2., 0., 0.]], atol=2e-6)
    assert result.objective == pytest.approx(11., abs=2e-6)


@pytest.mark.parametrize("value", [-1, 3, 1.5, True])
@pytest.mark.parametrize("name", ["get_ng_terminal", "get_ng_ineq_terminal"])
def test_invalid_terminal_counts(value, name):
    problem = TerminalOcp()
    setattr(problem, name, lambda: value)
    with pytest.raises((TypeError, ValueError), match=name):
        Solver(problem)


def test_terminal_defaults_and_zero_counts():
    problem = TerminalOcp()
    problem.get_ng_terminal = lambda: TrajOptAbstract.get_ng_terminal(problem)
    problem.get_ng_ineq_terminal = lambda: TrajOptAbstract.get_ng_ineq_terminal(problem)
    adapter = _TrajOptAbstractProblem(problem)
    assert adapter.dimensions["ng"] == [4, 2]
    assert adapter.dimensions["ng_ineq"] == [2, 2]
    problem.get_ng_terminal = problem.get_ng_ineq_terminal = lambda: 0
    solver = Solver(problem, OPTIONS)
    assert solver.dimensions["ng"] == (4, 0)
    result = solver.solve()
    assert result.success
    np.testing.assert_allclose(result.x, [[1., 2.], [.5, 2.]], atol=2e-6)
    np.testing.assert_allclose(result.u, [[-.5, 0., 0.]], atol=2e-6)


def test_initial_assembly_and_multiplier_offsets():
    problem = TerminalOcp()
    adapter = _TrajOptAbstractProblem(problem)
    primal = np.array([2., 0., 0., 1., 2., 3., 2.])
    primal.setflags(write=False)
    result = adapter._prepare(["eval_g", "eval_Gt"], primal, None, 1., None, 0.)
    np.testing.assert_array_equal(result["eval_g"][0], np.zeros(4))
    np.testing.assert_array_equal(result["eval_Gt"][0], np.eye(5)[:, 1:])
    np.testing.assert_array_equal(result["eval_Gt"][1], [[1.], [0.]])
    assert adapter.eq_offsets == [0, 4]
    assert adapter.dyn_offsets == [5] and adapter.ineq_offsets == [7, 9]
    seen = []

    def hessian(scale, u, x, ld, le, li, lam_initial):
        seen.append((ld.copy(), le.copy(), li.copy(), lam_initial.copy()))
        return np.zeros((2, 5, 5))

    adapter.batched["eval_RSQt"] = hessian
    lam = np.arange(10., dtype=float)
    lam.setflags(write=False)
    adapter._prepare(["eval_RSQt"], primal, lam, 1., None, 0.)
    ld, le, li, initial = seen[0]
    np.testing.assert_array_equal(le, [[0., 1.], [4., 0.]])
    np.testing.assert_array_equal(initial, [2., 3.])
    np.testing.assert_array_equal(ld, [[5., 6.], [0., 0.]])
    np.testing.assert_array_equal(li, [[7., 8.], [9., 0.]])


class NonlinearInitial(FixedIntegrator):
    target = np.array([0.])
    seen_initial_multipliers = None
    def batched_get_initial_xk(self): return np.array([[1.5]])
    def eval_g_initial(self, x0): return x0 ** 2 - 4.
    def eval_Gt_initial(self, x0): return np.array([[2. * x0[0]]])
    def batched_eval_RSQt(self, scale, u, x, ld, le, li, lam_initial):
        self.seen_initial_multipliers = lam_initial.copy()
        result = np.broadcast_to(scale * np.eye(2), (len(x), 2, 2)).copy()
        result[0, 1, 1] += 2. * lam_initial[0]
        return result


def test_nonlinear_initial_constraint_hessian_and_single_stage():
    problem = NonlinearInitial()
    solver = Solver(problem, OPTIONS)
    result = solver.solve()
    assert result.success
    np.testing.assert_allclose(result.x, [[2.]], atol=1e-8)
    assert result.u.shape == (0, 1)
    assert result.objective == pytest.approx(2., abs=1e-8)
    assert problem.seen_initial_multipliers is not None
    # Check the multiplier-weighted curvature separately from the solve.
    adapter = solver._problem
    hess = adapter._prepare(["eval_RSQt"], np.array([2.]), np.array([3.]), 2., None, 0.)
    np.testing.assert_array_equal(hess["eval_RSQt"][0], [[8.]])


@pytest.mark.parametrize("name,shape", [("eval_g_initial", (1, 1)), ("eval_Gt_initial", (1,))])
def test_initial_callback_shape_errors(name, shape):
    problem = FixedIntegrator()
    setattr(problem, name, lambda x: np.zeros(shape))
    with pytest.raises(ValueError, match=name + ".*expected shape"):
        Solver(problem, OPTIONS).solve()


@pytest.mark.parametrize("cls", [StageIntegrator, FixedIntegrator])
def test_wrong_shape_reports_callback(cls):
    problem = cls()
    if cls is StageIntegrator:
        problem.eval_b = lambda *a: np.zeros((1, 1))
        message = r"eval_b at stage 0: expected shape \(1,\)"
    else:
        problem.batched_eval_rq = lambda *a: np.zeros((2, 3))
        message = r"batched_eval_rq.*expected shape \(3, 2\)"
    with pytest.raises(ValueError, match=message):
        Solver(problem, OPTIONS).solve()


@pytest.mark.parametrize("cls", [StageIntegrator, FixedIntegrator])
@pytest.mark.parametrize("bad", [[0.], np.array([1j]), np.array(["bad"])])
def test_reject_non_numeric_or_non_array_returns(cls, bad):
    problem = cls()
    name = "get_initial_xk" if cls is StageIntegrator else "batched_get_initial_xk"
    if cls is FixedIntegrator and isinstance(bad, np.ndarray):
        bad = np.broadcast_to(bad, (3, 1))
    setattr(problem, name, lambda *a: bad)
    with pytest.raises(TypeError, match="get_initial_xk"):
        Solver(problem, OPTIONS).solve()


@pytest.mark.parametrize("cls", [StageIntegrator, FixedIntegrator])
@pytest.mark.parametrize("value", [-1, 1.5, True, 2**31])
def test_invalid_dimensions(cls, value):
    problem = cls()
    problem.get_nx = lambda *a: value
    with pytest.raises((TypeError, ValueError), match="get_nx"):
        Solver(problem)


@pytest.mark.parametrize("cls", [StageIntegrator, FixedIntegrator])
@pytest.mark.parametrize("bad", ["nan", "inverted"])
def test_invalid_bounds(cls, bad):
    problem = cls()
    name = "get_bounds" if cls is StageIntegrator else "batched_get_bounds"
    shape = (1,) if cls is StageIntegrator else (3, 1)
    lower = np.full(shape, np.nan if bad == "nan" else 2.)
    setattr(problem, name, lambda *a: (lower, np.ones(shape)))
    with pytest.raises(ValueError, match="get_bounds"):
        Solver(problem, OPTIONS).solve()


def test_equality_count_validation_includes_initial_block():
    problem = FixedIntegrator()
    problem.get_ng = lambda: 2
    problem.get_ng_terminal = lambda: 0
    with pytest.raises(ValueError, match="Stage 0.*equality count"):
        Solver(problem)


def test_strided_and_integer_callback_arrays():
    problem = StageIntegrator()
    problem.eval_BAt = lambda *a: np.ones((2, 2), dtype=np.int32)[:, ::2]
    problem.eval_RSQt = lambda scale, u, x, *a: (scale * np.eye(2 * (len(u) + 1)))[::2, ::2]
    assert Solver(problem, OPTIONS).solve().success


@pytest.mark.parametrize("cls,name", [
    (StageIntegrator, "get_initial_xk"), (FixedIntegrator, "batched_eval_b"),
    (FixedIntegrator, "eval_g_initial"), (FixedIntegrator, "eval_Gt_initial"),
])
def test_exception_propagates_and_cache_clears_for_retry(cls, name):
    problem = cls()
    original = getattr(problem, name)
    problem.fail = True

    def callback(*args):
        if problem.fail:
            raise LookupError("user callback failed")
        return original(*args)

    setattr(problem, name, callback)
    solver = Solver(problem, OPTIONS)
    with pytest.raises(LookupError, match="user callback failed"):
        solver.solve()
    problem.fail = False
    assert solver.solve().success


def test_unknown_option_and_iteration_limit():
    with pytest.raises(ValueError, match="Unknown Fatrop option"):
        Solver(FixedIntegrator(), {"typo_option": 1}).solve()
    result = Solver(FixedIntegrator(), {"max_iter": 0}).solve()
    assert result.status == Status.MaxIterExceeded and not result.success
    assert result.x.shape == (3, 1) and FATROP_VERSION == "1.2.0"


@pytest.mark.parametrize("base", [OcpAbstract, TrajOptAbstract])
@pytest.mark.parametrize("variable", ["x", "u"])
def test_default_retraction_rejects_different_dimensions(base, variable):
    prefix = "batched_" if base is TrajOptAbstract else ""
    callback = getattr(base, prefix + f"apply_retraction_{variable}k")
    shape = (3, 2) if prefix else (2,)
    step_shape = (3, 1) if prefix else (1,)
    args = (None, np.zeros(shape), np.zeros(step_shape), 1.)
    if not prefix:
        args += (0,)
    with pytest.raises(ValueError, match="custom retraction"):
        callback(*args)


class CircleIntegrator(FixedIntegrator):
    """S1 state with one tangent coordinate and two primal coordinates."""
    def get_nx(self): return 2
    def get_nx_tangent(self): return 1
    def batched_get_initial_xk(self):
        return np.broadcast_to([1., 0.], (len(self.target), 2))
    @staticmethod
    def angle(x): return np.arctan2(x[..., 1], x[..., 0])
    def batched_eval_b(self, xp, u, x):
        angle = self.angle(x) + u[:, 0] - self.angle(xp)
        return np.arctan2(np.sin(angle), np.cos(angle))[:, None]
    def batched_eval_L(self, scale, u, x):
        return scale * .5 * (u[:, 0] ** 2 + (self.angle(x) - self.target) ** 2)
    def batched_eval_rq(self, scale, u, x):
        return scale * np.stack([u[:, 0], self.angle(x) - self.target], axis=1)
    def batched_apply_retraction_xk(self, x, dx, alpha):
        angle = self.angle(x) + alpha * dx[:, 0]
        return np.stack([np.cos(angle), np.sin(angle)], axis=1)
    def eval_g_initial(self, x0): return np.array([self.angle(x0) - self.initial_state[0]])


class CircleControlIntegrator(CircleIntegrator):
    def get_nu(self): return 2
    def get_nu_tangent(self): return 1
    def batched_get_initial_uk(self):
        return np.broadcast_to([1., 0.], (len(self.target), 2))
    def batched_eval_b(self, xp, u, x):
        return (self.angle(x) + self.angle(u) - self.angle(xp))[:, None]
    def batched_eval_gineq(self, u, x): return self.angle(u)[:, None]
    def batched_eval_L(self, scale, u, x):
        return scale * .5 * (self.angle(u) ** 2 + (self.angle(x) - self.target) ** 2)
    def batched_eval_rq(self, scale, u, x):
        return scale * np.stack([self.angle(u), self.angle(x) - self.target], axis=1)
    def batched_apply_retraction_uk(self, u, du, alpha):
        angle = self.angle(u) + alpha * du[:, 0]
        return np.stack([np.cos(angle), np.sin(angle)], axis=1)


class StageCircleIntegrator(StageIntegrator):
    def get_nx(self, k): return 2
    def get_nx_tangent(self, k): return 1
    def get_initial_xk(self, k): return np.array([1., 0.])
    def eval_RSQt(self, scale, u, x, ld, le, li, k): return scale * np.eye(len(u) + 1)
    def eval_b(self, xp, u, x, k):
        return np.array([CircleIntegrator.angle(x) + u[0] - CircleIntegrator.angle(xp)])
    def eval_g(self, u, x, k):
        return np.array([CircleIntegrator.angle(x) - self.initial_state[0]]) if k == 0 else np.empty(0)
    def eval_L(self, scale, u, x, k):
        return scale * .5 * (u @ u + (CircleIntegrator.angle(x) - self.target[k]) ** 2)
    def eval_rq(self, scale, u, x, k):
        return scale * np.concatenate([u, [CircleIntegrator.angle(x) - self.target[k]]])
    def apply_retraction_xk(self, x, dx, alpha, k):
        angle = CircleIntegrator.angle(x) + alpha * dx[0]
        return np.array([np.cos(angle), np.sin(angle)])


def test_state_dependent_manifold_matches_fixed_mode():
    stage = Solver(StageCircleIntegrator(), OPTIONS).solve()
    fixed = Solver(CircleIntegrator(), OPTIONS).solve()
    assert stage.success and fixed.success
    np.testing.assert_allclose(np.stack(stage.x), fixed.x, atol=1e-8)
    np.testing.assert_allclose(np.concatenate(stage.u), fixed.u[:, 0], atol=1e-8)


@pytest.mark.parametrize("cls", [CircleIntegrator, CircleControlIntegrator])
def test_manifold_states_and_controls(cls):
    result = Solver(cls(), OPTIONS).solve()
    assert result.success
    np.testing.assert_allclose(np.linalg.norm(result.x, axis=1), 1., atol=1e-12)
    np.testing.assert_allclose(np.arctan2(result.x[:, 1], result.x[:, 0]), [.75, 1., 1.25], atol=2e-6)
    if cls is CircleControlIntegrator:
        assert result.u.shape == (2, 2)
        np.testing.assert_allclose(np.linalg.norm(result.u, axis=1), 1., atol=1e-12)
        np.testing.assert_allclose(np.arctan2(result.u[:, 1], result.u[:, 0]), [.25, .25], atol=2e-6)


@pytest.mark.parametrize("cls", [StageIntegrator, FixedIntegrator])
def test_single_stage_has_no_dynamics(cls):
    problem = cls()
    problem.target = np.array([2.])
    problem.initial_state = np.array([2.])

    def forbidden(*args):
        pytest.fail("A single-stage OCP has no dynamics callbacks")

    prefix = "batched_" if cls is FixedIntegrator else ""
    for name in ("eval_b", "eval_BAt"):
        setattr(problem, prefix + name, forbidden)
    result = Solver(problem, OPTIONS).solve()
    assert result.success
    np.testing.assert_allclose(np.stack(result.x), [[2.]], atol=1e-7)
    if cls is FixedIntegrator:
        assert result.u.shape == (0, problem.get_nu())
    else:
        assert result.u[-1].shape == (0,)


def test_single_stage_terminal_cropping_precedes_initial_append():
    problem = TerminalOcp()
    problem.get_horizon_length = lambda: 1
    problem.get_ng_terminal = lambda: 0
    problem.get_ng_ineq_terminal = lambda: 0
    problem.batched_get_initial_xk = lambda: np.zeros((1, 2))
    problem.batched_get_initial_uk = lambda: np.zeros((1, 3))
    problem.batched_get_bounds = lambda: (np.zeros((1, 2)), np.ones((1, 2)))
    problem.batched_eval_g = lambda u, x: np.full((1, 2), 1e20)
    problem.batched_eval_Gt = lambda u, x: np.full((1, 5, 2), 1e20)
    problem.batched_eval_gineq = lambda u, x: np.full((1, 2), 1e20)
    problem.batched_eval_Gt_ineq = lambda u, x: np.full((1, 5, 2), 1e20)
    problem.batched_eval_rq = lambda scale, u, x: scale * np.concatenate([u, x], axis=1)
    problem.batched_eval_RSQt = lambda scale, *a: np.array([scale * np.eye(5)])
    solver = Solver(problem, OPTIONS)
    assert solver.dimensions["ng"] == (2,)
    result = solver.solve()
    assert result.success
    np.testing.assert_allclose(result.x, [[1., 2.]], atol=1e-8)
    assert result.u.shape == (0, 3)


def test_stagewise_terminal_controls_are_preserved():
    problem = StageIntegrator()
    problem.get_nu = lambda k: 1
    result = Solver(problem, OPTIONS).solve()
    assert result.success
    assert all(u.shape == (1,) for u in result.u)
    np.testing.assert_allclose(result.u[-1], [0.], atol=1e-7)


def test_print_options_do_not_leak_between_solvers(capfd):
    Solver(FixedIntegrator(), OPTIONS).solve()
    assert capfd.readouterr().out == ""
    Solver(FixedIntegrator()).solve()
    assert capfd.readouterr().out != ""
@pytest.mark.parametrize("acceleration_limit", [1., 100.])
def test_double_integrator_example(acceleration_limit):
    problem = DoubleIntegrator()
    problem.max_acceleration = acceleration_limit
    result = Solver(problem, OPTIONS).solve()
    assert result.success
    states = np.stack(result.x)
    controls = np.concatenate(result.u)
    assert result.u[-1].shape == (0,)
    np.testing.assert_allclose(states[0], problem.initial_state, atol=1e-9)
    np.testing.assert_allclose(
        states[1:, 0],
        states[:-1, 0] + problem.dt * states[:-1, 1] + .5 * problem.dt**2 * controls,
        atol=1e-8,
    )
    np.testing.assert_allclose(states[1:, 1], states[:-1, 1] + problem.dt * controls, atol=1e-8)
    assert np.max(np.abs(controls)) <= acceleration_limit + 1e-6
    np.testing.assert_allclose(states[-1], problem.target, atol=1e-3)

    if acceleration_limit == 1.:
        assert np.max(np.abs(controls)) == pytest.approx(1., abs=1e-6)
    else:
        # Eliminate states using the double integrator's closed-form impulse
        # response, then solve the unconstrained quadratic in all accelerations.
        n = problem.horizon - 1
        hessian = problem.control_weight * np.eye(n)
        rhs = np.zeros(n)
        for k in range(problem.horizon):
            response = np.zeros((2, n))
            response[0, :k] = problem.dt**2 * (k - np.arange(k) - .5)
            response[1, :k] = problem.dt
            free_state = problem.initial_state + np.array([
                k * problem.dt * problem.initial_state[1], 0.
            ])
            weight = problem.Q_terminal if k == n else problem.Q
            hessian += response.T @ weight @ response
            rhs += response.T @ weight @ (problem.target - free_state)
        reference = np.linalg.solve(hessian, rhs)
        assert np.max(np.abs(reference)) < acceleration_limit
        np.testing.assert_allclose(controls, reference, atol=2e-6)



class VaryingOcp(StageIntegrator):
    target = np.array([0., 0., 1.])
    A = (np.array([[1.], [2.]]), np.array([[1., 1.]]))
    B = (np.array([[1.], [-1.]]), np.array([[1.]]))
    def get_ng(self, k): return 0
    def eval_g(self, u, x, k): return np.empty(0)
    def get_nx(self, k): return (1, 2, 1)[k]
    def get_ng_ineq(self, k): return 0
    def eval_BAt(self, xp, u, x, k): return np.vstack([self.B[k].T, self.A[k].T])
    def eval_b(self, xp, u, x, k): return self.A[k] @ x + self.B[k] @ u - xp
    def eval_Gt(self, u, x, k): return np.empty((len(u) + len(x), 0))
    def eval_Gt_ineq(self, u, x, k): return np.empty((len(u) + len(x), 0))
    def eval_gineq(self, u, x, k): return np.empty(0)
    def get_bounds(self, k): return np.empty(0), np.empty(0)
    def get_initial_xk(self, k): return np.zeros(self.get_nx(k))
    def eval_L(self, scale, u, x, k):
        return scale * .5 * (u @ u + (x - self.target[k]) @ (x - self.target[k]))
    def eval_rq(self, scale, u, x, k):
        return scale * np.concatenate([u, x - self.target[k]])


def test_varying_dimensions_and_rectangular_jacobians():
    result = Solver(VaryingOcp(), OPTIONS).solve()
    assert result.success
    # Condense the entire problem to independent variables z=[x0,u0,u1].
    map_x0 = np.array([[1., 0., 0.]])
    map_u0 = np.array([[0., 1., 0.]])
    map_x1 = VaryingOcp.A[0] @ map_x0 + VaryingOcp.B[0] @ map_u0
    map_u1 = np.array([[0., 0., 1.]])
    map_x2 = VaryingOcp.A[1] @ map_x1 + VaryingOcp.B[1] @ map_u1
    residual_map = np.vstack([map_x0, map_u0, map_x1, map_u1, map_x2])
    target = np.array([0., 0., 0., 0., 0., 1.])
    z = np.linalg.solve(residual_map.T @ residual_map, residual_map.T @ target)
    for actual, mapping in zip(result.x, [map_x0, map_x1, map_x2]):
        np.testing.assert_allclose(actual, mapping @ z, atol=1e-7)
