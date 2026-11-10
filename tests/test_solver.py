import gc
from collections import Counter

import numpy as np
import pytest

from cyfatrop import Solver, Status, FATROP_VERSION
from cyfatrop._solver import _CALLBACKS, _Problem
from examples.lqr import DoubleIntegrator


OPTIONS = {"tolerance": 1e-9, "max_iter": 100, "print_level": 0, "suppress_banner": True}


class Integrator:
    """Scalar regression fixture, independent of the runnable LQR example."""
    target = np.array([0., 0., 3.])

    def get_horizon_length(self): return len(self.target)
    def get_nx(self): return 1
    def get_nu(self): return 1
    def get_ng(self): return 0
    def get_ng_ineq(self): return 1
    def get_ng_ineq_terminal(self): return 0
    def eval_BAt(self, x_next, u, x, k): return np.ones((2, 1))
    def eval_b(self, x_next, u, x, k): return x + u - x_next
    def eval_RSQt(self, scale, u, x, lam_dyn, lam_eq, lam_ineq, k):
        return scale * np.eye(2)
    def eval_Gt(self, u, x, k): return np.empty((2, 0))
    def eval_g(self, u, x, k): return np.empty(0)
    def eval_Gt_ineq(self, u, x, k): return np.array([[1.], [0.]])
    def eval_gineq(self, u, x, k): return u.copy()
    def eval_L(self, scale, u, x, k):
        return float(scale * .5 * (u[0] ** 2 + (x[0] - self.target[k]) ** 2))
    def eval_rq(self, scale, u, x, k):
        return scale * np.array([u[0], x[0] - self.target[k]])
    def get_bounds(self, k): return np.array([-.25]), np.array([.25])
    def get_initial_xk(self, k): return np.zeros(1)
    def get_initial_uk(self, k): return np.zeros(1)


def batchify(problem, names=None):
    """Replace selected callbacks with batch-only implementations and count calls."""
    problem.batch_counts = Counter()
    problem.saved_inputs = []
    if names is None:
        names = [n for n in _CALLBACKS if hasattr(problem, n)]
    for name in names:
        original = getattr(problem, name)

        def callback(*args, name=name, original=original):
            problem.batch_counts[name] += 1
            count = problem.get_horizon_length() - (name in ("eval_BAt", "eval_b"))
            for v in args:
                if isinstance(v, np.ndarray):
                    assert not v.flags.writeable
                    assert v.shape[0] == count
                    problem.saved_inputs.append((v, v.copy()))
            rows = [original(*(v[k] if isinstance(v, np.ndarray) else v for v in args), k)
                    for k in range(count)]
            if name == "get_bounds":
                return tuple(np.stack([r[j] for r in rows]) for j in range(2))
            return np.stack(rows)

        setattr(problem, "batched_" + name, callback)
        setattr(problem, name, None)
    return problem


class VectorizedIntegrator(Integrator):
    def batched_eval_BAt(self, xp, u, x):
        return np.ones((len(x), 2, 1))

    def batched_eval_b(self, xp, u, x): return x + u - xp

    def batched_eval_RSQt(self, scale, u, x, ld, le, li):
        assert np.all(u[-1] == 0)
        assert np.all(ld[-1] == 0)
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
        return (np.full((self.get_horizon_length(), 1), -.25),
                np.full((self.get_horizon_length(), 1), .25))

    def batched_get_initial_xk(self): return np.zeros((self.get_horizon_length(), 1))
    def batched_get_initial_uk(self): return np.zeros((self.get_horizon_length(), 1))
    def batched_apply_retraction_xk(self, x, dx, alpha): return x + alpha * dx
    def batched_apply_retraction_uk(self, u, du, alpha): return u + alpha * du


class StageIntegrator:
    target = Integrator.target

    def get_horizon_length(self): return len(self.target)
    def get_nx(self, k): return 1
    def get_nu(self, k): return 1 if k < len(self.target) - 1 else 0
    def get_ng(self, k): return 0
    def get_ng_ineq(self, k): return self.get_nu(k)
    def eval_BAt(self, xp, u, x, k): return np.ones((2, 1))
    def eval_b(self, xp, u, x, k): return x + u - xp
    def eval_RSQt(self, scale, u, x, ld, le, li, k): return scale * np.eye(len(u) + len(x))
    def eval_Gt(self, u, x, k): return np.empty((len(u) + 1, 0))
    def eval_g(self, u, x, k): return np.empty(0)
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


@pytest.mark.parametrize("kind", ["stage", "fixed", "batch", "mixed", "vectorized"])
def test_analytic_bounded_ocp(kind):
    if kind == "stage":
        problem = StageIntegrator()
    elif kind == "vectorized":
        problem = VectorizedIntegrator()
    else:
        problem = Integrator()
        if kind in ("batch", "mixed"):
            batchify(problem, None if kind == "batch" else
                     ["eval_b", "eval_L", "eval_Gt_ineq", "get_initial_xk"])
    solver = Solver(problem, OPTIONS)
    result = solver.solve()
    assert result.success and result.status == Status.Success
    np.testing.assert_allclose(np.stack(result.x).ravel(), [.75, 1., 1.25], atol=2e-6)
    if kind == "stage":
        assert isinstance(result.x, tuple) and result.u[-1].shape == (0,)
        controls = np.concatenate(result.u)
    else:
        assert result.x.flags.owndata and result.u.flags.owndata
        assert result.u[-1, 0] == 0
        controls = result.u[:-1].ravel()
    np.testing.assert_allclose(controls, [.25, .25], atol=2e-6)
    assert result.objective == pytest.approx(2.375, abs=2e-6)
    assert result.stats["iterations"] > 0
    assert result.stats["solve_time"] > 0
    assert result.stats["eval_jacobian_time"] >= 0


def test_native_batch_cache_avoids_stage_dispatch(monkeypatch):
    problem = batchify(Integrator())
    original = _Problem._evaluate

    def evaluate(self, name, k, args):
        assert name not in self.batched, "Native code failed to consume its batch cache"
        return original(self, name, k, args)

    monkeypatch.setattr(_Problem, "_evaluate", evaluate)
    Solver(problem, OPTIONS).solve()
    counts = problem.batch_counts
    assert counts["eval_BAt"] == counts["eval_Gt"] == counts["eval_Gt_ineq"] > 0
    assert counts["eval_b"] == counts["eval_g"] == counts["eval_gineq"] > 0
    assert counts["get_initial_xk"] == counts["get_initial_uk"] == 1
    assert counts["eval_RSQt"] > 0 and counts["eval_rq"] > 0
    gc.collect()
    for retained, saved in problem.saved_inputs:
        np.testing.assert_array_equal(retained, saved)


def test_repeated_solves_refresh_primal_values():
    problem = batchify(Integrator())
    solver = Solver(problem, OPTIONS)
    first = solver.solve()
    old_x = first.x.copy()
    old_count = problem.batch_counts["eval_L"]
    problem.target = np.array([2., 2., 5.])
    second = solver.solve()
    np.testing.assert_allclose(second.x, first.x + 2., atol=2e-6)
    np.testing.assert_array_equal(first.x, old_x)
    assert problem.batch_counts["eval_L"] > old_count


class TerminalOcp:
    """Five independent equalities fix all five native primal variables."""
    def get_horizon_length(self): return 2
    def get_nx(self): return 2
    def get_nu(self): return 1
    def get_ng(self): return 2
    def get_ng_ineq(self): return 2
    def get_ng_terminal(self): return 1
    def get_ng_ineq_terminal(self): return 1
    def get_initial_xk(self, k): return np.zeros(2)
    def get_initial_uk(self, k): return np.array([1e20 if k == 1 else 0.])

    def eval_BAt(self, xp, u, x, k):
        return np.array([[1., 0.], [1., 0.], [0., 1.]])
    def eval_b(self, xp, u, x, k): return x + np.array([u[0], 0.]) - xp

    def eval_g(self, u, x, k):
        assert u.shape == (1,)
        if k == 1:
            assert u[0] == 0
            return np.array([x[0] - 3., 1e20])
        return x - [1., 2.]

    def eval_Gt(self, u, x, k):
        if k == 1:
            return np.array([[1e20, 1e20], [1., 1e20], [0., 1e20]])
        return np.array([[0., 0.], [1., 0.], [0., 1.]])

    def eval_gineq(self, u, x, k):
        return np.array([x[1], 1e20]) if k else np.array([u[0], x[1]])

    def eval_Gt_ineq(self, u, x, k):
        if k:
            return np.array([[1e20, 1e20], [0., 1e20], [1., 1e20]])
        return np.array([[1., 0.], [0., 0.], [0., 1.]])

    def get_bounds(self, k):
        if k:
            # Even inconsistent discarded bounds must have no effect.
            return np.array([1., 999.]), np.array([5., -999.])
        return np.array([-5., 1.]), np.array([5., 5.])

    def eval_L(self, scale, u, x, k): return scale * .5 * (u @ u + x @ x)

    def eval_rq(self, scale, u, x, k):
        return scale * np.concatenate([np.array([1e20]) if k else u, x])

    def eval_RSQt(self, scale, u, x, ld, le, li, k):
        if k:
            assert np.all(ld == 0) and le.shape == li.shape == (2,)
            assert le[1] == li[1] == 0
            result = scale * np.eye(3)
            result[0, :] = result[:, 0] = 1e20
            return result
        return scale * np.eye(3)

    def apply_retraction_uk(self, u, du, alpha, k):
        if k:
            assert np.all(u == 0) and np.all(du == 0)
            return np.array([1e20])
        return u + alpha * du


@pytest.mark.parametrize("batch", [False, True])
def test_terminal_counts_crop_and_discard_control_blocks(batch):
    problem = TerminalOcp()
    if batch:
        batchify(problem)
    solver = Solver(problem, OPTIONS)
    assert solver.dimensions["nu"] == (1, 0)
    assert solver.dimensions["ng"] == solver.dimensions["ng_ineq"] == (2, 1)
    result = solver.solve()
    assert result.success
    np.testing.assert_allclose(result.x, [[1., 2.], [3., 2.]], atol=2e-6)
    np.testing.assert_allclose(result.u, [[2.], [0.]], atol=2e-6)
    assert result.objective == pytest.approx(11., abs=2e-6)


class VaryingOcp(StageIntegrator):
    target = np.array([0., 0., 1.])
    A = (np.array([[1.], [2.]]), np.array([[1., 1.]]))
    B = (np.array([[1.], [-1.]]), np.array([[1.]]))
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


def test_constructor_does_not_execute_numerical_callbacks():
    problem = Integrator()
    problem.get_bounds = lambda k: pytest.fail("Premature numerical callback")
    Solver(problem)


@pytest.mark.parametrize("name", ["eval_Gt", "eval_g", "get_initial_uk", "get_bounds"])
def test_required_callbacks_including_empty_outputs(name):
    problem = Integrator()
    setattr(problem, name, None)
    with pytest.raises(TypeError, match=name):
        Solver(problem)


def test_stagewise_cannot_use_batch_only_callback():
    problem = StageIntegrator()
    problem.eval_g = None
    problem.batched_eval_g = lambda u, x: np.empty((len(x), 0))
    with pytest.raises(TypeError, match="eval_g"):
        Solver(problem)


@pytest.mark.parametrize("getter", [lambda k=0: 1, lambda *args: 1])
def test_reject_ambiguous_dimension_signatures(getter):
    problem = Integrator()
    problem.get_nx = getter
    with pytest.raises(TypeError, match="signature"):
        Solver(problem)


def test_reject_mixed_dimension_signatures():
    problem = Integrator()
    problem.get_nx = lambda k: 1
    with pytest.raises(TypeError, match="consistently"):
        Solver(problem)


@pytest.mark.parametrize("value", [-1, 3, 1.5, True])
@pytest.mark.parametrize("name", ["get_ng_terminal", "get_ng_ineq_terminal"])
def test_invalid_terminal_counts(value, name):
    problem = TerminalOcp()
    setattr(problem, name, lambda: value)
    with pytest.raises((TypeError, ValueError), match=name):
        Solver(problem)


def test_terminal_defaults_and_zero_counts():
    problem = TerminalOcp()
    problem.get_ng_terminal = problem.get_ng_ineq_terminal = None
    adapter = _Problem(problem)
    assert adapter.dimensions["ng"] == adapter.dimensions["ng_ineq"] == [2, 2]
    problem.get_ng_terminal = lambda: 0
    problem.get_ng_ineq_terminal = lambda: 0
    adapter = _Problem(problem)
    assert adapter.dimensions["ng"] == adapter.dimensions["ng_ineq"] == [2, 0]


@pytest.mark.parametrize("batch", [False, True])
def test_zero_terminal_constraint_counts_solve(batch):
    problem = TerminalOcp()
    problem.get_ng_terminal = problem.get_ng_ineq_terminal = lambda: 0
    if batch:
        batchify(problem)
    result = Solver(problem, OPTIONS).solve()
    assert result.success
    np.testing.assert_allclose(result.x, [[1., 2.], [.5, 2.]], atol=2e-6)
    np.testing.assert_allclose(result.u, [[-.5], [0.]], atol=2e-6)
    assert result.objective == pytest.approx(4.75, abs=2e-6)


def test_wrong_shape_reports_callback_and_stage():
    problem = Integrator()
    problem.eval_b = lambda xp, u, x, k: np.zeros((1, 1))
    with pytest.raises(ValueError, match=r"eval_b at stage 0: expected shape \(1,\)"):
        Solver(problem).solve()


def test_wrong_batch_shape():
    problem = Integrator()
    problem.batched_eval_rq = lambda scale, u, x: np.zeros((2, len(x)))
    with pytest.raises(ValueError, match=r"batched_eval_rq.*expected shape \(3, 2\)"):
        Solver(problem).solve()


@pytest.mark.parametrize("bad", [[0.], np.array([1j]), np.array(["bad"])])
def test_reject_non_numeric_or_non_array_returns(bad):
    problem = Integrator()
    problem.get_initial_xk = lambda k: bad
    with pytest.raises(TypeError, match="get_initial_xk"):
        Solver(problem).solve()


def test_strided_and_integer_callback_arrays():
    problem = Integrator()
    problem.eval_BAt = lambda xp, u, x, k: np.ones((2, 2), dtype=np.int32)[:, ::2]
    problem.eval_RSQt = lambda scale, u, x, ld, le, li, k: (scale * np.eye(4))[::2, ::2]
    assert Solver(problem, OPTIONS).solve().success


def test_exception_propagates_and_solver_can_retry():
    class CallbackError(Exception): pass
    problem = Integrator()
    problem.fail = True

    def initial(k):
        if problem.fail:
            raise CallbackError("user callback failed")
        return np.zeros(1)

    problem.get_initial_xk = initial
    solver = Solver(problem, OPTIONS)
    with pytest.raises(CallbackError, match="user callback failed"):
        solver.solve()
    problem.fail = False
    assert solver.solve().success


def test_batch_exception_clears_cache_for_retry():
    problem = VectorizedIntegrator()
    original = problem.batched_eval_b
    problem.fail = True

    def residual(*args):
        if problem.fail:
            raise LookupError("batch failed")
        return original(*args)

    problem.batched_eval_b = residual
    solver = Solver(problem, OPTIONS)
    with pytest.raises(LookupError, match="batch failed"):
        solver.solve()
    problem.fail = False
    assert solver.solve().success


def test_unknown_option_and_iteration_limit():
    with pytest.raises(ValueError, match="Unknown Fatrop option"):
        Solver(Integrator(), {"typo_option": 1}).solve()
    result = Solver(Integrator(), {"max_iter": 0}).solve()
    assert result.status == Status.MaxIterExceeded and not result.success
    assert result.x.shape == (3, 1)
    assert FATROP_VERSION == "1.2.0"


def test_manifold_dimension_requires_retraction():
    problem = Integrator()
    problem.get_nx = lambda: 2
    problem.get_nx_tangent = lambda: 1
    with pytest.raises(ValueError, match="apply_retraction_xk"):
        Solver(problem)


class CircleIntegrator(Integrator):
    """S1 represented by a unit vector, with one-dimensional tangent steps."""
    def get_nx(self): return 2
    def get_nx_tangent(self): return 1
    def get_initial_xk(self, k): return np.array([1., 0.])

    @staticmethod
    def angle(x): return np.arctan2(x[1], x[0])

    def eval_b(self, xp, u, x, k):
        angle = self.angle(x) + u[0] - self.angle(xp)
        return np.array([np.arctan2(np.sin(angle), np.cos(angle))])

    def eval_L(self, scale, u, x, k):
        return scale * .5 * (u[0] ** 2 + (self.angle(x) - self.target[k]) ** 2)

    def eval_rq(self, scale, u, x, k):
        return scale * np.array([u[0], self.angle(x) - self.target[k]])

    def apply_retraction_xk(self, x, dx, alpha, k):
        angle = self.angle(x) + alpha * dx[0]
        return np.array([np.cos(angle), np.sin(angle)])


@pytest.mark.parametrize("batch", [False, True])
def test_manifold_solve_with_unequal_primal_and_tangent_dimensions(batch):
    problem = CircleIntegrator()
    if batch:
        batchify(problem)
    result = Solver(problem, OPTIONS).solve()
    assert result.success
    np.testing.assert_allclose(np.linalg.norm(result.x, axis=1), 1., atol=1e-12)
    np.testing.assert_allclose(np.arctan2(result.x[:, 1], result.x[:, 0]),
                               [.75, 1., 1.25], atol=2e-6)


class CircleControlIntegrator(CircleIntegrator):
    def get_nu(self): return 2
    def get_nu_tangent(self): return 1
    def get_initial_uk(self, k): return np.array([1., 0.])
    def eval_b(self, xp, u, x, k):
        return np.array([self.angle(x) + self.angle(u) - self.angle(xp)])
    def eval_gineq(self, u, x, k): return np.array([self.angle(u)])
    def eval_L(self, scale, u, x, k):
        return scale * .5 * (self.angle(u) ** 2 + (self.angle(x) - self.target[k]) ** 2)
    def eval_rq(self, scale, u, x, k):
        return scale * np.array([self.angle(u), self.angle(x) - self.target[k]])
    def apply_retraction_uk(self, u, du, alpha, k):
        angle = self.angle(u) + alpha * du[0]
        return np.array([np.cos(angle), np.sin(angle)])


@pytest.mark.parametrize("batch", [False, True])
def test_manifold_controls_with_unequal_primal_and_tangent_dimensions(batch):
    problem = CircleControlIntegrator()
    if batch:
        batchify(problem)
    result = Solver(problem, OPTIONS).solve()
    assert result.success
    np.testing.assert_allclose(np.linalg.norm(result.u[:-1], axis=1), 1., atol=1e-12)
    np.testing.assert_allclose(np.arctan2(result.u[:-1, 1], result.u[:-1, 0]), [.25, .25], atol=2e-6)
    np.testing.assert_array_equal(result.u[-1], [0., 0.])


@pytest.mark.parametrize("batch", [False, True])
def test_single_stage_has_no_dynamics(batch):
    problem = Integrator()
    problem.target = np.array([2.])

    def forbidden(*args):
        pytest.fail("A single-stage OCP has no dynamics callbacks")

    problem.eval_b = problem.eval_BAt = forbidden
    if batch:
        batchify(problem)
    result = Solver(problem, OPTIONS).solve()
    assert result.success
    np.testing.assert_allclose(result.x, [[2.]], atol=1e-7)
    np.testing.assert_array_equal(result.u, [[0.]])


def test_stagewise_terminal_controls_are_preserved():
    problem = StageIntegrator()
    problem.get_nu = lambda k: 1
    result = Solver(problem, OPTIONS).solve()
    assert result.success
    assert all(u.shape == (1,) for u in result.u)
    np.testing.assert_allclose(result.u[-1], [0.], atol=1e-7)


def test_print_options_do_not_leak_between_solvers(capfd):
    Solver(Integrator(), OPTIONS).solve()
    assert capfd.readouterr().out == ""
    Solver(Integrator()).solve()
    assert capfd.readouterr().out != ""
