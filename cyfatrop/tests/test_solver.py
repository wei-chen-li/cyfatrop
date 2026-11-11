"""Solver regression tests using concrete problems and the public Python API.
"""

import gc
import os
import sys
import tempfile
import unittest
import weakref
from collections import Counter
from contextlib import contextmanager

import numpy as np

from cyfatrop import OcpAbstract, Solver, Status, TrajOptAbstract


OPTIONS = {"tolerance": 1e-9, "max_iter": 100, "print_level": 0, "suppress_banner": True}


class IntegratorData:
    """Problem data and observations, populated by ordinary callback methods."""

    def __init__(self, target=(0., 0., 3.), initial_state=.75, control_limit=.25):
        self.target = np.array(target, dtype=float)
        self.initial_state = np.array([initial_state], dtype=float)
        self.control_limit = control_limit
        self.calls = Counter()
        self.snapshots = []
        self.events = []

    def observe(self, name, *arrays):
        self.calls[name] += 1
        for array in arrays:
            if array.flags.writeable:
                raise AssertionError(f"{name} received a writable input")
            self.snapshots.append((array, array.copy()))

    def intermediate(self, iteration, x, u, stats):
        for trajectory in (x, u):
            blocks = trajectory if isinstance(trajectory, tuple) else (trajectory,)
            for block in blocks:
                if not block.flags.owndata:
                    raise AssertionError("Intermediate trajectories must own their storage")
                self.observe("intermediate", block)
        self.events.append((iteration, x, u, stats, stats.copy()))

    @staticmethod
    def state_values(x):
        return x

    @staticmethod
    def control_values(u):
        return u


class StageIntegrator(IntegratorData, OcpAbstract):
    """Minimize squared tracking error and effort, with x_next = x + u."""

    def __init__(self, *args, terminal_controls=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.terminal_controls = terminal_controls

    def get_horizon_length(self): return len(self.target)
    def get_nx(self, k): return 1
    def get_nu(self, k): return int(k < len(self.target) - 1 or self.terminal_controls)
    def get_ng(self, k): return int(k == 0)
    def get_ng_ineq(self, k): return self.get_nu(k)

    def stage_inputs(self, name, u, x, k, *extra):
        np.testing.assert_equal(u.shape, (self.get_nu(k),))
        np.testing.assert_equal(x.shape, (self.get_nx(k),))
        self.observe(name, u, x, *extra)

    def eval_BAt(self, x_next, u, x, k):
        self.stage_inputs("eval_BAt", u, x, k, x_next)
        np.testing.assert_equal(x_next.shape, (self.get_nx(k + 1),))
        return np.ones((2, 1))

    def eval_b(self, x_next, u, x, k):
        self.stage_inputs("eval_b", u, x, k, x_next)
        return self.state_values(x) + u - self.state_values(x_next)

    def eval_RSQt(self, scale, u, x, lam_dyn, lam_eq, lam_ineq, k):
        self.stage_inputs("eval_RSQt", u, x, k, lam_dyn, lam_eq, lam_ineq)
        np.testing.assert_equal(lam_dyn.shape,
                                (self.get_nx_tangent(k + 1) if k + 1 < len(self.target) else 0,))
        np.testing.assert_equal(lam_eq.shape, (self.get_ng(k),))
        np.testing.assert_equal(lam_ineq.shape, (self.get_ng_ineq(k),))
        return scale * np.eye(self.get_nu(k) + self.get_nx_tangent(k))

    def eval_Gt(self, u, x, k):
        self.stage_inputs("eval_Gt", u, x, k)
        if k == 0:
            return np.vstack([np.zeros((len(u), 1)), np.ones((1, 1))])
        return np.empty((len(u) + 1, 0))

    def eval_g(self, u, x, k):
        self.stage_inputs("eval_g", u, x, k)
        return self.state_values(x) - self.initial_state if k == 0 else np.empty(0)

    def eval_Gt_ineq(self, u, x, k):
        self.stage_inputs("eval_Gt_ineq", u, x, k)
        return np.vstack([np.eye(len(u)), np.zeros((1, len(u)))])

    def eval_gineq(self, u, x, k):
        self.stage_inputs("eval_gineq", u, x, k)
        return u.copy()

    def eval_L(self, scale, u, x, k):
        self.stage_inputs("eval_L", u, x, k)
        error = self.state_values(x) - self.target[k]
        return scale * .5 * (u @ u + error @ error)

    def eval_rq(self, scale, u, x, k):
        self.stage_inputs("eval_rq", u, x, k)
        return scale * np.concatenate([u, self.state_values(x) - self.target[k]])

    def get_bounds(self, k):
        self.observe("get_bounds")
        limit = np.full(self.get_ng_ineq(k), self.control_limit)
        return -limit, limit

    def get_initial_xk(self, k):
        self.observe("get_initial_xk")
        return np.zeros(self.get_nx(k))

    def get_initial_uk(self, k):
        self.observe("get_initial_uk")
        return np.zeros(self.get_nu(k))

    def apply_retraction_xk(self, x, dx, alpha, k):
        self.observe("apply_retraction_xk", x, dx)
        return super().apply_retraction_xk(x, dx, alpha, k)

    def apply_retraction_uk(self, u, du, alpha, k):
        self.observe("apply_retraction_uk", u, du)
        return super().apply_retraction_uk(u, du, alpha, k)


class BatchedIntegrator(IntegratorData, TrajOptAbstract):
    """The same scalar problem through horizon-wide callbacks."""

    def get_horizon_length(self): return len(self.target)
    def get_nx(self): return 1
    def get_nu(self): return 1
    def get_ng(self): return 0
    def get_ng_ineq(self): return 1
    def get_ng_ineq_terminal(self): return 0

    def batch_inputs(self, name, u, x, dynamics=False, *extra):
        K = len(self.target)
        np.testing.assert_equal(u.shape, (K - 1, self.get_nu()))
        np.testing.assert_equal(x.shape, (K - 1 if dynamics else K, self.get_nx()))
        self.observe(name, u, x, *extra)

    def batched_eval_BAt(self, x_next, u, x):
        self.batch_inputs("eval_BAt", u, x, True, x_next)
        np.testing.assert_equal(x_next.shape, x.shape)
        return np.ones((len(x), 2, 1))

    def batched_eval_b(self, x_next, u, x):
        self.batch_inputs("eval_b", u, x, True, x_next)
        np.testing.assert_equal(x_next.shape, x.shape)
        return self.state_values(x) + self.control_values(u) - self.state_values(x_next)

    def batched_eval_RSQt(self, scale, u, x, lam_dyn, lam_eq, lam_ineq, lam_initial):
        self.batch_inputs("eval_RSQt", u, x, False, lam_dyn, lam_eq, lam_ineq, lam_initial)
        K, tx = len(x), self.get_nx_tangent()
        np.testing.assert_equal(lam_dyn.shape, (K - 1, tx))
        np.testing.assert_equal(lam_eq.shape, (K, self.get_ng()))
        np.testing.assert_equal(lam_ineq.shape, (K, self.get_ng_ineq()))
        np.testing.assert_equal(lam_initial.shape, (tx,))
        np.testing.assert_array_equal(lam_ineq[-1], 0.)
        return np.broadcast_to(scale * np.eye(2), (K, 2, 2)).copy()

    def batched_eval_Gt(self, u, x):
        self.batch_inputs("eval_Gt", u, x)
        return np.empty((len(x), 2, 0))

    def batched_eval_g(self, u, x):
        self.batch_inputs("eval_g", u, x)
        return np.empty((len(x), 0))

    def batched_eval_Gt_ineq(self, u, x):
        self.batch_inputs("eval_Gt_ineq", u, x)
        return np.broadcast_to(np.array([[1.], [0.]]), (len(x), 2, 1))

    def batched_eval_gineq(self, u, x):
        self.batch_inputs("eval_gineq", u, x)
        return np.pad(self.control_values(u), ((0, 1), (0, 0)))

    def batched_eval_L(self, scale, u, x):
        self.batch_inputs("eval_L", u, x)
        values = (self.state_values(x)[:, 0] - self.target) ** 2
        values[:-1] += self.control_values(u)[:, 0] ** 2
        return scale * .5 * values

    def batched_eval_rq(self, scale, u, x):
        self.batch_inputs("eval_rq", u, x)
        return scale * np.concatenate([
            np.pad(self.control_values(u), ((0, 1), (0, 0))),
            self.state_values(x) - self.target[:, None],
        ], axis=1)

    def batched_get_bounds(self):
        self.observe("get_bounds")
        limit = np.full((len(self.target), 1), self.control_limit)
        return -limit, limit

    def batched_get_initial_xk(self):
        self.observe("get_initial_xk")
        return np.zeros((len(self.target), self.get_nx()))

    def batched_get_initial_uk(self):
        self.observe("get_initial_uk")
        return np.zeros((len(self.target) - 1, self.get_nu()))

    def batched_apply_retraction_xk(self, x, dx, alpha):
        np.testing.assert_equal(dx.shape, (len(self.target), self.get_nx_tangent()))
        self.observe("apply_retraction_xk", x, dx)
        return super().batched_apply_retraction_xk(x, dx, alpha)

    def batched_apply_retraction_uk(self, u, du, alpha):
        np.testing.assert_equal(u.shape, (len(self.target) - 1, self.get_nu()))
        np.testing.assert_equal(du.shape, (len(self.target) - 1, self.get_nu_tangent()))
        self.observe("apply_retraction_uk", u, du)
        return super().batched_apply_retraction_uk(u, du, alpha)

    def eval_g_initial(self, x0):
        self.observe("eval_g_initial", x0)
        return self.state_values(x0) - self.initial_state

    def eval_Gt_initial(self, x0):
        self.observe("eval_Gt_initial", x0)
        return np.eye(1)


def angle(value):
    return np.arctan2(value[..., 1:2], value[..., :1])


def circle(value):
    return np.concatenate([np.cos(value), np.sin(value)], axis=-1)


class StageCircleIntegrator(StageIntegrator):
    """S1 states: two primal coordinates, one tangent coordinate."""

    state_values = staticmethod(angle)

    def get_nx(self, k): return 2
    def get_nx_tangent(self, k): return 1

    def get_initial_xk(self, k):
        self.observe("get_initial_xk")
        return np.array([1., 0.])

    def apply_retraction_xk(self, x, dx, alpha, k):
        self.observe("apply_retraction_xk", x, dx)
        return circle(angle(x) + alpha * dx)


class BatchedCircleIntegrator(BatchedIntegrator):
    state_values = staticmethod(angle)

    def get_nx(self): return 2
    def get_nx_tangent(self): return 1

    def batched_get_initial_xk(self):
        self.observe("get_initial_xk")
        return np.broadcast_to([1., 0.], (len(self.target), 2))

    def batched_apply_retraction_xk(self, x, dx, alpha):
        np.testing.assert_equal(dx.shape, (len(self.target), 1))
        self.observe("apply_retraction_xk", x, dx)
        return circle(angle(x) + alpha * dx)


class CircleControlIntegrator(BatchedCircleIntegrator):
    control_values = staticmethod(angle)

    def get_nu(self): return 2
    def get_nu_tangent(self): return 1

    def batched_get_initial_uk(self):
        self.observe("get_initial_uk")
        return np.broadcast_to([1., 0.], (len(self.target) - 1, 2))

    def batched_apply_retraction_uk(self, u, du, alpha):
        np.testing.assert_equal(u.shape, (len(self.target) - 1, 2))
        np.testing.assert_equal(du.shape, (len(self.target) - 1, 1))
        self.observe("apply_retraction_uk", u, du)
        return circle(angle(u) + alpha * du)


class UnconstrainedStageIntegrator(StageIntegrator):
    def get_ng_ineq(self, k): return 0

    def eval_gineq(self, u, x, k):
        self.stage_inputs("eval_gineq", u, x, k)
        return np.empty(0)

    def eval_Gt_ineq(self, u, x, k):
        self.stage_inputs("eval_Gt_ineq", u, x, k)
        return np.empty((len(u) + 1, 0))


class UnconstrainedBatchIntegrator(BatchedIntegrator):
    def get_ng_ineq(self): return 0

    def batched_eval_gineq(self, u, x):
        self.batch_inputs("eval_gineq", u, x)
        return np.empty((len(x), 0))

    def batched_eval_Gt_ineq(self, u, x):
        self.batch_inputs("eval_Gt_ineq", u, x)
        return np.empty((len(x), 2, 0))

    def batched_get_bounds(self):
        self.observe("get_bounds")
        return np.empty((len(self.target), 0)), np.empty((len(self.target), 0))


class NonlinearInitial(BatchedIntegrator):
    """Minimize x²/2 subject to x²=4, starting near the positive root."""

    def __init__(self):
        super().__init__(target=(0.,))
        self.curvatures = []

    def batched_get_initial_xk(self):
        self.observe("get_initial_xk")
        return np.array([[1.5]])

    def eval_g_initial(self, x0):
        self.observe("eval_g_initial", x0)
        return x0 ** 2 - 4.

    def eval_Gt_initial(self, x0):
        self.observe("eval_Gt_initial", x0)
        return np.array([[2. * x0[0]]])

    def batched_eval_RSQt(self, scale, u, x, ld, le, li, initial):
        result = super().batched_eval_RSQt(scale, u, x, ld, le, li, initial)
        result[0, 1, 1] += 2. * initial[0]
        self.curvatures.append((scale, initial.copy(), result.copy()))
        return result


class TerminalProblem(IntegratorData, TrajOptAbstract):
    """Ordinary and initial equalities coexist with reduced terminal counts."""

    def __init__(self, horizon=2, terminal_equalities=1, terminal_inequalities=1):
        super().__init__(target=np.zeros(horizon))
        self.terminal_equalities = terminal_equalities
        self.terminal_inequalities = terminal_inequalities
        self.multipliers = []

    def get_horizon_length(self): return len(self.target)
    def get_nx(self): return 2
    def get_nu(self): return 3
    def get_ng(self): return 2
    def get_ng_ineq(self): return 2
    def get_ng_terminal(self): return self.terminal_equalities
    def get_ng_ineq_terminal(self): return self.terminal_inequalities
    def batched_get_initial_xk(self): return np.zeros((len(self.target), 2))
    def batched_get_initial_uk(self): return np.zeros((len(self.target) - 1, 3))

    def batched_eval_BAt(self, x_next, u, x):
        return np.array([[[1., 0.], [0., 1.], [0., 0.], [1., 0.], [0., 1.]]])

    def batched_eval_b(self, x_next, u, x): return x + u[:, :2] - x_next

    def batched_eval_g(self, u, x):
        result = np.full((len(x), 2), 1e20)
        if len(x) > 1:
            result[0] = u[0, 1:]
        result[-1, 0] = x[-1, 0] - 3.
        return result

    def batched_eval_Gt(self, u, x):
        result = np.zeros((len(x), 5, 2))
        result[0, 1, 0] = result[0, 2, 1] = 1.
        result[-1, :3, :] = result[-1, 3:, 1] = 1e20
        result[-1, 3, 0] = 1.
        return result

    def batched_eval_gineq(self, u, x):
        result = np.full((len(x), 2), 1e20)
        if len(x) > 1:
            result[0] = [u[0, 0], x[0, 1]]
        result[-1, 0] = x[-1, 1]
        return result

    def batched_eval_Gt_ineq(self, u, x):
        result = np.zeros((len(x), 5, 2))
        result[0, 0, 0] = result[0, 4, 1] = result[-1, 4, 0] = 1.
        result[-1, :3, :] = result[-1, 3:, 1] = 1e20
        return result

    def batched_get_bounds(self):
        lower = np.broadcast_to([-5., 1.], (len(self.target), 2)).copy()
        upper = np.full_like(lower, 5.)
        # Discarded terminal bounds deliberately contradict each other.
        lower[-1] = [1., 999.]
        upper[-1] = [5., -999.]
        return lower, upper

    def batched_eval_L(self, scale, u, x):
        result = np.sum(x * x, axis=1)
        result[:-1] += np.sum(u * u, axis=1)
        return scale * .5 * result

    def batched_eval_rq(self, scale, u, x):
        result = scale * np.concatenate([np.pad(u, ((0, 1), (0, 0))), x], axis=1)
        result[-1, :3] = 1e20
        return result

    def batched_eval_RSQt(self, scale, u, x, ld, le, li, initial):
        np.testing.assert_equal(u.shape, (len(x) - 1, 3))
        np.testing.assert_equal(ld.shape, (len(x) - 1, 2))
        np.testing.assert_equal(le.shape, (len(x), 2))
        np.testing.assert_equal(li.shape, (len(x), 2))
        np.testing.assert_equal(initial.shape, (2,))
        np.testing.assert_array_equal(le[-1, self.terminal_equalities:], 0.)
        np.testing.assert_array_equal(li[-1, self.terminal_inequalities:], 0.)
        self.observe("eval_RSQt", u, x, ld, le, li, initial)
        self.multipliers.append((ld.copy(), le.copy(), li.copy(), initial.copy()))
        result = np.broadcast_to(scale * np.eye(5), (len(x), 5, 5)).copy()
        result[-1, :3, :] = result[-1, :, :3] = 1e20
        return result

    def eval_g_initial(self, x0): return x0 - [1., 2.]
    def eval_Gt_initial(self, x0): return np.eye(2)


class VaryingProblem(StageIntegrator):
    """An unconstrained quadratic problem with state dimensions 1, 2, 1."""

    A = (np.array([[1.], [2.]]), np.array([[1., 1.]]))
    B = (np.array([[1.], [-1.]]), np.array([[1.]]))

    def __init__(self): super().__init__(target=(0., 0., 1.))
    def get_nx(self, k): return (1, 2, 1)[k]
    def get_ng(self, k): return 0
    def get_ng_ineq(self, k): return 0
    def eval_BAt(self, x_next, u, x, k): return np.vstack([self.B[k].T, self.A[k].T])
    def eval_b(self, x_next, u, x, k): return self.A[k] @ x + self.B[k] @ u - x_next
    def eval_g(self, u, x, k): return np.empty(0)
    def eval_Gt(self, u, x, k): return np.empty((len(u) + len(x), 0))
    def eval_gineq(self, u, x, k): return np.empty(0)
    def eval_Gt_ineq(self, u, x, k): return np.empty((len(u) + len(x), 0))


# Invalid problems use explicit overrides, rather than replacing bound methods.
class InvalidStageDimension(StageIntegrator):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def get_nx(self, k): return self.value


class InvalidBatchDimension(BatchedIntegrator):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def get_nx(self): return self.value


class InvalidHorizon(BatchedIntegrator):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def get_horizon_length(self): return self.value


class TooManyEqualities(BatchedIntegrator):
    def get_ng(self): return 2
    def get_ng_terminal(self): return 0


class BothInterfaces(BatchedIntegrator, StageIntegrator):
    pass


class InvalidStageInitial(StageIntegrator):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def get_initial_xk(self, k): return self.value


class InvalidBatchInitial(BatchedIntegrator):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def batched_get_initial_xk(self): return self.value


class InvalidStageDynamics(StageIntegrator):
    def eval_b(self, x_next, u, x, k): return np.zeros((1, 1))


class InvalidBatchGradient(BatchedIntegrator):
    def batched_eval_rq(self, scale, u, x): return np.zeros((2, 3))


class InvalidInitialResidual(BatchedIntegrator):
    def eval_g_initial(self, x0): return np.zeros((1, 1))


class InvalidInitialJacobian(BatchedIntegrator):
    def eval_Gt_initial(self, x0): return np.zeros(1)


class PaddedControlGuess(BatchedIntegrator):
    def batched_get_initial_uk(self): return np.zeros((len(self.target), 1))


class PaddedControlRetraction(BatchedIntegrator):
    def batched_apply_retraction_uk(self, u, du, alpha):
        return np.zeros((len(self.target), 1))


class InvalidStageBounds(StageIntegrator):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind

    def get_bounds(self, k):
        if self.kind == "pair":
            return np.zeros(self.get_ng_ineq(k))
        lower = np.full(self.get_ng_ineq(k), np.nan if self.kind == "nan" else 2.)
        return lower, np.ones_like(lower)


class InvalidBatchBounds(BatchedIntegrator):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind

    def batched_get_bounds(self):
        if self.kind == "pair":
            return np.zeros((len(self.target), 1))
        lower = np.full((len(self.target), 1), np.nan if self.kind == "nan" else 2.)
        return lower, np.ones_like(lower)


class StridedStageProblem(StageIntegrator):
    def eval_BAt(self, x_next, u, x, k):
        return np.ones((2, 2), dtype=np.int32)[:, ::2]

    def eval_RSQt(self, scale, u, x, ld, le, li, k):
        return (scale * np.eye(2 * (len(u) + 1)))[::2, ::2]


class StridedBatchProblem(BatchedIntegrator):
    def batched_eval_BAt(self, x_next, u, x):
        return np.ones((len(x), 2, 2), dtype=np.int32)[:, :, ::2]

    def batched_eval_RSQt(self, scale, u, x, ld, le, li, initial):
        return np.broadcast_to(scale * np.eye(4), (len(x), 4, 4))[:, ::2, ::2]


class CallbackFailure(Exception):
    pass


class FailingStageProblem(StageIntegrator):
    def __init__(self):
        super().__init__()
        self.fail = True

    def eval_b(self, x_next, u, x, k):
        if self.fail:
            raise CallbackFailure("dynamics failed")
        return super().eval_b(x_next, u, x, k)


class FailingBatchProblem(BatchedIntegrator):
    def __init__(self):
        super().__init__()
        self.fail = True

    def batched_eval_b(self, x_next, u, x):
        if self.fail:
            raise CallbackFailure("dynamics failed")
        return super().batched_eval_b(x_next, u, x)


class IntermediateBehavior:
    """Lifecycle controls implemented in an ordinary inherited callback."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fail = False
        self.recurse = False
        self.clear_stats = False
        self.solver = None

    def intermediate(self, iteration, x, u, stats):
        super().intermediate(iteration, x, u, stats)
        if self.fail:
            raise CallbackFailure("intermediate failed")
        if self.recurse:
            self.solver.solve()
        if self.clear_stats:
            stats.clear()
        return False


class ObservedStageProblem(IntermediateBehavior, StageIntegrator):
    pass


class ObservedBatchProblem(IntermediateBehavior, BatchedIntegrator):
    pass


class InheritedStageProblem(ObservedStageProblem):
    pass


class InheritedBatchProblem(ObservedBatchProblem):
    pass


@contextmanager
def native_output():
    """Capture native stdout/stderr as well as Python output; always restore FDs."""
    sys.stdout.flush()
    sys.stderr.flush()
    with tempfile.TemporaryFile(mode="w+b") as captured:
        saved_stdout, saved_stderr = os.dup(1), os.dup(2)
        try:
            os.dup2(captured.fileno(), 1)
            os.dup2(captured.fileno(), 2)
            yield captured
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.close(saved_stdout)
            os.close(saved_stderr)


def output_text(captured):
    captured.seek(0)
    return captured.read().decode("utf-8")


class NumericalSolveTests(unittest.TestCase):
    def test_bounded_integrator_and_interface_agreement(self):
        solutions = []
        for problem_type in (StageIntegrator, BatchedIntegrator):
            with self.subTest(interface=problem_type.__name__):
                result = Solver(problem_type(), OPTIONS).solve()
                self.assertTrue(result.success)
                self.assertEqual(result.status, Status.Success)
                np.testing.assert_allclose(np.stack(result.x).ravel(), [.75, 1., 1.25], atol=2e-6)
                controls = np.concatenate(result.u) if isinstance(result.u, tuple) else result.u.ravel()
                np.testing.assert_allclose(controls, [.25, .25], atol=2e-6)
                self.assertAlmostEqual(result.objective, 2.375, delta=2e-6)
                self.assertGreater(result.stats["iterations"], 0)
                self.assertGreater(result.stats["solve_time"], 0)
                solutions.append(result)
        np.testing.assert_allclose(np.stack(solutions[0].x), solutions[1].x, atol=1e-9)
        np.testing.assert_allclose(np.stack(solutions[0].u[:-1]), solutions[1].u, atol=1e-9)

    def test_unconstrained_quadratic_against_condensed_numpy_solution(self):
        target = np.array([0., 1., -1., 2., 3.])
        initial = .4
        # Eliminate states: x = initial + response @ u. The least-squares
        # normal equations are independent of either callback implementation.
        response = np.tril(np.ones((len(target), len(target) - 1)), k=-1)
        expected_u = np.linalg.solve(np.eye(len(target) - 1) + response.T @ response,
                                     response.T @ (target - initial))
        expected_x = initial + response @ expected_u
        expected_cost = .5 * (expected_u @ expected_u + np.sum((expected_x - target) ** 2))
        for cls in (UnconstrainedStageIntegrator, UnconstrainedBatchIntegrator):
            with self.subTest(interface=cls.__name__):
                result = Solver(cls(target=target, initial_state=initial), OPTIONS).solve()
                self.assertTrue(result.success)
                controls = np.concatenate(result.u) if isinstance(result.u, tuple) else result.u.ravel()
                np.testing.assert_allclose(controls, expected_u, atol=1e-8)
                np.testing.assert_allclose(np.stack(result.x).ravel(), expected_x, atol=1e-8)
                self.assertAlmostEqual(result.objective, expected_cost, delta=1e-8)

    def test_rectangular_dynamics_and_varying_dimensions(self):
        problem = VaryingProblem()
        result = Solver(problem, OPTIONS).solve()
        self.assertTrue(result.success)
        self.assertEqual([v.shape for v in result.x], [(1,), (2,), (1,)])
        # Independent variables z = [x0, u0, u1].
        x0, u0, u1 = np.eye(3)[0:1], np.eye(3)[1:2], np.eye(3)[2:3]
        x1 = problem.A[0] @ x0 + problem.B[0] @ u0
        x2 = problem.A[1] @ x1 + problem.B[1] @ u1
        residual = np.vstack([x0, u0, x1, u1, x2])
        target = np.array([0., 0., 0., 0., 0., 1.])
        z = np.linalg.solve(residual.T @ residual, residual.T @ target)
        for actual, mapping in zip(result.x, (x0, x1, x2)):
            np.testing.assert_allclose(actual, mapping @ z, atol=1e-8)
        np.testing.assert_allclose(np.concatenate(result.u), z[1:], atol=1e-8)

    def test_terminal_constraints_and_discarded_blocks(self):
        for count, expected_x, expected_u in (
            (1, [[1., 2.], [3., 2.]], [[2., 0., 0.]]),
            (0, [[1., 2.], [.5, 2.]], [[-.5, 0., 0.]]),
        ):
            with self.subTest(terminal_count=count):
                problem = TerminalProblem(terminal_equalities=count, terminal_inequalities=count)
                solver = Solver(problem, OPTIONS)
                self.assertEqual(solver.dimensions["nu"], (3, 0))
                self.assertEqual(solver.dimensions["ng"], (4, count))
                self.assertEqual(solver.dimensions["ng_ineq"], (2, count))
                result = solver.solve()
                self.assertTrue(result.success)
                np.testing.assert_allclose(result.x, expected_x, atol=2e-6)
                np.testing.assert_allclose(result.u, expected_u, atol=2e-6)
                self.assertTrue(problem.multipliers)
                if count:
                    self.assertAlmostEqual(result.objective, 11., delta=2e-6)

    def test_single_stage_terminal_cropping_and_initial_constraints(self):
        problem = TerminalProblem(horizon=1, terminal_equalities=0, terminal_inequalities=0)
        solver = Solver(problem, OPTIONS)
        self.assertEqual(solver.dimensions["ng"], (2,))
        result = solver.solve()
        self.assertTrue(result.success)
        np.testing.assert_allclose(result.x, [[1., 2.]], atol=1e-8)
        self.assertEqual(result.u.shape, (0, 3))

    def test_multiplier_batches_follow_constraint_order(self):
        problem = TerminalProblem()
        self.assertTrue(Solver(problem, OPTIONS).solve().success)
        dynamics, ordinary, inequalities, initial = problem.multipliers[-1]
        # Stationarity at x0=[1,2], x1=[3,2], u0=[2,0,0] determines
        # these multipliers. Every inequality is strictly inside its bounds.
        np.testing.assert_allclose(dynamics, [[-2., 2.]], atol=2e-6)
        np.testing.assert_allclose(ordinary, [[-2., 0.], [-5., 0.]], atol=2e-6)
        np.testing.assert_allclose(inequalities, 0., atol=2e-6)
        np.testing.assert_allclose(initial, [1., -4.], atol=2e-6)

    def test_stagewise_terminal_controls(self):
        result = Solver(StageIntegrator(terminal_controls=True), OPTIONS).solve()
        self.assertTrue(result.success)
        self.assertEqual([u.shape for u in result.u], [(1,), (1,), (1,)])
        np.testing.assert_allclose(result.u[-1], 0., atol=1e-8)

    def test_nonlinear_initial_curvature(self):
        problem = NonlinearInitial()
        result = Solver(problem, OPTIONS).solve()
        self.assertTrue(result.success)
        np.testing.assert_allclose(result.x, [[2.]], atol=1e-8)
        self.assertEqual(result.u.shape, (0, 1))
        self.assertAlmostEqual(result.objective, 2., delta=1e-8)
        self.assertTrue(problem.curvatures)
        # For x²/2 + lambda*(x²-4), stationarity at x=2 gives lambda=-1/2
        # and zero Lagrangian curvature, including the initial constraint.
        _, initial, hessian = problem.curvatures[-1]
        np.testing.assert_allclose(initial, [-.5], atol=2e-6)
        self.assertAlmostEqual(hessian[0, 1, 1], 0., delta=2e-6)

    def test_manifold_states_and_controls(self):
        for cls in (StageCircleIntegrator, BatchedCircleIntegrator, CircleControlIntegrator):
            with self.subTest(problem=cls.__name__):
                result = Solver(cls(), OPTIONS).solve()
                self.assertTrue(result.success)
                states = np.stack(result.x)
                np.testing.assert_allclose(np.linalg.norm(states, axis=1), 1., atol=1e-12)
                np.testing.assert_allclose(angle(states).ravel(), [.75, 1., 1.25], atol=2e-6)
                controls = np.concatenate(result.u) if isinstance(result.u, tuple) else result.u
                if cls is CircleControlIntegrator:
                    np.testing.assert_allclose(np.linalg.norm(controls, axis=1), 1., atol=1e-12)
                    controls = angle(controls)
                np.testing.assert_allclose(controls.ravel(), [.25, .25], atol=2e-6)


class CallbackContractTests(unittest.TestCase):
    def test_constructor_does_not_evaluate_callbacks(self):
        for cls in (StageIntegrator, BatchedIntegrator):
            with self.subTest(interface=cls.__name__):
                problem = cls()
                Solver(problem, OPTIONS)
                self.assertEqual(problem.calls, Counter())
                self.assertEqual(problem.events, [])

    def test_callback_shapes_ordering_and_single_stage(self):
        for cls in (StageIntegrator, BatchedIntegrator, CircleControlIntegrator):
            for horizon in (1, 3):
                with self.subTest(problem=cls.__name__, horizon=horizon):
                    problem = cls(target=(0., 0., 3.)[:horizon])
                    result = Solver(problem, OPTIONS).solve()
                    self.assertTrue(result.success)
                    # Callback methods themselves check all inputs and retain
                    # them, including the K-1 dynamics multiplier batches.
                    for name in ("eval_RSQt", "eval_rq", "eval_Gt", "eval_Gt_ineq",
                                 "eval_g", "eval_gineq", "eval_L",
                                 "get_initial_xk", "get_initial_uk"):
                        self.assertGreater(problem.calls[name], 0, name)
                    if horizon > 1:
                        self.assertGreater(problem.calls["get_bounds"], 0)
                    for name in ("eval_BAt", "eval_b"):
                        if horizon == 1:
                            self.assertEqual(problem.calls[name], 0)
                        else:
                            self.assertGreater(problem.calls[name], 0)
                    if isinstance(problem, TrajOptAbstract):
                        self.assertEqual(result.x.shape, (horizon, problem.get_nx()))
                        self.assertEqual(result.u.shape, (horizon - 1, problem.get_nu()))
                        self.assertGreater(problem.calls["apply_retraction_uk"], 0)
                    else:
                        self.assertIsInstance(result.x, tuple)
                        self.assertIsInstance(result.u, tuple)
                        self.assertEqual(result.u[-1].shape, (0,))

    def test_input_and_intermediate_snapshots_survive_solver_reuse(self):
        for cls in (StageIntegrator, BatchedIntegrator, CircleControlIntegrator):
            with self.subTest(problem=cls.__name__):
                problem = cls()
                solver = Solver(problem, OPTIONS)
                first = solver.solve()
                self.assertTrue(first.success)
                self.assertEqual([e[0] for e in problem.events], list(range(first.stats["iterations"])))
                self.assertEqual(len({id(e[3]) for e in problem.events}), len(problem.events))
                for _, x, u, stats, saved in problem.events:
                    self.assertEqual(set(stats), {"objective", "inf_pr", "inf_du", "mu"})
                    self.assertTrue(all(isinstance(v, float) and np.isfinite(v) for v in stats.values()))
                    self.assertGreaterEqual(stats["inf_pr"], 0.)
                    self.assertGreaterEqual(stats["inf_du"], 0.)
                    self.assertGreater(stats["mu"], 0.)
                    states = problem.state_values(np.stack(x)).ravel()
                    if isinstance(u, tuple):
                        controls = np.concatenate(u)
                    else:
                        controls = problem.control_values(u).ravel()
                    expected = .5 * (controls @ controls + np.sum((states - problem.target) ** 2))
                    self.assertAlmostEqual(stats["objective"], expected, delta=1e-12)
                    self.assertEqual(stats, saved)
                first_events = problem.events.copy()
                first_x = np.stack(first.x).copy()
                problem.events.clear()
                self.assertTrue(solver.solve().success)
                gc.collect()
                np.testing.assert_array_equal(np.stack(first.x), first_x)
                for retained, saved in problem.snapshots:
                    self.assertFalse(retained.flags.writeable)
                    np.testing.assert_array_equal(retained, saved)
                    if retained.size:
                        with self.assertRaises(ValueError):
                            retained.flat[0] = 99.
                for _, _, _, stats, saved in first_events + problem.events:
                    self.assertEqual(stats, saved)

    def test_strided_and_integer_callback_outputs(self):
        for cls in (StridedStageProblem, StridedBatchProblem):
            with self.subTest(problem=cls.__name__):
                result = Solver(cls(), OPTIONS).solve()
                self.assertTrue(result.success)
                np.testing.assert_allclose(np.stack(result.x).ravel(), [.75, 1., 1.25], atol=2e-6)

    def test_default_retractions_require_matching_dimensions(self):
        for base, prefix in ((OcpAbstract, ""), (TrajOptAbstract, "batched_")):
            for variable in ("x", "u"):
                with self.subTest(base=base.__name__, variable=variable):
                    value = np.zeros((3, 2) if prefix else (2,))
                    step = np.zeros((3, 1) if prefix else (1,))
                    method = getattr(base, prefix + f"apply_retraction_{variable}k")
                    args = (None, value, step, 1.) if prefix else (None, value, step, 1., 0)
                    with self.assertRaisesRegex(ValueError, "custom retraction"):
                        method(*args)


class ValidationTests(unittest.TestCase):
    def test_abstract_and_unsupported_problem_types(self):
        for base in (OcpAbstract, TrajOptAbstract):
            with self.subTest(base=base.__name__), self.assertRaises(TypeError):
                base()
        for problem in (object(), BothInterfaces()):
            with self.subTest(problem=type(problem).__name__):
                with self.assertRaisesRegex(TypeError, "exactly one"):
                    Solver(problem)

    def test_invalid_dimensions_and_horizons(self):
        for cls in (InvalidStageDimension, InvalidBatchDimension, InvalidHorizon):
            for value in (-1, 1.5, True, 2**31):
                with self.subTest(problem=cls.__name__, value=value):
                    with self.assertRaisesRegex((TypeError, ValueError), "get_nx|get_horizon_length"):
                        Solver(cls(value))
        with self.assertRaisesRegex(ValueError, "get_horizon_length"):
            Solver(InvalidHorizon(0))

    def test_terminal_counts_and_initial_equality_count(self):
        for value in (-1, 3, 1.5, True):
            for dimension in ("equalities", "inequalities"):
                with self.subTest(dimension=dimension, value=value):
                    problem = (TerminalProblem(terminal_equalities=value) if dimension == "equalities"
                               else TerminalProblem(terminal_inequalities=value))
                    with self.assertRaisesRegex((TypeError, ValueError), "get_ng.*terminal"):
                        Solver(problem)
        with self.assertRaisesRegex(ValueError, "Stage 0.*equality count"):
            Solver(TooManyEqualities())

    def test_callback_shape_errors_identify_the_callback(self):
        cases = (
            (InvalidStageDynamics, "eval_b"), (InvalidBatchGradient, "batched_eval_rq"),
            (InvalidInitialResidual, "eval_g_initial"), (InvalidInitialJacobian, "eval_Gt_initial"),
        )
        for cls, name in cases:
            with self.subTest(problem=cls.__name__):
                with self.assertRaisesRegex(ValueError, name + ".*expected shape"):
                    Solver(cls(), OPTIONS).solve()

    def test_control_outputs_must_exclude_terminal_row(self):
        for cls in (PaddedControlGuess, PaddedControlRetraction):
            for horizon in (1, 3):
                with self.subTest(problem=cls.__name__, horizon=horizon):
                    problem = cls(target=(0., 0., 3.)[:horizon])
                    with self.assertRaisesRegex(ValueError, rf".*expected shape \({horizon - 1}, 1\)"):
                        Solver(problem, OPTIONS).solve()

    def test_initial_outputs_must_be_real_arrays(self):
        for cls in (InvalidStageInitial, InvalidBatchInitial):
            for kind in ("list", "complex", "string"):
                with self.subTest(problem=cls.__name__, kind=kind):
                    shape = (1,) if cls is InvalidStageInitial else (3, 1)
                    value = ([0.] if kind == "list" else
                             np.full(shape, 1j if kind == "complex" else "bad"))
                    with self.assertRaisesRegex(TypeError, "get_initial_xk"):
                        Solver(cls(value), OPTIONS).solve()

    def test_invalid_bounds(self):
        for cls in (InvalidStageBounds, InvalidBatchBounds):
            for kind in ("nan", "inverted", "pair"):
                with self.subTest(problem=cls.__name__, kind=kind):
                    with self.assertRaisesRegex((TypeError, ValueError), "get_bounds"):
                        Solver(cls(kind), OPTIONS).solve()

    def test_unknown_solver_option(self):
        with self.assertRaisesRegex(ValueError, "Unknown Fatrop option"):
            Solver(BatchedIntegrator(), {"typo_option": 1}).solve()


class SolverLifecycleTests(unittest.TestCase):
    def test_repeated_solves_refresh_numerical_data(self):
        for cls in (StageIntegrator, BatchedIntegrator):
            with self.subTest(interface=cls.__name__):
                problem = cls()
                solver = Solver(problem, OPTIONS)
                first = solver.solve()
                saved = np.stack(first.x).copy()
                problem.target += 2.
                problem.initial_state += 2.
                second = solver.solve()
                self.assertTrue(first.success and second.success)
                np.testing.assert_allclose(np.stack(second.x), saved + 2., atol=2e-6)
                np.testing.assert_array_equal(np.stack(first.x), saved)

    def test_iteration_limit_and_no_intermediate_calls(self):
        for cls in (StageIntegrator, BatchedIntegrator):
            with self.subTest(interface=cls.__name__):
                problem = cls()
                result = Solver(problem, {**OPTIONS, "max_iter": 0}).solve()
                self.assertFalse(result.success)
                self.assertEqual(result.status, Status.MaxIterExceeded)
                self.assertEqual(result.stats["iterations"], 0)
                self.assertEqual(problem.events, [])

    def test_numerical_callback_exception_and_retry(self):
        for cls in (FailingStageProblem, FailingBatchProblem):
            with self.subTest(interface=cls.__name__):
                problem = cls()
                solver = Solver(problem, OPTIONS)
                with self.assertRaisesRegex(CallbackFailure, "dynamics failed"):
                    solver.solve()
                problem.fail = False
                result = solver.solve()
                self.assertTrue(result.success)
                self.assertAlmostEqual(result.objective, 2.375, delta=2e-6)

    def test_intermediate_exception_and_retry(self):
        for cls in (ObservedStageProblem, ObservedBatchProblem):
            with self.subTest(interface=cls.__name__):
                problem = cls()
                problem.fail = True
                solver = Solver(problem, OPTIONS)
                with self.assertRaisesRegex(CallbackFailure, "intermediate failed"):
                    solver.solve()
                self.assertEqual([e[0] for e in problem.events], [0])
                problem.fail = False
                problem.events.clear()
                result = solver.solve()
                self.assertTrue(result.success)
                self.assertEqual([e[0] for e in problem.events], list(range(result.stats["iterations"])))

    def test_recursive_solve_rejection_and_recovery(self):
        for cls in (ObservedStageProblem, ObservedBatchProblem):
            with self.subTest(interface=cls.__name__):
                problem = cls()
                problem.solver = Solver(problem, OPTIONS)
                problem.recurse = True
                with self.assertRaisesRegex(RuntimeError, "recursively"):
                    problem.solver.solve()
                problem.recurse = False
                self.assertTrue(problem.solver.solve().success)

    def test_inherited_intermediate_return_and_stats_mutation(self):
        for cls in (InheritedStageProblem, InheritedBatchProblem):
            with self.subTest(interface=cls.__name__):
                problem = cls()
                problem.clear_stats = True
                result = Solver(problem, OPTIONS).solve()
                self.assertTrue(result.success)
                self.assertEqual(len(problem.events), result.stats["iterations"])
                self.assertGreater(len(problem.events), 1)
                self.assertTrue(all(event[3] == {} for event in problem.events))
                self.assertAlmostEqual(result.objective, 2.375, delta=2e-6)

    def test_solver_problem_cycle_is_collectable(self):
        def create_cycle(cls):
            problem = cls()
            solver = Solver(problem, OPTIONS)
            problem.solver = solver
            self.assertTrue(solver.solve().success)
            return weakref.ref(problem), weakref.ref(solver)

        for cls in (ObservedStageProblem, ObservedBatchProblem):
            with self.subTest(interface=cls.__name__):
                refs = create_cycle(cls)
                gc.collect()
                self.assertTrue(all(ref() is None for ref in refs))

    def test_native_log_matches_intermediate_statistics(self):
        for cls in (StageIntegrator, BatchedIntegrator):
            with self.subTest(interface=cls.__name__):
                problem = cls()
                with native_output() as captured:
                    result = Solver(problem, {**OPTIONS, "print_level": 5}).solve()
                    rows = [line.split() for line in output_text(captured).splitlines()]
                rows = [row for row in rows if row and row[0].isdigit()]
                self.assertTrue(result.success)
                self.assertEqual(len(rows), len(problem.events) + 1)
                self.assertEqual(int(rows[-1][0]), result.stats["iterations"])
                for (iteration, _, _, stats, _), row in zip(problem.events, rows):
                    self.assertEqual(int(row[0]), iteration)
                    np.testing.assert_allclose(float(row[1]), stats["objective"], rtol=1e-8, atol=1e-12)
                    np.testing.assert_allclose(float(row[2]), stats["inf_pr"], rtol=.006, atol=1e-30)
                    np.testing.assert_allclose(float(row[3]), stats["inf_du"], rtol=.006, atol=1e-30)
                    self.assertAlmostEqual(float(row[4]), np.log10(stats["mu"]), delta=.051)

    def test_print_options_are_local_to_each_solver(self):
        with native_output() as captured:
            Solver(BatchedIntegrator(), OPTIONS).solve()
            quiet = output_text(captured)
        self.assertEqual(quiet, "")
        with native_output() as captured:
            Solver(BatchedIntegrator()).solve()
            verbose = output_text(captured)
        self.assertNotEqual(verbose, "")


