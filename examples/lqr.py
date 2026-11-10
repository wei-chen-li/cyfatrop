"""LQR tracking for a double integrator with acceleration bounds.

State x = [position, velocity], input u = [acceleration]. The system starts
at rest at position zero and tracks position one with zero velocity.
The example solves both callback interfaces and checks exact equality.
"""

import numpy as np

from cyfatrop import OcpAbstract, Solver, TrajOptAbstract


class DoubleIntegrator(OcpAbstract):
    def __init__(self, horizon=41, dt=0.1):
        self.horizon = horizon
        self.dt = dt
        self.A = np.array([[1., dt], [0., 1.]])
        self.B = np.array([[0.5 * dt**2], [dt]])
        self.initial_state = np.array([0., 0.])
        self.target = np.array([1., 0.])
        self.Q = np.diag([1., 0.1])
        self.Q_terminal = np.diag([10., 1.])
        self.control_weight = 0.01
        self.max_acceleration = 1.

    def get_horizon_length(self): return self.horizon
    def get_nx(self, k): return 2
    def get_nu(self, k): return 1 if k < self.horizon - 1 else 0
    def get_ng(self, k): return 2 if k == 0 else 0
    def get_ng_ineq(self, k): return self.get_nu(k)

    def eval_BAt(self, x_next, u, x, k):
        return np.vstack([self.B.T, self.A.T])

    def eval_b(self, x_next, u, x, k):
        return self._dynamics(u, x) - x_next

    def _state_weight(self, k):
        return self.Q_terminal if k == self.horizon - 1 else self.Q

    def eval_RSQt(self, scale, u, x, lam_dyn, lam_eq, lam_ineq, k):
        n = self.get_nu(k)
        hessian = np.zeros((n + 2, n + 2))
        hessian[:n, :n] = self.control_weight * np.eye(n)
        hessian[n:, n:] = self._state_weight(k)
        return scale * hessian

    def eval_Gt(self, u, x, k):
        if k == 0:
            return np.vstack([np.zeros((self.get_nu(k), 2)), np.eye(2)])
        return np.empty((self.get_nu(k) + 2, 0))

    def eval_g(self, u, x, k):
        return x - self.initial_state if k == 0 else np.empty(0)

    def eval_Gt_ineq(self, u, x, k):
        n = self.get_nu(k)
        return np.vstack([np.eye(n), np.zeros((2, n))])

    def eval_gineq(self, u, x, k): return u.copy()

    def _dynamics(self, u, x):
        return (np.einsum("ij,...j->...i", self.A, x)
                + np.einsum("ij,...j->...i", self.B, u))

    def _objective(self, scale, u, x, weight):
        error = x - self.target
        return scale * .5 * (
            self.control_weight * np.sum(u * u, axis=-1)
            + np.einsum("...i,...ij,...j->...", error, weight, error)
        )

    def _objective_gradient(self, scale, u, x, weight):
        state_gradient = np.einsum("...ij,...j->...i", weight, x - self.target)
        return scale * np.concatenate([self.control_weight * u, state_gradient], axis=-1)

    def eval_L(self, scale, u, x, k):
        return float(self._objective(scale, u, x, self._state_weight(k)))

    def eval_rq(self, scale, u, x, k):
        return self._objective_gradient(scale, u, x, self._state_weight(k))

    def get_bounds(self, k):
        n = self.get_ng_ineq(k)
        return np.full(n, -self.max_acceleration), np.full(n, self.max_acceleration)

    def get_initial_xk(self, k): return self.initial_state.copy()
    def get_initial_uk(self, k): return np.zeros(self.get_nu(k))


class DoubleIntegrator2(TrajOptAbstract):
    def __init__(self, horizon=41, dt=0.1):
        self.model = DoubleIntegrator(horizon, dt)

    def get_horizon_length(self): return self.model.horizon
    def get_nx(self): return 2
    def get_nu(self): return 1
    def get_ng(self): return 0
    def get_ng_ineq(self): return 1
    def get_ng_ineq_terminal(self): return 0

    def _weights(self):
        weights = np.broadcast_to(self.model.Q, (self.model.horizon, 2, 2)).copy()
        weights[-1] = self.model.Q_terminal
        return weights

    def batched_eval_BAt(self, x_next, u, x):
        return np.broadcast_to(np.vstack([self.model.B.T, self.model.A.T]), (len(x), 3, 2))

    def batched_eval_b(self, x_next, u, x):
        return self.model._dynamics(u, x) - x_next

    def batched_eval_RSQt(self, scale, u, x, lam_dyn, lam_eq, lam_ineq, lam_initial):
        # Dynamics, ordinary constraints, and the initial condition are linear.
        result = np.zeros((len(x), 3, 3))
        result[:, 0, 0] = scale * self.model.control_weight
        result[:, 1:, 1:] = scale * self._weights()
        return result

    def batched_eval_Gt(self, u, x): return np.empty((len(x), 3, 0))
    def batched_eval_g(self, u, x): return np.empty((len(x), 0))

    def batched_eval_Gt_ineq(self, u, x):
        return np.broadcast_to(np.array([[1.], [0.], [0.]]), (len(x), 3, 1))

    def batched_eval_gineq(self, u, x): return u.copy()

    def batched_eval_rq(self, scale, u, x):
        return self.model._objective_gradient(scale, u, x, self._weights())

    def batched_eval_L(self, scale, u, x):
        return self.model._objective(scale, u, x, self._weights())

    def batched_get_bounds(self):
        limit = np.full((self.model.horizon, 1), self.model.max_acceleration)
        return -limit, limit

    def batched_get_initial_xk(self):
        return np.broadcast_to(self.model.initial_state, (self.model.horizon, 2))

    def batched_get_initial_uk(self): return np.zeros((self.model.horizon, 1))
    def eval_g_initial(self, x0): return x0 - self.model.initial_state
    def eval_Gt_initial(self, x0): return np.eye(2)


if __name__ == "__main__":
    problem = DoubleIntegrator()
    problem2 = DoubleIntegrator2(problem.horizon, problem.dt)

    options = {"tolerance": 1e-9, "print_level": 0, "suppress_banner": True}
    result = Solver(problem, options).solve()
    result2 = Solver(problem2, options).solve()

    if not result.success or not result2.success:
        raise RuntimeError(f"Fatrop failed: {result.status}, {result2.status}")

    np.testing.assert_allclose(np.stack(result.x), result2.x)
    np.testing.assert_allclose(np.stack(result.u[:-1]), result2.u)
    np.testing.assert_allclose(result.objective, result2.objective)

    print(" time   position   velocity   acceleration")
    for k, (x, u) in enumerate(zip(result.x, result.u)):
        acceleration = f"{u[0]:12.5f}" if len(u) else "           -"
        print(f"{k * problem.dt:5.2f} {x[0]:10.5f} {x[1]:10.5f} {acceleration}")
