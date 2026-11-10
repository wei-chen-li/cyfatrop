"""LQR tracking for a double integrator with acceleration bounds.

State x = [position, velocity], input u = [acceleration]. The system starts
at rest at position zero and tracks position one with zero velocity.
"""

import numpy as np

from cyfatrop import Solver


class DoubleIntegrator:
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

    # Stage-dependent dimensions allow an equality only at the initial stage.
    def get_horizon_length(self): return self.horizon
    def get_nx(self, k): return 2
    def get_nu(self, k): return 1 if k < self.horizon - 1 else 0
    def get_ng(self, k): return 2 if k == 0 else 0
    def get_ng_ineq(self, k): return self.get_nu(k)

    def eval_BAt(self, x_next, u, x, k):
        return np.vstack([self.B.T, self.A.T])

    def eval_b(self, x_next, u, x, k):
        # p_next = p + dt*v + 0.5*dt**2*a; v_next = v + dt*a.
        return self.A @ x + self.B @ u - x_next

    def _state_weight(self, k):
        return self.Q_terminal if k == self.horizon - 1 else self.Q

    def eval_RSQt(self, scale, u, x, lam_dyn, lam_eq, lam_ineq, k):
        # All constraints are linear, so only the quadratic cost contributes.
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

    def eval_L(self, scale, u, x, k):
        error = x - self.target
        return float(scale * 0.5 * (
            self.control_weight * (u @ u) + error @ self._state_weight(k) @ error
        ))

    def eval_rq(self, scale, u, x, k):
        return scale * np.concatenate([
            self.control_weight * u, self._state_weight(k) @ (x - self.target)
        ])

    def get_bounds(self, k):
        n = self.get_ng_ineq(k)
        return np.full(n, -self.max_acceleration), np.full(n, self.max_acceleration)

    def get_initial_xk(self, k): return self.initial_state.copy()
    def get_initial_uk(self, k): return np.zeros(self.get_nu(k))


if __name__ == "__main__":
    problem = DoubleIntegrator()
    result = Solver(problem, {"tolerance": 1e-9}).solve()
    if not result.success:
        raise RuntimeError(f"Fatrop failed: {result.status}")
    print(result.status, "objective:", result.objective)
    print(" time   position   velocity   acceleration")
    for k, (x, u) in enumerate(zip(result.x, result.u)):
        acceleration = f"{u[0]:12.5f}" if len(u) else "           -"
        print(f"{k * problem.dt:5.2f} {x[0]:10.5f} {x[1]:10.5f} {acceleration}")
