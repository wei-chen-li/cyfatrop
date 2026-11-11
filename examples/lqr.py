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

    def _R(self, k):
        return self.control_weight * np.eye(self.get_nu(k))

    def _Q(self, k):
        return self.Q_terminal if k == self.horizon - 1 else self.Q

    def get_horizon_length(self): return self.horizon
    def get_nx(self, k): return 2
    def get_nu(self, k): return 1 if k < self.horizon - 1 else 0
    def get_ng(self, k): return 2 if k == 0 else 0
    def get_ng_ineq(self, k): return self.get_nu(k)

    def eval_BAt(self, x_next, u, x, k):
        return np.hstack([self.B, self.A]).T

    def eval_b(self, x_next, u, x, k):
        return self.A @ x + self.B @ u - x_next

    def eval_RSQt(self, scale, u, x, lam_dyn, lam_eq, lam_ineq, k):
        nu, nx = self.get_nu(k), self.get_nx(k)
        hessian = np.zeros((nu + nx, nu + nx))
        hessian[:nu, :nu] = self._R(k)
        hessian[nu:, nu:] = self._Q(k)
        return scale * hessian

    def eval_Gt(self, u, x, k):
        nu, nx = self.get_nu(k), self.get_nx(k)
        if k == 0:
            return np.hstack([np.zeros((nx, nu)), np.eye(nx)]).T
        return np.empty((0, nu + nx)).T

    def eval_g(self, u, x, k):
        return x - self.initial_state if k == 0 else np.empty(0)

    def eval_Gt_ineq(self, u, x, k):
        nu, nx = self.get_nu(k), self.get_nx(k)
        return np.hstack([np.eye(nu), np.zeros((nu, nx))]).T

    def eval_gineq(self, u, x, k):
        return u.copy()

    def eval_L(self, scale, u, x, k):
        error = x - self.target
        return scale * 0.5 * (u @ self._R(k) @ u + error @ self._Q(k) @ error)

    def eval_rq(self, scale, u, x, k):
        error = x - self.target
        return scale * np.concatenate([self._R(k) @ u, self._Q(k) @ error])

    def get_bounds(self, k):
        ng_ineq = self.get_ng_ineq(k)
        return (
            np.full(ng_ineq, -self.max_acceleration),
            np.full(ng_ineq, self.max_acceleration),
        )

    def get_initial_xk(self, k): return self.initial_state.copy()
    def get_initial_uk(self, k): return np.zeros(self.get_nu(k))


class DoubleIntegrator2(TrajOptAbstract):
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

    def _R(self):
        return np.broadcast_to(
            self.control_weight * np.eye(1),
            (self.horizon-1, 1, 1)
        )

    def _Q(self):
        Q = np.broadcast_to(self.Q, (self.horizon, 2, 2)).copy()
        Q[-1] = self.Q_terminal
        return Q

    def get_horizon_length(self): return self.horizon
    def get_nx(self): return 2
    def get_nu(self): return 1
    def get_ng(self): return 0
    def get_ng_ineq(self): return self.get_nu()
    def get_ng_ineq_terminal(self): return 0

    def batched_eval_BAt(self, x_next, u, x):
        return np.broadcast_to(
            np.hstack([self.B, self.A]).T,
            (len(x), 3, 2)
        )

    def batched_eval_b(self, x_next, u, x):
        return (
            np.einsum("ij,bj->bi", self.A, x)
            + np.einsum("ij,bj->bi", self.B, u)
            - x_next
        )

    def batched_eval_RSQt(self, scale, u, x, lam_dyn, lam_eq, lam_ineq, lam_initial):
        result = np.zeros((len(x), 3, 3))
        result[:-1, :1, :1] = scale * self._R()
        result[:, 1:, 1:] = scale * self._Q()
        return result

    def batched_eval_Gt(self, u, x): return np.empty((len(x), 3, 0))
    def batched_eval_g(self, u, x): return np.empty((len(x), 0))

    def batched_eval_Gt_ineq(self, u, x):
        return np.broadcast_to(np.array([[1., 0., 0.]]).T, (len(x), 3, 1))

    def batched_eval_gineq(self, u, x):
        values = np.zeros((len(x), 1))
        values[:-1] = u
        return values

    def batched_eval_rq(self, scale, u, x):
        result = np.zeros((len(x), 3))
        result[:-1, :1] = np.einsum("bij,bj->bi", self._R(), u)
        result[:, 1:] = np.einsum("bij,bj->bi", self._Q(), x - self.target)
        return scale * result

    def batched_eval_L(self, scale, u, x):
        error = x - self.target
        values = np.einsum("bi,bij,bj->b", error, self._Q(), error)
        values[:-1] += np.einsum("bi,bij,bj->b", u, self._R(), u)
        return scale * .5 * values

    def batched_get_bounds(self):
        limit = np.full((self.horizon, 1), self.max_acceleration)
        return -limit, limit

    def batched_get_initial_xk(self):
        return np.broadcast_to(self.initial_state, (self.horizon, 2))

    def batched_get_initial_uk(self):
        return np.zeros((self.horizon-1, 1))

    def eval_g_initial(self, x0): return x0 - self.initial_state
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
    print("Unbatched and batched callback problem give the same solution.\n")

    print(" time   position   velocity   acceleration")
    for k, (x, u) in enumerate(zip(result.x, result.u)):
        acceleration = f"{u[0]:12.5f}" if len(u) else "           -"
        print(f"{k * problem.dt:5.2f} {x[0]:10.5f} {x[1]:10.5f} {acceleration}")
