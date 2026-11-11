from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


class OcpAbstract(ABC):
    """OCP with stage index k on each callback.

    Dimensions and callback shapes are those of stage k, including the terminal
    stage. In particular, get_nu(K - 1) may be zero for terminal stage not
    having controls.
    """

    @abstractmethod
    def get_horizon_length(self) -> int:
        """Get the horizon length of the OCP."""

    @abstractmethod
    def get_nx(self, k: int) -> int:
        """Get the number of states for a given time step."""

    @abstractmethod
    def get_nu(self, k: int) -> int:
        """Get the number of inputs for a given time step."""

    @abstractmethod
    def get_ng(self, k: int) -> int:
        """Get the number of equality constraints for a given time step."""

    @abstractmethod
    def get_ng_ineq(self, k: int) -> int:
        """Get the number of inequality constraints for a given time step."""

    def get_nx_tangent(self, k: int) -> int:
        """Get the tangent-space dimension of the state at stage k."""
        return self.get_nx(k)

    def get_nu_tangent(self, k: int) -> int:
        """Get the tangent-space dimension of the controls at stage k."""
        return self.get_nu(k)

    @abstractmethod
    def eval_BAt(self, x_next: Array, u: Array, x: Array, k: int) -> Array:
        """Evaluate the tranposed dynamics Jacobian, shape (tu+tx, tx_next).

        This method evaluates [Bₖ Aₖ]ᵀ from the linearized dynamics
        δxₖ₊₁ = Aₖ δxₖ + Bₖ δuₖ + bₖ.
        """

    @abstractmethod
    def eval_RSQt(self, scale: float, u: Array, x: Array, lam_dyn: Array,
                  lam_eq: Array, lam_ineq: Array, k: int) -> Array:
        """Evaluate the Lagrangian Hessian, shape (tu+tx, tu+tx).

        This method evaluates the Hessian of the Lagrangian to the
        concatenation of (uₖ, xₖ).

        Multipliers have tx_next, ng, and ng_ineq entries, respectively.
        lam_dyn is empty at the terminal stage. Scale weights the objective.
        """

    @abstractmethod
    def eval_Gt(self, u: Array, x: Array, k: int) -> Array:
        """Evaluate the tranposed equality constraints Jacobian, shape (tu+tx, ng).

        This method evaluates the Jacobian of the equality constraints
        with respect to the concatenation of (uₖ, xₖ).
        """

    @abstractmethod
    def eval_Gt_ineq(self, u: Array, x: Array, k: int) -> Array:
        """Evaluate the tranposed inequality constraints Jacobian, shape (tu+tx, ng_ineq).

        This method evaluates the Jacobian of the inequality constraints
        with respect to the concatenation of (uₖ, xₖ).
        """

    @abstractmethod
    def eval_b(self, x_next: Array, u: Array, x: Array, k: int) -> Array:
        """Evaluate the dynamics constraint violation, shape (tx_next,).

        This method evaluates bₖ = -xₖ₊₁ + fₖ(uₖ, xₖ), for Euclidean states.
        For manifold states, express the residual in tangent coordinates.
        """

    @abstractmethod
    def eval_g(self, u: Array, x: Array, k: int) -> Array:
        """Evaluate the equality constraint violation, shape (ng,)."""

    @abstractmethod
    def eval_gineq(self, u: Array, x: Array, k: int) -> Array:
        """Evaluate the inequality constraint violation, shape (ng_ineq,)."""

    @abstractmethod
    def eval_rq(self, scale: float, u: Array, x: Array, k: int) -> Array:
        """Evaluate the scaled gradient of the objective function, shape (tu+tx,).

        This method evaluates the gradient of the objective function (not the Lagrangian!)
        with respect to the concatenation of (uₖ, xₖ).
        """

    @abstractmethod
    def eval_L(self, scale: float, u: Array, x: Array, k: int) -> float:
        """Evaluate the scaled objective function value."""

    @abstractmethod
    def get_bounds(self, k: int) -> tuple[Array, Array]:
        """Get the bounds of the inequalities at a given stage, 2-tuple of
        shape (ng_ineq,)."""

    @abstractmethod
    def get_initial_xk(self, k: int) -> Array:
        """Get the initial guess for the states of a given stage, shape (nx,)."""

    @abstractmethod
    def get_initial_uk(self, k: int) -> Array:
        """Get the initial guess for the inputs of a given stage, shape (nu,)."""

    def apply_retraction_xk(self, x: Array, delta_x: Array, alpha: float, k: int) -> Array:
        """Retract (nx,) state along (tx,) step at stage k."""
        return _euclidean_retraction(x, delta_x, alpha)

    def apply_retraction_uk(self, u: Array, delta_u: Array, alpha: float, k: int) -> Array:
        """Retract (nu,) control along (tu,) step at stage k."""
        return _euclidean_retraction(u, delta_u, alpha)

    def intermediate(self, iteration: int, x: tuple[Array, ...], u: tuple[Array, ...],
                     stats: dict[str, float]) -> None:
        """Observe the current iterate before the next solver step."""


class TrajOptAbstract(ABC):
    """Trajectory optimization problem with horizon-wide batch callbacks."""

    @abstractmethod
    def get_horizon_length(self) -> int:
        """Return the integer number of stages K (at least one).

        There are K-1 dynamics transitions.
        """

    @abstractmethod
    def get_nx(self) -> int:
        """Get the number of states."""

    @abstractmethod
    def get_nu(self) -> int:
        """Get the number of input.s"""

    @abstractmethod
    def get_ng(self) -> int:
        """Get the number of equality constraints, excluding initial constraints.
        Overidden by get_ng_terminal() for the terminal stage."""

    @abstractmethod
    def get_ng_ineq(self) -> int:
        """Get the number of inequality constraints.
        Overidden by get_ng_ineq_terminal() for the terminal stage."""

    def get_nx_tangent(self) -> int:
        """Get the tangent-space dimension of the state."""
        return self.get_nx()

    def get_nu_tangent(self) -> int:
        """Get the tangent-space dimension of the inputs."""
        return self.get_nu()

    def get_ng_terminal(self) -> int:
        """Get the number of terminal equality constraints, cannot exceed get_ng()."""
        return self.get_ng()

    def get_ng_ineq_terminal(self) -> int:
        """Get the number of terminal inequality constraints, cannot exceed get_ng_ineq()."""
        return self.get_ng_ineq()

    @abstractmethod
    def batched_eval_BAt(self, x_next: Array, u: Array, x: Array) -> Array:
        """Evaluate the tranposed dynamics Jacobian, shape (K-1, tu+tx, tx).

        This method evaluates in batch [Bₖ Aₖ]ᵀ from the linearized dynamics
        δxₖ₊₁ = Aₖ δxₖ + Bₖ δuₖ + bₖ.

        Argments:
            x_next: shape (K-1, nx)
            u: shape (K-1, nu)
            x: shape (K-1, nx)
        """

    @abstractmethod
    def batched_eval_RSQt(self, scale: float, u: Array, x: Array, lam_dyn: Array,
                          lam_eq: Array, lam_ineq: Array, lam_initial: Array) -> Array:

        """Evaluate the Lagrangian Hessian, shape (K, tu+tx, tu+tx).

        This method evaluates in batch the Hessian of the Lagrangian to the
        concatenation of (uₖ, xₖ). For the terminal stage Hessian, the first
        tu rows and tu columns are ignored.

        Aguments:
            scale: weighting for the objective
            u: shape (K-1, nu)
            x: shape (K, nx)
            lam_dyn: shape (K-1, tx)
            lam_eq: shape (K, ng), teminal stage padded if ng_terminal < ng.
            lam_ineq: shape (K, ng_ineq), teminal stage padded if ng_ineq_terminal < ng_ineq.
            lam_initial: shape (tx,)
        """

    @abstractmethod
    def batched_eval_Gt(self, u: Array, x: Array) -> Array:
        """Evaluate the transposed equality constraints Jacobian, shape (K, tu+tx, ng).

        This method evaluates in batch the Jacobian of the ordinary equality
        constraints with respect to the concatenation of (uₖ, xₖ). Initial
        constraints are supplied separately by eval_Gt_initial(). For the
        terminal stage, the first tu rows and columns beyond ng_terminal are
        ignored.

        Arguments:
            u: shape (K-1, nu)
            x: shape (K, nx)
        """

    @abstractmethod
    def batched_eval_Gt_ineq(self, u: Array, x: Array) -> Array:
        """Evaluate the transposed inequality constraints Jacobian, shape (K, tu+tx, ng_ineq).

        This method evaluates in batch the Jacobian of the inequality functions
        with respect to the concatenation of (uₖ, xₖ). For the terminal stage,
        the first tu rows and columns beyond ng_ineq_terminal are ignored.

        Arguments:
            u: shape (K-1, nu)
            x: shape (K, nx)
        """

    @abstractmethod
    def batched_eval_b(self, x_next: Array, u: Array, x: Array) -> Array:
        """Evaluate the dynamics constraint violation, shape (K-1, tx).

        This method evaluates in batch bₖ = -xₖ₊₁ + fₖ(uₖ, xₖ) for Euclidean
        states. For manifold states, express the residual in tangent coordinates.
        This callback is not invoked when K=1.

        Arguments:
            x_next: shape (K-1, nx)
            u: shape (K-1, nu)
            x: shape (K-1, nx)
        """

    @abstractmethod
    def batched_eval_g(self, u: Array, x: Array) -> Array:
        """Evaluate the equality constraint violation, shape (K, ng).

        This method evaluates in batch the ordinary equality functions.
        Initial constraints are supplied separately by eval_g_initial().
        For the terminal stage, entries beyond ng_terminal are ignored.

        Arguments:
            u: shape (K-1, nu)
            x: shape (K, nx)
        """

    @abstractmethod
    def batched_eval_gineq(self, u: Array, x: Array) -> Array:
        """Evaluate the inequality functions, shape (K, ng_ineq).

        This method evaluates in batch the inequality functions. For the
        terminal stage, entries beyond ng_ineq_terminal are ignored.

        Arguments:
            u: shape (K-1, nu)
            x: shape (K, nx)
        """

    @abstractmethod
    def batched_eval_rq(self, scale: float, u: Array, x: Array) -> Array:
        """Evaluate the scaled gradient of the objective function, shape (K, tu+tx).

        This method evaluates in batch the gradient of the objective function
        with respect to the concatenation of (uₖ, xₖ). For the terminal stage,
        the first tu entries are ignored.

        Arguments:
            scale: weighting for the objective
            u: shape (K-1, nu)
            x: shape (K, nx)
        """

    @abstractmethod
    def batched_eval_L(self, scale: float, u: Array, x: Array) -> Array:
        """Evaluate the scaled objective function values, shape (K,).

        This method evaluates in batch the stage costs, including the terminal
        state cost. The total objective is the sum of these values.

        Arguments:
            scale: weighting for the objective
            u: shape (K-1, nu)
            x: shape (K, nx)
        """

    @abstractmethod
    def batched_get_bounds(self) -> tuple[Array, Array]:
        """Get the inequality bounds, a 2-tuple of arrays with shape (K, ng_ineq).

        Return the lower and upper bounds for batched_eval_gineq(), in that
        order. For the terminal stage, entries beyond ng_ineq_terminal are
        ignored.
        """

    @abstractmethod
    def batched_get_initial_xk(self) -> Array:
        """Get the initial guess for the states, shape (K, nx)."""

    @abstractmethod
    def batched_get_initial_uk(self) -> Array:
        """Get the initial guess for the inputs, shape (K-1, nu)."""

    def batched_apply_retraction_xk(self, x: Array, delta_x: Array, alpha: float) -> Array:
        """Retract the states along tangent steps, returning shape (K, nx).

        Arguments:
            x: shape (K, nx)
            delta_x: shape (K, tx)
            alpha: scalar step length
        """
        return _euclidean_retraction(x, delta_x, alpha)

    def batched_apply_retraction_uk(self, u: Array, delta_u: Array, alpha: float) -> Array:
        """Retract the inputs along tangent steps, returning shape (K-1, nu).

        Arguments:
            u: shape (K-1, nu)
            delta_u: shape (K-1, tu)
            alpha: scalar step length
        """
        return _euclidean_retraction(u, delta_u, alpha)

    @abstractmethod
    def eval_Gt_initial(self, x0: Array) -> Array:
        """Evaluate the transposed initial constraint Jacobian, shape (tx, tx).

        This method evaluates the Jacobian of eval_g_initial() with respect to
        the initial state's tangent coordinates. Zero input rows are supplied
        internally, and these columns are appended after stage-0 ordinary
        equality columns. Include initial constraint curvature in stage 0 of
        batched_eval_RSQt(), weighted by lam_initial.

        Arguments:
            x0: shape (nx,)
        """

    @abstractmethod
    def eval_g_initial(self, x0: Array) -> Array:
        """Evaluate the initial equality constraint violation, shape (tx,).

        This method evaluates the initial state residual, which must be zero at
        a feasible trajectory. These entries are appended after stage-0 ordinary
        equalities and are excluded from get_ng().

        Arguments:
            x0: shape (nx,)
        """

    def intermediate(self, iteration: int, x: Array, u: Array,
                     stats: dict[str, float]) -> None:
        """Observe the current iterate before the next solver step."""


def _euclidean_retraction(value: Array, delta: Array, alpha: float) -> Array:
    if value.shape[-1] != delta.shape[-1]:
        raise ValueError("A custom retraction is required when primal and tangent dimensions differ")
    return value + alpha * delta
