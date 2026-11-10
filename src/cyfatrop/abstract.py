"""Public problem contracts for :class:`cyfatrop.Solver`.

Subclass exactly one of OcpAbstract or TrajOptAbstract. The base class selects
the internal adapter; inheriting both is unsupported. Python ABCs enforce
implementation of abstract methods, including callbacks returning empty arrays.
Signatures are not inspected and extra callbacks are ignored. Numerical
dimensions, callback output shapes/types, and bounds are validated at runtime.

Methods and dimensions are cached when constructing a solver. Construct a new
solver after changing dimensions or replacing methods. Callback-owned problem
data may change between solves. Dimensions must be nonnegative integers within
Fatrop's 32-bit index range; the horizon must be at least one and total problem
dimensions must also fit that range.

Callback array inputs are immutable snapshots that can safely be retained.
Return real numeric NumPy arrays of the documented shapes. Integer and strided
arrays are copied to contiguous float64 storage. Scalar stage objectives may
also be Python numbers. Active bounds must not contain NaN or have lower > upper.

Primal sizes are nx and nu; tangent sizes are tx and tu. Derivatives use tangent
variable ordering [u, x], with m = tu + tx. Tangent getters default to their
corresponding primal sizes. Retractions default to value + alpha * delta and
raise ValueError when primal and tangent lengths differ; manifold problems
override them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _euclidean_retraction(value: Array, delta: Array, alpha: float) -> Array:
    if value.shape[-1] != delta.shape[-1]:
        raise ValueError("A custom retraction is required when primal and tangent dimensions differ")
    return value + alpha * delta


class OcpAbstract(ABC):
    """State-dependent OCP with stage index k on each callback.

    Dimensions and callback shapes are those of stage k, including the terminal
    stage. Inputs x and u have shapes (nx,) and (nu,), while x_next has the next
    stage's primal state size. Let tx_next = get_nx_tangent(k + 1).

    Supply actual dimensions at every stage; there is no padding or cropping.
    In particular, get_nu(K - 1) may be zero or include terminal controls.
    Express initial equalities through eval_g and eval_Gt at stage 0.

    Result.x and Result.u are tuples of K owned arrays using actual stage sizes,
    including an empty terminal control array when get_nu(K - 1) is zero.
    """

    @abstractmethod
    def get_horizon_length(self) -> int:
        """Return the integer number of stages K (at least one).

        There are K-1 dynamics transitions. Dynamics callbacks are not invoked
        when K=1.
        """

    @abstractmethod
    def get_nx(self, k: int) -> int:
        """Return the primal state size at stage k."""

    @abstractmethod
    def get_nu(self, k: int) -> int:
        """Return the primal control size at stage k."""

    @abstractmethod
    def get_ng(self, k: int) -> int:
        """Return the equality count at stage k."""

    @abstractmethod
    def get_ng_ineq(self, k: int) -> int:
        """Return the inequality count at stage k."""

    def get_nx_tangent(self, k: int) -> int:
        """Return tx; defaults to get_nx(k)."""
        return self.get_nx(k)

    def get_nu_tangent(self, k: int) -> int:
        """Return tu; defaults to get_nu(k)."""
        return self.get_nu(k)

    @abstractmethod
    def eval_BAt(self, x_next: Array, u: Array, x: Array, k: int) -> Array:
        """Return the transposed dynamics Jacobian, shape (m, tx_next)."""

    @abstractmethod
    def eval_RSQt(self, scale: float, u: Array, x: Array, lam_dyn: Array,
                  lam_eq: Array, lam_ineq: Array, k: int) -> Array:
        """Return the Lagrangian Hessian (m, m).

        Multipliers have tx_next, ng, and ng_ineq entries, respectively.
        lam_dyn is empty at the terminal stage. Scale weights the objective only.
        """

    @abstractmethod
    def eval_Gt(self, u: Array, x: Array, k: int) -> Array:
        """Return the transposed equality Jacobian, shape (m, ng)."""

    @abstractmethod
    def eval_Gt_ineq(self, u: Array, x: Array, k: int) -> Array:
        """Return the transposed inequality Jacobian, shape (m, ng_ineq)."""

    @abstractmethod
    def eval_b(self, x_next: Array, u: Array, x: Array, k: int) -> Array:
        """Return the dynamics residual, shape (tx_next,)."""

    @abstractmethod
    def eval_g(self, u: Array, x: Array, k: int) -> Array:
        """Return the equality residual, shape (ng,)."""

    @abstractmethod
    def eval_gineq(self, u: Array, x: Array, k: int) -> Array:
        """Return the inequality values, shape (ng_ineq,)."""

    @abstractmethod
    def eval_rq(self, scale: float, u: Array, x: Array, k: int) -> Array:
        """Return the scaled objective gradient, shape (m,)."""

    @abstractmethod
    def eval_L(self, scale: float, u: Array, x: Array, k: int) -> float:
        """Return the scaled scalar stage objective."""

    @abstractmethod
    def get_bounds(self, k: int) -> tuple[Array, Array]:
        """Return lower and upper inequality bounds, each shape (ng_ineq,)."""

    @abstractmethod
    def get_initial_xk(self, k: int) -> Array:
        """Return the initial state guess, shape (nx,)."""

    @abstractmethod
    def get_initial_uk(self, k: int) -> Array:
        """Return the initial control guess, shape (nu,)."""

    def apply_retraction_xk(self, x: Array, delta_x: Array, alpha: float, k: int) -> Array:
        """Retract (nx,) state along (tx,) step; defaults to Euclidean addition."""
        return _euclidean_retraction(x, delta_x, alpha)

    def apply_retraction_uk(self, u: Array, delta_u: Array, alpha: float, k: int) -> Array:
        """Retract (nu,) control along (tu,) step; defaults to Euclidean addition."""
        return _euclidean_retraction(u, delta_u, alpha)


class TrajOptAbstract(ABC):
    """Fixed-dimension problem with horizon-wide batch callbacks.

    Implement batched callbacks; there is no per-stage fallback. Array inputs
    and outputs have a leading batch axis in ascending stage order. Callbacks
    for stage operations use K rows; dynamics callbacks use K-1 rows. Scale and
    alpha remain scalars. State batches have shape (K, nx), control batches
    (K, nu), and all callback arrays use nominal dimensions, including the
    terminal row.

    There are no native terminal control variables. Terminal control and control
    step inputs are zero placeholders, as are terminal dynamics multipliers.
    Inactive terminal ordinary equality and inequality multipliers are padded
    with zeros. Terminal residuals and bounds retain active leading entries;
    constraint Jacobians retain active leading columns and discard control rows.
    Terminal Hessians retain their state-state blocks, and objective gradients
    retain state entries. Terminal control initialization/retraction outputs
    are discarded. Return complete nominal shapes even for discarded entries,
    whose values may be arbitrary.

    Initial callbacks receive x0 with shape (nx,). Initial equalities are
    appended after ordinary stage-0 equalities; zero control rows are supplied
    internally for their transposed Jacobian. get_ng() excludes these tx initial
    constraints. Terminal count overrides and cropping apply before the append,
    including when K=1. Solver.dimensions["ng"] reports native counts including
    the initial block.

    Result.x is a stacked array with shape (K, nx). Result.u contains the actual
    controls with shape (K-1, nu); it has no padded terminal row. Callback padding
    is separate from the returned trajectories.
    """

    @abstractmethod
    def get_horizon_length(self) -> int:
        """Return the integer number of stages K (at least one).

        There are K-1 dynamics transitions. Dynamics callbacks are not invoked
        when K=1.
        """

    @abstractmethod
    def get_nx(self) -> int:
        """Return the nominal primal state size."""

    @abstractmethod
    def get_nu(self) -> int:
        """Return the nominal primal control size."""

    @abstractmethod
    def get_ng(self) -> int:
        """Return the ordinary equality count, excluding initial constraints."""

    @abstractmethod
    def get_ng_ineq(self) -> int:
        """Return the nominal inequality count."""

    def get_nx_tangent(self) -> int:
        """Return tx; defaults to get_nx()."""
        return self.get_nx()

    def get_nu_tangent(self) -> int:
        """Return tu; defaults to get_nu()."""
        return self.get_nu()

    def get_ng_terminal(self) -> int:
        """Return the terminal ordinary equality count; defaults to get_ng().

        Overrides may reduce this count, but cannot exceed get_ng().
        """
        return self.get_ng()

    def get_ng_ineq_terminal(self) -> int:
        """Return the terminal inequality count; defaults to get_ng_ineq().

        Overrides may reduce this count, but cannot exceed get_ng_ineq().
        """
        return self.get_ng_ineq()

    @abstractmethod
    def batched_eval_BAt(self, x_next: Array, u: Array, x: Array) -> Array:
        """Return transposed dynamics Jacobians, shape (K-1, m, tx)."""

    @abstractmethod
    def batched_eval_RSQt(self, scale: float, u: Array, x: Array, lam_dyn: Array,
                          lam_eq: Array, lam_ineq: Array, lam_initial: Array) -> Array:
        """Return Hessians (K, m, m), including initial curvature in row 0.

        Multiplier shapes are (K, tx), (K, ng), (K, ng_ineq), and (tx,).
        lam_eq contains only ordinary equalities; lam_initial contains only
        the appended initial multipliers. Include their multiplier-weighted
        constraint curvature in the stage-0 Hessian.
        Terminal dynamics and inactive terminal entries are zero-padded.
        Scale weights the objective only.
        """

    @abstractmethod
    def batched_eval_Gt(self, u: Array, x: Array) -> Array:
        """Return ordinary transposed equality Jacobians, shape (K, m, ng)."""

    @abstractmethod
    def batched_eval_Gt_ineq(self, u: Array, x: Array) -> Array:
        """Return transposed inequality Jacobians, shape (K, m, ng_ineq)."""

    @abstractmethod
    def batched_eval_b(self, x_next: Array, u: Array, x: Array) -> Array:
        """Return dynamics residuals, shape (K-1, tx)."""

    @abstractmethod
    def batched_eval_g(self, u: Array, x: Array) -> Array:
        """Return ordinary equality residuals, shape (K, ng)."""

    @abstractmethod
    def batched_eval_gineq(self, u: Array, x: Array) -> Array:
        """Return inequality values, shape (K, ng_ineq)."""

    @abstractmethod
    def batched_eval_rq(self, scale: float, u: Array, x: Array) -> Array:
        """Return scaled objective gradients, shape (K, m)."""

    @abstractmethod
    def batched_eval_L(self, scale: float, u: Array, x: Array) -> Array:
        """Return scaled stage objectives, shape (K,)."""

    @abstractmethod
    def batched_get_bounds(self) -> tuple[Array, Array]:
        """Return lower and upper bounds, each shape (K, ng_ineq)."""

    @abstractmethod
    def batched_get_initial_xk(self) -> Array:
        """Return initial state guesses, shape (K, nx)."""

    @abstractmethod
    def batched_get_initial_uk(self) -> Array:
        """Return initial control guesses, shape (K, nu)."""

    def batched_apply_retraction_xk(self, x: Array, delta_x: Array, alpha: float) -> Array:
        """Retract (K, nx) states along (K, tx) steps; defaults to addition."""
        return _euclidean_retraction(x, delta_x, alpha)

    def batched_apply_retraction_uk(self, u: Array, delta_u: Array, alpha: float) -> Array:
        """Retract (K, nu) controls along (K, tu) steps; defaults to addition."""
        return _euclidean_retraction(u, delta_u, alpha)

    @abstractmethod
    def eval_Gt_initial(self, x0: Array) -> Array:
        """Return the transposed initial state-tangent Jacobian, shape (tx, tx)."""

    @abstractmethod
    def eval_g_initial(self, x0: Array) -> Array:
        """Return the initial equality residual, shape (tx,)."""
