"""Optimal-control model definitions used by the figure scripts.

The models share the feedback/cost interface used by the PI scripts.
"""

from __future__ import annotations

import abc
import time

import numpy as np
from scipy import linalg as la
from scipy.integrate import solve_bvp


Array = np.ndarray


class Model(metaclass=abc.ABCMeta):
    """Abstract interface for control-affine benchmark models."""

    def __init__(self, state_weight: float, control_weight: float) -> None:
        """Store the quadratic state and control weights."""
        self.stateWeight   = state_weight
        self.controlWeight = control_weight

    @abc.abstractmethod
    def f(self, x: Array) -> Array:
        """Evaluate the uncontrolled vector field."""
        raise NotImplementedError

    @abc.abstractmethod
    def g(self, x: Array) -> Array:
        """Evaluate the control vector field."""
        raise NotImplementedError

    @abc.abstractmethod
    def jacobi_of_f_transposed_dot_p(self, x: Array, p: Array) -> Array:
        """Evaluate Df(x)^T p for the adjoint equation."""
        raise NotImplementedError

    def solve_open_loop_bvp(
        self,
        start_state: Array,
        end_time: float,
        number_of_eval_points: int,
        verbose: bool = False,
    ) -> tuple[Array, Array, float, Array]:
        """Compute a finite-horizon reference value by a PMP boundary problem."""
        n = start_state.shape[0]

        def ode_system(t: Array, y: Array) -> Array:
            """Evaluate state, adjoint, and accumulated-cost dynamics."""
            x       = y[:n, :]
            p       = y[n : 2 * n, :]
            u       = (-1 / (2 * self.controlWeight)) * self.g(x).T @ p
            x_dot   = self.f(x) + self.g(x) @ u
            p_dot   = -self.jacobi_of_f_transposed_dot_p(x, p) - 2 * x * self.stateWeight
            cost    = -self.stateWeight * np.sum(x**2, axis=0) - self.controlWeight * np.sum(u**2, axis=0)

            return np.r_[x_dot, p_dot, np.atleast_2d(cost)]

        def boundary_conditions(ya: Array, yb: Array) -> Array:
            """Enforce initial state and terminal adjoint/cost conditions."""
            return np.r_[ya[0:n] - start_state, yb[n : 2 * n + 1]]

        start_time = time.time()
        time_span  = np.linspace(0, end_time, number_of_eval_points)
        initial    = np.zeros((2 * n + 1, time_span.size))
        solution   = solve_bvp(
            ode_system,
            boundary_conditions,
            time_span,
            initial,
            max_nodes = 10_000_000,
            tol       = 1e-8,
        )

        if verbose:
            message = (
                f"Solved open-loop control with {solution.x.shape[0]} mesh points, "
                f"maximal residual error {np.amax(np.abs(solution.rms_residuals))}. "
                f"It took {time.time() - start_time:.2f} seconds."
            )
            print(message)

        return solution.y[0:n, :], solution.y[n : 2 * n, :], solution.y[-1, 0], solution.x[:]


class VanDerPol(Model):
    """Controlled two-dimensional Van der Pol oscillator."""

    def __init__(self, state_weight: float, control_weight: float) -> None:
        """Initialize the model and the stabilizing Riccati feedback."""
        super().__init__(state_weight, control_weight)

        linearized_matrix = np.c_[np.r_[0, 1], np.r_[1, 1]]
        control_matrix    = np.atleast_2d(np.array([0, 1])).T

        self.matrixKGain  = la.solve_continuous_are(
            linearized_matrix,
            control_matrix,
            state_weight * np.eye(2),
            control_weight,
            e        = None,
            s        = None,
            balanced = True,
        )

    def f(self, x: Array) -> Array:
        """Evaluate the uncontrolled Van der Pol vector field."""
        return np.array([x[1], -x[0] + x[1] * (1 - x[0]**2)])

    def g(self, x: Array) -> Array:
        """Evaluate the constant input vector field."""
        return np.array([[0, 1]]).T

    def jacobi_of_f_transposed_dot_p(self, x: Array, p: Array) -> Array:
        """Evaluate Df(x)^T p for the Van der Pol oscillator."""
        return np.array([-p[1] - 2 * p[1] * x[0] * x[1], p[0] + p[1] * (1 - x[0]**2)])

    def stable_control(self, x: Array) -> Array:
        """Evaluate the stabilizing Riccati feedback on column-wise states."""
        weighted_state = self.matrixKGain @ x
        feedback       = -(1 / self.controlWeight) * np.sum(self.g(x) * weighted_state, axis=0)

        return np.atleast_2d(feedback)

    def control_from_value_gradient(self, x: Array, value_gradient: Array) -> Array:
        """Evaluate the feedback induced by a value-function gradient."""
        return (-0.5 / self.controlWeight) * (self.g(x).T @ value_gradient)

    def closed_loop_rhs(self, x: Array, control: Array) -> Array:
        """Evaluate f(x) + g(x)u for column-wise states and controls."""
        return self.f(x) + self.g(x) @ control

    def running_cost(self, x: Array, control: Array) -> Array:
        """Evaluate the quadratic running cost on column-wise states."""
        state_cost   = self.stateWeight * np.sum(x**2, axis=0)
        control_cost = self.controlWeight * np.sum(control**2, axis=0)

        return state_cost + control_cost

    def linear_value_function(self, x: Array) -> Array:
        """Evaluate the quadratic value induced by the Riccati matrix."""
        return np.sum(x * (self.matrixKGain @ x), axis=0)


class EscapeCounterexample(Model):
    """Smooth example with finite-cost escape; initialize on the disk B_(1/2).

    PI uses the original state x. Reference BVPs use z=x/sqrt(1+|x|^2),
    d tau/dt=1-|z|^2, and w=u/sqrt(1-|z|^2). Then dz/dtau=Bw and the cost
    is integral(z.T Q z + |w|^2). Reaching |z|=1 represents escape as t->inf.
    """

    def __init__(self) -> None:
        super().__init__(1.0, 1.0)
        self.B = np.diag([1.0, 20.0])
        self.P = np.array([[1.0, 0.1], [0.1, 9/400]])
        self.Q = np.array([[1.0, 0.2], [0.2, 17/400]])
        self.M = self.B @ self.B.T

    def f(self, x: Array) -> Array:
        return np.zeros_like(x)

    def g(self, x: Array) -> Array:
        """Return g(x_j) by slices: shape (2, 2, number of points)."""
        matrices = np.eye(2)[:, :, None] + np.einsum('ik,jk->ijk', x, x)
        return np.einsum('ijk,jl->ilk', matrices, self.B)

    def jacobi_of_f_transposed_dot_p(self, x: Array, p: Array) -> Array:
        return np.zeros_like(p)

    def stable_control(self, x: Array) -> Array:
        """The initial closed loop is exactly dx/dt=-x."""
        return -np.linalg.solve(self.B, x)/(1 + np.sum(x*x, axis=0))

    def control_from_value_gradient(self, x: Array, gradient: Array) -> Array:
        return -0.5*self.B.T @ (gradient + x*np.sum(x*gradient, axis=0))

    def closed_loop_rhs(self, x: Array, control: Array) -> Array:
        bu = self.B @ control
        return bu + x*np.sum(x*bu, axis=0)

    def running_cost(self, x: Array, control: Array) -> Array:
        state_cost = np.sum(x*(self.Q @ x), axis=0)/(1 + np.sum(x*x, axis=0))**2
        return state_cost + np.sum(control*control, axis=0)

    def initial_value_function(self, x: Array) -> Array:
        return np.sum(x*(self.P @ x), axis=0)/(1 + np.sum(x*x, axis=0))

    def _pmp_bvp(self, z0: Array, horizon: float, nodes: int, sign: int = 0,
                 time_factor: float = 1.0):
        """PMP: z'=-Mp/2, p'=-2Qz, c'=-(z.T Q z+p.T M p/4).

        Origin: z(0)=z0, p(T)=0, c(T)=0, with prescribed large T.
        Escape: z(0)=z0, |z(T)|=1, c(T)=0 and free T. Terminal normality,
        H(T)=0 and outward motion give p(T)=-2 sqrt(z.T Q z/z.T M z) z(T).
        """
        mesh = np.linspace(0, 1 if sign else horizon, nodes)
        guess = np.zeros((5, nodes))
        parameters = None
        if sign:
            end = np.array([z0[0], sign*np.sqrt(1-z0[0]**2)])
            difference = end-z0
            mean_cost = (z0 @ self.Q @ z0 + z0 @ self.Q @ end + end @ self.Q @ end)/3
            time_guess = time_factor*np.sqrt(difference @ np.linalg.solve(self.M, difference)/mean_cost)
            guess[:2] = z0[:, None]*(1-mesh) + end[:, None]*mesh
            guess[2:4] = (-2*np.linalg.solve(self.M, difference)/time_guess)[:, None]
            parameters = np.array([time_guess])

        def equations(t, y, parameters=None):
            z, p = y[:2], y[2:4]
            cost = np.sum(z*(self.Q @ z), axis=0) + np.sum(p*(self.M @ p), axis=0)/4
            rhs = np.vstack([-0.5*self.M @ p, -2*self.Q @ z, -cost])
            return parameters[0]*rhs if sign else rhs

        def boundary(ya, yb, parameters=None):
            if not sign:
                return np.r_[ya[:2]-z0, yb[2:]]
            z, p = yb[:2], yb[2:4]
            normal = 2*np.sqrt((z @ self.Q @ z)/max(z @ self.M @ z, 1e-100))*z
            return np.r_[ya[:2]-z0, z @ z-1, p+normal, yb[4]]

        return solve_bvp(equations, boundary, mesh, guess, p=parameters,
                         tol=1e-8, max_nodes=5000)

    def solve_open_loop_bvp(self, start_state: Array, end_time: float = 60.0,
                           number_of_eval_points: int = 101, verbose: bool = False):
        """Select the least-cost converged, feasible PMP candidate.

        Returns (z, p_z, cost, tau): these trajectories are in the transformed
        coordinates, unlike VanDerPol's physical trajectories. The origin
        branch is a finite-horizon approximation; escape has free terminal tau.
        Both exit directions are tried. Nonconverged starts are never used.
        """
        x0 = np.asarray(start_state, dtype=float)
        z0 = x0/np.sqrt(1 + x0 @ x0)
        candidates = []

        def retain(solution, escape):
            if not solution.success or (escape and solution.p[0] <= 0):
                return False
            sample = solution.sol(np.linspace(solution.x[0], solution.x[-1], 1001))
            if np.max(np.sum(sample[:2]**2, axis=0)) > 1+1e-7 or solution.y[4, 0] < 0:
                return False
            candidates.append((solution, escape))
            return True

        retain(self._pmp_bvp(z0, end_time, number_of_eval_points), False)
        # The barrier estimate excludes every improving escape from this set;
        # its reference value is still computed by the origin PMP BVP above.
        if self.initial_value_function(x0[:, None])[0] >= np.linalg.eigvalsh(self.P)[0]/2:
            for sign in (-1, 1):
                for factor in (1.0, 3.0):
                    solution = self._pmp_bvp(z0, end_time, number_of_eval_points, sign, factor)
                    if retain(solution, True):
                        break
        if not candidates:
            raise RuntimeError(f'No feasible PMP reference found at {x0}.')
        best, escape = min(candidates, key=lambda item: item[0].y[4, 0])
        tau = best.x*best.p[0] if escape else best.x
        if verbose:
            print(f'PMP {"escape" if escape else "origin"}: cost={best.y[4, 0]:.10g}, '
                  f'max residual={np.max(best.rms_residuals):.2e}')
        return best.y[:2], best.y[2:4], float(best.y[4, 0]), tau
