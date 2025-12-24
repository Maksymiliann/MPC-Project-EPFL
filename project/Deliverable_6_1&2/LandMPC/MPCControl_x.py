import numpy as np
import cvxpy as cp
from .MPCControl_base import MPCControl_base


class MPCControl_x(MPCControl_base):
    x_ids: np.ndarray = np.array([1, 4, 6, 9])
    u_ids: np.ndarray = np.array([1])

    def _setup_controller(self) -> None:
        # Tuning
        Q = np.diag([10.0, 100.0, 10.0, 1.0]) # [vel, pos, ang, rate]
        R = np.diag([1.0])
        self.slack_penalty = 1.0

        u_max = 0.26 # ~15 deg
        angle_max = np.deg2rad(15)
        
        nx, nu, N = self.nx, self.nu, self.N
        
        # Variables
        X = cp.Variable((nx, N + 1))
        U = cp.Variable((nu, N))
        Slack = cp.Variable((1, N), nonneg=True)
        
        # Parameters expected by base.get_u
        x_init = cp.Parameter(nx)
        x_ref = cp.Parameter(nx) # <--- Required for tracking x=1 or y=0

        cost = 0
        constr = [X[:, 0] == x_init]

        for k in range(N):
            constr += [X[:, k + 1] == self.A @ X[:, k] + self.B @ U[:, k]]
            constr += [cp.abs(U[:, k]) <= u_max]
            
            # Soft Angle Constraint
            constr += [X[2, k] <= angle_max + Slack[0, k]]
            constr += [X[2, k] >= -angle_max - Slack[0, k]]
            
            # Cost uses x_ref
            state_err = X[:, k] - x_ref
            cost += cp.quad_form(state_err, Q) + cp.quad_form(U[:, k], R)
            cost += self.slack_penalty * Slack[0, k]

        # No Terminal Constraint
        cost += cp.quad_form(X[:, N] - x_ref, Q)

        self.ocp = cp.Problem(cp.Minimize(cost), constr)
        
        # Save handles for base class
        self._X = X
        self._U = U
        self._x_init = x_init
        self._x_ref = x_ref

    # def get_u(
    #     self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    # ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    #     #################################################
    #     # YOUR CODE HERE

    #     u0 = ...
    #     x_traj = ...
    #     u_traj = ...

    #     # YOUR CODE HERE
    #     #################################################

    #     return u0, x_traj, u_traj