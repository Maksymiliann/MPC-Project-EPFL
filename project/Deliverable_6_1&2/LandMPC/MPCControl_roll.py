import numpy as np
import cvxpy as cp
from .MPCControl_base import MPCControl_base


class MPCControl_roll(MPCControl_base):
    x_ids: np.ndarray = np.array([2, 5])
    u_ids: np.ndarray = np.array([3])

    def _setup_controller(self) -> None:
        Q = np.diag([200.0, 1.0]) # [angle, rate]
        R = np.diag([1.0])
        u_max = 0.35 # in rads, approx 20 degrees
        
        nx, nu, N = self.nx, self.nu, self.N
        X = cp.Variable((nx, N + 1))
        U = cp.Variable((nu, N))
        
        x_init = cp.Parameter(nx)
        x_ref = cp.Parameter(nx)

        cost = 0
        constr = [X[:, 0] == x_init]
        
        for k in range(N):
            constr += [X[:, k + 1] == self.A @ X[:, k] + self.B @ U[:, k]]
            constr += [cp.abs(U[:, k]) <= u_max]
            
            # Track x_ref (which will likely be 0 for roll)
            cost += cp.quad_form(X[:, k] - x_ref, Q) + cp.quad_form(U[:, k], R)
            
        cost += cp.quad_form(X[:, N] - x_ref, Q)

        self.ocp = cp.Problem(cp.Minimize(cost), constr)
        
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
