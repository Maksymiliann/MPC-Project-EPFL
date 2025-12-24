import cvxpy as cp
import numpy as np
from control import dlqr
from mpt4py import Polyhedron
from scipy.signal import cont2discrete


class MPCControl_base:
    """Complete states indices"""

    x_ids: np.ndarray
    u_ids: np.ndarray

    """Optimization system"""
    A: np.ndarray
    B: np.ndarray
    xs: np.ndarray
    us: np.ndarray
    nx: int
    nu: int
    Ts: float
    H: float
    N: int

    """Optimization problem"""
    ocp: cp.Problem

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        xs: np.ndarray,
        us: np.ndarray,
        Ts: float,
        H: float,
    ) -> None:
        self.Ts = Ts
        self.H = H
        self.N = int(H / Ts)
        self.nx = self.x_ids.shape[0]
        self.nu = self.u_ids.shape[0]

        # System definition
        xids_xi, xids_xj = np.meshgrid(self.x_ids, self.x_ids)
        A_red = A[xids_xi, xids_xj].T
        uids_xi, uids_xj = np.meshgrid(self.x_ids, self.u_ids)
        B_red = B[uids_xi, uids_xj].T

        self.A, self.B = self._discretize(A_red, B_red, Ts)
        self.xs = xs[self.x_ids]
        self.us = us[self.u_ids]

        self._setup_controller()

    # def _setup_controller(self) -> None:
    #     #################################################
    #     # YOUR CODE HERE

    #     self.ocp = ...

    #     # YOUR CODE HERE
    #     #################################################

    @staticmethod
    def _discretize(A: np.ndarray, B: np.ndarray, Ts: float):
        nx, nu = B.shape
        C = np.zeros((1, nx))
        D = np.zeros((1, nu))
        A_discrete, B_discrete, _, _, _ = cont2discrete(system=(A, B, C, D), dt=Ts)
        return A_discrete, B_discrete

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        dx0 = x0 - self.xs
        self._x_init.value = dx0
        
        if x_target is not None:
            if hasattr(self, '_x_ref'):
                self._x_ref.value = x_target - self.xs
        else:
            if hasattr(self, '_x_ref'):
                self._x_ref.value = np.zeros(self.nx)

        self.ocp.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        
        # Fallback if OSQP fails (rare with soft constraints)
        # if self.ocp.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        #      # Return trim input (delta u = 0)
        #      u0 = self.us
        #      x_traj = np.tile(self.xs.reshape(-1,1), (1, self.N+1))
        #      u_traj = np.tile(self.us.reshape(-1,1), (1, self.N))
        #      return u0, x_traj, u_traj

        du0 = self._U.value[:, 0]
        u0 = self.us + du0

        x_traj = self.xs.reshape(-1, 1) + self._X.value
        u_traj = self.us.reshape(-1, 1) + self._U.value

        return u0, x_traj, u_traj
