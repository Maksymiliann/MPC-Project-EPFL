import cvxpy as cp
import numpy as np
# from control import dlqr
# from mpt4py import Polyhedron
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

        # Defaults for tuning (can be overridden by subclasses)
        if not hasattr(self, "Q"):
            self.Q = np.eye(self.nx)
        if not hasattr(self, "R"):
            self.R = np.eye(self.nu)

        self._setup_controller()

    def _setup_controller(self) -> None:
        self.x_var = cp.Variable((self.nx, self.N + 1))
        self.u_var = cp.Variable((self.nu, self.N))
        self.x_init_param = cp.Parameter(self.nx)
        self.x_target_param = cp.Parameter(self.nx)
        self.u_target_param = cp.Parameter(self.nu)

        cost = 0.0
        constraints = [self.x_var[:, 0] == self.x_init_param]

        for k in range(self.N):
            constraints += [
                (self.x_var[:, k+1] - self.xs) == 
                self.A @ (self.x_var[:, k] - self.xs) + 
                self.B @ (self.u_var[:, k] - self.us)
            ]

            cost += self.get_stage_cost(self.x_var[:, k], self.u_var[:, k])
            constraints += self.get_stage_constraints(self.x_var[:, k], self.u_var[:, k])

        cost += self.get_terminal_cost(self.x_var[:, self.N])
        constraints += self.get_terminal_constraints(self.x_var[:, self.N])

        self.ocp = cp.Problem(cp.Minimize(cost), constraints)

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
        
        # # 1. Handle Default Targets (Regulation to Trim Point)
        # if x_target is None:
        #     print("MPC Info: No x_target provided, using steady state (xs).")
        #     x_target = self.xs  # Default: Steady state (xs)
        # if u_target is None:
        #     print("MPC Info: No u_target provided, using steady state input (us).")
        #     u_target = self.us  # Default: Steady state input (us)

        # 2. Update CVXPY Parameters
        self.x_init_param.value = x0
        self.x_target_param.value = x_target
        self.u_target_param.value = u_target

        # 3. Solve the Optimization Problem
        # warm_start=True reuses the previous solution to speed up calculation
        self.ocp.solve(solver=cp.OSQP, warm_start=True, verbose=False)

        # 4. Error Handling: Check if solver failed
        if self.u_var.value is None or self.ocp.status in [cp.INFEASIBLE, cp.UNBOUNDED]:
            print(f"MPC Warning: Solver failed with status '{self.ocp.status}' for class {self.__class__.__name__}")
            
            # Fallback strategy: Apply the steady-state (trim) input
            # This keeps the rocket hovering/stable-ish rather than crashing the code
            u0 = self.us.copy()
            
            x_traj = np.tile(self.xs[:, None], (1, self.N + 1))
            u_traj = np.tile(self.us[:, None], (1, self.N))
            
        else:
            u0 = self.u_var[:, 0].value
            x_traj = self.x_var.value
            u_traj = self.u_var.value

        return u0, x_traj, u_traj