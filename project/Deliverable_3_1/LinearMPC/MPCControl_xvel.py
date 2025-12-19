import numpy as np
import cvxpy as cp
from control import dlqr
from .MPCControl_base import MPCControl_base


class MPCControl_xvel(MPCControl_base):
    x_ids: np.ndarray = np.array([1, 4, 6])
    u_ids: np.ndarray = np.array([1])

    def __init__(self, A, B, xs, us, Ts, H):
        # Q: [w_y, beta, v_x]
        self.Q = np.diag([1.0, 10.0, 100.0])
        self.R = np.diag([5.0])

        super().__init__(A, B, xs, us, Ts, H)

    def get_stage_cost(self, x_k, u_k):
        """Returns cost at time step k: (x-xref)^T Q (x-xref) + (u-uref)^T R (u-uref)"""
        dx = x_k - self.x_target_param
        du = u_k - self.u_target_param
        return cp.quad_form(dx, self.Q) + cp.quad_form(du, self.R)

    def get_stage_constraints(self, x_k, u_k):
        """Returns constraints specific to this subsystem"""
        constraints = []
        
        # Input Constraint: Servo 2 limit +/- 15 deg (0.26 rad)
        constraints += [cp.abs(u_k[0]) <= np.deg2rad(15)]
        
        # State Constraint: Pitch Angle (Beta) limit +/- 10 deg (0.1745 rad)
        # Beta is index 1 in this subsystem [w_y, beta, v_x]
        constraints += [cp.abs(x_k[1]) <= np.deg2rad(10)]
        
        return constraints

    def get_terminal_cost(self, x_N):
        """Returns terminal cost: (x_N-xref)^T P (x_N-xref)"""
        if not hasattr(self, 'P'):
            K, self.P, E = dlqr(self.A, self.B, self.Q, self.R)
        dx = x_N - self.x_target_param
        return cp.quad_form(dx, self.P)

    def get_terminal_constraints(self, x_N):
        """Returns terminal equality constraint for recursive feasibility"""
        return [x_N == self.x_target_param]
    
    # def _setup_controller(self) -> None:
    #     #################################################
    #     # YOUR CODE HERE

    #     self.ocp = ...

    #     # YOUR CODE HERE
    #     #################################################

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
    