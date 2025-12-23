import numpy as np
import cvxpy as cp
from control import dlqr
from mpt4py import Polyhedron
from .MPCControl_base import MPCControl_base


class MPCControl_z(MPCControl_base):
    x_ids: np.ndarray = np.array([8, 11])
    u_ids: np.ndarray = np.array([2])

    # only useful for part 5 of the project
    # d_estimate: np.ndarray
    # d_gain: float

    @staticmethod
    def _max_invariant_set(A_cl: np.ndarray, X: Polyhedron, max_iter: int = 100) -> Polyhedron:
        """
        Computes the Maximal Positive Invariant Set O_inf for x+ = A_cl*x
        """
        O = X
        converged = False

        for _ in range(max_iter):
            O_prev = O
            # Pre(O) = {x | A_cl x in O}
            PreO = Polyhedron.from_Hrep(O.A @ A_cl, O.b)
            O = O.intersect(PreO)
            O.minHrep(True)
            if O == O_prev:
                converged = True
                break

        if not converged:
            print("[zvel] Warning: max invariant set did not converge within max_iter.")
        return O

    def _setup_controller(self) -> None:
        Q_lqr = np.diag([50.0, 100.0])  # High penalty on Z deviation
        R_lqr = np.diag([1.0])          # Moderate penalty on input
        
        # Weights for Nominal MPC
        Q_mpc = np.diag([10.0, 100.0])
        R_mpc = np.diag([1.0])

        #################################################
        # 2. ANCILLARY CONTROLLER (K) & TERMINAL COST
        #################################################
        # Compute LQR gain K and Terminal Cost Qf
        K_val, Qf, _ = dlqr(self.A, self.B, Q_lqr, R_lqr)
        self.K = -K_val  # Gain such that A+BK is stable
        self.Qf = Qf

        #################################################
        # 3. COMPUTE TUBE (mRPI SET E)
        #################################################
        # Disturbance w in [-15, 5]
        # Bw is the disturbance in state space.
        w_min, w_max = -15.0, 5.0
        H_w = np.array([[1.0], [-1.0]])
        h_w = np.array([w_max, -w_min])
        W_poly = Polyhedron.from_Hrep(H_w, h_w)
        
        # Map to state space: B * W
        BW_poly = W_poly.affine_map(self.B)
        
        # Compute mRPI Set E (Iterative approximation)
        # E = sum (A+BK)^i * (BW)
        E = BW_poly
        Ak = self.A + self.B @ self.K
        
        # Expand until convergence
        for _ in range(100):
            E_next = E.affine_map(Ak) + BW_poly  # Minkowski Sum
            E_next.minHrep(True)
            if E_next == E :
                break
            E = E_next
        self.E = E

        #################################################
        # 4. CONSTRAINT TIGHTENING
        #################################################
        # Define Original Constraints (Relative to Trim)
        # Trim is z=3. Constraint z >= 0 means delta_z >= -3.
        # vz is physically unconstrained, picking large bounds for Polyhedron.
        vz_bound = 50.0
        z_upper = 50.0
        z_lower = -3.0 # z >= 0 -> z_trim + dz >= 0 -> dz >= -3
        
        Hx = np.array([
            [1, 0], [-1, 0], # vz
            [0, 1], [0, -1]  # z
        ])
        hx = np.array([vz_bound, vz_bound, z_upper, -z_lower])
        X = Polyhedron.from_Hrep(Hx, hx)

        # Physical limits for P_avg
        p_min = 40.01
        p_max = 79.99
        
        u_trim = self.us[0] 
        
        # Calculate Delta Limits
        du_min = p_min - u_trim 
        du_max = p_max - u_trim
        
        Hu = np.array([[1.0], [-1.0]])
        hu = np.array([du_max, -du_min])
        U = Polyhedron.from_Hrep(Hu, hu)

        # Tightened Constraints
        self.X_bar = X - self.E
        
        KE = self.E.affine_map(self.K)
        self.U_bar = U - KE

        # Check for Empty Sets (Debugging)
        if self.X_bar.is_empty or self.U_bar.is_empty:
            raise ValueError("Tightened constraints are empty! Tube E is too big. Increase LQR Q/R.")

        #################################################
        # 5. NOMINAL TERMINAL SET
        #################################################
        # Compute Max Invariant Set for Nominal Dynamics (A+BK)
        X_bar_f = self.X_bar.intersect(
            Polyhedron.from_Hrep(self.U_bar.A @ self.K, self.U_bar.b)
        )
        self.Xf = self._max_invariant_set(Ak, X_bar_f)

        #################################################
        # 6. CVXPY PROBLEM SETUP
        #################################################
        nx, nu, N = self.nx, self.nu, self.N

        # Nominal Variables
        X_nom = cp.Variable((nx, N + 1))
        U_nom = cp.Variable((nu, N))
        
        # Parameter: REAL Initial State
        x_init = cp.Parameter(nx)

        cost = 0
        constr = []

        # --- TUBE MPC CONSTRAINT 1: INITIAL INCLUSION ---
        # The nominal start state X_nom[:,0] must be close enough to x_init.
        # x_init \in X_nom[:,0] \oplus E  <==>  x_init - X_nom[:,0] \in E
        constr += [self.E.A @ (x_init - X_nom[:, 0]) <= self.E.b]

        for k in range(N):
            # Nominal Dynamics
            constr += [X_nom[:, k + 1] == self.A @ X_nom[:, k] + self.B @ U_nom[:, k]]
            
            # Tightened Constraints
            constr += [self.X_bar.A @ X_nom[:, k] <= self.X_bar.b]
            constr += [self.U_bar.A @ U_nom[:, k] <= self.U_bar.b]
            
            # Cost on Nominal Path
            cost += cp.quad_form(X_nom[:, k], Q_mpc) + cp.quad_form(U_nom[:, k], R_mpc)

        # Terminal Components
        cost += cp.quad_form(X_nom[:, N], self.Qf)
        constr += [self.Xf.A @ X_nom[:, N] <= self.Xf.b]

        self.ocp = cp.Problem(cp.Minimize(cost), constr)
        
        # Save variables for access in get_u
        self._X_nom = X_nom
        self._U_nom = U_nom
        self._x_init = x_init

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        dx0 = x0 - self.xs
        self._x_init.value = dx0
        
        self.ocp.solve(solver=cp.OSQP, warm_start=True, verbose=False, max_iter=50000, eps_abs=1e-3, eps_rel=1e-3)
            
        if self.ocp.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            u_bar_0 = self._U_nom.value[:, 0]
            x_bar_0 = self._X_nom.value[:, 0]
            
            # Ancillary Feedback: u = u_nom + K(x - x_nom)
            u_feedback = self.K @ (dx0 - x_bar_0)
            du0 = u_bar_0 + u_feedback
            
            x_traj = self._X_nom.value + self.xs.reshape(-1, 1)
            u_traj = self._U_nom.value + self.us.reshape(-1, 1)
        
        u0 = self.us + du0
        u0 = np.clip(u0, 40.0, 80.0)

        return u0, x_traj, u_traj
