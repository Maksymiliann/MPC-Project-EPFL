import numpy as np
import cvxpy as cp
from control import dlqr
from mpt4py import Polyhedron

from .MPCControl_base import MPCControl_base


class MPCControl_xvel(MPCControl_base):
    # states: [wy, beta, vx], input: [d2]
    x_ids: np.ndarray = np.array([1, 4, 6])
    u_ids: np.ndarray = np.array([1])

    @staticmethod
    def _max_invariant_set(A_cl: np.ndarray, X: Polyhedron, max_iter: int = 50) -> Polyhedron:
        """
        Compute maximal positive invariant set for x+ = A_cl x inside X
        using: O_{k+1} = O_k ∩ Pre(O_k), where Pre(O) = {x | A_cl x ∈ O}.
        In H-rep: if O = {x | Hx <= h} then Pre(O) = {x | H A_cl x <= h}.
        """
        O = X
        converged = False

        for itr in range(max_iter):
            Oprev = O
            H, h = O.A, O.b  # Hx <= h
            PreO = Polyhedron.from_Hrep(H @ A_cl, h)
            O = O.intersect(PreO)
            O.minHrep(True)

            # (Same “touch Vrep once” trick as the exercise notebooks often need)
            _ = O.Vrep

            if O == Oprev:
                converged = True
                break

        if not converged:
            print("[xvel] Warning: max invariant set did not converge within max_iter.")
        return O

    def _setup_controller(self) -> None:
        # ----------------------------
        # 1) Choose weights (tune these!)
        # ----------------------------
        # Penalize beta strongly to stay within linearization validity, and vx for velocity regulation.
        Q = np.diag([20.0, 50.0, 10.0])
        R = np.diag([1.0])

        # LQR terminal ingredients (discrete-time)
        K_lqr, Qf, _ = dlqr(self.A, self.B, Q, R)
        K_lqr = -K_lqr  # dlqr returns u = -Kx in many conventions; we use u = Kx

        # ----------------------------
        # 2) Build constraint sets in delta-coordinates (Δx, Δu)
        # ----------------------------
        beta_max = np.deg2rad(9.9)
        d2_max = np.deg2rad(14.9)

        # Input constraint: |d2| <= 15deg  ->  us + Δu within [-d2_max, d2_max]
        # i.e. Δu <= d2_max - us,  -Δu <= d2_max + us
        Mu = np.array([[1.0], [-1.0]])
        mu = np.array([d2_max - self.us[0], d2_max + self.us[0]])
        U = Polyhedron.from_Hrep(Mu, mu)

        # State constraints:
        # Only beta is a "real" constraint from spec, but we also put reasonable bounds
        # on wy and vx so the terminal set is well-defined/bounded.
        wy_max = 5.0    # rad/s (tunable “design bound”)
        vx_max = 10.0   # m/s   (tunable “design bound”)

        # Constraints on actual beta: |beta| <= beta_max
        # beta = xs_beta + Δbeta
        xs_beta = self.xs[1]

        Fx = np.array([
            [ 1.0,  0.0,  0.0],   #  wy <= wy_max
            [-1.0,  0.0,  0.0],   # -wy <= wy_max
            [ 0.0,  1.0,  0.0],   #  beta <= beta_max
            [ 0.0, -1.0,  0.0],   # -beta <= beta_max
            [ 0.0,  0.0,  1.0],   #  vx <= vx_max
            [ 0.0,  0.0, -1.0],   # -vx <= vx_max
        ])
        fx = np.array([
            wy_max,
            wy_max,
            beta_max - xs_beta,   # Δbeta <= beta_max - xs_beta
            beta_max + xs_beta,   # -Δbeta <= beta_max + xs_beta
            vx_max,
            vx_max
        ])
        X = Polyhedron.from_Hrep(Fx, fx)

        # Terminal set: max invariant set inside X ∩ {x | Kx ∈ U}
        KU = Polyhedron.from_Hrep(U.A @ K_lqr, U.b)  # {x | U.A*(Kx) <= U.b}
        Xf = self._max_invariant_set(self.A + self.B @ K_lqr, X.intersect(KU))

        # Store for debugging/plotting later if you want
        self.K_lqr = K_lqr
        self.Q = Q
        self.R = R
        self.Qf = Qf
        self.X = X
        self.U = U
        self.Xf = Xf

        # ----------------------------
        # 3) CVXPY MPC problem
        # ----------------------------
        nx, nu, N = self.nx, self.nu, self.N

        Xvar = cp.Variable((nx, N + 1))
        Uvar = cp.Variable((nu, N))
        x0_param = cp.Parameter(nx)

        x_ref = cp.Parameter(nx)
        u_ref = cp.Parameter(nu)

        slack = cp.Variable((Fx.shape[0], N), nonneg=True)

        cost = 0
        constr = []

        # Initial condition
        constr += [Xvar[:, 0] == x0_param]

        for k in range(N):
            # dynamics
            constr += [Xvar[:, k + 1] == self.A @ Xvar[:, k] + self.B @ Uvar[:, k]]

            # stage costs
            cost += cp.quad_form(Xvar[:, k], Q) + cp.quad_form(Uvar[:, k], R)

            # state constraints: Fx (x+xr) <= fx + slack
            constr += [Fx @ (Xvar[:, k] + x_ref) <= fx + slack[:,k]]

            # input constraints: Mu u <= mu
            constr += [Mu @ (Uvar[:, k] + u_ref) <= mu]

            cost += 10000 * cp.sum(slack[:, k])

        # terminal cost
        cost += cp.quad_form(Xvar[:, N], Qf)

        # terminal constraint: Xf.A x_N <= Xf.b
        constr += [Xf.A @ (Xvar[:, N] + x_ref) <= Xf.b]

        # We allow the state to be outside the set by amount 'slack'
        # H * x <= h + slack
        # constr += [Xf.A @ (Xvar[:, N] + x_ref) <= Xf.b + slack[:,N]]
        
        # Make it extremely expensive so the solver only uses it if it HAS to.
        # cost += 10000 * cp.sum(slack[:, N])
        # cost += rho * cp.sum_squares(slack[:, N])  # quadratic

        self._Xvar = Xvar
        self._Uvar = Uvar
        self._x0_param = x0_param
        self._x_ref = x_ref
        self._u_ref = u_ref

        self.ocp = cp.Problem(cp.Minimize(cost), constr)

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        # convert to deviation-from-trim coordinates (dx = x - xs, du = u - us)
        dx0 = x0 - self.xs
        dxr = (x_target - self.xs) if x_target is not None else np.zeros_like(self.xs)
        dur = (u_target - self.us) if u_target is not None else np.zeros_like(self.us)

        # initial deviation to regulate
        dX0 = dx0 - dxr
        self._x0_param.value = dX0

        # Set tracking references (for constraint shifting)
        self._x_ref.value = dxr
        self._u_ref.value = dur

        # Solve
        try:
            self.ocp.solve(solver=cp.OSQP, warm_start=True, verbose=False, max_iter=50000)
        except Exception:
            # fallback (some setups use PIQP in exercises)
            print("[xvel] OSQP solver failed, falling back to PIQP.")
            self.ocp.solve(solver=cp.PIQP, verbose=False)

        if self.ocp.status in [cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE]:
            print(f"\n--- MPC INFEASIBLE at t={self._x0_param.value} ---")
            print(f"\n MPC CRASH REPORT: {self.__class__.__name__} ")
            print(f"Solver Status: {self.ocp.status}")
            
            # --- Check Angle Constraints (The usual suspect) ---
            # For xvel/yvel, the Angle is usually at index 1 of the subsystem state
            # xvel: [wy, beta, vx] -> beta is index 1
            # yvel: [wx, alpha, vy] -> alpha is index 1
            if len(x0) == 3: 
                angle_val = x0[1]
                limit_deg = 9.9  # The limit you set in _setup_controller
                print(f"-> Current Angle: {np.rad2deg(angle_val):.4f} deg")
                print(f"-> Constraint:    +/- {limit_deg} deg")
                
                if abs(np.rad2deg(angle_val)) > limit_deg:
                    print(" VIOLATION: State is outside Hard Constraints!")
                else:
                    print("/-/ State is inside constraints")

            # --- Check Velocity/Target ---
            if x_target is not None:
                # xvel/yvel velocity is usually index 2
                vel_current = x0[2] if len(x0) > 2 else x0[0]
                vel_target = x_target[2] if len(x_target) > 2 else x_target[0]
                print(f"-> Current Vel:   {vel_current:.4f} m/s")
                print(f"-> Target Vel:    {vel_target:.4f} m/s")

            print("----------------------------------------------\n")
        elif self.ocp.status == cp.USER_LIMIT:
            print("[x_vel] Warning: Solver hit user-defined limit.")
        if self.ocp.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE, cp.USER_LIMIT]:
            raise RuntimeError(f"[x_vel] MPC solve failed: status={self.ocp.status}")

        dU0 = self._Uvar.value[:, 0]
        du0 = dur + dU0
        u0 = self.us + du0

        x_traj = self.xs.reshape(-1,1) + (dxr.reshape(-1,1) + self._Xvar.value)
        u_traj = self.us.reshape(-1,1) + (dur.reshape(-1,1) + self._Uvar.value)

        return u0, x_traj, u_traj
