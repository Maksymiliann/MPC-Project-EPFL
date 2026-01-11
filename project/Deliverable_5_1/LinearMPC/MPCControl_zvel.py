import numpy as np
import cvxpy as cp
from control import dlqr
from scipy.signal import place_poles


from .MPCControl_base import MPCControl_base


class MPCControl_zvel(MPCControl_base):
    # Reduced subsystem: state = [vU] (z-velocity), input = [dF] (P_avg)
    # In the full rocket state, index 8 is vU; input index 2 is dF (P_avg).
    x_ids: np.ndarray = np.array([8])
    u_ids: np.ndarray = np.array([2])

    def _setup_controller(self) -> None:
        # ----------------------------
        # 1) Cost weights (tune)
        # ----------------------------
        # State is only vz. We want vz -> 0 quickly, but avoid crazy throttle changes.
        Q = np.array([[50.0]])
        R = np.array([[1.0]])

        # LQR terminal cost
        K_lqr, Qf, _ = dlqr(self.A, self.B, Q, R)
        K_lqr = -K_lqr  # use u = Kx convention

        # ----------------------------
        # 2) Constraints in delta coordinates (Δx, Δu)
        # ----------------------------
        # Input: P_avg in [40, 80]
        # We optimize Δu, and actual u = us + Δu.
        u_min = 40.1
        u_max = 79.9
        du_min = u_min - float(self.us[0])
        du_max = u_max - float(self.us[0])

        Mu = np.array([[1.0], [-1.0]])
        mu = np.array([du_max, -du_min])  # Δu <= du_max and -Δu <= -du_min

        # State: no explicit constraint in Part 3.1 for z-velocity besides “stay reasonable”.
        # For a well-defined terminal set, we add a design bound on vz (this is normal in the exercises).
        vz_max = 10.0  # m/s (design bound; tune if needed)
        Fx = np.array([[1.0], [-1.0]])
        fx = np.array([vz_max, vz_max])

        # store (optional, but useful for plotting/report)
        self.K_lqr = K_lqr
        self.Q = Q
        self.R = R
        self.Qf = Qf

        # ============
        # 3) Augmented model for offset-free MPC
        # ============
        # Deviation coordinates:
        #   dx_{k+1} = A dx_k + B du_k + Bd d_k
        #   d_{k+1}  = d_k  (constant disturbance)
        # Measurement: y = C dx (here vz is measured directly)
        C = np.array([[1.0]])

        # IMPORTANT: Bd tuning.
        # For vz, a constant acceleration bias integrates into velocity, so Bd ~ Ts is a good default.
        Bd = np.array([[self.Ts]])
        # Bd = np.array([[1]])

        Aaug = np.block([
            [self.A, Bd],
            [np.zeros((1, 1)), np.eye(1)]
        ])  # 2x2

        Baug = np.vstack([self.B, np.zeros((1, 1))])  # 2x1
        Caug = np.hstack([C, np.zeros((1, 1))])       # 1x2

        self.C = C
        self.Bd = Bd
        self.Aaug = Aaug
        self.Baug = Baug
        self.Caug = Caug

        # ============
        # 4) Observer gain via pole placement (same method as exercise)
        # ============
        # Poles inside unit circle. Closer to 1 -> slower/smoother; smaller -> faster/more aggressive.
        observer_poles = np.array([0.7, 0.9])
        res = place_poles(Aaug.T, Caug.T, observer_poles)
        L = res.gain_matrix.T  # 2x1
        self.L = L

        # Estimator memory (deviation coordinates)
        self._dx_hat = 0.0     # estimate of dx = vz - vz_trim
        self._d_hat = 0.0      # estimate of constant disturbance
        self._du_prev = 0.0    # last applied du (for observer prediction)

        # ----------------------------
        # 5) CVXPY MPC problem
        # ----------------------------
        nx, nu, N = self.nx, self.nu, self.N

        Xvar = cp.Variable((nx, N + 1))  # Δx trajectory
        Uvar = cp.Variable((nu, N))      # Δu trajectory
        x0_param = cp.Parameter(nx)      # Δx(0)

        x_ref = cp.Parameter(nx)
        u_ref = cp.Parameter(nu)

        cost = 0
        constr = []

        # Initial condition
        constr += [Xvar[:, 0] == x0_param]

        for k in range(N):
            # dynamics
            constr += [Xvar[:, k + 1] == self.A @ Xvar[:, k] + self.B @ Uvar[:, k]]

            # stage costs
            cost += cp.quad_form(Xvar[:, k], Q) + cp.quad_form(Uvar[:, k], R)

            # state constraints: Fx (x+xr) <= fx
            constr += [Fx @ (Xvar[:, k] + x_ref) <= fx]

            # input constraints: Mu u <= mu
            constr += [Mu @ (Uvar[:, k] + u_ref) <= mu]

        # terminal cost
        cost += cp.quad_form(Xvar[:, N], Qf)

        # Save handles for get_u()
        self._Xvar = Xvar
        self._Uvar = Uvar
        self._x0_param = x0_param
        self._x_ref = x_ref
        self._u_ref = u_ref
        self._observer_initialized = False
        self.d_hat_log = []
        self.time_log  = []

        self.ocp = cp.Problem(cp.Minimize(cost), constr)

    # -----------------------------
    # Observer step (augmented Luenberger)
    # -----------------------------
    def _observer_update(self, dx_meas: float) -> None:
        """
        One step of the augmented observer.

        Augmented model (deviation coords):
            [dx]_{k+1} = A*dx_k + B*du_k + Bd*d_k
            [ d]_{k+1} = d_k

        Measurement:
            y_k = C*dx_k   (here C = 1)
        We estimate x = [dx; d] with:
            x_hat+ = Aaug x_hat + Baug du_prev + L (y - y_hat)
            y_hat  = Caug x_hat
        """
        # Current augmented estimate x_hat = [dx_hat; d_hat]
        x_hat = np.array([[self._dx_hat],
                          [self._d_hat]])  # shape (2,1)

        # Predicted output y_hat = Caug x_hat
        y_hat = float((self.Caug @ x_hat)[0, 0])

        # Innovation (measurement residual)
        innov = float(dx_meas - y_hat)

        # Observer update (use last applied deviation input du_prev)
        x_hat_next = (self.Aaug @ x_hat) + (self.Baug * float(self._du_prev)) + (self.L * innov)

        # Store back as scalars
        self._dx_hat = float(x_hat_next[0, 0])
        self._d_hat  = float(x_hat_next[1, 0])


    def _target_selector(self, x_target: float) -> tuple[float, float]:
        """
        Disturbance-aware steady-state target computation (delta-formulation).

        We work in deviation-from-trim coordinates:
            dx_{k+1} = A dx_k + B du_k + Bd d_k
            y_k      = C dx_k + Cd d_k   (Cd often = 0)

        We want a steady-state (dxs, dus) such that:
            dxs = A dxs + B dus + Bd * d_hat
            r   = C dxs + Cd * d_hat

        This is equivalent to solving:
            [I - A,  -B] [dxs] = [Bd * d_hat]
            [C,      0] [dus]   [r - Cd*d_hat]
        """
        # Scalars (1D system)
        A  = float(self.A[0, 0])
        B  = float(self.B[0, 0])
        Bd = float(self.Bd[0, 0])

        # Output model: y = C dx + Cd d
        C  = 1.0
        Cd = 0.0  # for your vz measurement, usually no direct disturbance term in output

        d_hat = float(self._d_hat)

        # IMPORTANT: x_target is assumed ABSOLUTE vz target (in m/s),
        # so reference in deviation coords is r = vz_ref - vz_trim.
        vz_trim = float(self.xs[0])
        r = float(x_target - vz_trim)

        # Build and solve the 2x2 linear system
        M = np.array([
            [1.0 - A,  -B],
            [C,         0.0]
        ], dtype=float)

        b = np.array([
            [Bd * d_hat],
            [r - Cd * d_hat]
        ], dtype=float)

        sol = np.linalg.solve(M, b)
        dxs = float(sol[0, 0])
        dus = float(sol[1, 0])

        return dxs, dus


    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        print("\n[zvel MPC step]")

        # -------------------------------------------------
        # 1) measurement -> deviation from trim
        # -------------------------------------------------
        dx0 = float(x0) - float(self.xs)

        if not self._observer_initialized:
            self._dx_hat = dx0      # initialize state estimate
            self._d_hat  = 0.0      # still unknown
            self._observer_initialized = True
        else:
            self._observer_update(dx0)

        # log disturbance estimate
        self.d_hat_log.append(self._d_hat)

        print("[measurement]")
        print(f"  vz_meas        = {float(x0): .4f}")
        print(f"  vz_trim        = {float(self.xs): .4f}")
        print(f"  dx_meas        = {dx0: .4f}")

        # -------------------------------------------------
        # 2) observer update
        # -------------------------------------------------

        print("[observer]")
        print(f"  dx_hat (est)   = {self._dx_hat: .4f}")
        print(f"  d_hat  (est)   = {self._d_hat: .4f}")
        print(f"  du_prev        = {self._du_prev: .4f}")

        # -------------------------------------------------
        # 3) reference selection
        # -------------------------------------------------
        if x_target is None:
            vz_ref_abs = float(self.xs)
        else:
            vz_ref_abs = float(x_target)

        print("[reference]")
        print(f"  vz_target_abs  = {vz_ref_abs: .4f}")

        # -------------------------------------------------
        # 4) disturbance-aware steady state
        # -------------------------------------------------
        dxs, dus = self._target_selector(vz_ref_abs)

        print("[steady-state target]")
        print(f"  dxs (state ss) = {dxs: .4f}")
        print(f"  dus (input ss) = {dus: .4f}")

        # -------------------------------------------------
        # 5) MPC variables (same notation as before)
        # -------------------------------------------------
        dxr = np.array([dxs])
        dur = np.array([dus])

        dX0 = np.array([self._dx_hat - dxs])
        self._x0_param.value = dX0

        self._x_ref.value = dxr
        self._u_ref.value = dur

        print("[mpc initial condition]")
        print(f"  dX0            = {float(dX0[0]): .4f}")

        # -------------------------------------------------
        # 6) solve MPC
        # -------------------------------------------------
        try:
            self.ocp.solve(solver=cp.OSQP, warm_start=True, verbose=False, max_iter=10000)
        except Exception:
            print("[zvel] OSQP solver failed, falling back to PIQP.")
            self.ocp.solve(solver=cp.PIQP, verbose=False)

        if self.ocp.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            raise RuntimeError(f"[zvel] MPC solve failed: status={self.ocp.status}")

        # -------------------------------------------------
        # 7) recover input
        # -------------------------------------------------
        dU0 = self._Uvar.value[:, 0]
        du0 = dur + dU0
        u0  = self.us + du0

        self._du_prev = float(du0[0])

        print("[mpc output]")
        print(f"  du_cmd         = {float(dU0[0]): .4f}")
        print(f"  u_cmd (abs)    = {float(u0[0]): .4f}")
        print("-------------------------------")

        # -------------------------------------------------
        # 8) predicted trajectories
        # -------------------------------------------------
        x_traj = self.xs.reshape(-1, 1) + (dxr.reshape(-1, 1) + self._Xvar.value)
        u_traj = self.us.reshape(-1, 1) + (dur.reshape(-1, 1) + self._Uvar.value)

        return u0, x_traj, u_traj


