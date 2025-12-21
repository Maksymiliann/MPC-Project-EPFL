import numpy as np
import cvxpy as cp
from control import dlqr
from scipy.signal import place_poles

from .MPCControl_base import MPCControl_base


class MPCControl_zvel(MPCControl_base):
    # Reduced subsystem: state = [vU] (z-velocity), input = [dF] (P_avg)
    # In the full rocket state, index 8 is vU; input index 2 is dF (P_avg).
    x_ids: np.ndarray = np.array([8])   # vz
    u_ids: np.ndarray = np.array([2])   # Pavg

    # -----------------------------
    # Setup
    # -----------------------------
    def _setup_controller(self) -> None:
        # ============
        # 1) MPC weights
        # ============
        # Tune these (Q bigger -> faster vz regulation; R bigger -> smoother Pavg)
        Q = np.array([[50.0]])
        R = np.array([[1.0]])

        # Terminal COST is allowed (not a terminal set). Keep it to help stability.
        K_lqr, Qf, _ = dlqr(self.A, self.B, Q, R)
        K_lqr = -K_lqr

        self.Q = Q
        self.R = R
        self.Qf = Qf
        self.K_lqr = K_lqr

        # ============
        # 2) Constraints (in deviation coordinates around trim)
        # ============
        # Pavg physical bounds
        u_min = 40.1
        u_max = 79.9

        du_min = u_min - float(self.us[0])
        du_max = u_max - float(self.us[0])

        # Represent du in H-rep: Mu * du <= mu
        # du <= du_max
        # -du <= -du_min  <=> du >= du_min
        Mu = np.array([[1.0], [-1.0]])
        mu = np.array([du_max, -du_min])

        self.Mu, self.mu = Mu, mu

        # Optional design bound on vz deviation (keeps the polytope bounded / avoids weird solutions)
        # This is NOT a physical constraint; it's just to keep the optimization well-defined.
        vz_max = 10.0  # m/s
        Fx = np.array([[1.0], [-1.0]])
        fx = np.array([vz_max, vz_max])

        self.Fx, self.fx = Fx, fx

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
        # Bd = np.array([[self.Ts]])
        Bd = np.array([[1]])

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

        # ============
        # 5) CVXPY MPC (NO terminal set)
        # ============
        nx, nu, N = self.nx, self.nu, self.N

        # Decision variables are the error relative to a (disturbance-aware) steady-state:
        #   Xvar ~ (dx - dxs),   Uvar ~ (du - dus)
        Xvar = cp.Variable((nx, N + 1))
        Uvar = cp.Variable((nu, N))

        # Parameters:
        # x0_param = initial error: (dx_hat - dxs)
        x0_param = cp.Parameter(nx)

        # (dxs, dus) steady-state targets in deviation coordinates
        x_ref = cp.Parameter(nx)  # dxs
        u_ref = cp.Parameter(nu)  # dus

        cost = 0
        constr = [Xvar[:, 0] == x0_param]

        for k in range(N):
            # Linear dynamics in error coordinates (since steady-state cancels out):
            constr += [Xvar[:, k + 1] == self.A @ Xvar[:, k] + self.B @ Uvar[:, k]]

            cost += cp.quad_form(Xvar[:, k], Q) + cp.quad_form(Uvar[:, k], R)

            # Apply constraints around trim, shifted by steady-state (x_ref,u_ref):
            # actual deviation from trim is (x_ref + Xvar), (u_ref + Uvar)
            constr += [Fx @ (Xvar[:, k] + x_ref) <= fx]
            constr += [Mu @ (Uvar[:, k] + u_ref) <= mu]

        # Terminal COST
        cost += cp.quad_form(Xvar[:, N], Qf)

        self._Xvar = Xvar
        self._Uvar = Uvar
        self._x0_param = x0_param
        self._x_ref = x_ref
        self._u_ref = u_ref

        self.ocp = cp.Problem(cp.Minimize(cost), constr)

    # -----------------------------
    # Observer step
    # -----------------------------
    def _observer_update(self, dx_meas: float) -> None:
        """
        One step of the augmented Luenberger observer using measurement dx_meas = (vz_meas - vz_trim).
        """
        xaug = np.array([[self._dx_hat],
                         [self._d_hat]])  # 2x1

        # predicted output
        yhat = float((self.Caug @ xaug)[0, 0])
        innov = dx_meas - yhat

        # update
        xaug_next = (self.Aaug @ xaug) + (self.Baug * float(self._du_prev)) + (self.L * innov)

        self._dx_hat = float(xaug_next[0, 0])
        self._d_hat = float(xaug_next[1, 0])

    # # -----------------------------
    # # Target selector (disturbance-aware steady state)
    # # -----------------------------
    # def _target_selector(self, vz_ref_abs: float) -> tuple[float, float]:
    #     """
    #     Compute steady-state (dxs, dus) in deviation coordinates such that:
    #       dxs = A dxs + B dus + Bd * d_hat
    #       C dxs = (vz_ref_abs - vz_trim)
    #     """
    #     A = float(self.A[0, 0])
    #     B = float(self.B[0, 0])
    #     Bd = float(self.Bd[0, 0])
    #     C = 1.0

    #     vz_trim = float(self.xs[0])
    #     r_dx = float(vz_ref_abs - vz_trim)

    #     # Solve:
    #     # (1-A) dxs - B dus = Bd * d_hat
    #     # C dxs = r_dx
    #     M = np.array([[1.0 - A, -B],
    #                   [C,        0.0]])
    #     b = np.array([Bd * float(self._d_hat), r_dx])

    #     dxs, dus = np.linalg.solve(M, b)
    #     return float(dxs), float(dus)

    def _target_selector(self, vz_ref_abs: float) -> tuple[float, float]:
        """
        CVXPY version (exercise-style):
        Find steady-state (dxs, dus) in deviation coordinates such that:
            dxs = A dxs + B dus + Bd * d_hat
            C dxs = r_dx
        and minimize dus^2, subject to input bounds.
        """
        # Scalars (1D)
        A = float(self.A[0, 0])
        B = float(self.B[0, 0])
        Bd = float(self.Bd[0, 0])
        C = 1.0

        vz_trim = float(self.xs[0])
        r_dx = float(vz_ref_abs - vz_trim)   # desired deviation in vz


        # --- DEBUG PRINTS (ADD HERE) ---
        du_req = float(-(Bd / B) * self._d_hat) if abs(B) > 1e-9 else 0.0
        du_max = float(self.mu[0])
        du_min = float(-self.mu[1])

        print("\n[zvel target selector]")
        print(f"  vz_ref_abs = {vz_ref_abs:.3f}")
        print(f"  d_hat      = {self._d_hat:.4f}")
        print(f"  du_req     = {du_req:.4f}")
        print(f"  du bounds  = [{du_min:.4f}, {du_max:.4f}]")
        print(f"  Pavg_req   = {self.us[0] + du_req:.2f}")
        print(f"  Pavg bounds= [40, 80]")
        print("--------------------------")


        # Decision variables
        dxs_var = cp.Variable(1)  # steady-state deviation state
        dus_var = cp.Variable(1)  # steady-state deviation input

        # Objective: minimize input effort (like exercise)
        obj = cp.Minimize(cp.sum_squares(dus_var))

        # Constraints: steady-state + output equation + input bounds
        cons = []

        # dxs = A dxs + B dus + Bd d_hat   -> (1-A)dxs - B dus = Bd d_hat
        cons += [(1.0 - A) * dxs_var - B * dus_var == Bd * float(self._d_hat)]

        # C dxs = r_dx
        cons += [C * dxs_var == r_dx]

        # Input bounds in deviation coordinates: Mu*(dus) <= mu
        # (same Mu, mu you already built)
        cons += [self.Mu @ dus_var <= self.mu]

        prob = cp.Problem(obj, cons)
        prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)

        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            raise RuntimeError(f"[zvel] steady-state target solve failed: status={prob.status}")

        dxs = float(dxs_var.value.item())
        dus = float(dus_var.value.item())
        return dxs, dus

    # -----------------------------
    # MPC call
    # -----------------------------
    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        x0 is treated as a MEASUREMENT.
        We:
          1) update observer -> (dx_hat, d_hat)
          2) compute disturbance-aware steady-state (dxs, dus) for desired vz
          3) solve MPC around that steady-state
        Returns:
          u0 (absolute), x_traj (absolute), u_traj (absolute)
        """
        # ---- measurement to deviation dx_meas
        if x0.size == 1:
            vz_meas = float(x0[0])
        else:
            vz_meas = float(x0[8])  # full state vector case
        dx_meas = vz_meas - float(self.xs[0])

        # ---- observer update
        self._observer_update(dx_meas)

        # ---- choose vz reference (absolute)
        if x_target is None:
            vz_ref_abs = float(self.xs[0])  # hold trim
        else:
            if x_target.size == 1:
                vz_ref_abs = float(x_target[0])
            else:
                vz_ref_abs = float(x_target[8])

        # ---- disturbance-aware steady-state (in deviation coords)
        dxs, dus = self._target_selector(vz_ref_abs)

        # Set parameters
        self._x_ref.value = np.array([dxs])
        self._u_ref.value = np.array([dus])

        # Initial error relative to steady-state
        self._x0_param.value = np.array([self._dx_hat - dxs])

        # ---- solve
        try:
            self.ocp.solve(solver=cp.OSQP, warm_start=True, verbose=False, max_iter=20000)
        except Exception:
            self.ocp.solve(solver=cp.PIQP, verbose=False)

        if self.ocp.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            raise RuntimeError(f"[zvel] MPC solve failed: status={self.ocp.status}")

        # ---- recover absolute input
        dU0 = float(self._Uvar.value[:, 0])     # (du - dus) at k=0
        du0 = dus + dU0                        # total du
        u0 = float(self.us[0] + du0)           # absolute Pavg

        # store applied du for next observer prediction
        self._du_prev = du0

        # Predicted trajectories (absolute)
        dx_traj = dxs + self._Xvar.value       # deviation from trim along horizon
        du_traj = dus + self._Uvar.value

        x_traj = self.xs.reshape(-1, 1) + dx_traj
        u_traj = self.us.reshape(-1, 1) + du_traj

        return np.array([u0]), x_traj, u_traj
