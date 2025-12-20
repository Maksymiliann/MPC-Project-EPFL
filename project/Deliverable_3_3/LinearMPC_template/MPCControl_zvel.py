import numpy as np
import cvxpy as cp
from control import dlqr
from mpt4py import Polyhedron

from .MPCControl_base import MPCControl_base


class MPCControl_zvel(MPCControl_base):
    # Reduced subsystem: state = [vU] (z-velocity), input = [dF] (P_avg)
    # In the full rocket state, index 8 is vU; input index 2 is dF (P_avg).
    x_ids: np.ndarray = np.array([8])
    u_ids: np.ndarray = np.array([2])

    @staticmethod
    def _max_invariant_set(A_cl: np.ndarray, X: Polyhedron, max_iter: int = 50) -> Polyhedron:
        """
        Maximal positive invariant set for x+ = A_cl x inside X:
            O_{k+1} = O_k ∩ Pre(O_k),  Pre(O) = {x | A_cl x ∈ O}.
        If O = {x | Hx <= h} => Pre(O) = {x | H A_cl x <= h}.
        """
        O = X
        converged = False

        for _ in range(max_iter):
            Oprev = O
            H, h = O.A, O.b
            PreO = Polyhedron.from_Hrep(H @ A_cl, h)
            O = O.intersect(PreO)
            O.minHrep(True)

            # helps some mpt4py backends finalize representation
            _ = O.Vrep

            if O == Oprev:
                converged = True
                break

        if not converged:
            print("[zvel] Warning: max invariant set did not converge within max_iter.")
        return O

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
        U = Polyhedron.from_Hrep(Mu, mu)

        # State: no explicit constraint in Part 3.1 for z-velocity besides “stay reasonable”.
        # For a well-defined terminal set, we add a design bound on vz (this is normal in the exercises).
        vz_max = 10.0  # m/s (design bound; tune if needed)
        Fx = np.array([[1.0], [-1.0]])
        fx = np.array([vz_max, vz_max])
        X = Polyhedron.from_Hrep(Fx, fx)

        # Terminal set: Xf = maximal invariant subset of X ∩ {x | Kx ∈ U}
        KU = Polyhedron.from_Hrep(U.A @ K_lqr, U.b)   # {x | U.A*(Kx) <= U.b}
        Xf0 = X.intersect(KU)
        Acl = self.A + self.B @ K_lqr
        Xf = self._max_invariant_set(Acl, Xf0)

        # store (optional, but useful for plotting/report)
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

        # terminal constraint: Xf.A x_N <= Xf.b
        constr += [Xf.A @ (Xvar[:, N] + x_ref) <= Xf.b]

        # Save handles for get_u()
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
            self.ocp.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        except Exception:
            # fallback (some setups use PIQP in exercises)
            self.ocp.solve(solver=cp.PIQP, verbose=False)

        if self.ocp.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            raise RuntimeError(f"[xvel] MPC solve failed: status={self.ocp.status}")

        dU0 = self._Uvar.value[:, 0]
        du0 = dur + dU0
        u0 = self.us + du0

        x_traj = self.xs.reshape(-1,1) + (dxr.reshape(-1,1) + self._Xvar.value)
        u_traj = self.us.reshape(-1,1) + (dur.reshape(-1,1) + self._Uvar.value)

        return u0, x_traj, u_traj
