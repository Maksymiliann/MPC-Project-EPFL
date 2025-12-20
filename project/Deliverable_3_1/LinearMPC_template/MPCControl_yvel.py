import numpy as np
import cvxpy as cp
from control import dlqr
from mpt4py import Polyhedron

from .MPCControl_base import MPCControl_base


class MPCControl_yvel(MPCControl_base):
    # states: [wx, alpha, vy], input: [d1]
    x_ids: np.ndarray = np.array([0, 3, 7])
    u_ids: np.ndarray = np.array([0])

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
        # Penalize alpha strongly to stay within linearization validity, and vy for velocity regulation.
        Q = np.diag([1.0, 50.0, 10.0])
        R = np.diag([1.0])

        # LQR terminal ingredients (discrete-time)
        K_lqr, Qf, _ = dlqr(self.A, self.B, Q, R)
        K_lqr = -K_lqr  # dlqr returns u = -Kx in many conventions; we use u = Kx

        # ----------------------------
        # 2) Build constraint sets in delta-coordinates (Δx, Δu)
        # ----------------------------
        alpha_max = np.deg2rad(9.9)
        d1_max = np.deg2rad(14.9)

        # Input constraint: |d1| <= 15deg  ->  us + Δu within [-d1_max, d1_max]
        # i.e. Δu <= d1_max - us,  -Δu <= d1_max + us
        Mu = np.array([[1.0], [-1.0]])
        mu = np.array([d1_max - self.us[0], d1_max + self.us[0]])
        U = Polyhedron.from_Hrep(Mu, mu)

        # State constraints:
        # Only alpha is a "real" constraint from spec, but we also put reasonable bounds
        # on wx and vy so the terminal set is well-defined/bounded.
        wx_max = 5.0    # rad/s (tunable “design bound”)
        vy_max = 10.0   # m/s   (tunable “design bound”)

        # Constraints on actual alpha: |alpha| <= alpha_max
        # alpha = xs_alpha + Δalpha
        xs_alpha = self.xs[1]

        Fx = np.array([
            [ 1.0,  0.0,  0.0],   #  wx <= wx_max
            [-1.0,  0.0,  0.0],   # -wx <= wx_max
            [ 0.0,  1.0,  0.0],   #  alpha <= alpha_max
            [ 0.0, -1.0,  0.0],   # -alpha <= alpha_max
            [ 0.0,  0.0,  1.0],   #  vy <= vy_max
            [ 0.0,  0.0, -1.0],   # -vy <= vy_max
        ])
        fx = np.array([
            wx_max,
            wx_max,
            alpha_max - xs_alpha,   # Δalpha <= alpha_max - xs_alpha
            alpha_max + xs_alpha,   # -Δalpha <= alpha_max + xs_alpha
            vy_max,
            vy_max
        ])
        X = Polyhedron.from_Hrep(Fx, fx)

        # Terminal set: max invariant set inside X ∩ {x | Kx ∈ U}
        KU = Polyhedron.from_Hrep(U.A @ K_lqr, U.b)  # {x | U.A*(Kx) <= U.b}
        Xf0 = X.intersect(KU)
        Acl = self.A + self.B @ K_lqr
        Xf = self._max_invariant_set(Acl, Xf0)
        # Xf = self._max_invariant_set(self.A + self.B @ K_lqr, X.intersect(KU))

        # Store for debugging/plotting later if you want
        self.K_lqr = K_lqr
        self.Q = Q
        self.R = R
        self.Qf = Qf
        self.X = X
        self.U = U
        self.Xf = Xf

        # ----------------------------
        # 3) CvyPY MPC problem
        # ----------------------------
        nx, nu, N = self.nx, self.nu, self.N

        Xvar = cp.Variable((nx, N + 1))
        Uvar = cp.Variable((nu, N))
        x0_param = cp.Parameter(nx)

        cost = 0
        constr = []

        # Initial condition
        constr += [Xvar[:, 0] == x0_param]

        for k in range(N):
            # dynamics
            constr += [Xvar[:, k + 1] == self.A @ Xvar[:, k] + self.B @ Uvar[:, k]]

            # stage costs
            cost += cp.quad_form(Xvar[:, k], Q) + cp.quad_form(Uvar[:, k], R)

            # state constraints: Fx x <= fx
            constr += [Fx @ Xvar[:, k] <= fx]

            # input constraints: Mu u <= mu
            constr += [Mu @ Uvar[:, k] <= mu]

        # terminal cost
        cost += cp.quad_form(Xvar[:, N], Qf)

        # terminal constraint: Xf.A x_N <= Xf.b
        constr += [Xf.A @ Xvar[:, N] <= Xf.b]

        self._Xvar = Xvar
        self._Uvar = Uvar
        self._x0_param = x0_param

        self.ocp = cp.Problem(cp.Minimize(cost), constr)

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        dx0 = x0 - self.xs
        self._x0_param.value = dx0

        # Solve
        try:
            self.ocp.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        except Exception:
            # fallback (some setups use PIQP in exercises)
            self.ocp.solve(solver=cp.PIQP, verbose=False)

        if self.ocp.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            raise RuntimeError(f"[xvel] MPC solve failed: status={self.ocp.status}")

        du0 = self._Uvar.value[:, 0]
        u0 = self.us + du0  # convert Δu -> u

        # Return predicted trajectories in *actual* coordinates
        x_traj = self.xs.reshape(-1, 1) + self._Xvar.value
        u_traj = self.us.reshape(-1, 1) + self._Uvar.value

        return u0, x_traj, u_traj
