import numpy as np
import cvxpy as cp
from control import dlqr
from mpt4py import Polyhedron

from .MPCControl_base import MPCControl_base


class MPCControl_roll(MPCControl_base):
    # Reduced subsystem: states = [wz, gamma], input = [Pdiff]
    # In full rocket state: wz index 2, gamma index 5; input Pdiff index 3
    x_ids: np.ndarray = np.array([2, 5])
    u_ids: np.ndarray = np.array([3])

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
            print("[roll] Warning: max invariant set did not converge within max_iter.")
        return O

    def _setup_controller(self) -> None:
        # ----------------------------
        # 1) Cost weights (tune)
        # ----------------------------
        # States: [wz, gamma]. Primary goal: gamma -> 0. Penalize wz to damp oscillations.
        Q = np.diag([1.0, 30.0])
        R = np.diag([1.0])

        # LQR terminal ingredients (discrete-time)
        K_lqr, Qf, _ = dlqr(self.A, self.B, Q, R)
        K_lqr = -K_lqr  # use u = Kx convention

        # ----------------------------
        # 2) Constraints in delta coordinates (Δx, Δu)
        # ----------------------------
        # Input constraint: |Pdiff| <= 20
        u_min = -19.9
        u_max = 19.9
        du_min = u_min - float(self.us[0])
        du_max = u_max - float(self.us[0])

        Mu = np.array([[1.0], [-1.0]])
        mu = np.array([du_max, -du_min])  # Δu <= du_max and -Δu <= -du_min
        U = Polyhedron.from_Hrep(Mu, mu)

        # Roll subsystem is valid for any gamma in the project statement,
        # but for terminal set computation we add *design bounds* (optional but recommended).
        wz_max = np.deg2rad(200.0)    # rad/s design bound
        gamma_max = np.deg2rad(180.0) # rad design bound (wide; roll can be large)

        Fx = np.array([
            [ 1.0,  0.0],   #  wz <= wz_max
            [-1.0,  0.0],   # -wz <= wz_max
            [ 0.0,  1.0],   #  gamma <= gamma_max
            [ 0.0, -1.0],   # -gamma <= gamma_max
        ])
        fx = np.array([wz_max, wz_max, gamma_max, gamma_max])
        X = Polyhedron.from_Hrep(Fx, fx)

        # Terminal set: maximal invariant subset of X ∩ {x | Kx ∈ U}
        KU = Polyhedron.from_Hrep(U.A @ K_lqr, U.b)  # {x | U.A*(Kx) <= U.b}
        Xf0 = X.intersect(KU)
        Acl = self.A + self.B @ K_lqr
        Xf = self._max_invariant_set(Acl, Xf0)

        # Store (optional, useful for plotting/report)
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

        cost = 0
        constr = [Xvar[:, 0] == x0_param]

        for k in range(N):
            # dynamics
            constr += [Xvar[:, k + 1] == self.A @ Xvar[:, k] + self.B @ Uvar[:, k]]

            # stage cost
            cost += cp.quad_form(Xvar[:, k], Q) + cp.quad_form(Uvar[:, k], R)

            # constraints
            constr += [Fx @ Xvar[:, k] <= fx]
            constr += [Mu @ Uvar[:, k] <= mu]

        # terminal cost + terminal constraint
        cost += cp.quad_form(Xvar[:, N], Qf)
        constr += [Xf.A @ Xvar[:, N] <= Xf.b]

        # Save handles for get_u()
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
            self.ocp.solve(verbose=False)

        if self.ocp.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            raise RuntimeError(f"[roll] MPC solve failed: status={self.ocp.status}")

        du0 = self._Uvar.value[:, 0]
        u0 = self.us + du0  # convert Δu -> actual u

        # Predicted trajectories in actual coordinates
        x_traj = self.xs.reshape(-1, 1) + self._Xvar.value
        u_traj = self.us.reshape(-1, 1) + self._Uvar.value

        return u0, x_traj, u_traj
