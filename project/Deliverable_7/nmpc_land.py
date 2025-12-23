import numpy as np
import casadi as ca
from control import dlqr
from scipy.signal import cont2discrete
from typing import Tuple


class NmpcCtrl:
    """
    Nonlinear MPC controller.
    get_u should provide this functionality: u0, x_ol, u_ol, t_ol = mpc_z_rob.get_u(t0, x0).
    - x_ol shape: (12, N+1); u_ol shape: (4, N); t_ol shape: (N+1,)
    You are free to modify other parts    
    """

    def __init__(self, rocket, H: float, xs: np.ndarray, us: np.ndarray):
        """
        Hint: As in our NMPC exercise, you can evaluate the dynamics of the rocket using 
            CASADI variables x and u via the call rocket.f_symbolic(x,u).
            We create a self.f for you: x_dot = self.f(x,u)
        """    
        self.rocket = rocket
        self.Ts = float(rocket.Ts)
        self.H = float(H)
        self.N = int(round(self.H / self.Ts))

        self.xs = np.array(xs, dtype=float).reshape(12,)
        self.us = np.array(us, dtype=float).reshape(4,)

        # Symbolic continuous dynamics from rocket
        self.f = lambda x, u: rocket.f_symbolic(x, u)[0]

        # For warm-start
        self._prev_X = None
        self._prev_U = None

        self._setup_controller()

    # -------------------------------------------------
    # Discretization (RK4)
    # -------------------------------------------------
    def _rk4_step(self, x, u):
        Ts = self.Ts
        k1 = self.f(x, u)
        k2 = self.f(x + 0.5 * Ts * k1, u)
        k3 = self.f(x + 0.5 * Ts * k2, u)
        k4 = self.f(x + Ts * k3, u)
        return x + (Ts / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # -------------------------------------------------
    # Build OCP (CasADi Opti)
    # -------------------------------------------------
    def _setup_controller(self) -> None:
        nx, nu, N = 12, 4, self.N
        Ts = self.Ts

        opti = ca.Opti()

        # Decision variables
        X = opti.variable(nx, N + 1)
        U = opti.variable(nu, N)

        # Parameter: current measured state
        X0 = opti.parameter(nx, 1)

        # ---- Cost weights (reasonable defaults; you can tune)
        # State is [w(3), phi(3), v(3), p(3)]
        Q = np.diag([
            10.0, 10.0, 10.0,          # angular rates w
            50.0, 100.0, 50.0,      # angles phi (penalize beta more if you want)
            10.0, 10.0, 100.0,       # linear velocities v
            200.0, 200.0, 200.0       # positions p (z strongest)
        ])
        R = np.diag([50.0, 50.0, 1.0, 1.0])  # inputs: [d1, d2, Pavg, Pdiff] 

        # 1) continuous-time linearization: xdot = A_c x + B_c u
        A_c, B_c = self.rocket.linearize(self.xs, self.us)  # (12x12), (12x4) 

        # 2) discretize (same method as in MPCControl_base) 
        C = np.zeros((1, nx))
        D = np.zeros((1, nu))
        A_d, B_d, _, _, _ = cont2discrete((A_c, B_c, C, D), Ts)

        # 3) LQR terminal matrix Qf = P (Riccati solution)
        # dlqr returns (K, S, E) where S is the Riccati matrix P
        K_lqr, Qf, _ = dlqr(A_d, B_d, Q, R)

        self.Q_np  = np.array(Q)
        self.R_np  = np.array(R)
        self.Qf_np = np.array(Qf)

        # Convert to CasADi for use in the Opti objective
        Q  = ca.DM(Q)
        R  = ca.DM(R)
        Qf = ca.DM(Qf)

        # (optional) store for debugging/report
        self.Qf = Qf
        self.K_lqr = -np.array(K_lqr)  # if you want u = Kx convention


        xs = ca.DM(self.xs).reshape((nx, 1))
        us = ca.DM(self.us).reshape((nu, 1))

        # ---- Constraints
        opti.subject_to(X[:, 0] == X0)

        beta_max = float(80.0 * np.pi / 180.0)
        vz_max = 6.0  # m/s
        vz_land = 0.3  # m/s

        # Input bounds (typical from your project)
        # Assumption: u = [d1, d2, P_avg, P_roll]
        Pdiff_max = 20.0
        Pavg_min, Pavg_max = 40.0, 80.0
        d_max = 0.26
        

        for k in range(N):
            # Discrete dynamics
            x_next = self._rk4_step(X[:, k], U[:, k])
            opti.subject_to(X[:, k + 1] == x_next)

            # Constraint (9): z >= 0  (z is p_z = state index 11)
            opti.subject_to(X[11, k] >= 0.0)

            # |beta| <= 80deg (beta is phi_y = state index 4)
            opti.subject_to(X[4, k] <= beta_max)
            opti.subject_to(X[4, k] >= -beta_max)

            # vz <= 5 m/s
            opti.subject_to(X[8, k] <=  vz_max)
            opti.subject_to(X[8, k] >= -vz_max)

            # Input constraints
            opti.subject_to(U[2, k] <= Pavg_max)
            opti.subject_to(U[2, k] >= Pavg_min)

            opti.subject_to(U[0, k] <= d_max)
            opti.subject_to(U[0, k] >= -d_max)

            opti.subject_to(U[1, k] <= d_max)
            opti.subject_to(U[1, k] >= -d_max)

            opti.subject_to(U[3, k] <= Pdiff_max)
            opti.subject_to(U[3, k] >= -Pdiff_max)

        # Also apply z/beta constraint at terminal point
        opti.subject_to(X[11, N] >= 0.0)
        opti.subject_to(X[4, N] <= beta_max)
        opti.subject_to(X[4, N] >= -beta_max)
        opti.subject_to(X[8, N] <=  vz_land)
        opti.subject_to(X[8, N] >= -vz_land)

        # ---- Objective (tracking around the trim steady-state)
        cost = 0
        for k in range(N):
            dx = (X[:, k] - xs)
            du = (U[:, k] - us)
            cost += ca.mtimes([dx.T, Q, dx]) + ca.mtimes([du.T, R, du])

        # Terminal cost
        dxN = (X[:, N] - xs)
        cost += ca.mtimes([dxN.T, Qf, dxN])

        opti.minimize(cost)

        # ---- Solver options (fast-ish)
        opts = {
            "expand": True,
            "print_time": False,
        }
        opti.solver("ipopt", opts)

        # Save
        self.opti = opti
        self._X = X
        self._U = U
        self._X0 = X0


    def _state_cost_breakdown(self, x, Qmat):
        dx = (x - self.xs).reshape(-1, 1)

        # helper block cost (diagonal block only)
        def qslice(i0, i1):
            d = dx[i0:i1, :]
            Qs = Qmat[i0:i1, i0:i1]
            return float(d.T @ Qs @ d)

        out = {}
        out["w"]   = qslice(0, 3)
        out["phi"] = qslice(3, 6)
        out["v"]   = qslice(6, 9)
        out["p"]   = qslice(9, 12)
        out["vz_only"] = float(Qmat[8, 8] * dx[8, 0]**2)
        out["z_only"]  = float(Qmat[11,11] * dx[11,0]**2)

        # full quadratic form
        total = float(dx.T @ Qmat @ dx)

        # diagonal-only contribution
        diag = np.diag(Qmat)
        diag_part = float((dx[:,0]**2 * diag).sum())

        # cross-term contribution = total - diag_part
        cross_part = total - diag_part

        out["total"] = total
        out["diag_part"] = diag_part
        out["cross_part"] = cross_part
        out["blocks_sum"] = out["w"] + out["phi"] + out["v"] + out["p"]

        return out


    def _input_cost_breakdown(self, u):
        du = u - self.us
        costs = {
            "d1":    self.R_np[0, 0] * du[0]**2,
            "d2":    self.R_np[1, 1] * du[1]**2,
            "Pavg":  self.R_np[2, 2] * du[2]**2,
            "Pdiff": self.R_np[3, 3] * du[3]**2,
        }
        costs["total"] = sum(costs.values())
        return {k: float(v) for k, v in costs.items()}

    # -------------------------------------------------
    # MPC call
    # -------------------------------------------------
    def get_u(self, t0: float, x0: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        
        x0 = np.array(x0, dtype=float).reshape(12,)
        print("[parameter injection]")
        print(f"  X0.shape = {x0.reshape(12,1).shape}")
        print(f"  X0(z,beta) = ({x0[11]:.3f}, {x0[4]*180/np.pi:.2f}deg)")
        self.opti.set_value(self._X0, x0.reshape(12, 1))

        print("\n[NMPC get_u]")
        print(f"  t0 = {t0:.2f}")

        print("[initial state]")
        print(f"  z      = {x0[11]: .4f}")
        print(f"  beta   = {x0[4]*180/np.pi: .2f} deg")
        print(f"  vz     = {x0[8]: .4f}")
        print(f"  Pavg(trim) = {self.us[2]: .4f}")

        # Warm start
        if self._prev_X is None:
            # simple initial guess: hold state, hold trim input
            self.opti.set_initial(self._X, np.tile(x0.reshape(12, 1), (1, self.N + 1)))
            self.opti.set_initial(self._U, np.tile(self.us.reshape(4, 1), (1, self.N)))
            print("[warm start] using trim / hold")
            print(f"  U_init = {self.us}")
        else:
            # shift previous solution
            Xg = np.hstack([self._prev_X[:, 1:], self._prev_X[:, -1:]])
            Ug = np.hstack([self._prev_U[:, 1:], self._prev_U[:, -1:]])
            self.opti.set_initial(self._X, Xg)
            self.opti.set_initial(self._U, Ug)
            print("[warm start] shifted previous solution")
            print(f"  U_prev[0] = {self._prev_U[:,0]}")

        try:
            sol = self.opti.solve()
        except RuntimeError as e:
            print("\n[NMPC SOLVER FAILED]")
            print("Reason:", e)

            print("\n[DEBUG VALUES]")
            Xdbg = self.opti.debug.value(self._X)
            Udbg = self.opti.debug.value(self._U)
            X0dbg = self.opti.debug.value(self._X0)

            print(f"  X0(z)     = {X0dbg[11]}")
            print(f"  X0(beta) = {X0dbg[4]*180/np.pi:.2f} deg")
            print(f"  U0 guess = {Udbg[:,0]}")

            raise

        X_ol = np.array(sol.value(self._X))
        U_ol = np.array(sol.value(self._U))

        # ============================
        # DETAILED COST BREAKDOWN
        # ============================
        state_costs_sum = {
            "w": 0.0, "phi": 0.0, "v": 0.0, "p": 0.0, "total": 0.0,
            "vz_only": 0.0, "z_only": 0.0
        }
        input_costs_sum = {
            "d1": 0, "d2": 0, "Pavg": 0, "Pdiff": 0, "total": 0
        }

        for k in range(self.N):
            sc = self._state_cost_breakdown(X_ol[:, k], self.Q_np)
            ic = self._input_cost_breakdown(U_ol[:, k])

            for key in state_costs_sum:
                state_costs_sum[key] += sc[key]

            for key in input_costs_sum:
                input_costs_sum[key] += ic[key]

        # terminal cost breakdown
        terminal_sc = self._state_cost_breakdown(X_ol[:, -1], self.Qf_np)

        # ============================
        # PRINT RESULTS
        # ============================
        print("\n[COST BREAKDOWN – STAGE (STATE)]")
        print(f"  w       : {state_costs_sum['w']:.2f}")
        print(f"  phi     : {state_costs_sum['phi']:.2f}")
        print(f"  v       : {state_costs_sum['v']:.2f}   (vz_only={state_costs_sum['vz_only']:.2f})")
        print(f"  p       : {state_costs_sum['p']:.2f}   (z_only={state_costs_sum['z_only']:.2f})")
        print(f"  total   : {state_costs_sum['total']:.2f}")

        print("\n[COST BREAKDOWN – STAGE (INPUT)]")
        for k, v in input_costs_sum.items():
            print(f"  {k:6s}: {v:8.2f}")

        print("\n[COST BREAKDOWN – TERMINAL (STATE)]")
        print(f"  w       : {terminal_sc['w']:.2f}")
        print(f"  phi     : {terminal_sc['phi']:.2f}")
        print(f"  v       : {terminal_sc['v']:.2f}   (vz_only={terminal_sc['vz_only']:.2f})")
        print(f"  p       : {terminal_sc['p']:.2f}   (z_only={terminal_sc['z_only']:.2f})")
        print(f"  total   : {terminal_sc['total']:.2f}")
        print(f"  blocks_sum : {terminal_sc['blocks_sum']:.2f}")
        print(f"  diag_part  : {terminal_sc['diag_part']:.2f}")
        print(f"  cross_part : {terminal_sc['cross_part']:.2f}")
        print(f"  total      : {terminal_sc['total']:.2f}")

        total_stage = state_costs_sum["total"] + input_costs_sum["total"]
        total_cost  = total_stage + terminal_sc["total"]

        print("\n[COST SUMMARY]")
        print(f"  Stage total    : {total_stage:.2f}")
        print(f"  Terminal total : {terminal_sc['total']:.2f}")
        print(f"  TOTAL COST     : {total_cost:.2f}")
        print(f"  Terminal ratio : {100*terminal_sc['total']/(total_cost+1e-9):.1f}%")

        # store for next warm-start
        self._prev_X = X_ol
        self._prev_U = U_ol

        u0 = U_ol[:, 0].copy()
        t_ol = t0 + np.arange(self.N + 1) * self.Ts

        return u0, X_ol, U_ol, t_ol
