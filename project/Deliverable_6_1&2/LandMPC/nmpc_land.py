import numpy as np
import casadi as ca
from typing import Tuple


class NmpcCtrl:
    """
    Nonlinear MPC controller.
    get_u should provide this functionality: u0, x_ol, u_ol, t_ol = mpc_z_rob.get_u(t0, x0).
    - x_ol shape: (12, N+1); u_ol shape: (4, N); t_ol shape: (N+1,)
    You are free to modify other parts    
    """



    def __init__(self, rocket):
        """
        Hint: As in our NMPC exercise, you can evaluate the dynamics of the rocket using 
            CASADI variables x and u via the call rocket.f_symbolic(x,u).
            We create a self.f for you: x_dot = self.f(x,u)
        """        
        # symbolic dynamics f(x,u) from rocket
        self.f = lambda x,u: rocket.f_symbolic(x,u)[0]

    def _setup_controller(self) -> None:
        
        self.ocp = ...

    def get_u(self, t0: float, x0: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        u0 = ...
        x_ol = ...
        u_ol = ...
        t_ol = ... 

        return u0, x_ol, u_ol, t_ol