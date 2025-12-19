from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import ipywidgets as widgets
from IPython.display import display

import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from webcolors import names
from src.rocket import Rocket


def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return a


@dataclass
class RocketVis:
    rocket: Rocket           # pass in the Rocket object (so we can call eul2mat)
    mesh_path: str           # path to rocket mesh, e.g. "Cartoon_rocket.obj"
    color_traj: str = "blue"
    anim_rate: float = 1.0   # speed of animation (1.0 = real time)
    
    # Colors (approximate mappings)
    color_ref: str = "orange"
    color_ol: str = "orange" 
    color_traj: str = "blue"

    axes_length = 1.0
    thrust_length = 2.5

    x_to_plot_lower: float = -0.5  # lower index of states to plot in time series (default: all)
    x_to_plot_upper: float = 5.0  # upper index of states to plot in time series (default: all)

    y_ranges: Dict[str, Tuple[float, float]] = None

    scene_objects: Dict[str, Any] = None
    plotter: pv.Plotter = None


    def __post_init__(self) -> None:
        self.y_ranges = {
            "dR": (-20, +20),
            "dP": (-20, +20),
            "dF": (30, 100),
            "dFdiff": (-30, 30),
            "wx": (-30, 30),
            "wy": (-30, 30),
            "wz": (-30, 30),
            "roll": (-20, 20),
            "pitch": (-20, 20),
            "yaw": (-100, 100),
            "vE": (-20, 20),
            "vN": (-20, 20),
            "vU": (-20, 20),
            "posE": (-10, 50),
            "posN": (-10, 50),
            "posU": (-10, 50),
        }


    def _plot_into_axes(
        self,
        ax_list: List[plt.Axes],
        names: Dict[str, Any],
        T: np.ndarray,
        X: np.ndarray,
        U: np.ndarray,
        X_ref: Optional[np.ndarray] = None,
        color: Optional[str] = None,
    ) -> List[Any]:
        """Plot time-series into provided axes following MATLAB mapping."""

        hnd: Dict[str, Any] = {}
        # Inputs
        for idx, lbl in enumerate(["dR", "dP"]):
            ax_list[idx].set_ylim(self.y_ranges[lbl])
            h_cl = ax_list[idx].plot(T, np.rad2deg(U[idx, :]), color=color or self.color_traj)[0]
            h_ol = ax_list[idx].plot([], [], color=color or self.color_ol)[0]
            p = ax_list[idx].plot(T[0], np.rad2deg(U[idx, 0]), marker="o", color="red")[0]
            # ax_list[idx].set_ylabel(lbl)
            hnd[f"cl_{lbl}"] = h_cl
            hnd[f"ol_{lbl}"] = h_ol
            hnd[f"p_{lbl}"] = p

        for idx, lbl in enumerate(["dF", "dFdiff"]):
            ax_list[2+idx].set_ylim(self.y_ranges[lbl])
            h_cl = ax_list[2+idx].plot(T, U[2+idx, :], color=color or self.color_traj)[0]
            h_ol = ax_list[2+idx].plot([], [], color=color or self.color_ol)[0]
            p = ax_list[2+idx].plot(T[0], U[2+idx, 0], marker="o", color="red")[0]
            # ax_list[2+idx].set_ylabel(lbl)
            hnd[f"cl_{lbl}"] = h_cl
            hnd[f"ol_{lbl}"] = h_ol
            hnd[f"p_{lbl}"] = p

        # vN, vE, vD
        for idx, lbl in zip([6, 7, 8], ["vE", "vN", "vU"]):
            ax_list[4+idx].get_xaxis().set_visible(False)
            ax_list[4+idx].set_ylim(self.y_ranges[lbl])
            h_cl = ax_list[4+idx].plot(T, X[idx, :] if X.size else [], color=color or self.color_traj)[0]
            h_ol = ax_list[4+idx].plot([], [], color=color or self.color_ol)[0]
            h_ref = ax_list[4+idx].plot(T, X_ref[idx, :] if X_ref.size else [], linestyle="--", color=self.color_ref)[0]
            p = ax_list[4+idx].plot(T[0], X[idx, 0], marker="o", color="red")[0]
            # ax_list[4+idx].set_ylabel(f"{lbl} (m/s)")
            hnd[f"cl_{lbl}"] = h_cl
            hnd[f"ol_{lbl}"] = h_ol
            hnd[f"ref_{lbl}"] = h_ref
            hnd[f"p_{lbl}"] = p
        # wx, wy, wz
        for idx, lbl in zip([0, 1, 2], ["wx", "wy", "wz"]):
            ax_list[4+idx].get_xaxis().set_visible(False)
            ax_list[4+idx].set_ylim(self.y_ranges[lbl])
            h_cl = ax_list[4+idx].plot(T, np.rad2deg(X[idx, :]) if X.size else [], color=color or self.color_traj)[0]
            h_ol = ax_list[4+idx].plot([], [], color=color or self.color_ol)[0]
            h_ref = ax_list[4+idx].plot(T, np.rad2deg(X_ref[idx, :]) if X_ref.size else [], linestyle="--", color=self.color_ref)[0]
            p = ax_list[4+idx].plot(T[0], np.rad2deg(X[idx, 0]), marker="o", color="red")[0]
            # ax_list[4+idx].set_ylabel(f"{lbl} (deg/s)")
            hnd[f"cl_{lbl}"] = h_cl
            hnd[f"ol_{lbl}"] = h_ol
            hnd[f"ref_{lbl}"] = h_ref
            hnd[f"p_{lbl}"] = p
        # posN, posE, posD
        for idx, lbl in zip([9, 10, 11], ["posE", "posN", "posU"]):
            ax_list[4+idx].get_xaxis().set_visible(False)
            ax_list[4+idx].set_ylim(self.y_ranges[lbl])
            h_cl = ax_list[4+idx].plot(T, X[idx, :] if X.size else [], color=color or self.color_traj)[0]
            p = ax_list[4+idx].plot(T[0], X[idx, 0], marker="o", color="red")[0]
            # ax_list[4+idx].set_ylabel(f"{lbl} (m)")
            hnd[f"cl_{lbl}"] = h_cl
            hnd[f"p_{lbl}"] = p
        # roll, pitch, yaw
        for idx, lbl in zip([3, 4, 5], ["roll", "pitch", "yaw"]):
            ax_list[4+idx].get_xaxis().set_visible(False)
            ax_list[4+idx].set_ylim(self.y_ranges[lbl])
            h_cl = ax_list[4+idx].plot(T, np.rad2deg(X[idx, :]) if X.size else [], color=color or self.color_traj)[0]
            h_ol = ax_list[4+idx].plot([], [], color=color or self.color_ol)[0]
            h_ref = ax_list[4+idx].plot(T, np.rad2deg(X_ref[idx, :]) if X_ref.size else [], linestyle="--", color=self.color_ref)[0]
            p = ax_list[4+idx].plot(T[0], np.rad2deg(X[idx, 0]), marker="o", color="red")[0]
            # ax_list[4+idx].set_ylabel(f"{lbl} (deg)")
            hnd[f"cl_{lbl}"] = h_cl
            hnd[f"ol_{lbl}"] = h_ol
            hnd[f"ref_{lbl}"] = h_ref
            hnd[f"p_{lbl}"] = p

        return hnd


    def _create_rocket_mesh(self, plotter) -> Any:
        """Create a 3D rocket mesh using PyVista primitives."""        
        # Combine all parts
        rocket_mesh = pv.read(self.mesh_path)
        rocket_mesh.translate([0,-1.0,0], inplace=True)
        rocket_mesh.rotate_x(90, point=(0, 0, 0), inplace=True)
        mesh_points = np.array(rocket_mesh.points)
        rocket_actor = plotter.add_mesh(rocket_mesh)
    
        return rocket_actor, mesh_points
    
    def _create_rocket_axis(self, plotter) -> Tuple[pv.Actor, pv.Actor, pv.Actor]:
        """Create 3D body axes using PyVista lines."""
        x_axis = pv.PolyData([[0, 0, 0], [self.axes_length, 0, 0]], lines=[[2, 0, 1]])
        y_axis = pv.PolyData([[0, 0, 0], [0, self.axes_length, 0]], lines=[[2, 0, 1]])
        z_axis = pv.PolyData([[0, 0, 0], [0, 0, self.axes_length]], lines=[[2, 0, 1]])

        x_axis_actor = plotter.add_mesh(x_axis, color='red', line_width=10)
        y_axis_actor = plotter.add_mesh(y_axis, color='green', line_width=10)
        z_axis_actor = plotter.add_mesh(z_axis, color='blue', line_width=10)

        x_axis_mesh_points = np.array(x_axis.points)
        y_axis_mesh_points = np.array(y_axis.points)
        z_axis_mesh_points = np.array(z_axis.points)

        return x_axis_actor, y_axis_actor, z_axis_actor, x_axis_mesh_points, y_axis_mesh_points, z_axis_mesh_points

    def _create_trajectory(self, plotter) -> pv.Actor:
        """Create 3D trajectory using PyVista PolyData."""
        trajectory_points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
        lines = np.array([[0]])
        trajectory = pv.PolyData(trajectory_points, lines=lines)

        trajectory_actor = plotter.add_mesh(trajectory, render_lines_as_tubes=False, line_width=1, color='gray')
        
        return trajectory_actor
    
    def _create_thrustvector(self, plotter) -> pv.Actor:
        """Create 3D thrust vector using PyVista line."""
        thrust_line = pv.PolyData([[0, 0, 0], [0, 0, -self.thrust_length]], lines=[[2, 0, 1]])
        thrust_actor = plotter.add_mesh(thrust_line, color='purple', render_lines_as_tubes=False, line_width=10)
        thrust_mesh_points = np.array(thrust_line.points)

        return thrust_actor, thrust_mesh_points


    def _create_3d_scene(self) -> Tuple[Any, Dict[str, Any]]:
        """Create PyVista 3D scene with rocket and trajectory elements."""
        plotter = pv.Plotter(title="Rocket Trajectory", notebook=True)
        plotter.camera.position = (30, 0, 10)
        plotter.camera.focal_point = (0, 0, 5)

        # Add ground plane
        ground = pv.Plane(center=[0, 0, 0], direction=[0, 0, 1], 
                         i_size=50, j_size=50, i_resolution=10, j_resolution=10)
        plotter.add_mesh(ground, color='lightgreen', opacity=0.5)
        
        # Create rocket actors
        rocket_actor, rocket_mesh_points = self._create_rocket_mesh(plotter)
        x_axis_actor, y_axis_actor, z_axis_actor, x_axis_mesh_points, y_axis_mesh_points, z_axis_mesh_points = self._create_rocket_axis(plotter)
        trajectory_actor = self._create_trajectory(plotter)
        # trajectory_actor = None
        thrust_actor, thrust_mesh_points = self._create_thrustvector(plotter)
        
        # Setup camera and axes
        plotter.add_axes(line_width=2, labels_off=False)
        plotter.set_background('white')
        
        # Store references for updates
        scene_objects = {
            'rocket_actor': rocket_actor,
            'rocket_mesh_points': rocket_mesh_points,
            'x_axis_actor': x_axis_actor,
            'y_axis_actor': y_axis_actor, 
            'z_axis_actor': z_axis_actor,
            'x_axis_mesh_points': x_axis_mesh_points,
            'y_axis_mesh_points': y_axis_mesh_points,
            'z_axis_mesh_points': z_axis_mesh_points,
            'trajectory_actor': trajectory_actor,
            'thrust_actor': thrust_actor,
            'thrust_mesh_points': thrust_mesh_points,
        }
        
        return plotter, scene_objects

    def _update_plot(self, hnd: Dict[str, Any], axes: List[plt.Axes], x: np.ndarray, u: np.ndarray, x_ol: np.ndarray = None, u_ol: np.ndarray = None, t_ol: np.ndarray = None) -> None:
        for ax in axes:
            ax.set_xlim(t_ol[0]+self.x_to_plot_lower, t_ol[0]+self.x_to_plot_upper)
        
        for idx, lbl in enumerate(["dR", "dP"]):
            if u_ol is not None:
                hnd[f'ol_{lbl}'].set_data(t_ol[:-1], np.rad2deg(u_ol[idx,:]))
            hnd[f'p_{lbl}'].set_data(t_ol[0:1], np.rad2deg(u[idx:idx+1]))

        for idx, lbl in enumerate(["dF", "dFdiff"]):
            if u_ol is not None:
                hnd[f'ol_{lbl}'].set_data(t_ol[:-1], u_ol[2+idx,:])
            hnd[f'p_{lbl}'].set_data(t_ol[0:1], u[2+idx:2+idx+1])

        for idx, lbl in zip([6,7,8], ["vE", "vN", "vU"]):
            if x_ol is not None:
                hnd[f'ol_{lbl}'].set_data(t_ol, x_ol[idx,:])
            hnd[f'p_{lbl}'].set_data(t_ol[0:1], x[idx:idx+1])

        for idx, lbl in zip([9,10,11], ["posE", "posN", "posU"]):
            hnd[f'p_{lbl}'].set_data(t_ol[0:1], x[idx:idx+1])
        
        for idx, lbl in zip([0,1,2,3,4,5], ["wx", "wy", "wz", "roll", "pitch", "yaw"]):
            if x_ol is not None:
                hnd[f'ol_{lbl}'].set_data(t_ol, np.rad2deg(x_ol[idx,:]))
            hnd[f'p_{lbl}'].set_data(t_ol[0:1], np.rad2deg(x[idx:idx+1]))


    def _update_rocket_pose(self, scene_objects: Dict[str, Any], pos: np.ndarray, r: np.ndarray, dR, dP, Pavg) -> None:
        """Update rocket pose in PyVista scene by updating the actor's mesh."""
        R = Rocket.eul2mat(np.array(r))

        # Transform the points of the mesh
        rocket_mesh_points = scene_objects['rocket_mesh_points']
        rocket_current_points = rocket_mesh_points @ R.T + pos
        scene_objects['rocket_actor'].mapper.dataset.points = rocket_current_points

        # Update body axes
        origin = pos.astype(np.float32)
        x_end = scene_objects['x_axis_mesh_points'] @ R.T + pos
        y_end = scene_objects['y_axis_mesh_points'] @ R.T + pos
        z_end = scene_objects['z_axis_mesh_points'] @ R.T + pos

        scene_objects['x_axis_actor'].mapper.dataset.points = x_end
        scene_objects['y_axis_actor'].mapper.dataset.points = y_end
        scene_objects['z_axis_actor'].mapper.dataset.points = z_end

        Rr = np.array([
            [1, 0, 0],
            [0, np.cos(-dR), -np.sin(-dR)],
            [0, np.sin(-dR),  np.cos(-dR)]
        ])
        Rp = np.array([
            [np.cos(-dP), 0, np.sin(-dP)],
            [0, 1, 0],
            [-np.sin(-dP), 0, np.cos(-dP)]
        ])
        thrust_R = R @ Rp @ Rr
        thrust_end = scene_objects['thrust_mesh_points'] @ thrust_R.T + pos 
        scene_objects['thrust_actor'].mapper.dataset.points = thrust_end
        


    def _update_trajectory(self, scene_objects: Dict[str, Any], traj_points: np.ndarray) -> None:
        """Update trajectory in PyVista scene."""
        if traj_points.shape[0] < 2:
            return
            
        # Create lines connecting consecutive trajectory points
        n_points = traj_points.shape[0]
        lines = []
        for i in range(n_points - 1):
            lines.append([2, i, i + 1])  # Line from point i to point i+1
        
        # trajectory = pv.PolyData(traj_points, lines=lines)
        scene_objects['trajectory_actor'].mapper.dataset.points = traj_points
        scene_objects['trajectory_actor'].mapper.dataset.lines = lines


    # def update_callback(self, change):
    #     value = change['new']
    #     pos = X[9:12, value]        # (x, y, z)
    #     angles = self.X[3:6, value]      # (alpha, beta, gamma)
    #     self._update_rocket_pose(self.scene, pos, angles)
# 
    #     traj_points = np.column_stack([self.X[9, :], self.X[10, :], self.X[11, :]])
    #     self._update_trajectory(self.scene, traj_points[:value + 1])
    #     self.plotter.render()


    def animate(
        self,
        T: np.ndarray,
        X: np.ndarray,
        U: np.ndarray,
        Ref: Optional[np.ndarray] = None,
        T_ol: Optional[np.ndarray] = None,
        X_ol: Optional[np.ndarray] = None,
        U_ol: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """PyVista-enhanced equivalent of MATLAB plotvis.m."""
        # Map Ref to state-space if provided
        X_ref = np.full_like(X, np.nan)
        if Ref is not None:
            Ref = _ensure_2d(Ref)
            if Ref.shape[0] == 12:
                X_ref = Ref
            else:
                raise ValueError("Ref must have 12 rows corresponding to full state vector.")

        # Create matplotlib subplot for time series
        nrow, ncol = 5, 4  # 16 subplots total to match original layout
        with plt.ioff():
            fig = plt.figure()
            
        gs = fig.add_gridspec(nrows=nrow, ncols=ncol)
        axes: List[plt.Axes] = []
        
        # Mapping for time series plots
        state_mapping = [(0,1),(0,0),(0,3),(1,1),(1,0),(1,3),(2,0),(2,1),(2,2),(3,0),(3,1),(3,2)]
        inputs_mapping = [(4,0),(4,1),(4,2),(4,3)]
        for r, c in inputs_mapping + state_mapping:
            ax = fig.add_subplot(gs[r, c], sharex = axes[0] if axes else None)
            ax.grid(True)
            axes.append(ax)


        # Plot time series data
        hnd = self._plot_into_axes(axes, self.rocket.sys, T, X, U, X_ref)
        
        # Set titles
        axes[5].set_title("Subsystem X")
        axes[4].set_title("Subsystem Y")
        axes[12].set_title("Subsystem Z")
        axes[6].set_title("Subsystem Roll")

        # Set y-labels
        axes[0].set_ylabel("inputs")
        axes[5].set_ylabel(r"$\omega_{\alpha\beta\gamma}$ (deg/s)")
        axes[8].set_ylabel(r"$\alpha\beta\gamma$ (deg)")
        axes[10].set_ylabel(r"$v$ (m/s)")
        axes[13].set_ylabel(r"$\text{pos}$ (m)")
        

        # matplotlib_widget = widgets.Output()
        # with matplotlib_widget:
        #     display(fig)

        # Create PyVista 3D scene if available
        

        def update_callback(change):
            # with plotter:
            if self._update_running:
                print("Skipping update to avoid overlap")
                return
            
            self._update_running = True
            value = change['new']
            pos = np.array([X[9, value], X[10, value], X[11, value]])  # E,N,U 
            angles = X[3:6, value]      # (alpha, beta, gamma)
            self._update_rocket_pose(self.scene_objects, pos, angles, U[0, value], U[1, value], U[2, value])

            traj_points = np.column_stack([X[9, :], X[10, :], X[11, :]])
            self._update_trajectory(self.scene_objects, traj_points[:value + 1])
            self.plotter.render()

            if T_ol is not None and X_ol is not None and U_ol is not None:
                self._update_plot(hnd, axes, X[:, value], U[:, value], X_ol[..., value], U_ol[..., value], T_ol[..., value])
            else:
                self._update_plot(hnd, axes, X[:, value], U[:, value], t_ol=T[value:value+1])
            self._update_running = False

        self._update_running = False
        self.plotter, self.scene_objects = self._create_3d_scene()
        pyvista_widget = self.plotter.show(auto_close=False, interactive_update=True, return_viewer=True, jupyter_backend='client')

        Ts = (T[1]-T[0]) / self.anim_rate
        max_fps = 10.0
        step = max(1, np.ceil(1 / (Ts * max_fps)).astype(int))
        play = widgets.Play(value=0, min=0, max=T.shape[0]-1, step=step, interval=1000*step*Ts, description="Press play", disabled=False)
        slider = widgets.IntSlider(min=0, max=T.shape[0]-1)
        widgets.jslink((play, 'value'), (slider, 'value'))

        slider.observe(update_callback, names='value')

        layout = widgets.AppLayout(
            left_sidebar=fig.canvas, 
            right_sidebar=pyvista_widget, 
            footer=widgets.HBox([play, slider]),
            pane_heights=[0, 5, '30px']
        )
        display(layout)

        # Add trajectory to PyVista scene
        if X.shape[0] >= 9:
            # !!!CHECK!!! if it needs to be ENU
            traj_points = np.column_stack([X[9, :], X[10, :], X[11, :]])  # E,N,U 
            self._update_trajectory(self.scene_objects, traj_points)
        
        # Set initial rocket pose
        if X.shape[0] >= 12 and T.size > 0:
            pos = np.array([X[9, 0], X[10, 0], X[11, 0]])
            r = np.array([X[3, 0], X[4, 0], X[5, 0]])
            dR = U[0,0] if U.shape[0]>0 else 0.0
            dP = U[1,0] if U.shape[0]>1 else 0.0
            Pavg = U[2,0] if U.shape[0]>1 else 0.0
            self._update_rocket_pose(self.scene_objects, pos, r, dR, dP, Pavg)

        self.plotter.render()
        self.plotter.reset_camera()

        if T_ol is not None and X_ol is not None and U_ol is not None:
            self._update_plot(hnd, axes, X[:, 0], U[:, 0], X_ol[..., 0], U_ol[..., 0], T_ol[..., 0])
        else:
            self._update_plot(hnd, axes, X[:, 0], U[:, 0], t_ol=T[0:1])


        return {
            "fig": fig,
            "axes": axes,
            "plotter": self.plotter,
            "scene_objects": self.scene_objects,
            # "slider": slider,
            # "time_text": time_text,
        }


def plot_static_states_inputs(T, X, U, Ref=None, type=None):
    """Plot rocket subsystem states and inputs with constraints."""

    if Ref is not None and Ref.ndim == 1:
        N_cl = U.shape[1]
        Ref = np.tile(Ref[:, None], (1, N_cl))

    # --- constraint dictionary ---
    constraints = {
        "z":        [0],  # lower bound only
        "P_diff":   [-20, 20],
        "P_avg":    [40, 80],
        "δ₁":       [-15*np.pi/180, 15*np.pi/180],
        "δ₂":       [-15*np.pi/180, 15*np.pi/180],
    }

    # --- helper to plot data + optional ref + constraint lines ---
    def plot_with_unit(ax, t, data, label, unit="", ref_data=None):
        if unit in ("deg", "deg/s"):
            data = np.rad2deg(data)
            if ref_data is not None:
                ref_data = np.rad2deg(ref_data)
        ax.plot(t, data, label=label)
        if ref_data is not None:
            ax.plot(t, ref_data, "--", color="k", label="ref")

        # add constraints if defined
        if label in constraints:
            for val in constraints[label]:
                ax.axhline(y=val, linestyle='--', color='k', linewidth=0.8)

        ax.set_ylabel(f"{label} ({unit})")

    # Otherwise: plot all subsystems
    fig, axes = plt.subplots(5, 4, figsize=(12, 6), sharex=True)
    axes = axes.T  # transpose: each column is a subsystem

    # sys_x
    labels_x = ["ω_y", "β", "v_x", "x"]
    units_x  = ["deg/s", "deg", "m/s", "m"]
    idx_x = [1, 4, 6, 9]
    for i, (lab, unit, idx) in enumerate(zip(labels_x, units_x, idx_x)):
        ref_seq = Ref[idx, :] if (Ref is not None and lab == "x") else None
        plot_with_unit(axes[0][i], T, X[idx, :], lab, unit, ref_seq)
    plot_with_unit(axes[0][-1], T, U[1, :], "δ₂", "rad")

    # sys_y
    labels_y = ["ω_x", "α", "v_y", "y"]
    units_y  = ["deg/s", "deg", "m/s", "m"]
    idx_y = [0, 3, 7, 10]
    for i, (lab, unit, idx) in enumerate(zip(labels_y, units_y, idx_y)):
        ref_seq = Ref[idx, :] if (Ref is not None and lab == "y") else None
        plot_with_unit(axes[1][i], T, X[idx, :], lab, unit, ref_seq)
    plot_with_unit(axes[1][-1], T, U[0, :], "δ₁", "rad")

    # sys_z
    labels_z = ["v_z", "z"]
    units_z  = ["m/s", "m"]
    idx_z = [8, 11]
    axe_z = [2,3]
    for i, (lab, unit, idx, axe) in enumerate(zip(labels_z, units_z, idx_z, axe_z)):
        ref_seq = Ref[idx, :] if (Ref is not None and lab == "z") else None
        plot_with_unit(axes[2][axe], T, X[idx, :], lab, unit, ref_seq)
    plot_with_unit(axes[2][-1], T, U[2, :], "P_avg", "%")

    # sys_roll
    labels_r = ["ω_z", "γ"]
    units_r  = ["deg/s", "deg"]
    idx_r = [2, 5]
    for i, (lab, unit, idx) in enumerate(zip(labels_r, units_r, idx_r)):
        ref_seq = Ref[idx, :] if (Ref is not None and lab=="γ") else None
        plot_with_unit(axes[3][i], T, X[idx, :], lab, unit, ref_seq)
    plot_with_unit(axes[3][-1], T, U[3, :], "P_diff", "%")

    # Add time label for bottom row
    for ax in axes[:, -1]:
        ax.set_xlabel("Time (s)")

    # --- Add subsystem subtitles ---
    subsystem_titles = ["sys_x", "sys_y", "sys_z", "sys_roll"]
    for i, title in enumerate(subsystem_titles):
        axes[i][0].set_title(title, fontsize=11, loc='left', pad=8)        

    fig.tight_layout()
    plt.show()
