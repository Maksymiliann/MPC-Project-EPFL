"""
RocketParams: Python counterpart to MATLAB RocketParams.m

Provides a container for rocket parameters and a loader from a YAML file
following the same keys used by the MATLAB version.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RocketParams:
    # Defaults match MATLAB RocketParams.m
    g: float = 9.806
    rho: float = 1.1589

    # To be loaded from YAML
    mass: Optional[float] = None
    J: Optional[np.array] = None
    rz_thrust: Optional[float] = None
    thrust_coeff: Optional[np.ndarray] = None  # [c2, c1, c0]
    torque_coeff: Optional[np.ndarray] = None  # [c2, c1, c0]
    angle_g: Optional[np.ndarray] = None  # [c1, c0]

    def load_params_from_yaml(self, yaml_filepath: str) -> "RocketParams":
        try:
            import yaml  # type: ignore
        except Exception as e:  # pragma: no cover - import error path
            raise ImportError(
                "PyYAML is required to load YAML files. Install it with 'pip install pyyaml'."
            ) from e

        with open(yaml_filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Mirror MATLAB key structure
        inertia = data.get("inertia", {})
        propulsion = data.get("propulsion", {})
        angle = data.get("angle", {})

        self.mass = float(inertia.get("mass"))
        self.fuel_rate = float(inertia.get("fuel_rate"))
        
        self.J =  np.diag(
            [
                float(inertia.get("Ixx")),
                float(inertia.get("Iyy")),
                float(inertia.get("Izz")),
            ]
        )

        self.rz_thrust = float(propulsion.get("rz_thrust"))

        self.thrust_coeff = np.array(
            [
                float(propulsion.get("thrust_c2")),
                float(propulsion.get("thrust_c1")),
                float(propulsion.get("thrust_c0")),
            ]
        )

        self.torque_coeff = np.array(
            [
                float(propulsion.get("torque_c2")),
                float(propulsion.get("torque_c1")),
                float(propulsion.get("torque_c0")),
            ]
        )

        self.angle_g = np.array(
            [
                float(angle.get("angle_c1")),
                float(angle.get("angle_c0")),
            ]
        )

        # Allow overriding g/rho from YAML if present
        if "g" in data:
            self.g = float(data["g"]) 
        if "rho" in data:
            self.rho = float(data["rho"]) 

        return self


__all__ = ["RocketParams"]
