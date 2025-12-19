"""
RocketBase: Python counterpart to MATLAB RocketBase.m

Provides system sizes, names/units, default states/inputs, physical bounds,
and integrates with RocketParams to load model parameters from YAML. The
subclass must implement `dynamics_impl`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.rocket_params import RocketParams

@dataclass
class RocketBase(RocketParams):
    # Public attributes to mirror MATLAB. These are not included in the
    # generated __init__ to avoid dataclass field ordering issues for
    # subclasses that declare required (positional) fields.
    nx: int = field(default=12, init=False)
    nu: int = field(default=4, init=False)
    defaultState: np.ndarray = field(default_factory=lambda: np.zeros(12), init=False)
    defaultControl: np.ndarray = field(default_factory=lambda: np.zeros(4), init=False)
    UBU: np.ndarray = field(default_factory=lambda: np.array([np.deg2rad(15), np.deg2rad(15), 80, 20]), init=False)
    LBU: np.ndarray = field(default_factory=lambda: np.array([np.deg2rad(-15), np.deg2rad(-15), 50, -20]), init=False)
    sys: Dict[str, Any] = field(default_factory=dict, init=False)

    # Optional path for loading additional model parameters (not part of
    # the public constructor signature)
    model_params_filepath: Optional[str] = field(default=None, init=False)
    Ts: Optional[float] = field(default=None, init=True)

    def __post_init__(self) -> None:
        # Initialize default sizes/values (kept here to mirror the
        # original MATLAB structure and allow subclasses to override)
        nx = 12
        nu = 4

        sys: Dict[str, Any] = {}
        sys["StateName"] = [
            "wx", "wy", "wz",
            "alpha", "beta", "gamma",
            "vx", "vy", "vz",
            "x", "y", "z"
        ]
        sys["StateUnit"] = [
            "rad/s", "rad/s", "rad/s",
            "rad", "rad", "rad",
            "m/s", "m/s", "m/s",
            "m", "m", "m"
        ]

        sys["InputName"] = ["d1", "d2", "Pavg", "Pdiff"]
        sys["InputUnit"] = ["rad", "rad", "%", "%"]

        sys["OutputName"] = sys["StateName"] + [
            "spec_nongrav_force_x",
            "spec_nongrav_force_y",
            "spec_nongrav_force_z",
        ]
        sys["OutputUnit"] = sys["StateUnit"] + ["N/kg", "N/kg", "N/kg"]

        phyUBU = np.array([np.pi, np.pi/2.01, np.pi], dtype=float)
        phyLBU = -phyUBU

        defaultState = np.array(
            [
                0.01,
                0.01,
                0.02,
                0.005,
                0.005,
                0.0,
                0.0,
                0.001,
                0.001,
                0.0,
                0.0,
                -2.0,
            ],
            dtype=float,
        )
        defaultControl = np.array([0.0, 0.0, 0.7, 0.0], dtype=float)

        # Assign to instance attributes
        self.nx = nx
        self.nu = nu
        self.sys = sys
        self.defaultState = defaultState
        self.defaultControl = defaultControl
        self.phyUBU = phyUBU
        self.phyLBU = phyLBU

        if self.model_params_filepath:
            self.load_params_from_yaml(self.model_params_filepath)


    def dynamics(self, x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.dynamics_impl(x, u)

    # To be implemented by subclass
    def dynamics_impl(self, x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:  # pragma: no cover - abstract behavior
        raise NotImplementedError("dynamics_impl must be implemented by subclass")


    __all__ = ["RocketBase"]
