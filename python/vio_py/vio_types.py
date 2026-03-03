from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Point3D:
    position: np.ndarray  # shape (3,), dtype float64
    color: np.ndarray  # shape (3,), dtype float64, RGB in [0, 1]


@dataclass
class CameraPose:
    timestamp: float
    T_wc: np.ndarray  # shape (4, 4), dtype float64


Trajectory = List[CameraPose]
PointCloud = List[Point3D]


@dataclass
class GeneratorConfig:
    # Trajectory
    num_poses: int = 300
    helix_radius: float = 4.0
    helix_height: float = 3.0
    helix_turns: float = 2.0

    # Point cloud
    num_points: int = 8000
    cloud_extent: float = 6.0
    cloud_height: float = 4.0


@dataclass
class RecordConfig:
    output_path: str = "output.mp4"
    width: int = 1920
    height: int = 1080
    fps: int = 60
    camera_distance: float = 3.0
    camera_height: float = 1.5
    smoothing: float = 0.05
    poses_per_frame: int = 1
