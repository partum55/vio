from __future__ import annotations

import math

import numpy as np

from .vio_types import CameraPose, GeneratorConfig, Point3D, PointCloud, Trajectory


def generate_trajectory(config: GeneratorConfig | None = None) -> Trajectory:
    if config is None:
        config = GeneratorConfig()

    traj: Trajectory = []
    if config.num_poses <= 0:
        return traj

    denom = max(config.num_poses - 1, 1)
    for i in range(config.num_poses):
        t = float(i) / float(denom)
        angle = 2.0 * math.pi * config.helix_turns * t

        x = config.helix_radius * math.cos(angle)
        z = config.helix_radius * math.sin(angle)
        y = config.helix_height * t

        pos = np.array([x, y, z], dtype=np.float64)
        target = np.array([0.0, config.helix_height * 0.5, 0.0], dtype=np.float64)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        forward = target - pos
        forward /= np.linalg.norm(forward) + 1e-12
        right = np.cross(forward, up)
        right /= np.linalg.norm(right) + 1e-12
        cam_up = np.cross(right, forward)
        cam_up /= np.linalg.norm(cam_up) + 1e-12

        T_wc = np.eye(4, dtype=np.float64)
        T_wc[:3, 0] = right
        T_wc[:3, 1] = cam_up
        T_wc[:3, 2] = -forward
        T_wc[:3, 3] = pos

        traj.append(CameraPose(timestamp=t, T_wc=T_wc))

    return traj


def generate_point_cloud(config: GeneratorConfig | None = None) -> PointCloud:
    if config is None:
        config = GeneratorConfig()

    cloud: PointCloud = []
    if config.num_points <= 0:
        return cloud

    rng = np.random.default_rng(seed=42)
    xs = rng.uniform(-config.cloud_extent, config.cloud_extent, size=config.num_points)
    ys = rng.uniform(0.0, config.cloud_height, size=config.num_points)
    zs = rng.uniform(-config.cloud_extent, config.cloud_extent, size=config.num_points)

    for x, y, z in zip(xs, ys, zs):
        t = float(y / config.cloud_height) if config.cloud_height > 0.0 else 0.0
        if t < 0.5:
            s = t * 2.0
            color = np.array([0.0, s, 1.0 - s], dtype=np.float64)
        else:
            s = (t - 0.5) * 2.0
            color = np.array([s, 1.0 - s, 0.0], dtype=np.float64)

        cloud.append(
            Point3D(
                position=np.array([x, y, z], dtype=np.float64),
                color=np.clip(color, 0.0, 1.0),
            )
        )

    return cloud
