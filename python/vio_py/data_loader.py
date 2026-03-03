from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from .vio_types import CameraPose, Point3D, PointCloud, Trajectory


def _quat_to_rot_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    qx, qy, qz, qw = q / norm

    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _iter_non_comment_lines(path: Path) -> Iterable[tuple[int, str]]:
    with path.open("r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            yield line_num, line


def load_trajectory_tum(filepath: str) -> Trajectory:
    path = Path(filepath)
    trajectory: Trajectory = []
    if not path.is_file():
        print(f"Error: could not open trajectory file: {filepath}")
        return trajectory

    for line_num, line in _iter_non_comment_lines(path):
        parts = line.split()
        if len(parts) < 8:
            print(f"Warning: skipping malformed line {line_num} in {filepath}")
            continue

        try:
            ts, tx, ty, tz, qx, qy, qz, qw = map(float, parts[:8])
        except ValueError:
            print(f"Warning: skipping malformed line {line_num} in {filepath}")
            continue

        T_wc = np.eye(4, dtype=np.float64)
        T_wc[:3, :3] = _quat_to_rot_matrix(qx, qy, qz, qw)
        T_wc[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
        trajectory.append(CameraPose(timestamp=ts, T_wc=T_wc))

    print(f"Loaded {len(trajectory)} poses from {filepath}")
    return trajectory


def load_point_cloud_xyz(filepath: str) -> PointCloud:
    path = Path(filepath)
    cloud: PointCloud = []
    if not path.is_file():
        print(f"Error: could not open point cloud file: {filepath}")
        return cloud

    for line_num, line in _iter_non_comment_lines(path):
        parts = line.split()
        if len(parts) < 3:
            print(f"Warning: skipping malformed line {line_num} in {filepath}")
            continue

        try:
            x, y, z = map(float, parts[:3])
        except ValueError:
            print(f"Warning: skipping malformed line {line_num} in {filepath}")
            continue

        if len(parts) >= 6:
            try:
                r, g, b = map(float, parts[3:6])
                color = np.array([r / 255.0, g / 255.0, b / 255.0], dtype=np.float64)
            except ValueError:
                color = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        else:
            color = np.array([1.0, 1.0, 1.0], dtype=np.float64)

        cloud.append(
            Point3D(
                position=np.array([x, y, z], dtype=np.float64),
                color=np.clip(color, 0.0, 1.0),
            )
        )

    print(f"Loaded {len(cloud)} points from {filepath}")
    return cloud
