from __future__ import annotations

import argparse
from typing import Sequence

from .data_generator import generate_point_cloud, generate_trajectory
from .data_loader import load_point_cloud_xyz, load_trajectory_tum
from .vio_types import GeneratorConfig, RecordConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m vio_py.cli",
        description="VIO viewer and recorder (Python implementation)",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "File formats:\n"
            "  Trajectory (TUM):  timestamp tx ty tz qx qy qz qw\n"
            "  Point cloud (XYZ): x y z [r g b]   (RGB 0-255, optional)\n"
            "  Lines starting with '#' are treated as comments.\n\n"
            "Examples:\n"
            "  python -m vio_py.cli\n"
            "  python -m vio_py.cli --backend mpl\n"
            "  python -m vio_py.cli --trajectory traj.tum\n"
            "  python -m vio_py.cli --cloud points.xyz\n"
            "  python -m vio_py.cli --trajectory t.tum --cloud c.xyz\n"
            "  python -m vio_py.cli --record out.mp4 --trajectory t.tum --backend auto\n"
        ),
    )

    parser.add_argument("--trajectory", type=str, default="", help="Load trajectory from TUM format file")
    parser.add_argument("--cloud", type=str, default="", help="Load point cloud from XYZ text file")
    parser.add_argument("--record", type=str, default="", help="Record to video file (default: output.mp4)")
    parser.add_argument("--width", type=int, default=1920, help="Video width (default: 1920)")
    parser.add_argument("--height", type=int, default=1080, help="Video height (default: 1080)")
    parser.add_argument("--fps", type=int, default=60, help="Video FPS (default: 60)")
    parser.add_argument(
        "--backend",
        type=str,
        choices=("auto", "open3d", "mpl"),
        default="auto",
        help="Rendering backend: auto/open3d/mpl (default: auto)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    use_synthetic = (not args.trajectory) and (not args.cloud)
    if use_synthetic:
        print("No data files specified - running synthetic demo.")
        gen_cfg = GeneratorConfig()
        trajectory = generate_trajectory(gen_cfg)
        cloud = generate_point_cloud(gen_cfg)
    else:
        trajectory = load_trajectory_tum(args.trajectory) if args.trajectory else []
        cloud = load_point_cloud_xyz(args.cloud) if args.cloud else []

    try:
        from .viewer import Viewer
    except ImportError as exc:
        print(f"Error: failed to import viewer dependencies: {exc}")
        print("Install dependencies from python/requirements.txt.")
        return 1

    viewer = Viewer(trajectory=trajectory, cloud=cloud)
    if args.record:
        rec_cfg = RecordConfig(
            output_path=args.record,
            width=args.width,
            height=args.height,
            fps=args.fps,
        )
        print(f"Recording video to: {rec_cfg.output_path}")
        ok = viewer.record(rec_cfg, backend=args.backend)
    else:
        ok = viewer.run(backend=args.backend)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
