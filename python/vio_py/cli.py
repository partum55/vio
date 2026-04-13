from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Receive VIO samples and log them to Rerun.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9877)
    parser.add_argument("--keep-alive", action="store_true", help="Keep the process alive after the producer disconnects.")
    return parser.parse_args()


def build_grid_segments(extent: int = 10, step: int = 1, z: float = 0.0) -> list[list[list[float]]]:
    segments: list[list[list[float]]] = []
    for value in range(-extent, extent + 1, step):
        segments.append([[float(value), float(-extent), z], [float(value), float(extent), z]])
        segments.append([[float(-extent), float(value), z], [float(extent), float(value), z]])
    return segments


def try_import_runtime():
    try:
        import rerun as rr  # type: ignore
        import rerun.blueprint as rrb  # type: ignore
    except Exception as exc:  # pragma: no cover
        print(f"Receiver dependency error: {exc}", file=sys.stderr)
        return None, None
    return rr, rrb


def run_server(args: argparse.Namespace) -> int:
    rr, rrb = try_import_runtime()
    if rr is None or rrb is None:
        return 2

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                origin="/",
                name="3D World",
                background=rrb.Background(kind=rrb.components.BackgroundKind.GradientBright),
                line_grid=rrb.LineGrid3D(
                    visible=True,
                    color=[235, 245, 255, 90],
                    stroke_width=1.5,
                ),
                spatial_information=rrb.SpatialInformation(
                    target_frame="tf#/",
                    show_axes=True,
                    show_bounding_box=False,
                ),
            ),
            rrb.Vertical(
                rrb.Spatial2DView(
                    origin="camera/first_view",
                    name="First-person camera",
                ),
                rrb.TextDocumentView(origin="world/dataset", name="Dataset info"),
                rrb.TextDocumentView(origin="stats/tracks", name="Tracks"),
                row_shares=[3, 1, 1],
            ),
            column_shares=[3, 1],
        ),
    )
    rr.init("vio")
    rr.spawn(hide_welcome_screen=True)
    rr.send_blueprint(blueprint, make_active=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    visual_scale = 1.0
    ground_z = 0.0
    rr.log(
        "world/grid",
        rr.LineStrips3D(
            build_grid_segments(z=ground_z),
            colors=[[80, 80, 80]],
        ),
    )

    _TRAJ_STEP = 10       # subsample: log 1-in-N samples (200 Hz → 20 Hz)
    _TRAJ_CHUNK = 200     # points per sealed chunk
    sample_seq: int = 0
    traj_buffer: list[list[float]] = []
    traj_chunk_id: int = 0
    last_image_path: Path | None = None

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(1)
        print(f"Rerun receiver listening on {args.host}:{args.port}")

        conn, addr = server.accept()
        print(f"Connected from {addr[0]}:{addr[1]}")
        with conn, conn.makefile("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                message = json.loads(line)
                msg_type = message.get("type")

                if msg_type == "init":
                    visual_scale = max(1.0, float(message.get("visual_scale", 1.0)))
                    ground_z = float(message.get("ground_z", 0.0))
                    grid_extent = max(10, int(round(10.0 * visual_scale)))
                    grid_step = max(1, int(round(visual_scale)))
                    rr.log(
                        "world/grid",
                        rr.LineStrips3D(
                            build_grid_segments(extent=grid_extent, step=grid_step, z=ground_z),
                            colors=[[80, 80, 80]],
                        ),
                    )
                    rr.log(
                        "world/dataset",
                        rr.TextDocument(
                            f"Dataset: {message.get('dataset_root', '')}\n"
                            f"Resolution: {message.get('image_width')}x{message.get('image_height')}\n"
                            f"Intrinsics: fx={message.get('fx')}, fy={message.get('fy')}, "
                            f"cx={message.get('cx')}, cy={message.get('cy')}\n"
                            f"Visual scale: {visual_scale:g}x\n"
                            f"Ground z: {ground_z:.2f}"
                        ),
                    )
                    continue

                if msg_type == "cloud":
                    positions = message.get("positions", [])
                    colors = message.get("colors", [])
                    if positions:
                        rr.log(
                            "world/cloud",
                            rr.Points3D(positions, colors=colors, radii=[0.03 * visual_scale]),
                        )
                    continue

                if msg_type == "sample":
                    timestamp = float(message["timestamp"])
                    if hasattr(rr, "set_time"):
                        rr.set_time("sim_time", timestamp=timestamp)
                    elif hasattr(rr, "set_time_seconds"):
                        rr.set_time_seconds("sim_time", timestamp)

                    sample_seq += 1
                    position = [float(v) for v in message["position"]]
                    quat = [float(v) for v in message["orientation_xyzw"]]

                    if sample_seq % _TRAJ_STEP == 0:
                        traj_buffer.append(position)
                        if len(traj_buffer) > _TRAJ_CHUNK:
                            rr.log(
                                f"world/trajectory/c{traj_chunk_id:06d}",
                                rr.LineStrips3D([traj_buffer], colors=[[48, 220, 128]]),
                            )
                            traj_chunk_id += 1
                            traj_buffer = [position]
                        else:
                            rr.log(
                                "world/trajectory/live",
                                rr.LineStrips3D([traj_buffer], colors=[[48, 220, 128]]),
                            )

                        rr.log(
                            "world/agent",
                            rr.Points3D([position], colors=[[255, 220, 64]], radii=[0.08 * visual_scale]),
                        )

                        heading_len = 0.35 * visual_scale
                        direction_tip = [
                            position[0] + heading_len * (2.0 * (quat[0] * quat[2] + quat[3] * quat[1])),
                            position[1] + heading_len * (2.0 * (quat[1] * quat[2] - quat[3] * quat[0])),
                            position[2] + heading_len * (1.0 - 2.0 * (quat[0] * quat[0] + quat[1] * quat[1])),
                        ]
                        rr.log(
                            "world/agent_heading",
                            rr.LineStrips3D([[position, direction_tip]], colors=[[255, 220, 64]]),
                        )

                    rr.log(
                        "stats/tracks",
                        rr.TextDocument(f"Tracked features: {int(message.get('track_count', 0))}"),
                    )

                    image_path = Path(message.get("image_path", ""))
                    if image_path != last_image_path and image_path.is_file():
                        rr.log("camera/first_view", rr.EncodedImage(contents=image_path.read_bytes(), media_type="image/jpeg"))
                        last_image_path = image_path
                    continue

                if msg_type == "done":
                    rr.log(
                        "world/status",
                        rr.TextDocument(f"Processing finished. Frames: {message.get('num_samples', 0)}"),
                    )
                    break

    if args.keep_alive:
        print("Producer disconnected. Keeping Rerun scene alive; press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            return 0

    return 0


def main() -> int:
    return run_server(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
