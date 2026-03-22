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


def build_grid_segments(extent: int = 10, step: int = 1) -> list[list[list[float]]]:
    segments: list[list[list[float]]] = []
    for value in range(-extent, extent + 1, step):
        segments.append([[float(value), 0.0, float(-extent)], [float(value), 0.0, float(extent)]])
        segments.append([[float(-extent), 0.0, float(value)], [float(extent), 0.0, float(value)]])
    return segments


def try_import_runtime():
    try:
        import rerun as rr  # type: ignore
    except Exception as exc:  # pragma: no cover
        print(f"Receiver dependency error: {exc}", file=sys.stderr)
        return None, None

    try:
        import cv2  # type: ignore
    except Exception:  # pragma: no cover
        cv2 = None

    return rr, cv2


def run_server(args: argparse.Namespace) -> int:
    rr, cv2 = try_import_runtime()
    if rr is None or cv2 is None:
        return 2

    rr.init("vio")
    rr.spawn()
    rr.log(
        "world/grid",
        rr.LineStrips3D(
            build_grid_segments(),
            colors=[[80, 80, 80]],
        ),
    )

    trajectory: list[list[float]] = []

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
                    rr.log(
                        "world/scene",
                        rr.TextDocument(
                            f"Scene: {message.get('dataset_root', '')}\n"
                            f"Resolution: {message.get('image_width')}x{message.get('image_height')}\n"
                            f"Intrinsics: fx={message.get('fx')}, fy={message.get('fy')}, "
                            f"cx={message.get('cx')}, cy={message.get('cy')}"
                        ),
                    )
                    continue

                if msg_type == "cloud":
                    positions = message.get("positions", [])
                    colors = message.get("colors", [])
                    if positions:
                        rr.log(
                            "world/cloud",
                            rr.Points3D(positions, colors=colors, radii=[0.03]),
                        )
                    continue

                if msg_type == "sample":
                    timestamp = float(message["timestamp"])
                    if hasattr(rr, "set_time_seconds"):
                        rr.set_time_seconds("sim_time", timestamp)

                    position = [float(v) for v in message["position"]]
                    quat = [float(v) for v in message["orientation_xyzw"]]
                    trajectory.append(position)

                    rr.log(
                        "world/trajectory",
                        rr.LineStrips3D([trajectory], colors=[[48, 220, 128]]),
                    )
                    rr.log(
                        "world/agent",
                        rr.Points3D([position], colors=[[255, 220, 64]], radii=[0.08]),
                    )

                    direction_tip = [
                        position[0] + 0.35 * (2.0 * (quat[0] * quat[2] + quat[3] * quat[1])),
                        position[1] + 0.35 * (2.0 * (quat[1] * quat[2] - quat[3] * quat[0])),
                        position[2] + 0.35 * (1.0 - 2.0 * (quat[0] * quat[0] + quat[1] * quat[1])),
                    ]
                    rr.log(
                        "world/agent_heading",
                        rr.LineStrips3D([[position, direction_tip]], colors=[[255, 220, 64]]),
                    )

                    rr.log(
                        "sensor/imu_acc",
                        rr.LineStrips3D(
                            [[[0.0, 0.0, 0.0], [float(v) for v in message["acc"]]]],
                            colors=[[255, 96, 96]],
                        ),
                    )
                    rr.log(
                        "sensor/imu_gyro",
                        rr.LineStrips3D(
                            [[[0.0, 0.0, 0.0], [float(v) for v in message["gyro"]]]],
                            colors=[[96, 96, 255]],
                        ),
                    )
                    rr.log(
                        "stats/tracks",
                        rr.TextDocument(f"Tracked features: {int(message.get('track_count', 0))}"),
                    )

                    image_path = Path(message.get("image_path", ""))
                    if cv2 is not None and image_path.is_file():
                        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                        if image is not None:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            rr.log("camera/first_view", rr.Image(image))
                    continue

                if msg_type == "done":
                    rr.log(
                        "world/status",
                        rr.TextDocument(f"Streaming finished. Frames: {message.get('num_samples', 0)}"),
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
