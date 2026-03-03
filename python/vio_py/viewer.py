from __future__ import annotations

import shutil
import subprocess
from typing import Iterable

import numpy as np
import open3d as o3d

from .vio_types import PointCloud, RecordConfig, Trajectory

SUPPORTED_BACKENDS = ("auto", "open3d", "mpl")


class Viewer:
    keyframe_step = 15
    frustum_scale = 0.15

    def __init__(self, trajectory: Trajectory, cloud: PointCloud) -> None:
        self.trajectory = trajectory
        self.cloud = cloud

    def run(self, backend: str = "auto") -> bool:
        try:
            backend = _normalize_backend(backend)
        except ValueError as exc:
            print(f"Error: {exc}")
            return False

        if backend == "mpl":
            return self._run_with_matplotlib_interactive()

        if backend == "open3d":
            if self._run_with_open3d_interactive():
                return True
            self._print_open3d_backend_error(mode="interactive")
            return False

        if self._run_with_open3d_interactive():
            return True

        print("Open3D interactive renderer unavailable. Falling back to matplotlib interactive viewer.")
        return self._run_with_matplotlib_interactive()

    def record(self, cfg: RecordConfig | None = None, backend: str = "auto") -> bool:
        try:
            backend = _normalize_backend(backend)
        except ValueError as exc:
            print(f"Error: {exc}")
            return False

        if cfg is None:
            cfg = RecordConfig()

        if shutil.which("ffmpeg") is None:
            print("Failed to open ffmpeg pipe. Is ffmpeg installed?")
            return False

        width = int(cfg.width)
        height = int(cfg.height)
        fps = int(cfg.fps)
        poses_per_frame = max(int(cfg.poses_per_frame), 1)

        if backend == "mpl":
            return self._record_with_matplotlib(
                cfg=cfg, width=width, height=height, fps=fps, poses_per_frame=poses_per_frame
            )

        success, renderer_available = self._record_with_open3d(
            cfg=cfg,
            width=width,
            height=height,
            fps=fps,
            poses_per_frame=poses_per_frame,
        )
        if success:
            return True

        if backend == "open3d":
            if not renderer_available:
                self._print_open3d_backend_error(mode="recording")
            return False

        if not renderer_available:
            print("Open3D offscreen renderer unavailable. Falling back to matplotlib software rendering.")
            return self._record_with_matplotlib(
                cfg=cfg, width=width, height=height, fps=fps, poses_per_frame=poses_per_frame
            )

        return False

    def _run_with_open3d_interactive(self) -> bool:
        width = 1280
        height = 720
        vis = o3d.visualization.Visualizer()
        created = vis.create_window("VIO Viewer", width=width, height=height, visible=True)
        if not created:
            return False
        if not self._configure_render(vis):
            vis.destroy_window()
            return False

        grid = self._create_grid_lineset()
        cloud = self._create_point_cloud_geometry()
        trajectory = self._create_trajectory_lineset(up_to_pose=-1)
        keyframe_frustums = self._create_keyframe_frustums_lineset()

        vis.add_geometry(grid, reset_bounding_box=False)
        vis.add_geometry(cloud, reset_bounding_box=False)
        vis.add_geometry(trajectory, reset_bounding_box=False)
        vis.add_geometry(keyframe_frustums, reset_bounding_box=False)

        self._set_camera_pose(
            vis=vis,
            eye=np.array([8.0, 6.0, 8.0], dtype=np.float64),
            lookat=np.array([0.0, 1.5, 0.0], dtype=np.float64),
            width=width,
            height=height,
            fx=500.0,
            fy=500.0,
        )

        try:
            while True:
                if not vis.poll_events():
                    break
                vis.update_renderer()
        finally:
            vis.destroy_window()

        return True

    def _run_with_matplotlib_interactive(self) -> bool:
        import matplotlib

        using_webagg = False
        webagg_error = None
        if matplotlib.get_backend().lower() == "agg":
            for candidate in ("TkAgg", "QtAgg", "GTK3Agg", "WebAgg"):
                try:
                    matplotlib.use(candidate, force=True)
                    using_webagg = candidate.lower() == "webagg"
                    break
                except Exception as exc:
                    if candidate.lower() == "webagg":
                        webagg_error = exc
                    continue

        from matplotlib import pyplot as plt

        if matplotlib.get_backend().lower() == "agg":
            print("Error: no interactive matplotlib backend is available in this environment.")
            print("Install a GUI backend (for example Tk/Qt), or use `--record ... --backend mpl`.")
            if webagg_error is not None:
                print(f"WebAgg fallback unavailable: {webagg_error}")
                print("Install `tornado` to enable browser-based interactive fallback.")
            return False

        width = 1280
        height = 720
        dpi = 100
        fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        try:
            fig.canvas.manager.set_window_title("VIO Viewer (matplotlib fallback)")
        except Exception:
            pass
        fig.patch.set_facecolor((0.05, 0.05, 0.08))
        ax = fig.add_subplot(111, projection="3d")

        cloud_points = (
            np.asarray([pt.position for pt in self.cloud], dtype=np.float64)
            if self.cloud
            else np.empty((0, 3), dtype=np.float64)
        )
        cloud_colors = (
            np.asarray([pt.color for pt in self.cloud], dtype=np.float64)
            if self.cloud
            else np.empty((0, 3), dtype=np.float64)
        )
        center, radius = self._compute_scene_bounds(cloud_points)
        self._configure_mpl_axes(ax, center=center, radius=radius)
        self._draw_grid_mpl(ax)

        if cloud_points.shape[0] > 0:
            cloud_plot = _world_to_mpl(cloud_points)
            ax.scatter(
                cloud_plot[:, 0],
                cloud_plot[:, 1],
                cloud_plot[:, 2],
                c=cloud_colors,
                s=1.0,
                depthshade=False,
            )

        traj_points, traj_lines, traj_colors = self._trajectory_arrays(up_to_pose=-1)
        self._draw_lineset_mpl(
            ax=ax,
            points_world=traj_points,
            lines=traj_lines,
            colors=traj_colors,
            linewidth=1.8,
        )

        key_points, key_lines, key_colors = self._frustum_arrays(
            pose_indices=range(0, len(self.trajectory), self.keyframe_step),
            scale=self.frustum_scale,
            color=np.array([0.4, 0.7, 1.0], dtype=np.float64),
        )
        self._draw_lineset_mpl(
            ax=ax,
            points_world=key_points,
            lines=key_lines,
            colors=key_colors,
            linewidth=0.9,
        )

        current_idx = max(len(self.trajectory) - 1, 0)
        cur_points, cur_lines, cur_colors = self._frustum_arrays(
            pose_indices=[current_idx],
            scale=self.frustum_scale * 2.0,
            color=np.array([1.0, 0.9, 0.2], dtype=np.float64),
        )
        self._draw_lineset_mpl(
            ax=ax,
            points_world=cur_points,
            lines=cur_lines,
            colors=cur_colors,
            linewidth=1.6,
        )
        self._set_mpl_camera(
            ax=ax,
            eye_world=np.array([8.0, 6.0, 8.0], dtype=np.float64),
            lookat_world=np.array([0.0, 1.5, 0.0], dtype=np.float64),
        )

        if using_webagg:
            print("Using matplotlib WebAgg interactive fallback (browser-based).")
        else:
            print("Using matplotlib interactive viewer fallback.")
        plt.show()
        return True

    def _record_with_open3d(
        self,
        cfg: RecordConfig,
        width: int,
        height: int,
        fps: int,
        poses_per_frame: int,
    ) -> tuple[bool, bool]:
        vis = o3d.visualization.Visualizer()
        created = vis.create_window("VIO Recorder", width=width, height=height, visible=False)
        if not created or not self._configure_render(vis):
            if created:
                vis.destroy_window()
            return False, False

        grid = self._create_grid_lineset()
        cloud = self._create_point_cloud_geometry()
        keyframe_frustums = self._create_keyframe_frustums_lineset()
        trail = self._create_trajectory_lineset(up_to_pose=0)
        current = self._create_current_frustum_lineset(0)

        vis.add_geometry(grid, reset_bounding_box=False)
        vis.add_geometry(cloud, reset_bounding_box=False)
        vis.add_geometry(keyframe_frustums, reset_bounding_box=False)
        vis.add_geometry(trail, reset_bounding_box=False)
        vis.add_geometry(current, reset_bounding_box=False)

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            cfg.output_path,
        ]

        try:
            ffmpeg_proc = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError:
            vis.destroy_window()
            print("Failed to open ffmpeg pipe. Is ffmpeg installed?")
            return False, True

        total_poses = len(self.trajectory)
        frame = 0
        cam_pos = np.array([8.0, 6.0, 8.0], dtype=np.float64)
        cam_look = np.array([0.0, 1.5, 0.0], dtype=np.float64)
        smoothing = float(np.clip(cfg.smoothing, 0.0, 1.0))
        alpha = 1.0 - smoothing
        fx = 0.7 * width
        fy = 0.7 * width

        print(f"Recording {total_poses} poses to {cfg.output_path} ...")
        return_code = 0

        try:
            for pose_idx in range(0, total_poses, poses_per_frame):
                T_wc = self.trajectory[pose_idx].T_wc
                agent_pos = T_wc[:3, 3]
                agent_forward = -T_wc[:3, 2]
                desired_cam_pos = (
                    agent_pos
                    - agent_forward * float(cfg.camera_distance)
                    + np.array([0.0, float(cfg.camera_height), 0.0], dtype=np.float64)
                )

                cam_pos = cam_pos * (1.0 - alpha) + desired_cam_pos * alpha
                cam_look = cam_look * (1.0 - alpha) + agent_pos * alpha

                self._set_camera_pose(
                    vis=vis,
                    eye=cam_pos,
                    lookat=cam_look,
                    width=width,
                    height=height,
                    fx=fx,
                    fy=fy,
                )

                self._update_trajectory_lineset(trail, pose_idx)
                self._update_current_frustum_lineset(current, pose_idx)
                vis.update_geometry(trail)
                vis.update_geometry(current)
                vis.poll_events()
                vis.update_renderer()

                frame_rgb = np.asarray(vis.capture_screen_float_buffer(do_render=False))
                frame_rgb = np.clip(frame_rgb[:, :, :3] * 255.0, 0.0, 255.0).astype(np.uint8)

                if ffmpeg_proc.stdin is None:
                    break
                try:
                    ffmpeg_proc.stdin.write(frame_rgb.tobytes())
                except (BrokenPipeError, OSError):
                    print("\nWarning: ffmpeg pipe closed before all frames were written.")
                    break
                frame += 1

                if frame % 30 == 0:
                    approx_total = total_poses // poses_per_frame
                    print(f"  frame {frame} / ~{approx_total}", end="\r", flush=True)
        finally:
            if ffmpeg_proc.stdin is not None:
                ffmpeg_proc.stdin.close()
            return_code = ffmpeg_proc.wait()
            vis.destroy_window()

        if return_code != 0:
            print(f"Warning: ffmpeg exited with non-zero status ({return_code}).")
            return False, True

        print(f"\nDone! Wrote {frame} frames to {cfg.output_path}")
        return True, True

    def _print_open3d_backend_error(self, mode: str) -> None:
        print(f"Error: Open3D {mode} backend is unavailable in this environment.")
        print("This is usually a Wayland/OpenGL context issue.")
        print("Try `--backend mpl`, or run under X11/XWayland for Open3D.")

    def _configure_render(self, vis: o3d.visualization.Visualizer) -> bool:
        render_option = vis.get_render_option()
        if render_option is None:
            return False
        render_option.background_color = np.array([0.05, 0.05, 0.08], dtype=np.float64)
        render_option.point_size = 2.0
        return True

    def _record_with_matplotlib(
        self,
        cfg: RecordConfig,
        width: int,
        height: int,
        fps: int,
        poses_per_frame: int,
    ) -> bool:
        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib import pyplot as plt

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            cfg.output_path,
        ]

        try:
            ffmpeg_proc = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError:
            print("Failed to open ffmpeg pipe. Is ffmpeg installed?")
            return False

        total_poses = len(self.trajectory)
        frame = 0
        cam_pos = np.array([8.0, 6.0, 8.0], dtype=np.float64)
        cam_look = np.array([0.0, 1.5, 0.0], dtype=np.float64)
        smoothing = float(np.clip(cfg.smoothing, 0.0, 1.0))
        alpha = 1.0 - smoothing

        cloud_points = (
            np.asarray([pt.position for pt in self.cloud], dtype=np.float64)
            if self.cloud
            else np.empty((0, 3), dtype=np.float64)
        )
        cloud_colors = (
            np.asarray([pt.color for pt in self.cloud], dtype=np.float64)
            if self.cloud
            else np.empty((0, 3), dtype=np.float64)
        )
        key_points, key_lines, key_colors = self._frustum_arrays(
            pose_indices=range(0, len(self.trajectory), self.keyframe_step),
            scale=self.frustum_scale,
            color=np.array([0.4, 0.7, 1.0], dtype=np.float64),
        )

        center, radius = self._compute_scene_bounds(cloud_points)

        dpi = 100
        fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        fig.patch.set_facecolor((0.05, 0.05, 0.08))
        ax = fig.add_subplot(111, projection="3d")

        print(f"Recording {total_poses} poses to {cfg.output_path} ...")
        return_code = 0

        try:
            for pose_idx in range(0, total_poses, poses_per_frame):
                T_wc = self.trajectory[pose_idx].T_wc
                agent_pos = T_wc[:3, 3]
                agent_forward = -T_wc[:3, 2]
                desired_cam_pos = (
                    agent_pos
                    - agent_forward * float(cfg.camera_distance)
                    + np.array([0.0, float(cfg.camera_height), 0.0], dtype=np.float64)
                )

                cam_pos = cam_pos * (1.0 - alpha) + desired_cam_pos * alpha
                cam_look = cam_look * (1.0 - alpha) + agent_pos * alpha

                ax.cla()
                self._configure_mpl_axes(ax, center=center, radius=radius)
                self._draw_grid_mpl(ax)

                if cloud_points.shape[0] > 0:
                    cloud_plot = _world_to_mpl(cloud_points)
                    ax.scatter(
                        cloud_plot[:, 0],
                        cloud_plot[:, 1],
                        cloud_plot[:, 2],
                        c=cloud_colors,
                        s=1.0,
                        depthshade=False,
                    )

                traj_points, traj_lines, traj_colors = self._trajectory_arrays(up_to_pose=pose_idx)
                self._draw_lineset_mpl(
                    ax=ax,
                    points_world=traj_points,
                    lines=traj_lines,
                    colors=traj_colors,
                    linewidth=1.8,
                )
                self._draw_lineset_mpl(
                    ax=ax,
                    points_world=key_points,
                    lines=key_lines,
                    colors=key_colors,
                    linewidth=0.9,
                )

                cur_points, cur_lines, cur_colors = self._frustum_arrays(
                    pose_indices=[pose_idx],
                    scale=self.frustum_scale * 2.0,
                    color=np.array([1.0, 0.9, 0.2], dtype=np.float64),
                )
                self._draw_lineset_mpl(
                    ax=ax,
                    points_world=cur_points,
                    lines=cur_lines,
                    colors=cur_colors,
                    linewidth=1.6,
                )

                self._set_mpl_camera(ax=ax, eye_world=cam_pos, lookat_world=cam_look)

                fig.canvas.draw()
                frame_rgb = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[:, :, :3]

                if ffmpeg_proc.stdin is None:
                    break
                try:
                    ffmpeg_proc.stdin.write(frame_rgb.tobytes())
                except (BrokenPipeError, OSError):
                    print("\nWarning: ffmpeg pipe closed before all frames were written.")
                    break

                frame += 1
                if frame % 30 == 0:
                    approx_total = total_poses // poses_per_frame
                    print(f"  frame {frame} / ~{approx_total}", end="\r", flush=True)
        finally:
            plt.close(fig)
            if ffmpeg_proc.stdin is not None:
                ffmpeg_proc.stdin.close()
            return_code = ffmpeg_proc.wait()

        if return_code != 0:
            print(f"Warning: ffmpeg exited with non-zero status ({return_code}).")
            return False
        print(f"\nDone! Wrote {frame} frames to {cfg.output_path}")
        return True

    def _compute_scene_bounds(self, cloud_points: np.ndarray) -> tuple[np.ndarray, float]:
        candidates = []
        if cloud_points.shape[0] > 0:
            candidates.append(cloud_points)
        if self.trajectory:
            traj_points = np.asarray([pose.T_wc[:3, 3] for pose in self.trajectory], dtype=np.float64)
            candidates.append(traj_points)

        if not candidates:
            return np.array([0.0, 0.0, 0.0], dtype=np.float64), 6.0

        all_points = np.vstack(candidates)
        min_v = all_points.min(axis=0)
        max_v = all_points.max(axis=0)
        center = 0.5 * (min_v + max_v)
        span = float(np.max(max_v - min_v))
        radius = max(4.0, 0.65 * span + 1.0)
        return center, radius

    def _configure_mpl_axes(self, ax, center: np.ndarray, radius: float) -> None:
        ax.set_facecolor((0.05, 0.05, 0.08))
        ax.grid(False)
        ax.set_axis_off()

        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[2] - radius, center[2] + radius)
        ax.set_zlim(max(0.0, center[1] - radius * 0.2), center[1] + radius)
        try:
            ax.set_box_aspect((1.0, 1.0, 0.8))
        except Exception:
            pass

    def _set_mpl_camera(self, ax, eye_world: np.ndarray, lookat_world: np.ndarray) -> None:
        view = np.asarray(lookat_world, dtype=np.float64) - np.asarray(eye_world, dtype=np.float64)
        view_mpl = np.array([view[0], view[2], view[1]], dtype=np.float64)
        if np.linalg.norm(view_mpl) < 1e-12:
            ax.view_init(elev=20.0, azim=-60.0)
            return
        xy_norm = np.linalg.norm(view_mpl[:2])
        elev = np.degrees(np.arctan2(view_mpl[2], max(xy_norm, 1e-12)))
        azim = np.degrees(np.arctan2(view_mpl[1], view_mpl[0]))
        ax.view_init(elev=float(elev), azim=float(azim))

    def _draw_grid_mpl(self, ax) -> None:
        extent = 10.0
        step = 1.0
        values = np.arange(-extent, extent + 1e-6, step, dtype=np.float64)
        for x in values:
            seg = np.array([[x, 0.0, -extent], [x, 0.0, extent]], dtype=np.float64)
            seg = _world_to_mpl(seg)
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=(0.3, 0.3, 0.3), alpha=0.5, linewidth=0.5)
        for z in values:
            seg = np.array([[-extent, 0.0, z], [extent, 0.0, z]], dtype=np.float64)
            seg = _world_to_mpl(seg)
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=(0.3, 0.3, 0.3), alpha=0.5, linewidth=0.5)

    def _draw_lineset_mpl(
        self,
        ax,
        points_world: np.ndarray,
        lines: np.ndarray,
        colors: np.ndarray,
        linewidth: float,
    ) -> None:
        if points_world.shape[0] == 0 or lines.shape[0] == 0:
            return
        points = _world_to_mpl(points_world)
        default_color = (0.9, 0.9, 0.9)
        for i, (a, b) in enumerate(lines):
            seg = points[[a, b]]
            color = tuple(colors[i]) if i < colors.shape[0] else default_color
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=color, linewidth=linewidth)

    def _set_camera_pose(
        self,
        vis: o3d.visualization.Visualizer,
        eye: np.ndarray,
        lookat: np.ndarray,
        width: int,
        height: int,
        fx: float,
        fy: float,
    ) -> None:
        extrinsic = _lookat_extrinsic(eye=eye, target=lookat, up=np.array([0.0, 1.0, 0.0]))

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width,
            height,
            float(fx),
            float(fy),
            width / 2.0,
            height / 2.0,
        )
        params = o3d.camera.PinholeCameraParameters()
        params.intrinsic = intrinsic
        params.extrinsic = extrinsic

        controller = vis.get_view_control()
        try:
            controller.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
        except TypeError:
            controller.convert_from_pinhole_camera_parameters(params)

    def _create_point_cloud_geometry(self) -> o3d.geometry.PointCloud:
        geometry = o3d.geometry.PointCloud()
        if not self.cloud:
            return geometry

        points = np.asarray([pt.position for pt in self.cloud], dtype=np.float64)
        colors = np.asarray([pt.color for pt in self.cloud], dtype=np.float64)

        geometry.points = o3d.utility.Vector3dVector(points)
        geometry.colors = o3d.utility.Vector3dVector(colors)
        return geometry

    def _create_grid_lineset(self) -> o3d.geometry.LineSet:
        extent = 10.0
        step = 1.0
        values = np.arange(-extent, extent + 1e-6, step, dtype=np.float64)

        points = []
        lines = []
        for x in values:
            idx = len(points)
            points.append([x, 0.0, -extent])
            points.append([x, 0.0, extent])
            lines.append([idx, idx + 1])

        for z in values:
            idx = len(points)
            points.append([-extent, 0.0, z])
            points.append([extent, 0.0, z])
            lines.append([idx, idx + 1])

        colors = np.tile(np.array([[0.3, 0.3, 0.3]], dtype=np.float64), (len(lines), 1))
        return _build_lineset(np.asarray(points), np.asarray(lines, dtype=np.int32), colors)

    def _create_trajectory_lineset(self, up_to_pose: int) -> o3d.geometry.LineSet:
        points, lines, colors = self._trajectory_arrays(up_to_pose)
        return _build_lineset(points, lines, colors)

    def _update_trajectory_lineset(self, line_set: o3d.geometry.LineSet, up_to_pose: int) -> None:
        points, lines, colors = self._trajectory_arrays(up_to_pose)
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)

    def _create_keyframe_frustums_lineset(self) -> o3d.geometry.LineSet:
        indices = range(0, len(self.trajectory), self.keyframe_step)
        points, lines, colors = self._frustum_arrays(
            pose_indices=indices,
            scale=self.frustum_scale,
            color=np.array([0.4, 0.7, 1.0], dtype=np.float64),
        )
        return _build_lineset(points, lines, colors)

    def _create_current_frustum_lineset(self, pose_idx: int) -> o3d.geometry.LineSet:
        points, lines, colors = self._frustum_arrays(
            pose_indices=[pose_idx],
            scale=self.frustum_scale * 2.0,
            color=np.array([1.0, 0.9, 0.2], dtype=np.float64),
        )
        return _build_lineset(points, lines, colors)

    def _update_current_frustum_lineset(
        self, line_set: o3d.geometry.LineSet, pose_idx: int
    ) -> None:
        points, lines, colors = self._frustum_arrays(
            pose_indices=[pose_idx],
            scale=self.frustum_scale * 2.0,
            color=np.array([1.0, 0.9, 0.2], dtype=np.float64),
        )
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)

    def _trajectory_arrays(self, up_to_pose: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.trajectory:
            return _empty_lineset_arrays()

        if up_to_pose < 0:
            limit = len(self.trajectory)
        else:
            limit = min(up_to_pose + 1, len(self.trajectory))

        if limit <= 0:
            return _empty_lineset_arrays()

        points = np.asarray(
            [self.trajectory[i].T_wc[:3, 3] for i in range(limit)],
            dtype=np.float64,
        )

        if limit < 2:
            return points, np.empty((0, 2), dtype=np.int32), np.empty((0, 3), dtype=np.float64)

        lines = np.column_stack(
            [np.arange(0, limit - 1, dtype=np.int32), np.arange(1, limit, dtype=np.int32)]
        )
        colors = np.tile(np.array([[0.2, 0.9, 0.4]], dtype=np.float64), (lines.shape[0], 1))
        return points, lines, colors

    def _frustum_arrays(
        self,
        pose_indices: Iterable[int],
        scale: float,
        color: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        frustum_local_points = _frustum_points(scale=scale)
        frustum_local_lines = _frustum_lines()

        all_points = []
        all_lines = []
        all_colors = []
        offset = 0

        for pose_idx in pose_indices:
            if pose_idx < 0 or pose_idx >= len(self.trajectory):
                continue
            T_wc = self.trajectory[pose_idx].T_wc
            world_points = _transform_points(T_wc, frustum_local_points)

            all_points.append(world_points)
            all_lines.append(frustum_local_lines + offset)
            all_colors.append(np.tile(color.reshape(1, 3), (frustum_local_lines.shape[0], 1)))
            offset += world_points.shape[0]

        if not all_points:
            return _empty_lineset_arrays()

        points = np.vstack(all_points)
        lines = np.vstack(all_lines).astype(np.int32)
        colors = np.vstack(all_colors)
        return points, lines, colors


def _lookat_extrinsic(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    eye = np.asarray(eye, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    up = np.asarray(up, dtype=np.float64).reshape(3)

    forward = target - eye
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-12:
        forward = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    else:
        forward = forward / forward_norm

    up_norm = np.linalg.norm(up)
    if up_norm < 1e-12:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        up = up / up_norm

    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-12:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        right = np.cross(forward, up)
        right_norm = np.linalg.norm(right)
    right = right / (right_norm + 1e-12)
    cam_up = np.cross(right, forward)
    cam_up = cam_up / (np.linalg.norm(cam_up) + 1e-12)

    extrinsic = np.eye(4, dtype=np.float64)
    extrinsic[:3, :3] = np.vstack([right, cam_up, -forward])
    extrinsic[:3, 3] = -extrinsic[:3, :3] @ eye
    return extrinsic


def _frustum_points(scale: float) -> np.ndarray:
    w = scale
    h = w * 0.75
    z = w * 0.6
    return np.asarray(
        [
            [0.0, 0.0, 0.0],
            [w, h, z],
            [w, -h, z],
            [-w, -h, z],
            [-w, h, z],
        ],
        dtype=np.float64,
    )


def _frustum_lines() -> np.ndarray:
    return np.asarray(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 1],
        ],
        dtype=np.int32,
    )


def _transform_points(T_wc: np.ndarray, points_c: np.ndarray) -> np.ndarray:
    ones = np.ones((points_c.shape[0], 1), dtype=np.float64)
    points_h = np.hstack([points_c, ones])
    transformed = (np.asarray(T_wc, dtype=np.float64) @ points_h.T).T
    return transformed[:, :3]


def _empty_lineset_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.empty((0, 3), dtype=np.float64),
        np.empty((0, 2), dtype=np.int32),
        np.empty((0, 3), dtype=np.float64),
    )


def _build_lineset(points: np.ndarray, lines: np.ndarray, colors: np.ndarray) -> o3d.geometry.LineSet:
    geometry = o3d.geometry.LineSet()
    geometry.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    geometry.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    geometry.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    return geometry


def _world_to_mpl(points_world: np.ndarray) -> np.ndarray:
    points_world = np.asarray(points_world, dtype=np.float64)
    if points_world.ndim != 2 or points_world.shape[1] != 3:
        return np.empty((0, 3), dtype=np.float64)
    # Map world (x, y-up, z) -> matplotlib (x, y, z-up)
    return np.column_stack([points_world[:, 0], points_world[:, 2], points_world[:, 1]])


def _normalize_backend(backend: str) -> str:
    name = str(backend).strip().lower()
    if name not in SUPPORTED_BACKENDS:
        valid = ", ".join(SUPPORTED_BACKENDS)
        raise ValueError(f"Invalid backend '{backend}'. Expected one of: {valid}")
    return name
