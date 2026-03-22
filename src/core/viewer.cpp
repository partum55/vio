#include "core/viewer.h"

#include <pangolin/pangolin.h>
#include <GL/gl.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

namespace vio {

Viewer::Viewer(const Trajectory& trajectory, const PointCloud& cloud)
    : trajectory_(trajectory), cloud_(cloud) {}

void Viewer::run() {
    pangolin::CreateWindowAndBind("VIO Viewer", 1280, 720);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1280, 720, 500, 500, 640, 360, 0.1, 100),
        pangolin::ModelViewLookAt(8, 6, 8, 0, 1.5, 0, pangolin::AxisY));

    pangolin::View& d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, 0.0, 1.0, -1280.0 / 720.0)
        .SetHandler(new pangolin::Handler3D(s_cam));

    while (!pangolin::ShouldQuit()) {
        glClearColor(0.05f, 0.05f, 0.08f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);

        drawScene();
        pangolin::FinishFrame();
    }
}

void Viewer::record(const RecordConfig& cfg) {
    const int W = cfg.width;
    const int H = cfg.height;

    pangolin::CreateWindowAndBind("VIO Recorder", W, H);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    const double aspect = static_cast<double>(W) / H;
    const double fx = 0.7 * W;
    const double fy = 0.7 * W;

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(W, H, fx, fy, W / 2.0, H / 2.0, 0.1, 200),
        pangolin::ModelViewLookAt(8, 6, 8, 0, 1.5, 0, pangolin::AxisY));

    pangolin::View& d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, 0.0, 1.0, -aspect);

    char cmd[1024];
    std::snprintf(cmd, sizeof(cmd),
                  "ffmpeg -y -f rawvideo -vcodec rawvideo -pix_fmt rgb24 "
                  "-s %dx%d -r %d -i - "
                  "-c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p "
                  "-movflags +faststart \"%s\" 2>/dev/null",
                  W, H, cfg.fps, cfg.output_path.c_str());

    FILE* ffmpeg = popen(cmd, "w");
    if (!ffmpeg) {
        std::cerr << "Failed to open ffmpeg pipe. Is ffmpeg installed?\n";
        return;
    }

    std::vector<unsigned char> pixels(W * H * 3);
    Eigen::Vector3d cam_pos(8, 6, 8);
    Eigen::Vector3d cam_look(0, 1.5, 0);

    const int total_poses = static_cast<int>(trajectory_.size());
    int frame = 0;

    std::cout << "Recording " << total_poses << " poses to " << cfg.output_path << " ...\n";

    for (int pose_idx = 0; pose_idx < total_poses; pose_idx += cfg.poses_per_frame) {
        if (pangolin::ShouldQuit()) {
            break;
        }

        const Eigen::Matrix4d& T = trajectory_[pose_idx].T_wc;
        Eigen::Vector3d agent_pos = T.block<3, 1>(0, 3);
        Eigen::Vector3d agent_forward = -T.block<3, 1>(0, 2);
        Eigen::Vector3d desired_cam_pos = agent_pos - agent_forward * cfg.camera_distance
                                        + Eigen::Vector3d(0, cfg.camera_height, 0);

        double alpha = 1.0 - cfg.smoothing;
        cam_pos = cam_pos * (1.0 - alpha) + desired_cam_pos * alpha;
        cam_look = cam_look * (1.0 - alpha) + agent_pos * alpha;

        s_cam.SetModelViewMatrix(
            pangolin::ModelViewLookAt(
                cam_pos.x(), cam_pos.y(), cam_pos.z(),
                cam_look.x(), cam_look.y(), cam_look.z(),
                pangolin::AxisY));

        glClearColor(0.05f, 0.05f, 0.08f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);

        drawGrid();
        drawPointCloud();
        drawTrajectory(pose_idx);
        drawCurrentFrustum(pose_idx);
        drawCameraFrustums();

        pangolin::FinishFrame();

        glReadPixels(0, 0, W, H, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
        const int row_bytes = W * 3;
        std::vector<unsigned char> row_buf(row_bytes);
        for (int y = 0; y < H / 2; ++y) {
            unsigned char* top = pixels.data() + y * row_bytes;
            unsigned char* bot = pixels.data() + (H - 1 - y) * row_bytes;
            std::memcpy(row_buf.data(), top, row_bytes);
            std::memcpy(top, bot, row_bytes);
            std::memcpy(bot, row_buf.data(), row_bytes);
        }

        fwrite(pixels.data(), 1, pixels.size(), ffmpeg);
        ++frame;

        if (frame % 30 == 0) {
            std::cout << "  frame " << frame << " / ~"
                      << total_poses / cfg.poses_per_frame << "\r" << std::flush;
        }
    }

    pclose(ffmpeg);
    std::cout << "\nDone! Wrote " << frame << " frames to " << cfg.output_path << "\n";

    pangolin::DestroyWindow("VIO Recorder");
}

void Viewer::drawScene() {
    drawGrid();
    drawPointCloud();
    drawTrajectory();
    drawCameraFrustums();
}

void Viewer::drawPointCloud() {
    glPointSize(2.0f);
    glBegin(GL_POINTS);
    for (const auto& pt : cloud_) {
        glColor3f(pt.color.x(), pt.color.y(), pt.color.z());
        glVertex3d(pt.position.x(), pt.position.y(), pt.position.z());
    }
    glEnd();
}

void Viewer::drawTrajectory(int up_to_pose) {
    int limit = (up_to_pose < 0) ? static_cast<int>(trajectory_.size())
                                 : std::min(up_to_pose + 1, static_cast<int>(trajectory_.size()));

    glLineWidth(2.0f);
    glColor3f(0.2f, 0.9f, 0.4f);
    glBegin(GL_LINE_STRIP);
    for (int i = 0; i < limit; ++i) {
        Eigen::Vector3d p = trajectory_[i].T_wc.block<3, 1>(0, 3);
        glVertex3d(p.x(), p.y(), p.z());
    }
    glEnd();
}

void Viewer::drawCurrentFrustum(int pose_idx) {
    if (pose_idx < 0 || pose_idx >= static_cast<int>(trajectory_.size())) {
        return;
    }

    const float w = frustum_scale_ * 2.0f;
    const float h = w * 0.75f;
    const float z = w * 0.6f;

    const Eigen::Matrix4d& T = trajectory_[pose_idx].T_wc;

    glLineWidth(2.5f);
    glColor3f(1.0f, 0.9f, 0.2f);

    glPushMatrix();
    glMultMatrixd(T.data());

    glBegin(GL_LINES);
    glVertex3f(0, 0, 0); glVertex3f( w,  h, z);
    glVertex3f(0, 0, 0); glVertex3f( w, -h, z);
    glVertex3f(0, 0, 0); glVertex3f(-w, -h, z);
    glVertex3f(0, 0, 0); glVertex3f(-w,  h, z);
    glVertex3f( w,  h, z); glVertex3f( w, -h, z);
    glVertex3f( w, -h, z); glVertex3f(-w, -h, z);
    glVertex3f(-w, -h, z); glVertex3f(-w,  h, z);
    glVertex3f(-w,  h, z); glVertex3f( w,  h, z);
    glEnd();

    glPopMatrix();
}

void Viewer::drawCameraFrustums() {
    const float w = frustum_scale_;
    const float h = w * 0.75f;
    const float z = w * 0.6f;

    glLineWidth(1.0f);
    glColor3f(0.4f, 0.7f, 1.0f);

    for (size_t i = 0; i < trajectory_.size(); i += keyframe_step_) {
        const Eigen::Matrix4d& T = trajectory_[i].T_wc;

        glPushMatrix();
        glMultMatrixd(T.data());

        glBegin(GL_LINES);
        glVertex3f(0, 0, 0); glVertex3f( w,  h, z);
        glVertex3f(0, 0, 0); glVertex3f( w, -h, z);
        glVertex3f(0, 0, 0); glVertex3f(-w, -h, z);
        glVertex3f(0, 0, 0); glVertex3f(-w,  h, z);
        glVertex3f( w,  h, z); glVertex3f( w, -h, z);
        glVertex3f( w, -h, z); glVertex3f(-w, -h, z);
        glVertex3f(-w, -h, z); glVertex3f(-w,  h, z);
        glVertex3f(-w,  h, z); glVertex3f( w,  h, z);
        glEnd();

        glPopMatrix();
    }
}

void Viewer::drawGrid() {
    const float extent = 10.0f;
    const float step = 1.0f;

    glLineWidth(1.0f);
    glColor4f(0.3f, 0.3f, 0.3f, 0.5f);
    glBegin(GL_LINES);
    for (float x = -extent; x <= extent; x += step) {
        glVertex3f(x, 0.0f, -extent);
        glVertex3f(x, 0.0f,  extent);
    }
    for (float z = -extent; z <= extent; z += step) {
        glVertex3f(-extent, 0.0f, z);
        glVertex3f( extent, 0.0f, z);
    }
    glEnd();
}

} // namespace vio
