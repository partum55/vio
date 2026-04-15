#include "io/tracked_output_writer.hpp"

#include <fstream>
#include <iomanip>

bool writeFrameStatesCsv(
    const std::string& path,
    const std::vector<vio::TrackedFrame>& sequence
)
{
    std::ofstream out(path);
    if (!out.is_open()) {
        return false;
    }

    out << "frame_id,timestamp,px,py,pz,vx,vy,vz,qw,qx,qy,qz\n";
    out << std::fixed << std::setprecision(9);

    for (const auto& frame : sequence) {
        const auto& s = frame.state;

        out << s.frame_id << ","
            << s.timestamp << ","
            << s.t_wc.x() << "," << s.t_wc.y() << "," << s.t_wc.z() << ","
            << s.v_w.x() << "," << s.v_w.y() << "," << s.v_w.z() << ","
            << s.q_wc.w() << "," << s.q_wc.x() << "," << s.q_wc.y() << "," << s.q_wc.z()
            << "\n";
    }

    return true;
}

bool writeObservationsCsv(
    const std::string& path,
    const std::vector<vio::TrackedFrame>& sequence
)
{
    std::ofstream out(path);
    if (!out.is_open()) {
        return false;
    }

    out << "frame_id,timestamp,track_id,x,y,valid\n";
    out << std::fixed << std::setprecision(9);

    for (const auto& frame : sequence) {
        for (const auto& obs : frame.observations) {
            out << obs.frame_id << ","
                << frame.state.timestamp << ","
                << obs.track_id << ","
                << obs.uv.x() << ","
                << obs.uv.y() << ","
                << (obs.valid ? 1 : 0)
                << "\n";
        }
    }

    return true;
}