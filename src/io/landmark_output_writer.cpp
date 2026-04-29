#include "io/landmark_output_writer.hpp"

#include <fstream>
#include <iomanip>

bool writeLandmarksCsv(
    const std::string& path,
    const std::vector<vio::Landmark>& landmarks
) {
    std::ofstream out(path);
    if (!out.is_open()) {
        return false;
    }

    out << "track_id,X,Y,Z,valid,reprojection_error,num_observations\n";
    out << std::fixed << std::setprecision(9);

    for (const auto& landmark : landmarks) {
        out << landmark.track_id << ","
            << landmark.p_w.x() << ","
            << landmark.p_w.y() << ","
            << landmark.p_w.z() << ","
            << (landmark.valid ? 1 : 0) << ","
            << landmark.reprojection_error << ","
            << landmark.num_observations
            << "\n";
    }

    return true;
}
