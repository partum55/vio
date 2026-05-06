#include "io/dataset_loader.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iostream>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace vio {
namespace {

std::string readTextFile(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("could not open file: " + path.string());
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

std::filesystem::path stripTrailingSeparators(std::filesystem::path path) {
    while (!path.empty() && path.filename().empty() && path.has_parent_path()) {
        path = path.parent_path();
    }
    return path;
}

std::vector<double> parseNumberListAfterKey(const std::string& text,
                                            const std::string& key,
                                            std::size_t expected_min) {
    const std::size_t key_pos = text.find(key);
    if (key_pos == std::string::npos) {
        throw std::runtime_error("missing key '" + key + "' in calibration file");
    }

    const std::size_t open = text.find('[', key_pos);
    const std::size_t close = text.find(']', open);
    if (open == std::string::npos || close == std::string::npos || close <= open) {
        throw std::runtime_error("malformed array for key '" + key + "'");
    }

    const std::string body = text.substr(open + 1, close - open - 1);
    static const std::regex number_re(R"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)");

    std::vector<double> values;
    for (std::sregex_iterator it(body.begin(), body.end(), number_re), end; it != end; ++it) {
        values.push_back(std::stod(it->str()));
    }

    if (values.size() < expected_min) {
        throw std::runtime_error("not enough values for key '" + key + "'");
    }
    return values;
}

CameraCalibration loadCameraCalibration(const std::filesystem::path& path) {
    const std::string text = readTextFile(path);
    CameraCalibration calibration;

    const std::vector<double> intrinsics = parseNumberListAfterKey(text, "intrinsics", 4);
    calibration.fx = intrinsics[0];
    calibration.fy = intrinsics[1];
    calibration.cx = intrinsics[2];
    calibration.cy = intrinsics[3];

    const std::vector<double> resolution = parseNumberListAfterKey(text, "resolution", 2);
    calibration.width = static_cast<int>(resolution[0]);
    calibration.height = static_cast<int>(resolution[1]);

    const std::vector<double> distortion = parseNumberListAfterKey(text, "distortion_coefficients", 4);
    calibration.distortion_coeffs = distortion;

    const std::size_t model_pos = text.find("distortion_model");
    if (model_pos != std::string::npos) {
        const std::size_t colon = text.find(':', model_pos);
        const std::size_t end = text.find('\n', colon);
        if (colon != std::string::npos) {
            calibration.distortion_model = text.substr(
                colon + 1,
                end == std::string::npos ? std::string::npos : end - colon - 1);
            calibration.distortion_model.erase(
                std::remove_if(calibration.distortion_model.begin(),
                               calibration.distortion_model.end(),
                               [](unsigned char ch) { return std::isspace(ch) != 0; }),
                calibration.distortion_model.end());
        }
    }

    const std::vector<double> t_bs = parseNumberListAfterKey(text, "T_BS", 16);
    calibration.T_BS = Eigen::Matrix4d::Identity();
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            calibration.T_BS(row, col) = t_bs[static_cast<std::size_t>(row * 4 + col)];
        }
    }
    return calibration;
}

ImuCalibration loadImuCalibration(const std::filesystem::path& path);

std::optional<CameraCalibration> tryLoadCameraCalibration(
    const std::filesystem::path& camera_dir
) {
    const std::filesystem::path undistorted_yaml =
        camera_dir / "sensor-undistorted.yaml";
    if (std::filesystem::exists(undistorted_yaml)) {
        return loadCameraCalibration(undistorted_yaml);
    }

    const std::filesystem::path sensor_yaml = camera_dir / "sensor.yaml";
    if (std::filesystem::exists(sensor_yaml)) {
        return loadCameraCalibration(sensor_yaml);
    }

    return std::nullopt;
}

std::optional<ImuCalibration> tryLoadImuCalibration(
    const std::filesystem::path& dataset_root
) {
    const std::filesystem::path imu_yaml =
        dataset_root / "imu0" / "sensor.yaml";
    if (std::filesystem::exists(imu_yaml)) {
        return loadImuCalibration(imu_yaml);
    }

    return std::nullopt;
}

ImuCalibration loadImuCalibration(const std::filesystem::path& path) {
    const std::string text = readTextFile(path);
    ImuCalibration calibration;
    const std::vector<double> t_bs = parseNumberListAfterKey(text, "T_BS", 16);
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            calibration.T_BS(row, col) = t_bs[static_cast<std::size_t>(row * 4 + col)];
        }
    }
    return calibration;
}

std::vector<DatasetFrame> loadFramesFromCsv(const std::filesystem::path& csv_path,
                                            const std::filesystem::path& image_dir) {
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        throw std::runtime_error("could not open image list: " + csv_path.string());
    }

    std::vector<DatasetFrame> frames;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::stringstream ss(line);
        std::string ts_field;
        std::string filename_field;
        if (!std::getline(ss, ts_field, ',')) {
            continue;
        }
        if (!std::getline(ss, filename_field)) {
            continue;
        }

        const std::int64_t timestamp_ns = std::stoll(ts_field);
        DatasetFrame frame;
        frame.timestamp_ns = timestamp_ns;
        frame.timestamp_s = static_cast<double>(timestamp_ns) * 1e-9;
        frame.frame_index = frames.size();
        frame.image_path = image_dir / filename_field;
        frames.push_back(frame);
    }

    return frames;
}

std::vector<DatasetFrame> scanFramesFromDirectory(const std::filesystem::path& image_dir) {
    std::vector<DatasetFrame> frames;
    for (const auto& entry : std::filesystem::directory_iterator(image_dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const std::filesystem::path path = entry.path();
        const std::string ext = path.extension().string();
        if (ext != ".png" && ext != ".jpg" && ext != ".jpeg") {
            continue;
        }

        DatasetFrame frame;
        frame.timestamp_ns = std::stoll(path.stem().string());
        frame.timestamp_s = static_cast<double>(frame.timestamp_ns) * 1e-9;
        frame.image_path = path;
        frames.push_back(frame);
    }

    std::sort(frames.begin(), frames.end(), [](const DatasetFrame& a, const DatasetFrame& b) {
        return a.timestamp_ns < b.timestamp_ns;
    });
    for (std::size_t i = 0; i < frames.size(); ++i) {
        frames[i].frame_index = i;
    }
    return frames;
}

std::vector<double> loadFrameTimestamps(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("could not open frame timestamps: " + path.string());
    }

    std::vector<double> timestamps;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::stringstream ss(line);
        std::string first_field;
        if (!std::getline(ss, first_field, ',')) {
            continue;
        }

        const double raw = std::stod(first_field);
        timestamps.push_back(raw > 1.0e12 ? raw * 1.0e-9 : raw);
    }

    return timestamps;
}

CameraCalibration toCameraCalibration(const CameraIntrinsics& intrinsics) {
    CameraCalibration calibration;
    calibration.width = intrinsics.width;
    calibration.height = intrinsics.height;
    calibration.fx = intrinsics.fx;
    calibration.fy = intrinsics.fy;
    calibration.cx = intrinsics.cx;
    calibration.cy = intrinsics.cy;
    return calibration;
}

} // namespace

Dataset loadDataset(const DatasetLoadOptions& options) {
    if (options.imu_csv_path.empty()) {
        throw std::runtime_error("DatasetLoadOptions::imu_csv_path is empty");
    }
    if (options.images_dir.empty()) {
        throw std::runtime_error("DatasetLoadOptions::images_dir is empty");
    }
    if (!options.camera_intrinsics.isValid()) {
        throw std::runtime_error("DatasetLoadOptions::camera_intrinsics is invalid");
    }

    Dataset dataset;
    const std::filesystem::path image_dir_abs =
        std::filesystem::absolute(stripTrailingSeparators(options.images_dir));
    const std::filesystem::path camera_dir = image_dir_abs.parent_path();
    const std::filesystem::path dataset_root = camera_dir.parent_path();
    dataset.root = dataset_root;
    dataset.camera = toCameraCalibration(options.camera_intrinsics);

    if (const std::optional<CameraCalibration> camera_calibration =
            tryLoadCameraCalibration(camera_dir)) {
        dataset.camera = *camera_calibration;
    }

    if (const std::optional<ImuCalibration> imu_calibration =
            tryLoadImuCalibration(dataset_root)) {
        dataset.imu = *imu_calibration;
    }

    dataset.frames = scanFramesFromDirectory(options.images_dir);
    if (dataset.frames.empty()) {
        throw std::runtime_error("no camera frames found in " + options.images_dir.string());
    }

    if (!options.frame_timestamps_path.empty()) {
        const std::vector<double> timestamps =
            loadFrameTimestamps(options.frame_timestamps_path);
        if (timestamps.size() != dataset.frames.size()) {
            throw std::runtime_error(
                "frame timestamp count does not match image count: " +
                std::to_string(timestamps.size()) + " timestamps vs " +
                std::to_string(dataset.frames.size()) + " images"
            );
        }

        for (std::size_t i = 0; i < dataset.frames.size(); ++i) {
            dataset.frames[i].timestamp_s = timestamps[i];
            dataset.frames[i].timestamp_ns =
                static_cast<std::int64_t>(std::llround(timestamps[i] * 1.0e9));
            dataset.frames[i].frame_index = i;
        }
    }

    if (!loadImuCsv(options.imu_csv_path.string(), dataset.imu_samples) ||
        dataset.imu_samples.empty()) {
        throw std::runtime_error("failed to load IMU CSV: " + options.imu_csv_path.string());
    }

    std::sort(dataset.imu_samples.begin(), dataset.imu_samples.end(),
        [](const ImuSample& a, const ImuSample& b) { return a.t < b.t; });

    return dataset;
}

Dataset loadEurocDataset(const std::filesystem::path& root) {
    Dataset dataset;
    dataset.root = std::filesystem::absolute(root);

    const auto cam_dir = dataset.root / "cam0";
    const auto imu_dir = dataset.root / "imu0";
    const auto image_dir = cam_dir / "data";
    const auto image_csv = cam_dir / "data.csv";
    const auto imu_csv = imu_dir / "data.csv";
    const auto cam_yaml = cam_dir / "sensor.yaml";
    const auto imu_yaml = imu_dir / "sensor.yaml";

    if (!std::filesystem::exists(cam_dir) || !std::filesystem::exists(imu_dir)) {
        throw std::runtime_error("expected EuRoC-like dataset with cam0/ and imu0/ directories");
    }
    if (!std::filesystem::exists(image_dir)) {
        throw std::runtime_error("missing camera image directory: " + image_dir.string());
    }
    if (!std::filesystem::exists(imu_csv)) {
        throw std::runtime_error("missing IMU csv: " + imu_csv.string());
    }
    if (!std::filesystem::exists(cam_yaml) || !std::filesystem::exists(imu_yaml)) {
        throw std::runtime_error("missing sensor calibration yaml inside dataset");
    }

    dataset.camera = loadCameraCalibration(cam_yaml);
    dataset.imu = loadImuCalibration(imu_yaml);

    if (std::filesystem::exists(image_csv)) {
        dataset.frames = loadFramesFromCsv(image_csv, image_dir);
    } else {
        dataset.frames = scanFramesFromDirectory(image_dir);
    }
    if (dataset.frames.empty()) {
        throw std::runtime_error("no camera frames found in dataset");
    }

    if (!loadImuCsv(imu_csv.string(), dataset.imu_samples) || dataset.imu_samples.empty()) {
        throw std::runtime_error("failed to load IMU samples from " + imu_csv.string());
    }

    return dataset;
}

} // namespace vio
