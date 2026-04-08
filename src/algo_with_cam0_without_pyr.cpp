#include "lk_tracker_without_pyr.hpp"
#include "tracking_vis.hpp"
#include "feature_refresh.hpp"

#include <filesystem>
#include <iostream>
#include <fstream>
#include <chrono>
#include <string>

int main()
{
    namespace fs = std::filesystem;

    const std::string imagesDir   = "../cam0/undistorted_alpha0";
    const std::string outputVideo = "tracking_visualization.mp4";
    const std::string outputCsv   = "tracking.csv";
    const double fps = 20.0;

    std::vector<std::string> imagePaths;
    for (const auto& entry : fs::directory_iterator(imagesDir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".png") {
            imagePaths.push_back(entry.path().string());
        }
    }

    std::sort(imagePaths.begin(), imagePaths.end());

    if (imagePaths.empty()) {
        std::cerr << "No PNG images found in directory: " << imagesDir << "\n";
        return 1;
    }

    std::ofstream csv(outputCsv);
    if (!csv.is_open()) {
        std::cerr << "Cannot open CSV output file\n";
        return 1;
    }

    csv << "frame_idx,point_id,x,y,status\n";

    cv::Mat prevFrame = cv::imread(imagePaths[0], cv::IMREAD_COLOR);
    if (prevFrame.empty()) {
        std::cerr << "Failed to read first image: " << imagePaths[0] << "\n";
        return 1;
    }

    const int width = prevFrame.cols;
    const int height = prevFrame.rows;

    cv::VideoWriter writer(
        outputVideo,
        cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
        fps,
        cv::Size(width, height)
    );

    if (!writer.isOpened()) {
        std::cerr << "Cannot open output video writer\n";
        return 1;
    }

    cv::Mat prevGray;
    cv::cvtColor(prevFrame, prevGray, cv::COLOR_BGR2GRAY);

    std::vector<cv::Point2f> initialPts;
    cv::goodFeaturesToTrack(prevGray, initialPts, 100, 0.01, 10.0);

    if (initialPts.empty()) {
        std::cerr << "No features found in the first frame.\n";
        return 1;
    }

    std::vector<Track> tracks;
    int nextTrackId = 0;

    for (const auto& p : initialPts) {
        Track t;
        t.id = nextTrackId++;
        t.pt = p;
        t.history.push_back(p);
        tracks.push_back(t);
    }

    FeatureRefreshParams refreshParams;
    refreshParams.minTrackedFeatures = 50;
    refreshParams.targetFeatures = 100;
    refreshParams.suppressionRadius = 10.0f;
    refreshParams.qualityLevel = 0.01;
    refreshParams.minDistance = 10.0;

    int frameIdx = 0;

    for (const auto& t : tracks) {
        csv << frameIdx << "," << t.id << "," << t.pt.x << "," << t.pt.y << ",1\n";
    }

    writer.write(drawTrackingVisualization(prevFrame, tracks));

    for (size_t imgIdx = 1; imgIdx < imagePaths.size(); ++imgIdx) {
        cv::Mat currFrame = cv::imread(imagePaths[imgIdx], cv::IMREAD_COLOR);
        if (currFrame.empty()) {
            std::cerr << "Failed to read image: " << imagePaths[imgIdx] << "\n";
            continue;
        }

        ++frameIdx;

        cv::Mat currGray;
        cv::cvtColor(currFrame, currGray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> ptsPrev;
        ptsPrev.reserve(tracks.size());
        for (const auto& t : tracks) {
            ptsPrev.push_back(t.pt);
        }

        std::vector<cv::Point2f> ptsCurr;
        std::vector<uchar> status;
        std::vector<float> err;

        try {
            auto t1 = std::chrono::high_resolution_clock::now();

            trackPointsLKSingleLevel(
                prevGray,
                currGray,
                ptsPrev,
                ptsCurr,
                status,
                err,
                9,
                10,
                1e-3f
            );

            auto t2 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
            std::cout << "Frame " << frameIdx << " tracking time: " << ms << " ms\n";
        } catch (const std::exception& e) {
            std::cerr << "Tracking error: " << e.what() << std::endl;
            return 1;
        }

        std::vector<Track> newTracks;
        newTracks.reserve(tracks.size());

        for (size_t i = 0; i < tracks.size(); ++i) {
            if (status[i]) {
                Track updated = tracks[i];
                updated.pt = ptsCurr[i];
                updated.history.push_back(ptsCurr[i]);
                newTracks.push_back(updated);

                csv << frameIdx << "," << updated.id << "," << updated.pt.x << "," << updated.pt.y << ",1\n";
            } else {
                csv << frameIdx << "," << tracks[i].id << ",-1,-1,0\n";
            }
        }

        tracks = std::move(newTracks);

        refreshTracksIfNeeded(currGray, tracks, nextTrackId, refreshParams);

        for (const auto& t : tracks) {
            if (t.history.size() == 1) {
                csv << frameIdx << "," << t.id << "," << t.pt.x << "," << t.pt.y << ",1\n";
            }
        }

        cv::Mat vis = drawTrackingVisualization(currFrame, tracks, 15);
        writer.write(vis);

        prevGray = currGray.clone();

        if (tracks.empty()) {
            std::cerr << "No points left to track.\n";
            break;
        }
    }

    writer.release();
    csv.close();

    std::cout << "Done.\n";
    std::cout << "Saved video: " << outputVideo << "\n";
    std::cout << "Saved tracking: " << outputCsv << "\n";

    return 0;
}