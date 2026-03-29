#include "lk_tracker.hpp"
#include "tracking_vis.hpp"

#include <filesystem>
#include <iostream>
#include <fstream>
#include <chrono>

int main()
{
    const std::string inputVideo  = "../IMG_4273.MOV";
    const std::string outputVideo = "tracking_visualization.mp4";
    const std::string outputCsv   = "tracking.csv";

    cv::VideoCapture cap(inputVideo);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open video source\n";
        return 1;
    }

    const double fps = cap.get(cv::CAP_PROP_FPS);
    const int width  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    const int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    cv::VideoWriter writer(
        outputVideo,
        cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
        (fps > 0.0 ? fps : 25.0),
        cv::Size(width, height)
    );

    if (!writer.isOpened()) {
        std::cerr << "Cannot open output video writer\n";
        return 1;
    }

    std::ofstream csv(outputCsv);
    if (!csv.is_open()) {
        std::cerr << "Cannot open CSV output file\n";
        return 1;
    }

    csv << "frame_idx,point_id,x,y,status\n";

    cv::Mat prevFrame;
    if (!cap.read(prevFrame) || prevFrame.empty()) {
        std::cerr << "Failed to read first frame from video.\n";
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

    int frameIdx = 0;

    for (const auto& t : tracks) {
        csv << frameIdx << "," << t.id << "," << t.pt.x << "," << t.pt.y << ",1\n";
    }

    writer.write(drawTrackingVisualization(prevFrame, tracks));

    while (true) {
        cv::Mat currFrame;
        if (!cap.read(currFrame) || currFrame.empty()) {
            break;
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
            trackPointsPyramidalLK(prevGray, currGray, ptsPrev, ptsCurr, status, err, 9, 3, 10, 1e-3f);
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

        if (tracks.size() < 50) {
            addNewTracks(currGray, tracks, nextTrackId, 100, 0.01, 10.0);
        }

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

    cap.release();
    writer.release();
    csv.close();

    std::cout << "Done.\n";
    std::cout << "Saved video: " << outputVideo << "\n";
    std::cout << "Saved tracking: " << outputCsv << "\n";

    return 0;
}
