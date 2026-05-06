<claude-mem-context>
# Memory Context

# [vio] recent context, 2026-05-06 12:37am GMT+3

Legend: 🎯session 🔴bugfix 🟣feature 🔄refactor ✅change 🔵discovery ⚖️decision 🚨security_alert 🔐security_note
Format: ID TIME TYPE TITLE
Fetch details: get_observations([IDs]) | Search: mem-search skill

Stats: 50 obs (21,744t read) | 1,239,116t work | 98% savings

### Apr 27, 2026
S3 VIO pipeline redesign with pivot/triangulation/PnP cycle — decision on whether TBB flow_graph is needed for concurrency (Apr 27, 3:52 AM)
S2 VIO pipeline redesign: pivot-based init → 2-view triangulation → PnP cycle, and whether Intel TBB flow_graph is needed (Apr 27, 3:56 AM)
S4 Specific TBB and C++20 usage guidance for the VIO pipeline — which TBB primitives to use, where, and why (Apr 27, 3:57 AM)
S6 TBB vs oneTBB API compatibility — concurrent_bounded_queue::abort() removed in oneTBB, poison pill pattern required instead (Apr 27, 3:58 AM)
S7 Environment audit and optimization package selection for VIO project — what to install and what CMakeLists.txt changes are needed for maximum performance (Apr 27, 3:59 AM)
S5 TBB vs oneTBB API difference — concurrent_bounded_queue::abort() removed in oneTBB, corrected shutdown pattern (Apr 27, 3:59 AM)
S8 Implement DatasetStreamer — parallel IMU + camera streaming producer-consumer refactor for the VIO C++ project (Apr 27, 4:12 AM)
S12 GPU-прискорення VIO проекту: аналіз коду що можна розпаралелити на GPU і чому (Apr 27, 1:28 PM)
### Apr 28, 2026
S13 GPU acceleration for VIO pipeline algorithms on Intel Iris XE — auto-detection + CPU fallback via environment variable (Apr 28, 2:11 PM)
66 2:13p 🔵 Build Directory Reveals Old CMake Compiled klt_tracker.cpp; System OpenCV Has No CUDA Modules
67 2:14p 🔵 System OpenCV 4.6.0 Confirmed: No CUDA Libraries Installed — Custom CUDA Kernels Required
68 " 🔵 Full Frontend Tracking Call Chain Mapped: No Virtual Dispatch, Direct Function Calls — Easy GPU Substitution Points
69 " 🔵 Subagent Incorrectly Claimed System Has OpenCV CUDA Modules — Only CPU Optflow Available
70 11:19p 🔵 VIO Project Structure — Matrix/Convolution Headers Located
71 " 🔵 VisionComputeBackend — Existing GPU/CPU Architecture for Matrix Ops
72 " 🔵 Hardware Capabilities & TODO in lk_tracker Matching User's Task
73 11:20p 🔵 keypoint_extraction/ Has AVX2-Optimized + Banded CPU Convolution — Best Available CPU Path
74 " ⚖️ Refactor Plan Confirmed: Extend VisionComputeBackend With Matrix Convolution Methods
75 11:21p 🔵 FeatureTracker Is the Live Consumer of VisionComputeBackend
76 " 🔵 CustomShiTomasiDetector Instantiated Directly in 5 Call Sites — Bypasses VisionComputeBackend
77 " ⚖️ Implementation Plan Written: agile-humming-puffin — Extend VisionComputeBackend With gaussianBlur + sobelGradients
78 11:22p 🔵 CustomShiTomasiDetector Matrix Ops Bypass GPU Backend
79 " ⚖️ Plan: Route ShiTomasi Matrix Ops Through VisionComputeBackend (GPU/CPU)
### Apr 29, 2026
80 12:01a 🔵 ThreadPool Created Per-Call at All CustomShiTomasiDetector Instantiation Sites
81 " 🔵 vio_runner.cpp Does Not Call refreshTracksIfNeeded — Uses Detector Directly
83 " 🔵 OpenCL Backend Internal Architecture: ClImage Buffers Used Inline, No Reusable Upload/Download Helpers
84 " 🔵 VisionComputeBackend::createAuto() Controlled by VIO_GPU Env Var with Three Modes
82 " 🔵 refreshTracksIfNeeded Called Only from visual_frontend.cpp, Not vio_runner or imu_tracking_pipeline
85 " 🔄 VisionComputeBackend Interface Extended with gaussianBlur and sobelGradients Pure Virtuals
86 12:03a 🔄 vision_compute_backend.cpp Gains Includes for CPU Matrix Operation Implementations
87 12:04a 🟣 CpuVisionComputeBackend Implements gaussianBlur and sobelGradients with Persistent ThreadPool
88 " 🟣 gaussian_blur OpenCL Kernel Added to kOpenClSource
89 12:05a 🟣 OpenClVisionComputeBackend Implements gaussianBlur and sobelGradients via GPU
90 " 🔄 CustomShiTomasiDetector Constructor Changed from ABCThreadPool to VisionComputeBackend
91 " 🔄 CustomShiTomasiDetector Refactored to Use VisionComputeBackend for Matrix Ops
92 " 🔄 CustomShiTomasiDetector Fully Migrated — All 6 Matrix Op Calls Now Route Through VisionComputeBackend
94 " 🔴 Build Fails: vio Target Missing vision_compute_backend.cpp, imu_tracking_app Has Wrong gaussian_blur/sobel Sources
95 " 🔴 CMakeLists.txt Fixed: vision_compute_backend.cpp Added to vio, keypoints/ Sources Added to imu_tracking_app
93 12:06a 🔄 vio_runner.cpp Both Detector Sites Migrated to VisionComputeBackend::createAuto()
96 5:17p 🔵 VIO Pipeline Codebase Audit Initiated
97 " 🔵 VIO App Bypasses VioPipeline — Uses ImuTrackingPipeline Directly
98 " 🔵 VIO Project Has Significantly Expanded File Structure vs Intended Design
99 " 🔵 VioPipeline Has Clean 3-Stage State Machine and Is Used Internally by ImuTrackingPipeline
100 " 🔵 Core Types Use Consistent World-Camera Convention; RigidTransform Has No Frame Label
101 " 🔵 ImuTrackingPipeline::loadInputs() Has Duplicate loadImagePaths() Call Bug
102 5:20p 🔵 IMU Processor Has Proper 3-Second Bias Estimation and Kalman-Filtered Integration
103 " 🔵 Coordinate Frame Conversion in frame_pose_sync Uses pose.q.conjugate() — Potential Frame Confusion
104 " 🔵 Triangulator and PnPSolver Both Use Correct World-Camera Frame Conventions
105 " 🔵 configs/ Directory Does Not Exist — All Thresholds Are Hardcoded; Build Passes
106 " 🔵 DatasetLoader Loads EuRoC Format with T_BS Calibration; DatasetStreamer Uses Two-Pointer Merge
107 " 🔵 Sandbox Permission Errors — Tool Calls Failing Without require_escalated
108 " ⚖️ Audit Complete — Session Transitioning to Minimal Patch Planning for apps/vio_app.cpp
109 5:22p 🟣 Minimal Patch Applied: apps/vio_app.cpp Now Uses VioPipeline::runConfigured() API
110 5:24p 🟣 Patch Confirmed Applied via apply_patch; Build Started Successfully at 55%
111 " 🟣 Post-Patch Build Passes: [100%] Built target vio
112 5:25p 🔵 CTest Confirms Zero Tests Registered in Build System
113 " 🔵 Git Status Reveals Codebase Is Mid-Refactor — Many Core Files New/Deleted Since Last Commit
114 " ✅ Patch Also Fixed Broken Include: vio_pipeline.hpp Changed from deleted tracked_frame.hpp to types.hpp, and TrackedFrame Moved Into types.hpp
115 " 🟣 Final Clean Rebuild Passes: [100%] Built target vio — BUILD_OK
S14 VIO pipeline audit and refactor: second pass cutting custom keypoint_extraction/ and tracking/ subsystems from include chain, replacing with direct OpenCV calls (Apr 29, 5:25 PM)

Access 1239k tokens of past work via get_observations([IDs]) or mem-search skill.
</claude-mem-context>