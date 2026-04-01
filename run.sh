#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
VENV_DIR="${ROOT_DIR}/.venv"
LOG_DIR="${ROOT_DIR}/.run_logs"
PY_REQ="${ROOT_DIR}/python/requirements.txt"
DEFAULT_OPENCV_DIR="/usr/lib/x86_64-linux-gnu/cmake/opencv4"
CONFIGURE_LOG="${LOG_DIR}/configure.log"
BUILD_LOG="${LOG_DIR}/build.log"
PIP_LOG="${LOG_DIR}/pip.log"

mkdir -p "${LOG_DIR}"

say() {
  printf '[run] %s\n' "$1"
}

run_quiet() {
  local label="$1"
  local logfile="$2"
  shift 2

  if "$@" >"${logfile}" 2>&1; then
    say "${label}: ok"
  else
    say "${label}: failed"
    echo "---- ${label} log ----"
    tail -n 80 "${logfile}" || true
    exit 1
  fi
}

if [[ $# -lt 1 ]]; then
  echo "Usage: ./run.sh <dataset_path> [vio options...]"
  echo "   or: ./run.sh --demo [vio options...]"
  echo "   or: ./run.sh --demo-l [vio options...]"
  echo "   or: ./run.sh --vicon-demo [.] [vio options...]"
  exit 1
fi

if [[ "$1" == "--help" || "$1" == "-h" ]]; then
  echo "Usage: ./run.sh <dataset_path> [vio options...]"
  echo "   or: ./run.sh --demo [vio options...]"
  echo "   or: ./run.sh --demo-l [vio options...]"
  echo "   or: ./run.sh --vicon-demo [.] [vio options...]"
  echo "Example: ./run.sh /path/to/euroc_dataset --record output.mp4"
  echo "Example: ./run.sh --demo"
  echo "Example: ./run.sh --demo-l"
  echo "Example: ./run.sh --vicon-demo . --rate 2.0"
  exit 0
fi

RUN_ARGS=()
if [[ "$1" == "--demo" ]]; then
  RUN_ARGS+=(--demo)
  shift || true
elif [[ "$1" == "--demo-l" || "$1" == "--vicon-demo" || "$1" == "--vicon-live-demo" || "$1" == "--vicon-live.demo" ]]; then
  shift || true
  if [[ "${1-}" == "." ]]; then
    shift || true
  fi
  DATASET_PATH="${ROOT_DIR}/dataset/mav0"
  if [[ ! -d "${DATASET_PATH}" ]]; then
    echo "Hardcoded Vicon demo dataset does not exist: ${DATASET_PATH}"
    exit 1
  fi
  RUN_ARGS+=(--vicon-demo "${DATASET_PATH}")
else
  DATASET_PATH="$1"
  shift || true
  if [[ ! -d "${DATASET_PATH}" ]]; then
    echo "Dataset path does not exist: ${DATASET_PATH}"
    exit 1
  fi
  RUN_ARGS+=("${DATASET_PATH}")
fi

OPENCV_DIR="${OpenCV_DIR:-${DEFAULT_OPENCV_DIR}}"
if [[ ! -f "${OPENCV_DIR}/OpenCVConfig.cmake" ]]; then
  DETECTED_OPENCV_DIR="$(find /usr /usr/local /opt -name OpenCVConfig.cmake 2>/dev/null | head -n 1 | xargs -r dirname)"
  if [[ -n "${DETECTED_OPENCV_DIR}" ]]; then
    OPENCV_DIR="${DETECTED_OPENCV_DIR}"
  fi
fi

if [[ ! -f "${OPENCV_DIR}/OpenCVConfig.cmake" ]]; then
  echo "OpenCVConfig.cmake not found."
  echo "Install OpenCV or export OpenCV_DIR before running this script."
  exit 1
fi
say "OpenCV: ${OPENCV_DIR}"

if [[ ! -d "${VENV_DIR}" ]]; then
  say "Creating virtualenv"
  python3 -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
export PATH="${VENV_DIR}/bin:${PATH}"
export VIO_LOG_DIR="${LOG_DIR}"
say "Python env: ${VENV_DIR}"

if ! python3 -c "import rerun, cv2" >/dev/null 2>&1; then
  say "Installing Python dependencies"
  run_quiet "Python dependencies" "${PIP_LOG}" pip install -r "${PY_REQ}"
else
  say "Python dependencies: ready"
fi

run_quiet "Configure" "${CONFIGURE_LOG}" cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DOpenCV_DIR="${OPENCV_DIR}"
run_quiet "Build" "${BUILD_LOG}" cmake --build "${BUILD_DIR}" -j

say "Runtime logs: ${LOG_DIR}"
if [[ "${RUN_ARGS[*]-}" == "--demo" ]]; then
  say "Launching: synthetic live demo"
elif [[ "${RUN_ARGS[0]-}" == "--vicon-demo" ]]; then
  say "Launching: Vicon demo from ${RUN_ARGS[1]}"
else
  say "Launching: dataset ${RUN_ARGS[0]}"
fi
exec "${BUILD_DIR}/vio" "${RUN_ARGS[@]}" "$@"
