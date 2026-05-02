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

if [[ "${1-}" == "--help" || "${1-}" == "-h" ]]; then
  echo "Usage: ./run.sh [mode] [dataset_path] [vio options...]"
  echo ""
  echo "Modes:"
  echo "  --plain, --simple, --no-vis    Run pipeline without visualization. This is the default."
  echo "  --rerun, --vis                 Run pipeline with live Rerun window."
  echo "  --video, --points              Generate MP4 points overlay from dataset images."
  echo "  --both                         Run Rerun window and generate MP4."
  echo ""
  echo "   or: ./run.sh --demo [vio options...]"
  echo "   or: ./run.sh --demo-l [vio options...]"
  echo "   or: ./run.sh --vicon-demo [.] [vio options...]"
  echo "Example: ./run.sh --plain"
  echo "Example: ./run.sh --rerun"
  echo "Example: ./run.sh --video --video-out ../results/points.mp4"
  echo "Example: ./run.sh --both /path/to/euroc_dataset --video-out ../results/points.mp4"
  echo "Example: ./run.sh --points"
  exit 0
fi

RUN_ARGS=()
if [[ "${1-}" == "--demo" ]]; then
  RUN_ARGS+=(--demo)
  shift || true
elif [[ "${1-}" == "--demo-l" || "${1-}" == "--vicon-demo" || "${1-}" == "--vicon-live-demo" || "${1-}" == "--vicon-live.demo" ]]; then
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
elif [[ $# -gt 0 && "${1}" != --* ]]; then
  DATASET_PATH="$1"
  shift || true
  if [[ ! -d "${DATASET_PATH}" ]]; then
    echo "Dataset path does not exist: ${DATASET_PATH}"
    exit 1
  fi
  DATASET_PATH="$(cd "${DATASET_PATH}" && pwd)"
  RUN_ARGS+=("${DATASET_PATH}")
fi

RUN_RERUN="${VIO_START_RERUN:-0}"
RUN_POINTS=0
FORWARD_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --plain|--simple|--no-vis)
      RUN_RERUN=0
      RUN_POINTS=0
      ;;
    --rerun|--vis|--visualization)
      RUN_RERUN=1
      RUN_POINTS=0
      ;;
    --video|--points|--points-video)
      RUN_RERUN=0
      RUN_POINTS=1
      ;;
    --both|--all-vis)
      RUN_RERUN=1
      RUN_POINTS=1
      ;;
    --no-rerun)
      RUN_RERUN=0
      ;;
    *)
      FORWARD_ARGS+=("$1")
      ;;
  esac
  shift || true
done

if [[ "${RUN_POINTS}" == "1" ]]; then
  FORWARD_ARGS+=(--points)
fi

if [[ "${RUN_RERUN}" == "0" ]]; then
  FORWARD_ARGS+=(--no-rerun)
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

export VIO_LOG_DIR="${LOG_DIR}"

if [[ "${RUN_RERUN}" != "0" ]]; then
  if [[ ! -d "${VENV_DIR}" ]]; then
    say "Creating virtualenv"
    python3 -m venv "${VENV_DIR}"
  fi

  source "${VENV_DIR}/bin/activate"
  export PATH="${VENV_DIR}/bin:${PATH}"
  say "Python env: ${VENV_DIR}"

  if ! python3 -c "import rerun, cv2" >/dev/null 2>&1; then
    say "Installing Python dependencies"
    run_quiet "Python dependencies" "${PIP_LOG}" pip install -r "${PY_REQ}"
  else
    say "Python dependencies: ready"
  fi
else
  say "Rerun: disabled"
fi

run_quiet "Configure" "${CONFIGURE_LOG}" cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DOpenCV_DIR="${OPENCV_DIR}"
run_quiet "Build" "${BUILD_LOG}" cmake --build "${BUILD_DIR}" -j

RERUN_HOST="${VIO_RERUN_HOST:-127.0.0.1}"
RERUN_PORT="${VIO_RERUN_PORT:-9877}"

if [[ "${RUN_RERUN}" != "0" ]]; then
  say "Launching Rerun window on ${RERUN_HOST}:${RERUN_PORT}"
  python3 "${ROOT_DIR}/python/vio_py/cli.py" --host "${RERUN_HOST}" --port "${RERUN_PORT}" \
      >"${LOG_DIR}/rerun_receiver.log" 2>&1 &
  RERUN_PID=$!
  for _ in {1..40}; do
    if grep -q "Rerun receiver listening" "${LOG_DIR}/rerun_receiver.log" 2>/dev/null; then
      break
    fi
    if ! kill -0 "${RERUN_PID}" 2>/dev/null; then
      break
    fi
    sleep 0.25
  done
  if ! kill -0 "${RERUN_PID}" 2>/dev/null; then
    say "Rerun receiver failed"
    echo "---- Rerun receiver log ----"
    tail -n 80 "${LOG_DIR}/rerun_receiver.log" || true
    exit 1
  fi
fi

say "Runtime logs: ${LOG_DIR}"
if [[ "${RUN_ARGS[*]-}" == "--demo" ]]; then
  say "Launching: synthetic live demo"
elif [[ "${RUN_ARGS[0]-}" == "--vicon-demo" ]]; then
  say "Launching: Vicon demo from ${RUN_ARGS[1]}"
else
  say "Launching: ${RUN_ARGS[*]-default configured dataset}"
fi
cd "${BUILD_DIR}"
exec ./vio "${RUN_ARGS[@]}" "${FORWARD_ARGS[@]}"
