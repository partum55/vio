#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
LOG_DIR="${ROOT_DIR}/.run_logs"

CONFIGURE_LOG="${LOG_DIR}/configure.log"
BUILD_LOG="${LOG_DIR}/build.log"

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
    tail -n 120 "${logfile}" || true
    exit 1
  fi
}

OPENCV_DIR_DEFAULT="/usr/lib/x86_64-linux-gnu/cmake/opencv4"
OPENCV_DIR="${OpenCV_DIR:-${OPENCV_DIR_DEFAULT}}"

if [[ ! -f "${OPENCV_DIR}/OpenCVConfig.cmake" ]]; then
  DETECTED_OPENCV_DIR="$(
    find /usr /usr/local /opt -name OpenCVConfig.cmake 2>/dev/null \
      | head -n 1 \
      | xargs -r dirname
  )"

  if [[ -n "${DETECTED_OPENCV_DIR}" ]]; then
    OPENCV_DIR="${DETECTED_OPENCV_DIR}"
  fi
fi

if [[ ! -f "${OPENCV_DIR}/OpenCVConfig.cmake" ]]; then
  echo "OpenCVConfig.cmake not found."
  echo "Install OpenCV or run:"
  echo "  export OpenCV_DIR=/path/to/opencv4"
  exit 1
fi

say "OpenCV: ${OPENCV_DIR}"

run_quiet \
  "Configure" \
  "${CONFIGURE_LOG}" \
  cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DOpenCV_DIR="${OPENCV_DIR}"

run_quiet \
  "Build" \
  "${BUILD_LOG}" \
  cmake --build "${BUILD_DIR}" -j

say "Build finished"
say "Executable: ${BUILD_DIR}/vio"

if [[ $# -gt 0 ]]; then
  say "Running vio with arguments: $*"
  exec "${BUILD_DIR}/vio" "$@"
else
  say "No runtime arguments provided. Build only."
  say "Example:"
  say "  ./run.sh --demo"
fi