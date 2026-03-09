#!/bin/bash

set -e

BUILD_DIR=build

echo "Creating build directory..."
mkdir -p $BUILD_DIR

echo "Running CMake..."
cmake -S . -B $BUILD_DIR

echo "Building project..."
cmake --build $BUILD_DIR

echo "Build finished!"