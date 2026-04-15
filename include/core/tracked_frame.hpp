#pragma once
#include "core/types.hpp"
#include <vector>

namespace vio {

struct TrackedFrame {
    FrameState state;
    std::vector<Observation> observations;
};

}