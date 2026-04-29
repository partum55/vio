#pragma once

#include "core/tracked_frame.hpp"

namespace vio {

    class PipelinePivot {
    public:
        void set(const TrackedFrame& frame);
        void reset();

        bool valid() const;

        const TrackedFrame& frame() const;
        const FrameState& state() const;
        const std::vector<Observation>& observations() const;

    private:
        bool valid_ = false;
        TrackedFrame frame_;
    };

}