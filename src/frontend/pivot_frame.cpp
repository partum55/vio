#include "frontend/pivot_frame.hpp"

#include <stdexcept>

namespace vio {

    void PivotFrame::set(const TrackedFrame& frame) {
        frame_ = frame;
        valid_ = true;
    }

    void PivotFrame::reset() {
        frame_ = TrackedFrame{};
        valid_ = false;
    }

    bool PivotFrame::valid() const {
        return valid_;
    }

    const TrackedFrame& PivotFrame::frame() const {
        if (!valid_) {
            throw std::runtime_error("PivotFrame is not valid");
        }

        return frame_;
    }

    const FrameState& PivotFrame::state() const {
        return frame().state;
    }

    const std::vector<Observation>& PivotFrame::observations() const {
        return frame().observations;
    }

}