#include "pipeline/pipeline_pivot.hpp"

#include <stdexcept>

namespace vio {

    void PipelinePivot::set(const TrackedFrame& frame)
    {
        frame_ = frame;
        valid_ = true;
    }

    void PipelinePivot::reset()
    {
        frame_ = TrackedFrame{};
        valid_ = false;
    }

    bool PipelinePivot::valid() const
    {
        return valid_;
    }

    const TrackedFrame& PipelinePivot::frame() const
    {
        if (!valid_) {
            throw std::runtime_error("PipelinePivot is not valid");
        }

        return frame_;
    }

    const FrameState& PipelinePivot::state() const
    {
        return frame().state;
    }

    const std::vector<Observation>& PipelinePivot::observations() const
    {
        return frame().observations;
    }

}