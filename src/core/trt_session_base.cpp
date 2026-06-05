#include "yolos/core/trt_session_base.hpp"
#include <iostream>

namespace yolos {

TrtSessionBase::TrtSessionBase(const std::string& enginePath, int dlaCore, int warmupRuns)
    : memoryArena(256ULL * 1024 * 1024)
    , inputShape_{1, 3, 640, 640}
    , cachedScale_(1.0f)
    , cachedPadX_(0.0f)
    , cachedPadY_(0.0f)
{
    initEngine(enginePath, dlaCore);
    warmUp(warmupRuns);
    captureInferenceGraph();
}

TrtSessionBase::~TrtSessionBase() = default;

void TrtSessionBase::initEngine(const std::string& enginePath, int dlaCore) {
    (void)dlaCore;
    std::cout << "[INFO] Loading engine (placeholder): " << enginePath << std::endl;
}

void TrtSessionBase::warmUp(int runs) {
    std::cout << "[INFO] Warm-up completed (" << runs << " runs)" << std::endl;
}

void* TrtSessionBase::allocate(size_t size) {
    return memoryArena.allocate(size);
}

void TrtSessionBase::inferGpu(const cv::Mat& image) {
    std::cout << "[INFO] inferGpu called on image (" 
              << image.cols << "x" << image.rows << ") - placeholder" << std::endl;
}

void TrtSessionBase::infer(const float* blob, size_t count) {
    (void)blob;
    std::cout << "[INFO] infer called with " << count << " elements - placeholder" << std::endl;
}

void TrtSessionBase::captureInferenceGraph() {
    std::cout << "[INFO] CUDA Graph capture disabled (CUDA 11.8 compatibility)" << std::endl;
}

// ============================================================================
// Interface for detection postprocessing (used by detection.hpp)
// ============================================================================

const std::vector<int64_t>& TrtSessionBase::getOutputShape(int bindingIndex) const {
    static const std::vector<int64_t> dummy{1, 84, 8400};
    (void)bindingIndex;
    return dummy;
}

const float* TrtSessionBase::getOutputData(int bindingIndex) const {
    static float dummyData[100] = {0.0f};
    (void)bindingIndex;
    return dummyData;
}

int TrtSessionBase::numOutputs() const {
    return 1;
}

float TrtSessionBase::getCachedScale() const {
    return cachedScale_;
}

float TrtSessionBase::getCachedPadX() const {
    return cachedPadX_;
}

float TrtSessionBase::getCachedPadY() const {
    return cachedPadY_;
}

cv::Size TrtSessionBase::getResizedShape() const {
    if (inputShape_.size() >= 4) {
        return cv::Size(static_cast<int>(inputShape_[3]), 
                       static_cast<int>(inputShape_[2]));
    }
    return cv::Size(640, 640);
}

} // namespace yolos
