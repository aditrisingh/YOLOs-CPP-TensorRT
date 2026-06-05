#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "yolos/core/MemoryArena.hpp"

namespace yolos {

class TrtSessionBase {
public:
    TrtSessionBase(const std::string& enginePath, int dlaCore = -1, int warmupRuns = 10);
    virtual ~TrtSessionBase();

    void inferGpu(const cv::Mat& image);
    void infer(const float* blob, size_t count);
    void* allocate(size_t size);

    // Interface for detection / classification
    virtual const std::vector<int64_t>& getOutputShape(int bindingIndex = 0) const;
    virtual const float* getOutputData(int bindingIndex = 0) const;
    virtual int numOutputs() const;
    virtual float getCachedScale() const;
    virtual float getCachedPadX() const;
    virtual float getCachedPadY() const;
    cv::Size getResizedShape() const;

protected:
    std::vector<int64_t> inputShape_{1, 3, 640, 640};

    float cachedScale_ = 1.0f;
    float cachedPadX_ = 0.0f;
    float cachedPadY_ = 0.0f;

private:
    MemoryArena memoryArena;           // moved after protected members to fix reorder warning

    void initEngine(const std::string& enginePath, int dlaCore);
    void warmUp(int runs);
    void captureInferenceGraph();
};

} // namespace yolos
