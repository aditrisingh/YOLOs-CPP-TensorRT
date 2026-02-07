<!-- Shields -->
<p align="center">
  <a href="https://github.com/Geekgineer/YOLOs-CPP/actions"><img src="https://github.com/Geekgineer/YOLOs-CPP/actions/workflows/main.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-AGPL--3.0-blue.svg" alt="License: AGPL-3.0"></a>
  <a href="https://developer.nvidia.com/tensorrt"><img src="https://img.shields.io/badge/TensorRT-%E2%89%A5%2010.0-76B900?logo=nvidia" alt="TensorRT"></a>
  <a href="https://developer.nvidia.com/cuda-toolkit"><img src="https://img.shields.io/badge/CUDA-%E2%89%A5%2012.0-76B900?logo=nvidia" alt="CUDA"></a>
  <img src="https://img.shields.io/badge/C%2B%2B-17-blue.svg?logo=cplusplus" alt="C++17">
</p>

# YOLOs-TRT &mdash; High-Performance YOLO Inference with TensorRT

A **C++17 header-only library** (plus CUDA kernels) for running YOLO object-detection, segmentation, pose-estimation, oriented-bounding-box, and classification models at maximum throughput on NVIDIA GPUs using **TensorRT**.

---

## Highlights

- **Multi-version support** &mdash; YOLOv5, v6, v7, v8, v9, v10, v11, v12, YOLO26, YOLO-NAS
- **Five task types** &mdash; Detection, Instance Segmentation, Pose Estimation, OBB (Oriented Bounding Box), Classification
- **Runtime auto-detection** of YOLO version from output tensor shapes &mdash; no manual config needed
- **Zero CPU preprocessing in the hot path** &mdash; single CUDA kernel performs letterbox + BGR-to-RGB + normalize directly into the TRT input buffer
- **CUDA graph capture** eliminates per-frame kernel launch overhead
- **Pinned staging buffers** for truly async host-to-device transfers
- **GPU-only execution** &mdash; TensorRT is always GPU
- **Jetson support** &mdash; Xavier (sm_72) and Orin (sm_87) with optional DLA offload
- **Header-only API** &mdash; include the headers you need, link TensorRT and CUDA, done

---

## Benchmark

Measured on an **NVIDIA RTX 2000 Ada Laptop GPU** with YOLOv11n (640x640 input, 1000 inference iterations after 10-iteration warm-up):

| Precision | FPS | Avg Latency | P50 Latency | P99 Latency | Peak GPU Memory |
|:---------:|:---:|:-----------:|:-----------:|:-----------:|:---------------:|
| **FP32**  | 466 | 2.144 ms    | 2.040 ms    | 3.028 ms    | 529.5 MB        |
| **FP16**  | 479 | 2.086 ms    | 1.976 ms    | 2.913 ms    | 535.7 MB        |
| **INT8**  | 530 | 1.886 ms    | 1.775 ms    | 2.701 ms    | 443.7 MB        |

Run the included benchmark suite to reproduce:

```bash
cd build
./yolo_unified_benchmark --engine ../models/yolo11n.trt --labels ../models/coco.names --task detect
```

---

## Architecture

```
cv::Mat (host BGR)
  |
  |  memcpy into pinned staging buffer
  v
  pinned host  --cudaMemcpyAsync-->  device uint8 (raw BGR)
  |
  |  letterboxNormalizeKernel (single CUDA kernel)
  |    letterbox resize  +  BGR-to-RGB  +  /255 normalize
  v
  TRT input buffer (device float32 NCHW)
  |
  |  enqueueV3  /  cudaGraphLaunch
  v
  TRT output buffer(s) (device)
  |
  |  cudaMemcpyAsync D2H
  v
  host output  -->  postprocess (CPU)
```

### Core Components

| Component | Description |
|-----------|-------------|
| **`TrtSessionBase`** | Engine deserialization, I/O buffer allocation, warm-up, CUDA graph capture, async inference pipeline |
| **CUDA preprocessing** (`cuda_preprocessing.cu`) | Single kernel: bilinear letterbox + BGR-to-RGB + normalize. Writes directly into TRT input buffer. |
| **CUDA graph capture** | For fixed-shape models the entire `enqueueV3` call graph is captured once and replayed via `cudaGraphLaunch`, eliminating ~0.1-0.3 ms of per-frame dispatch overhead |
| **Pinned staging buffers** | Raw BGR pixels are copied into a CUDA pinned buffer before async H2D transfer, overlapping copy with compute |
| **10-iteration warm-up** | Lets the TensorRT autotuner converge to optimal kernel selections before timing begins |

---

## Supported Models and Tasks

| Task | Class | Method | Models |
|------|-------|--------|--------|
| Detection | `YOLODetector` | `detect()` | YOLOv5, v7, v8, v9, v10, v11, v12, v26, NAS |
| Segmentation | `YOLOSegDetector` | `segment()` | YOLOv8-seg, v11-seg, v26-seg |
| Pose Estimation | `YOLOPoseDetector` | `detect()` | YOLOv8-pose, v11-pose, v26-pose |
| OBB | `YOLOOBBDetector` | `detect()` | YOLOv8-obb, v11-obb, v26-obb |
| Classification | `YOLOClassifier` | `classify()` | YOLOv8-cls, v11-cls, v12-cls, v26-cls |

---

## Prerequisites

| Dependency | Minimum Version | Notes |
|------------|:---------------:|-------|
| NVIDIA GPU | Compute Capability >= 7.5 | Turing, Ampere, Ada, Hopper, or Jetson Xavier/Orin |
| CUDA Toolkit | >= 12.0 | |
| TensorRT | >= 10.0 | Tensor-based API (`enqueueV3`) |
| OpenCV | >= 4.5 | Used for image I/O and postprocess visualization |
| CMake | >= 3.18 | CUDA language support required |
| C++ compiler | C++17 | GCC 9+, Clang 10+ |
| Python | >= 3.8 | Only needed for ONNX export and TensorRT conversion |

---

## Installation

### 1. Install TensorRT

**Ubuntu (x86_64):**

```bash
sudo apt update
sudo apt install tensorrt
```

**Jetson (JetPack 6.x):**

TensorRT is pre-installed with JetPack. Verify:

```bash
dpkg -l | grep tensorrt
```

**Manual install:**

Download from <https://developer.nvidia.com/tensorrt> and set `TENSORRT_DIR` when building.

### 2. Clone the repository

```bash
git clone https://github.com/Geekgineer/YOLOs-CPP.git
cd YOLOs-CPP
```

### 3. Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

To specify a custom TensorRT location:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DTENSORRT_DIR=/path/to/tensorrt
```

To build the example applications as well:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=ON
```

### 4. Verify

```bash
./image_inference ../models/yolo11n.trt ../data/dog.jpg ../models/coco.names
```

---

## Model Conversion

### Export ONNX from Ultralytics

```bash
pip install -r requirements.txt   # or: uv pip install -r requirements.txt
python models/export_onnx.py --model yolo11n
```

### Convert ONNX to TensorRT engine

```bash
# FP16 (recommended for most GPUs)
python trt-files/scripts/convert_to_tensorrt.py --onnx models/yolo11n.onnx --fp16

# INT8 with calibration data
python trt-files/scripts/convert_to_tensorrt.py --onnx models/yolo11n.onnx --int8 \
    --calib-data trt-files/scripts/calibration_data/

# FP32 (baseline)
python trt-files/scripts/convert_to_tensorrt.py --onnx models/yolo11n.onnx
```

The converter outputs a `.trt` engine file ready for inference.

---

## Quick Start

```cpp
#include "yolos/tasks/detection.hpp"

using namespace yolos::det;

int main() {
    // Load TensorRT engine — version is auto-detected from output tensor shape
    YOLODetector detector("yolo11n.trt", "coco.names");

    cv::Mat image = cv::imread("dog.jpg");
    auto detections = detector.detect(image);

    detector.drawDetections(image, detections);
    cv::imshow("Result", image);
    cv::waitKey(0);
}
```

### Segmentation

```cpp
#include "yolos/tasks/segmentation.hpp"

yolos::seg::YOLOSegDetector seg("yolo11n-seg.trt", "coco.names");
auto results = seg.segment(image);
seg.drawSegmentations(image, results);
```

### Pose Estimation

```cpp
#include "yolos/tasks/pose.hpp"

yolos::pose::YOLOPoseDetector pose("yolo11n-pose.trt");
auto results = pose.detect(image);
pose.drawPoses(image, results);
```

### Oriented Bounding Box (OBB)

```cpp
#include "yolos/tasks/obb.hpp"

yolos::obb::YOLOOBBDetector obb("yolo11n-obb.trt", "Dota.names");
auto results = obb.detect(image);
obb.drawDetections(image, results);
```

### Classification

```cpp
#include "yolos/tasks/classification.hpp"

yolos::cls::YOLOClassifier cls("yolov8n-cls.trt", "ImageNet.names");
auto result = cls.classify(image);
std::cout << result.className << ": " << result.confidence * 100 << "%" << std::endl;
```

---

## Executables

The build produces five ready-to-use binaries:

| Binary | Description | Usage |
|--------|-------------|-------|
| `image_inference` | Single image or folder detection | `./image_inference <engine> [image] [labels]` |
| `video_inference` | Video file detection (multi-threaded capture/detect/write) | `./video_inference <engine> [video] [output] [labels]` |
| `camera_inference` | Live camera detection (multi-threaded) | `./camera_inference <engine> [source] [labels]` |
| `batch_image_inference` | Batch detection over a folder of images | `./batch_image_inference <engine> [folder] [labels]` |
| `class_image_inference` | Image classification | `./class_image_inference <engine> [image] [labels]` |

**Examples:**

```bash
# Detect objects in an image
./image_inference ../models/yolo11n.trt ../data/dog.jpg ../models/coco.names

# Process a video file
./video_inference ../models/yolo11n.trt ../data/dogs.mp4 ../data/output.mp4 ../models/coco.names

# Live camera feed (V4L2)
./camera_inference ../models/yolo11n.trt /dev/video0 ../models/coco.names

# Classify an image
./class_image_inference ../models/yolov8n-cls.trt ../data/dog.jpg ../models/ImageNet.names
```

---

## Docker

### Data-center GPU

```bash
docker build -f Dockerfile.tensorrt -t yolos-trt .

docker run --gpus all \
    -v /path/to/models:/app/models \
    -v /path/to/data:/app/data \
    yolos-trt \
    "./image_inference ./models/yolo11n.trt ./data/dog.jpg ./models/coco.names"
```

### NVIDIA Jetson (JetPack 6.x)

```bash
docker build -f Dockerfile.tensorrt.jetson -t yolos-trt-jetson .

docker run --runtime nvidia \
    -v /path/to/models:/app/models \
    -v /path/to/data:/app/data \
    yolos-trt-jetson \
    "./image_inference ./models/yolo11n.trt ./data/dog.jpg ./models/coco.names"
```

---

## Project Structure

```
YOLOs-CPP/
├── CMakeLists.txt                  # Top-level build (CXX + CUDA)
├── include/
│   └── yolos/
│       ├── yolos.hpp               # Master include (all tasks)
│       ├── core/
│       │   ├── trt_session_base.hpp # Engine load, buffers, graph, async infer
│       │   ├── trt_utils.hpp        # CUDA check macros, logger, memory RAII
│       │   ├── cuda_preprocessing.hpp  # Host-side API for CUDA preprocessing
│       │   ├── cuda_preprocessing.cu   # Letterbox + BGR→RGB + normalize kernel
│       │   ├── preprocessing.hpp    # CPU fallback preprocessing
│       │   ├── nms.hpp              # Batched NMS
│       │   ├── drawing.hpp          # Bounding box / mask / skeleton drawing
│       │   ├── types.hpp            # BoundingBox, KeyPoint, OrientedBoundingBox
│       │   ├── utils.hpp            # File I/O, label parsing
│       │   └── version.hpp          # YOLOVersion enum, auto-detection logic
│       └── tasks/
│           ├── detection.hpp        # YOLODetector + version-specific subclasses
│           ├── segmentation.hpp     # YOLOSegDetector (instance seg with masks)
│           ├── pose.hpp             # YOLOPoseDetector (keypoint estimation)
│           ├── obb.hpp              # YOLOOBBDetector (oriented bounding boxes)
│           └── classification.hpp   # YOLOClassifier + version subclasses
├── src/
│   ├── image_inference.cpp          # Single-image / folder detection
│   ├── video_inference.cpp          # Multi-threaded video pipeline
│   ├── camera_inference.cpp         # Live camera pipeline
│   ├── batch_image_inference.cpp    # Batch folder detection
│   └── class_image_inference.cpp    # Image classification
├── models/
│   ├── export_onnx.py               # Ultralytics ONNX exporter
│   ├── coco.names                   # COCO 80-class labels
│   ├── Dota.names                   # DOTA OBB labels
│   └── ImageNet.names               # ImageNet 1000-class labels
├── trt-files/
│   └── scripts/
│       ├── convert_to_tensorrt.py   # ONNX → TensorRT engine converter
│       ├── generate_calibration_data.py  # INT8 calibration image generator
│       └── calibration_data/        # Sample calibration images
├── benchmarks/
│   ├── CMakeLists.txt
│   └── yolo_unified_benchmark.cpp   # Multi-task benchmark suite
├── tests/                           # Per-task C++/Python validation suites
├── examples/                        # Per-task example apps (image/video/camera)
├── data/                            # Sample images
├── Dockerfile.tensorrt              # Multi-stage Docker for data-center GPUs
├── Dockerfile.tensorrt.jetson       # Multi-stage Docker for Jetson
└── doc/                             # Additional documentation
```

---

## API Reference

### Common Constructor Pattern

All task classes inherit from `TrtSessionBase` and follow the same constructor pattern:

```cpp
ClassName(const std::string& enginePath,
          const std::string& labelsPath,
          [YOLOVersion version = YOLOVersion::Auto,]   // detection only
          int dlaCore = -1);
```

| Parameter | Description |
|-----------|-------------|
| `enginePath` | Path to the serialized TensorRT engine (`.trt` / `.engine`) |
| `labelsPath` | Path to newline-delimited class names file |
| `version` | Explicit YOLO version override; default `Auto` resolves from output tensor shape |
| `dlaCore` | DLA core index for Jetson (`-1` = GPU, `0` or `1` = DLA) |

### Task Classes

| Class | Namespace | Primary Method | Return Type |
|-------|-----------|----------------|-------------|
| `YOLODetector` | `yolos::det` | `detect(image, confThresh, iouThresh)` | `std::vector<Detection>` |
| `YOLOSegDetector` | `yolos::seg` | `segment(image, confThresh, iouThresh)` | `std::vector<Segmentation>` |
| `YOLOPoseDetector` | `yolos::pose` | `detect(image, confThresh, iouThresh)` | `std::vector<PoseResult>` |
| `YOLOOBBDetector` | `yolos::obb` | `detect(image, confThresh, iouThresh)` | `std::vector<OBBResult>` |
| `YOLOClassifier` | `yolos::cls` | `classify(image)` | `ClassificationResult` |

### Factory Functions

```cpp
// Detection — auto-detect or specify version
auto det = yolos::det::createDetector("yolo11n.trt", "coco.names");
auto det = yolos::det::createDetector("yolo11n.trt", "coco.names", yolos::YOLOVersion::V11);

// Classification
auto cls = yolos::cls::createClassifier("yolov8n-cls.trt", "ImageNet.names");
auto cls = yolos::cls::createClassifier("yolov8n-cls.trt", "ImageNet.names", yolos::YOLOVersion::V12);
```

### Version-Specific Detector Subclasses

For cases where auto-detection is not desired, explicit subclasses are provided:

| Subclass | Version | Postprocessing |
|----------|---------|----------------|
| `YOLOv7Detector` | v7 | `[batch, boxes, features]` with NMS |
| `YOLOv8Detector` | v8 | `[batch, features, boxes]` with NMS |
| `YOLOv10Detector` | v10 | `[batch, boxes, 6]` end-to-end (no NMS) |
| `YOLOv11Detector` | v11 | `[batch, features, boxes]` with NMS |
| `YOLO26Detector` | v26 | `[batch, boxes, 6]` end-to-end (no NMS) |
| `YOLONASDetector` | NAS | Two-output (boxes + scores) with NMS |

---

## DLA on Jetson

To offload inference to a DLA core on Jetson Orin/Xavier:

```cpp
// Use DLA core 0
YOLODetector detector("yolo11n.trt", "coco.names", YOLOVersion::Auto, /*dlaCore=*/0);
```

Build the engine with DLA support during conversion for best compatibility.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `TensorRT not found` | Set `-DTENSORRT_DIR=/path/to/tensorrt` in CMake, or install via `sudo apt install tensorrt` |
| `CUDA graph capture failed` | This is normal for dynamic-shape models; inference falls back to `enqueueV3` automatically |
| Engine built on different GPU | TensorRT engines are GPU-specific. Rebuild the engine on the target device. |
| Low FPS on first frames | The 10-iteration warm-up handles this. If benchmarking, discard the first few frames. |

---

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLO model ecosystem
- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) for the high-performance inference runtime
