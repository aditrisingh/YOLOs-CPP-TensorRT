# YOLOs-CPP-TensorRT

**High-performance multi-version YOLO inference engine in C++ using TensorRT**

A fast, modular, and production-ready C++ framework for running YOLOv7, YOLOv8, YOLOv10, YOLOv11, YOLOv26, YOLO-NAS and more on NVIDIA GPUs with TensorRT.

---

## Features

- **Multi-Version Support**: Automatic YOLO version detection + explicit mode
- **End-to-End GPU Pipeline**: CUDA preprocessing (letterbox) → TensorRT inference → Postprocessing
- **Custom Memory Management**: Linear arena allocator for low fragmentation
- **Multiple Inference Modes**: Image, Video, and Batch inference
- **Clean Architecture**: `TrtSessionBase`, `YOLODetector`, `YOLOClassifier`
- **CMake + CUDA Ready**

## Supported Models
- YOLOv7, YOLOv8, YOLOv10, YOLOv11, YOLOv26, YOLO-NAS

## Tech Stack
- **C++17**
- **TensorRT 8.6**
- **CUDA 11.8**
- **OpenCV 4**
- **CMake**

## Build Instructions

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
Run Examples
Bash# Image inference
./image_inference ../models/yolov8n.engine ../models/coco.names ../images/test.jpg

# Video inference
./video_inference ../models/yolov8n.engine ../models/coco.names ../videos/test.mp4

# Batch inference
./batch_image_inference ../models/yolov8n.engine ../models/coco.names ../images/
Project Structure
text├── include/yolos/
│   ├── core/              # TrtSessionBase, MemoryArena, utils, etc.
│   ├── tasks/             # detection.hpp, classification.hpp
│   └── ...
├── src/
│   ├── core/
│   └── tasks/
├── models/
├── images/
└── README.md
Current Status
Build Status: ✅ Compiles successfully
Inference Status: Placeholder backend ready (real TensorRT loading in progress)

Future Improvements

Full TensorRT engine loading & optimization
CUDA Graph capture
ONNX → TensorRT conversion tool
Python bindings
More model support


Made with ❤️ for learning high-performance Edge AI / Computer Vision

