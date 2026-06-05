# YOLOs-CPP-TensorRT

**High-performance multi-version YOLO inference engine in C++ using TensorRT**

A fast, modular C++ framework for running YOLOv7, YOLOv8, YOLOv10, YOLOv11, YOLOv26, YOLO-NAS and more on NVIDIA GPUs.

---

## Features

- **Multi-Version Support**: Automatic YOLO version detection + explicit mode support
- **End-to-End GPU Pipeline**: CUDA letterbox preprocessing → TensorRT inference → Postprocessing
- **Custom Memory Management**: Linear arena allocator for low fragmentation and efficient buffer handling
- **Multiple Inference Modes**: Image, Video, and Batch inference executables
- **Clean Architecture**: `TrtSessionBase`, `YOLODetector`, `YOLOClassifier` with factory pattern
- **Build System**: Modern CMake with CUDA support

## Supported Models

- YOLOv7, YOLOv8, YOLOv10, YOLOv11, YOLOv26, YOLO-NAS

## Tech Stack

- C++17
- TensorRT 8.6
- CUDA 11.8
- OpenCV 4
- CMake

## Build Instructions

```bash
mkdir -p build && cd build
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
│   ├── core/          # TrtSessionBase, MemoryArena, utils, preprocessing
│   ├── tasks/         # detection, classification
│   └── ...
├── src/
│   ├── core/
│   └── tasks/
├── models/
├── images/
├── videos/
└── README.md
Current Status

Build Status: ✅ Successfully compiles
Inference Status: Placeholder backend ready (full TensorRT engine loading in progress)

Future Improvements

Full TensorRT engine loading and optimization
CUDA Graph capture support
ONNX to TensorRT conversion utilities
Python bindings
Additional model support


Made with ❤️ for learning high-performance Edge AI / Computer Vision

