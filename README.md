# Road Line and Object Detection Using YOLO and OpenCV

This project demonstrates road line detection using Canny Edge Detection and Hough Line Transform, as well as object detection using the YOLO (You Only Look Once) model. The system processes video frames to detect road lanes and objects, drawing bounding boxes around detected objects.

## Features

- **Road Line Detection**: Detects lanes on the road using Canny edge detection and the Hough Line Transform.
- **Object Detection**: Uses YOLOv3 to detect objects from a predefined set of COCO classes in video frames.
- **Real-time Video Processing**: Processes video files frame by frame and displays the detected lines and objects in real time.

## Prerequisites

Before running this project, ensure you have the following software installed:

- Python 3.x
- OpenCV (`opencv-python` and `opencv-python-headless`)
- NumPy (`numpy`)
- YOLOv3 weights and configuration files:
  - `yolov3.weights`: Pre-trained YOLO model weights
  - `yolov3.cfg`: YOLO model configuration file
  - `coco.names.txt`: List of object classes used in COCO dataset

To install the required libraries, run:

```bash
pip install opencv-python opencv-python-headless numpy
