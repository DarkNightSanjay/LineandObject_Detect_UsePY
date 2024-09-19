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

    pip install opencv-python opencv-python-headless numpy

### Files
- yolov3.weights: YOLO model weights.
- **yolov3.cfg:** YOLO model configuration.
- **coco.names.txt:** COCO dataset class labels.
- **car lane.mp4:** Example video to process.
### How to Run
- Clone the repository and navigate to the project directory.
- Ensure you have the YOLO files (`yolov3.weights, yolov3.cfg, coco.names.txt`) in the specified paths.
- Replace the video_path in the code with the path to your video file (e.g., car lane.mp4).
### Run the Python script:

    python detect_road_in_video.py
The program will open a window displaying the road line and object detection results in real time. Press 'q' to exit the video processing.
### Code Explanation
- **YOLO Object Detection:** The detect_objects_yolo() function uses OpenCV's dnn module to run the YOLO model and detect objects in the video frames.
- **Road Line Detection:** The process_frame() function uses Canny Edge Detection and Hough Line Transformation to detect road lines.
- **Video Processing:** The detect_road_in_video() function processes each frame of the input video, applying both YOLO and road line detection to each frame.
### YOLO Configuration
To use YOLOv3 for object detection, download the following files:

- **yolov3.weights** 
- **yolov3.cfg** 
- **coco.names.txt**

### Troubleshooting
Ensure the paths to yolov3.weights, yolov3.cfg, coco.names.txt, and your video file are correct.
If road lines are not detected properly, try adjusting the Canny edge detection thresholds or the Hough Line Transform parameters.
