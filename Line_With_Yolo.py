import cv2
import numpy as np

# Load YOLO model and COCO class labels
net = cv2.dnn.readNet("C:\\Users\\4a Freeboard\\Desktop\\Result\\yolov3.weights", "C:\\Users\\4a Freeboard\\Desktop\\Result\\yolov3.cfg")
with open("C:\\Users\\4a Freeboard\\Desktop\\Result\\coco.names.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

def process_frame(frame):
    # Resize the frame for faster processing
    frame_resized = cv2.resize(frame, (640, 480))
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    
    # Gaussian Blur to smooth out the image and reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adjust the Canny Edge Detection thresholds for better line detection
    edges = cv2.Canny(blurred, 100, 200)  # Increased thresholds
    
    # Define a mask region to focus on the road area
    mask = np.zeros_like(edges)
    height, width = edges.shape
    polygon = np.array([[
        (0, height),
        (width, height),
        (int(width * 0.6), int(height * 0.6)),
        (int(width * 0.4), int(height * 0.6))
    ]], np.int32)
    
    cv2.fillPoly(mask, polygon, 255)
    
    # Apply the mask to the edges
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Detect lines using Hough Line Transformation with tuned parameters
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,           # Lower threshold for line detection sensitivity
        minLineLength=40,        # Adjusted minimum line length
        maxLineGap=200           # Adjusted max line gap
    )
    
    # Create an image to draw the lines
    line_image = np.zeros_like(frame_resized)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)  # Draw detected road lines
    
    # Overlay the line image on the original frame
    combo_image = cv2.addWeighted(frame_resized, 0.8, line_image, 1, 1)
    
    return combo_image


def detect_objects_yolo(frame):
    height, width, _ = frame.shape
    
    # Create a blob from the frame and perform a forward pass through the YOLO network
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    # Store detected objects
    class_ids = []
    confidences = []
    boxes = []
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # Filter out weak detections
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                # Only proceed if the class_id is valid
                if class_id < len(classes):
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
    
    # Perform Non-Maximum Suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    # Draw bounding boxes and labels
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])  # This is now safe
            confidence = confidences[i]
            color = (0, 255, 0)  # Green for bounding boxes
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {int(confidence * 100)}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame

def detect_road_in_video(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Process each frame for road detection
        road_detected_frame = process_frame(frame)
        
        # Perform YOLO object detection
        yolo_detected_frame = detect_objects_yolo(road_detected_frame)
        
        # Display the frame
        cv2.imshow('Road and Object Detection', yolo_detected_frame)
        
        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Call the function with the path to your video file
video_path = "C:\\Users\\4a Freeboard\\Videos\\car lane.mp4" # Replace with your video path
detect_road_in_video(video_path)
