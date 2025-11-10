import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
import os
from torch_bytetrack import ByteTrack
from yolo_detector import YOLODetector  # Ensure this file is in the same directory!

# --- CONFIGURATION ---
# Set the path to the folder containing your MOT16 sequence images (e.g., 'MOT16-02/img1')
IMAGE_FOLDER = 'MOT16/train/MOT16-02/img1'

# The frame range to process (adjust these to match your sequence)
START_FRAME_INDEX = 1 
END_FRAME_INDEX = 500 

# Tracker and Detector configuration
CONFIDENCE_THRESHOLD = 0.1
DISPLAY_WINDOW_NAME = "ByteTrack MOT16 Viewer"
WAIT_TIME_MS = 1 # Milliseconds to wait between frames (set higher for slower playback)

# --- 1. INITIALIZE TRACKER AND DETECTOR ---

# Initialize the ByteTrack object
tracker = ByteTrack()

# Initialize the YOLO Detector 
# Assuming default weights, change 'yolov8n.pt' if you're using a specific model
yolo_processor = YOLODetector(model_weights='yolov8m.pt', confidence_threshold=CONFIDENCE_THRESHOLD) 
detector_function = yolo_processor.mydetector

print(f"Starting real-time tracking from {IMAGE_FOLDER}...")
cv2.namedWindow(DISPLAY_WINDOW_NAME, cv2.WINDOW_NORMAL)

# --- 2. IMAGE PROCESSING LOOP ---

current_frame_count = 0

# Loop through the desired range of frame numbers
for i in range(START_FRAME_INDEX, END_FRAME_INDEX + 1):
    # Construct the filename using the MOT16 zero-padded format (e.g., '000001.jpg')
    frame_filename = f"{i:06d}.jpg"
    frame_path = os.path.join(IMAGE_FOLDER, frame_filename)
    
    if not os.path.exists(frame_path):
        print(f"Warning: Frame {frame_path} not found. Stopping sequence.")
        break
        
    # Read the frame
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"Error reading image file: {frame_path}. Skipping.")
        continue
    
    current_frame_count += 1
    
    # 3. RUN DETECTION on the current frame
    detections = detector_function(frame) 
    # if detections.shape[1] == 6:  # [x1, y1, x2, y2, conf, cls]
    #     detections = detections[:, :5]
    
    # 4. RUN TRACKING: Update the tracker's state
    tracks = tracker.update_tracks(detections, format="xyxy")

    # 5. PROCESS AND VISUALIZE TRACKING RESULTS
    for track in tracks:
        if track.is_confirmed():
            box = track.to_ltrb()
            track_id = track.track_id 
            
            x1, y1, x2, y2 = map(int, box)
            
            # Simple color mapping based on ID
            color_id = (track_id * 30 % 255, track_id * 50 % 255, track_id * 70 % 255)
            
            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_id, 2)
            
            # Draw the Track ID
            label = f"ID: {track_id}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_id, 2)

    # 6. DISPLAY THE FRAME
    cv2.imshow(DISPLAY_WINDOW_NAME, frame)
    
    # Check for exit key ('q') or wait time
    key = cv2.waitKey(WAIT_TIME_MS) & 0xFF
    if key == ord('q'):
        print("Stopping video playback...")
        break

# --- 7. CLEAN UP ---
cv2.destroyAllWindows()
print(f"\nTracking finished. {current_frame_count} frames processed.")