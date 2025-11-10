
import numpy as np
from ultralytics import YOLO
import cv2

class YOLODetector:
    """
    Object detection class using YOLOv8 to generate detections for ByteTrack.
    The output format is specifically tailored for the torch-bytetrack update_tracks method.
    """
    def __init__(self, model_weights: str = 'yolov8m.pt', confidence_threshold: float = 0.5):
        self.model = YOLO(model_weights)
        self.confidence_threshold = confidence_threshold
        self.target_class_id = 0 # Default: 'person' in COCO

        if "football" in model_weights:
            print("Assuming custom model is loaded. Class indices may differ.")
            
    def mydetector(self, frame: np.ndarray) -> np.ndarray:
        """
        Performs object detection on a single frame and formats the results.
        
        Returns:
            A NumPy array of detections in the REQUIRED ByteTrack format:
            [[confidence, class_idx, x1, y1, x2, y2], ...]
        """
        # 1. Run inference on the frame
        results = self.model.predict(frame, conf=self.confidence_threshold, verbose=False)
        
        # 2. Extract and format detections
        if not results or not results[0].boxes:
            return np.array([], dtype=np.float32).reshape(0, 6)

        boxes = results[0].boxes.cpu().numpy()
        
        xyxy = boxes.xyxy
        conf = boxes.conf
        cls  = boxes.cls
        
        # 3. Filter for 'person' (or 'player') class
        person_mask = (cls == self.target_class_id)
        
        filtered_xyxy = xyxy[person_mask]
        filtered_conf = conf[person_mask]
        filtered_cls  = cls[person_mask]

        # 4. Construct the final output array in the CRITICAL ORDER
        detections_list = []
        for bbox, score, class_id in zip(filtered_xyxy, filtered_conf, filtered_cls):
            
            # Ensure coordinates are integers for drawing stability
            x1, y1, x2, y2 = bbox.astype(np.int32)
            
            # --- CRITICAL FIX: Match ByteTrack's Expected Order ---
            # [confidence, class_idx, x1, y1, x2, y2]
            detections_list.append([score, class_id, x1, y1, x2, y2])
            
        return np.array(detections_list, dtype=np.float32)

# --- Example of how to use this in your main file ---
if __name__ == '__main__':
    # This block is for testing the detector standalone
    detector = YOLODetector()
    
    # Create a dummy frame (e.g., a black 640x480 image)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Since we can't show a real detection without an image/video, 
    # we'll just test the function call.
    print("Testing YOLODetector with dummy frame (no actual detection will occur).")
    detections_output = detector.mydetector(dummy_frame)
    
    print(f"Output shape (Expected: N x 6): {detections_output.shape}")
    print(f"Output type: {detections_output.dtype}")
    print("Integration successful. Now use YOLODetector in your main tracking script.")