import numpy as np

class YOLODetector:
    def __init__(self, model_name="yolov8n.pt", person_only=True):
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_name)
            self.ok = True
        except Exception as e:
            print("[WARN] YOLO not available:", e)
            self.model = None
            self.ok = False
        self.person_only = person_only

    def detect(self, frame):
        if not self.ok:
            return [], []
        res = self.model(frame, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy().astype(np.float32)
        confs = res.boxes.conf.cpu().numpy().astype(np.float32)
        clss = res.boxes.cls.cpu().numpy().astype(np.int32)
        if self.person_only and clss.size:
            keep = clss == 0
            boxes, confs = boxes[keep], confs[keep]
        return boxes.tolist(), confs.tolist()
