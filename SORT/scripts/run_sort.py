import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import numpy as np
from tqdm import tqdm
from sort.tracker import Sort
from sort.utils import load_mot_detections, save_mot_results
import os

# ==== 🔧 USER SETTINGS ====
DET_PATH = r"data/MOT16/train/MOT16-02/det/det.txt"
SEQ_PATH = r"data/MOT16/train/MOT16-02/img1"
OUT_TXT = r"results/MOT16-02_sort.txt"
OUT_VIDEO = r"results/MOT16-02_sort.mp4"
MAX_AGE = 30
MIN_HITS = 3
IOU_THR = 0.3
FPS = 30  # You can adjust based on seqinfo.ini
# ===========================


def main():
    frames = load_mot_detections(DET_PATH)
    tracker = Sort(max_age=MAX_AGE, min_hits=MIN_HITS, iou_threshold=IOU_THR)

    results = []
    video_writer = None

    # Initialize video writer if needed
    if OUT_VIDEO and SEQ_PATH:
        first_frame_path = os.path.join(SEQ_PATH, f"{1:06d}.jpg")
        if not os.path.exists(first_frame_path):
            raise FileNotFoundError(f"Image not found: {first_frame_path}")
        frame = cv2.imread(first_frame_path)
        h, w = frame.shape[:2]
        os.makedirs(os.path.dirname(OUT_VIDEO), exist_ok=True)
        video_writer = cv2.VideoWriter(
            OUT_VIDEO,
            cv2.VideoWriter_fourcc(*'mp4v'),
            FPS,
            (w, h)
        )

    for f in tqdm(sorted(frames.keys())):
        dets = frames[f]  # (N,5)
        tracked = tracker.update(dets)  # (M,6): x1,y1,x2,y2,id,score

        # Save MOT results
        for t in tracked:
            x1, y1, x2, y2, tid, score = t
            w, h = x2 - x1, y2 - y1
            results.append([f, int(tid), x1, y1, w, h, score, -1, -1])

        # Visualization
        if SEQ_PATH:
            img_path = os.path.join(SEQ_PATH, f"{f:06d}.jpg")
            if os.path.exists(img_path):
                frame = cv2.imread(img_path)
                for t in tracked:
                    x1, y1, x2, y2, tid, score = t
                    color = (int(tid * 37 % 255), int(tid * 17 % 255), int(tid * 29 % 255))
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(frame, f'ID {int(tid)}', (int(x1), int(y1) - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.imshow("SORT Tracking", frame)
                if video_writer:
                    video_writer.write(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    save_mot_results(OUT_TXT, results)
    print(f"✅ Results saved to {OUT_TXT}")
    if OUT_VIDEO:
        print(f"🎥 Video saved to {OUT_VIDEO}")


if __name__ == "__main__":
    main()
