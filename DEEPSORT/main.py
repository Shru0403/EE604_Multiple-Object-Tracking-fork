# # import argparse
# # import cv2
# # from io.detections_json import load_detections
# # from io.video import open_video_writer
# # from detector import YOLODetector
# # from tracker import DeepSORT
# # from visualization.drawer import draw_tracks

# # def parse_args():
# #     ap = argparse.ArgumentParser()
# #     ap.add_argument("--video", required=True, help="Path to input video")
# #     ap.add_argument("--use-yolo", type=int, default=0, help="1 to use YOLO live")
# #     ap.add_argument("--detections", type=str, default=None, help="Path to dets.json")
# #     ap.add_argument("--save", type=str, default=None, help="Optional output video path")
# #     return ap.parse_args()

# # def main():
# #     args = parse_args()
# #     cap = cv2.VideoCapture(args.video)
# #     assert cap.isOpened(), f"Cannot open video: {args.video}"

# #     det_map = load_detections(args.detections) if args.detections else None
# #     detector = YOLODetector() if args.use_yolo else None

# #     tracker = DeepSORT()
# #     writer = open_video_writer(args.save, cap) if args.save else None

# #     frame_idx = 0
# #     while True:
# #         ok, frame = cap.read()
# #         if not ok:
# #             break

# #         boxes, scores = [], []
# #         if detector is not None:
# #             boxes, scores = detector.detect(frame)
# #         elif det_map is not None:
# #             arr = det_map.get(str(frame_idx), [])
# #             boxes = [a[:4] for a in arr]
# #             scores = [a[4] if len(a) > 4 else 1.0 for a in arr]

# #         tracks = tracker.update(frame, boxes, scores)
# #         vis = draw_tracks(frame.copy(), tracks, fps_text=True)

# #         cv2.imshow("DeepSORT_from_scratch", vis)
# #         if writer is not None:
# #             writer.write(vis)
# #         if cv2.waitKey(1) == 27:  # ESC
# #             break

# #         frame_idx += 1

# #     cap.release()
# #     if writer is not None:
# #         writer.release()
# #     cv2.destroyAllWindows()

# # if __name__ == "__main__":
# #     main()
# import os
# import cv2
# import numpy as np
# from tracker import DeepSORT
# from visualization.drawer import draw_tracks
# from io_utils.video import open_video_writer
# from io_utils.detections_json import load_detections

# # >>>> MODIFY THIS PATH <<<<
# SEQ_DIR = "datasets/MOT16/train/MOT16-02"
# IMG_DIR = os.path.join(SEQ_DIR, "img1")
# DET_FILE = os.path.join(SEQ_DIR, "det", "det.txt")
# SAVE_PATH = "mot16_MOT16-02_output.mp4"

# def load_mot16_detections(det_file):
#     """Loads MOT16 det.txt and returns a dict: frame_idx -> [ [x1,y1,x2,y2,conf], ... ]"""
#     dets = np.loadtxt(det_file, delimiter=",")
#     det_map = {}
#     for d in dets:
#         frame, _, x, y, w, h, conf = d[:7]
#         frame = int(frame)
#         box = [x, y, x + w, y + h, conf]
#         det_map.setdefault(frame, []).append(box)
#     return det_map

# def main():
#     # load detections
#     print(f"Loading detections from: {DET_FILE}")
#     det_map = load_mot16_detections(DET_FILE)

#     # list image files
#     img_files = sorted(
#         [f for f in os.listdir(IMG_DIR) if f.endswith(".jpg") or f.endswith(".png")]
#     )
#     assert len(img_files) > 0, "No frames found in img1/"

#     # read first frame to get video info
#     first_frame = cv2.imread(os.path.join(IMG_DIR, img_files[0]))
#     h, w = first_frame.shape[:2]
#     fps = 30
#     cap_stub = type("cap", (), {"get": lambda s, x: fps, "CAP_PROP_FPS": 0, "CAP_PROP_FRAME_WIDTH": w, "CAP_PROP_FRAME_HEIGHT": h})()
#     writer = open_video_writer(SAVE_PATH, cap_stub)

#     # initialize tracker
#     tracker = DeepSORT()
#     total_frames = len(img_files)
#     print(f"Running DeepSORT on {total_frames} frames...")

#     for idx, fname in enumerate(img_files, start=1):
#         frame_path = os.path.join(IMG_DIR, fname)
#         frame = cv2.imread(frame_path)
#         detections = det_map.get(idx, [])

#         boxes = [d[:4] for d in detections]
#         scores = [d[4] for d in detections]

#         tracks = tracker.update(frame, boxes, scores)
#         vis = draw_tracks(frame, tracks)

#         writer.write(vis)
#         cv2.imshow("DeepSORT MOT16-02", vis)
#         if cv2.waitKey(1) == 27:  # ESC
#             break

#         print(f"Processed frame {idx}/{total_frames}", end="\r")

#     writer.release()
#     cv2.destroyAllWindows()
#     print("\n✅ Tracking complete! Output saved to:", SAVE_PATH)

# if __name__ == "__main__":
#     main()
import os
import cv2
import numpy as np
from tracker import DeepSORT
from visualization import draw_tracks
from io_utils.video import open_video_writer

SEQ_DIR = "datasets/MOT16/train/MOT16-02"
IMG_DIR = os.path.join(SEQ_DIR, "img1")
DET_FILE = os.path.join(SEQ_DIR, "det", "det.txt")
SEQINFO  = os.path.join(SEQ_DIR, "seqinfo.ini")
SAVE_PATH = os.path.join("outputs", "mot16_MOT16-02_output.mp4")

def read_fps_from_seqinfo(seqinfo_path, default_fps=30.0):
    fps = default_fps
    if os.path.exists(seqinfo_path):
        with open(seqinfo_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.lower().startswith("framerate="):
                    try:
                        fps = float(line.split("=", 1)[1])
                    except:
                        pass
                    break
    return fps

def load_mot16_detections(det_file):
    dets = np.loadtxt(det_file, delimiter=",")
    det_map = {}
    for d in dets:
        frame, _, x, y, w, h, conf = d[:7]
        frame = int(frame)
        det_map.setdefault(frame, []).append([x, y, x + w, y + h, conf])
    return det_map

def main():
    print(f"Loading detections from: {DET_FILE}")
    det_map = load_mot16_detections(DET_FILE)

    img_files = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg", ".png"))])
    assert img_files, "No frames found in img1/"

    first = cv2.imread(os.path.join(IMG_DIR, img_files[0]))
    assert first is not None, "Failed to read first frame."
    H, W = first.shape[:2]
    fps = read_fps_from_seqinfo(SEQINFO, default_fps=30.0)

    # Create outputs/ and open writer with explicit params
    writer = open_video_writer(SAVE_PATH, fps=fps, width=W, height=H, codec="mp4v")
    print(f"Writing to: {SAVE_PATH}  (fps={fps}, size=({W},{H}))")

    tracker = DeepSORT()

    for idx, fname in enumerate(img_files, start=1):
        frame = cv2.imread(os.path.join(IMG_DIR, fname))
        if frame is None:
            continue  # skip unreadable frames
        # ensure exact size
        if frame.shape[1] != W or frame.shape[0] != H:
            frame = cv2.resize(frame, (W, H))

        dets = det_map.get(idx, [])
        boxes = [d[:4] for d in dets]
        scores = [d[4] for d in dets]

        tracks = tracker.update(frame, boxes, scores)
        vis = draw_tracks(frame, tracks)

        writer.write(vis)
        cv2.imshow("DeepSORT MOT16-02", vis)
        if cv2.waitKey(1) == 27:  # ESC
            break

        if idx % 50 == 0:
            print(f"Processed {idx}/{len(img_files)} frames...")

    writer.release()
    cv2.destroyAllWindows()
    print("Done. Saved:", SAVE_PATH)

if __name__ == "__main__":
    main()
