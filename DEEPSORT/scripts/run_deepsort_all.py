# scripts/run_deepsort_all.py
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import os
import glob
import numpy as np
import cv2
from tracker import DeepSORT

ROOT = os.path.dirname(os.path.dirname(__file__))  # project root
DATA_ROOT = os.path.join(ROOT, "datasets", "MOT16", "train")
OUT_ROOT = os.path.join(ROOT, "outputs", "MOT16-train")
os.makedirs(OUT_ROOT, exist_ok=True)

def list_sequences(data_root):
    seqs = []
    for p in sorted(glob.glob(os.path.join(data_root, "*"))):
        if os.path.isdir(os.path.join(p, "img1")):
            seqs.append(os.path.basename(p))
    return seqs

def load_mot16_detections(det_file):
    """
    det.txt columns: frame, id(-1), x, y, w, h, conf, -, -, -
    Return: dict[int -> list[[x1,y1,x2,y2,conf], ...]]
    """
    dets = np.loadtxt(det_file, delimiter=",")
    det_map = {}
    for d in dets:
        f, _, x, y, w, h, conf = d[:7]
        f = int(f)
        det_map.setdefault(f, []).append([x, y, x + w, y + h, float(conf)])
    return det_map

def run_sequence(seq_dir, out_path):
    img_dir = os.path.join(seq_dir, "img1")
    det_file = os.path.join(seq_dir, "det", "det.txt")

    det_map = load_mot16_detections(det_file)
    img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png"))])
    assert img_files, f"No frames in {img_dir}"

    tracker = DeepSORT()
    results = []  # rows: frame,id,x,y,w,h,score,-1,-1,-1

    for idx, fname in enumerate(img_files, start=1):
        frame = cv2.imread(os.path.join(img_dir, fname))
        if frame is None:
            continue
        H, W = frame.shape[:2]

        dets = det_map.get(idx, [])
        boxes = [d[:4] for d in dets]
        scores = [d[4] for d in dets]

        tracks = tracker.update(frame, boxes, scores)
        for tid, tlwh in tracks:
            x, y, w, h = map(float, tlwh)
            # clamp
            x = max(0.0, min(x, W - 1))
            y = max(0.0, min(y, H - 1))
            w = max(0.0, min(w, W - x))
            h = max(0.0, min(h, H - y))
            results.append([idx, tid, x, y, w, h, 1.0, -1, -1, -1])

        if idx % 50 == 0:
            print(f"{os.path.basename(seq_dir)}: {idx}/{len(img_files)} frames")

    if results:
        arr = np.asarray(results, dtype=np.float32)
        np.savetxt(
            out_path,
            arr,
            fmt="%.0f,%.0f,%.2f,%.2f,%.2f,%.2f,%.4f,%.0f,%.0f,%.0f"
        )
    else:
        # write empty file to signal “no outputs”
        open(out_path, "w").close()

def main():
    sequences = list_sequences(DATA_ROOT)
    print("Found sequences:", sequences)
    if not sequences:
        print("No sequences found in", DATA_ROOT)
        return

    for seq in sequences:
        seq_dir = os.path.join(DATA_ROOT, seq)
        out_txt = os.path.join(OUT_ROOT, f"{seq}.txt")
        print(f"\n=== Running DeepSORT on {seq} ===")
        run_sequence(seq_dir, out_txt)
        print("Saved:", out_txt)

if __name__ == "__main__":
    main()
