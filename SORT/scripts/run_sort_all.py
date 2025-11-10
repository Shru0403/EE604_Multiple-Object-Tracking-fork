import os
from tqdm import tqdm
from sort.tracker import Sort
from sort.utils import load_mot_detections, save_mot_results

# ==== USER CONFIG ====
DATA_ROOT = r"data/MOT16/train"
OUT_ROOT = r"results"
SEQ_NAMES = ["MOT16-02", "MOT16-04", "MOT16-05", "MOT16-09", "MOT16-10", "MOT16-11", "MOT16-13"]

MAX_AGE = 20
MIN_HITS = 2
IOU_THR = 0.4
CONF_THR = 0.4  # Filter weak detections
# ======================

os.makedirs(OUT_ROOT, exist_ok=True)

for seq in SEQ_NAMES:
    print(f"\n=== Processing {seq} ===")

    det_path = os.path.join(DATA_ROOT, seq, "det", "det.txt")
    if not os.path.exists(det_path):
        print(f"[SKIP] Missing det file: {det_path}")
        continue

    frames = load_mot_detections(det_path)
    tracker = Sort(max_age=MAX_AGE, min_hits=MIN_HITS, iou_threshold=IOU_THR)

    results = []
    for f in tqdm(sorted(frames.keys()), desc=f"Tracking {seq}"):
        dets = frames[f]
        if dets.size:
            dets = dets[dets[:, 4] >= CONF_THR]  # filter low-confidence
        tracked = tracker.update(dets)
        for t in tracked:
            x1, y1, x2, y2, tid, score = t
            w, h = x2 - x1, y2 - y1
            results.append([f, int(tid), x1, y1, w, h, score, -1, -1])

    out_path = os.path.join(OUT_ROOT, f"{seq}_sort.txt")
    save_mot_results(out_path, results)
    print(f"[DONE] Saved results for {seq} → {out_path}")
