import numpy as np
import os

def load_mot_detections(det_file):
    """
    Reads a MOTChallenge det file (per sequence) with rows:
    frame, id(-1), x, y, w, h, conf, class(-1), vis(-1)
    Returns dict: frame_idx -> ndarray(N,5) with [x1,y1,x2,y2,score]
    """
    data = np.loadtxt(det_file, delimiter=',')
    frames = {}
    for row in data:
        f = int(row[0])
        x, y, w, h = row[2], row[3], row[4], row[5]
        score = row[6]
        bbox = np.array([x, y, x + w, y + h, score])
        frames.setdefault(f, []).append(bbox)
    for k in frames:
        frames[k] = np.vstack(frames[k])
    return frames

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def save_mot_results(path, tracks):
    """
    tracks: list of (frame, id, x, y, w, h, score)
    """
    ensure_dir(os.path.dirname(path))
    arr = np.array(tracks, dtype=float)
    np.savetxt(path, arr, fmt="%.6f", delimiter=',')
