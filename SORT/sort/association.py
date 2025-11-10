import numpy as np
from scipy.optimize import linear_sum_assignment

def iou(b1, b2):
    # b: [x1,y1,x2,y2]
    xx1 = np.maximum(b1[0], b2[0])
    yy1 = np.maximum(b1[1], b2[1])
    xx2 = np.minimum(b1[2], b2[2])
    yy2 = np.minimum(b1[3], b2[3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    union = a1 + a2 - inter + 1e-6
    return inter / union

def iou_cost_matrix(dets, trks):
    if len(dets) == 0 or len(trks) == 0:
        return np.empty((len(dets), len(trks)))
    C = np.zeros((len(dets), len(trks)), dtype=float)
    for i, d in enumerate(dets):
        for j, t in enumerate(trks):
            C[i, j] = 1.0 - iou(d, t)  # cost = 1 - IoU
    return C

def associate_detections_to_trackers(dets, trks, iou_threshold=0.3):
    """
    Args:
        dets: (N,4) ndarray of detections
        trks: (M,4) ndarray of predicted boxes from trackers
    Returns:
        matches: [(det_idx, trk_idx), ...]
        unmatched_dets: [indices]
        unmatched_trks: [indices]
    """
    if len(trks) == 0:
        return [], list(range(len(dets))), []

    C = iou_cost_matrix(dets, trks)
    row_ind, col_ind = linear_sum_assignment(C)

    matches, unmatched_dets, unmatched_trks = [], [], []
    for r, c in zip(row_ind, col_ind):
        if 1.0 - C[r, c] < iou_threshold:
            unmatched_dets.append(r)
            unmatched_trks.append(c)
        else:
            matches.append((r, c))

    unmatched_dets += [d for d in range(len(dets)) if d not in row_ind]
    unmatched_trks += [t for t in range(len(trks)) if t not in col_ind]
    return matches, unmatched_dets, unmatched_trks
