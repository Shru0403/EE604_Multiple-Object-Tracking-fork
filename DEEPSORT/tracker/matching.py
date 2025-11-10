import numpy as np
from scipy.optimize import linear_sum_assignment
from .metrics import iou, cosine_distance, CHI2INV_95_DF4
from .utils import tlwh_to_xyah

def iou_cost(tracks, detections_tlwh):
    """
    Compute IoU-based cost matrix between existing tracks and new detections.
    Cost = 1 - IoU  (lower is better)
    """
    C = np.zeros((len(tracks), len(detections_tlwh)), dtype=np.float32)
    for i, t in enumerate(tracks):
        tbox = t.tlwh
        for j, d in enumerate(detections_tlwh):
            C[i, j] = 1.0 - iou(tbox, d)
    return C


def gate_cost_matrix(kf, tracks, detections_xyah, C):
    """
    Apply gating: if a detection is too far from the predicted track location
    (based on Kalman filter uncertainty), assign a very high cost.
    """
    for i, t in enumerate(tracks):
        gating_dist = kf.gating_distance(t.mean, t.covariance, detections_xyah, only_position=False)
        C[i, gating_dist > CHI2INV_95_DF4] = 1e5  # reject impossible matches
    return C


def assign_with_cost(C, high_cost=1e4):
    """
    Solve the assignment problem (Hungarian algorithm) with cost threshold.
    Returns:
        matches      : list of (track_idx, det_idx)
        u_trk, u_det : lists of unmatched indices
    """
    if C.size == 0:
        return [], list(range(C.shape[0])), list(range(C.shape[1]))
    row_ind, col_ind = linear_sum_assignment(C)
    matches, assigned_r, assigned_c = [], set(), set()
    for r, c in zip(row_ind, col_ind):
        if C[r, c] > high_cost:
            continue
        matches.append((r, c))
        assigned_r.add(r)
        assigned_c.add(c)
    u_trk = [r for r in range(C.shape[0]) if r not in assigned_r]
    u_det = [c for c in range(C.shape[1]) if c not in assigned_c]
    return matches, u_trk, u_det


def two_stage_matching(kf, tracks, detections, det_feats, max_cosine=0.2, max_iou_dist=0.7):
    """
    Perform DeepSORT's two-stage association:
        1) Match confirmed tracks by appearance (cosine distance + gating)
        2) Match remaining (unconfirmed + unmatched confirmed) by IoU
    """

    det_tlwh = np.array([d for d in detections], dtype=np.float32)
    det_xyah = np.array([tlwh_to_xyah(d) for d in det_tlwh], dtype=np.float32)

    # Divide tracks by confirmation state
    confirmed_idx = [i for i, t in enumerate(tracks) if t.is_confirmed]
    unconfirmed_idx = [i for i, t in enumerate(tracks) if not t.is_confirmed]

    matches_a, unmatched_conf, unmatched_det = [], confirmed_idx, list(range(len(det_tlwh)))

    # ---------- Stage 1: Appearance (for confirmed tracks only) ----------
    if confirmed_idx and len(det_tlwh):
        trk_feats = []
        for i in confirmed_idx:
            if tracks[i].features:
                trk_feats.append(tracks[i].features[-1])
            else:
                trk_feats.append(np.zeros_like(det_feats[0]))
        trk_feats = np.stack(trk_feats, axis=0)

        # Cosine distance → cost matrix
        C = cosine_distance(trk_feats, det_feats)
        C = gate_cost_matrix(kf, [tracks[i] for i in confirmed_idx], det_xyah, C)
        C[C > max_cosine] = 1e5  # reject visually dissimilar matches

        ma, uc, ud = assign_with_cost(C)
        matches_a = [(confirmed_idx[r], unmatched_det[c]) for r, c in ma]
        unmatched_conf = [confirmed_idx[r] for r in uc]
        unmatched_det = [unmatched_det[c] for c in ud]

    # ---------- Stage 2: IoU (for remaining + unconfirmed) ----------
    remaining_trk = unconfirmed_idx + unmatched_conf
    matches_b = []
    if remaining_trk and unmatched_det:
        C = iou_cost([tracks[i] for i in remaining_trk], [det_tlwh[j] for j in unmatched_det])
        C[C > (1 - max_iou_dist)] = 1e5
        mb, ut, ud = assign_with_cost(C)
        matches_b = [(remaining_trk[r], unmatched_det[c]) for r, c in mb]
        u_trk = [remaining_trk[r] for r in ut]
        u_det = [unmatched_det[c] for c in ud]
    else:
        u_trk = remaining_trk
        u_det = unmatched_det

    matches = matches_a + matches_b
    return matches, u_trk, u_det
