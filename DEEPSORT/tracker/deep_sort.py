# tracker/deep_sort.py
import numpy as np
from .kalman_filter import KalmanFilter
from .track import Track
from .matching import two_stage_matching
from .utils import xyxy_to_tlwh, tlwh_to_xyah
from .appearance import SimpleAppearance

class DeepSORT:
    """
    Minimal, readable Deep SORT tracker.

    Pipeline per frame:
      1) Encode detections with lightweight appearance features
      2) Kalman predict existing tracks
      3) Two-stage association:
            A) appearance for confirmed tracks (cosine distance + gating)
            B) IoU for unmatched + unconfirmed
      4) Update matched, spawn new from unmatched detections, age unmatched tracks
      5) Return confirmed tracks updated this frame as (track_id, tlwh)
    """

    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 3,
        max_iou_distance: float = 0.7,
        max_cosine_distance: float = 0.2,
        appearance_bins: int = 16,
    ):
        self.kf = KalmanFilter()
        self.tracks = []
        self._next_id = 1

        self.max_age = max_age
        self.n_init = n_init
        self.max_iou_distance = max_iou_distance
        self.max_cosine_distance = max_cosine_distance

        # Lightweight, non-deep appearance encoder (HSV histograms)
        self.encoder = SimpleAppearance(bins=appearance_bins)

    # -----------------------
    # Public API
    # -----------------------
    def update(self, frame, boxes_xyxy, scores):
        """
        Args:
            frame (np.ndarray): BGR image
            boxes_xyxy (List[List[float]] or np.ndarray): [x1,y1,x2,y2] detections
            scores (List[float] or np.ndarray): detection confidences (unused for matching here)

        Returns:
            List[Tuple[int, np.ndarray]]: [(track_id, tlwh), ...] for tracks updated this frame
        """
        # 0) Encode detections → (tlwhs, feat vectors)
        det_tlwhs, det_feats = self._gather_detections(frame, boxes_xyxy)

        # 1) Predict track states
        self._predict()

        # 2) Associate detections to tracks
        matches, u_trk, u_det = two_stage_matching(
            self.kf, self.tracks, det_tlwhs, det_feats,
            max_cosine=self.max_cosine_distance, max_iou_dist=self.max_iou_distance
        )

        # 3) Update matched tracks
        for ti, di in matches:
            self.tracks[ti].update(self.kf, tlwh_to_xyah(det_tlwhs[di]), det_feats[di])

        # 4) Spawn new tracks from unmatched detections
        for di in u_det:
            mean, cov = self.kf.initiate(tlwh_to_xyah(det_tlwhs[di]))
            trk = Track(mean, cov, self._next_id, n_init=self.n_init, max_age=self.max_age)
            # feed first measurement immediately to accumulate a feature and reset timers
            trk.update(self.kf, tlwh_to_xyah(det_tlwhs[di]), det_feats[di])
            self.tracks.append(trk)
            self._next_id += 1

        # 5) Age unmatched tracks
        for ti in u_trk:
            self.tracks[ti].mark_missed()

        # 6) Remove deleted tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted]

        # 7) Output confirmed tracks updated this frame
        outs = []
        for t in self.tracks:
            if t.is_confirmed and t.time_since_update == 0:
                outs.append((t.track_id, t.tlwh))
        return outs

    # -----------------------
    # Internal helpers
    # -----------------------
    def _predict(self):
        for t in self.tracks:
            t.predict(self.kf)

    def _gather_detections(self, frame, boxes_xyxy):
        """
        Convert xyxy boxes to tlwh and compute appearance features from crops.
        """
        if isinstance(boxes_xyxy, list):
            boxes = np.array(boxes_xyxy, dtype=np.float32) if len(boxes_xyxy) else np.zeros((0, 4), dtype=np.float32)
        else:
            boxes = boxes_xyxy.astype(np.float32) if boxes_xyxy is not None else np.zeros((0, 4), dtype=np.float32)

        h, w = frame.shape[:2]
        crops, tlwhs = [], []
        for b in boxes:
            x1, y1, x2, y2 = [int(v) for v in b]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            crop = frame[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else None
            crops.append(crop)
            tlwhs.append(xyxy_to_tlwh(b))

        feats = self.encoder.encode(crops)
        return tlwhs, feats
