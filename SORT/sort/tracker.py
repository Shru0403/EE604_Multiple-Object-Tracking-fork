import numpy as np
from .kalman_filter import KalmanBoxTracker
from .association import associate_detections_to_trackers

class Sort:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets):
        """
        dets: ndarray (N, 5) -> [x1,y1,x2,y2,score] or (N,4) without score
        Returns: ndarray of tracked objects: [x1,y1,x2,y2,id,score]
        """
        self.frame_count += 1
        dets = np.asarray(dets, dtype=float)
        scores = dets[:, 4] if dets.shape[1] == 5 else np.ones(len(dets))
        det_boxes = dets[:, :4] if len(dets) else dets.reshape(0, 4)

        # 1) Predict
        trk_boxes = []
        for t in self.trackers:
            trk_boxes.append(t.predict())
        trk_boxes = np.asarray(trk_boxes).reshape(-1, 4)

        # 2) Associate D->T by IoU
        matches, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            det_boxes, trk_boxes, self.iou_threshold
        )

        # 3) Update matched trackers
        for det_idx, trk_idx in matches:
            self.trackers[trk_idx].update(det_boxes[det_idx], scores[det_idx])

        # 4) Create new trackers for unmatched detections
        for idx in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(det_boxes[idx], scores[idx]))

        # 5) Collect results and prune dead tracks
        ret = []
        alive = []
        for t in self.trackers:
            if t.time_since_update < 1:
                d = t.get_state()
                out = [*d[:4], t.id, (t.score if t.score is not None else 1.0)]
                # Emit once track is mature (or always if you prefer)
                if (t.hit_streak >= self.min_hits) or (self.frame_count <= self.min_hits):
                    ret.append(out)
            # keep tracker if not too old
            if t.time_since_update <= self.max_age:
                alive.append(t)
        self.trackers = alive

        return np.asarray(ret)
