import numpy as np

# Utilities for bbox <-> measurement
def bbox_to_z(bbox):
    # bbox: [x1,y1,x2,y2] -> z: [cx, cy, s, r]
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.0
    cy = bbox[1] + h / 2.0
    s = w * h            # scale (area)
    r = w / (h + 1e-6)   # aspect ratio
    return np.array([cx, cy, s, r], dtype=float)

def x_to_bbox(x, score=None):
    # state x -> [x1,y1,x2,y2,(score)]
    cx, cy, s, r = x[0], x[1], x[2], x[3]
    w = np.sqrt(s * r)
    h = s / (w + 1e-6)
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    if score is None:
        return np.array([x1, y1, x2, y2])
    return np.array([x1, y1, x2, y2, score])

class KalmanBoxTracker:
    """Simple Kalman filter for bounding boxes with constant velocity in (cx,cy,s)."""
    count = 0

    def __init__(self, bbox, score=None):
        # State: [cx, cy, s, r, vx, vy, vs]
        self.x = np.zeros((7, 1), dtype=float)
        z = bbox_to_z(bbox).reshape(4, 1)
        self.x[:4] = z
        # State covariance
        self.P = np.eye(7)
        # Motion model (dt=1)
        self.F = np.eye(7)
        self.F[0, 4] = 1.0
        self.F[1, 5] = 1.0
        self.F[2, 6] = 1.0
        # Measurement model
        self.H = np.zeros((4, 7))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = self.H[3, 3] = 1.0
        # Process / measurement noise (tuned, small but nonzero)
        self.Q = np.eye(7) * 1e-2
        self.R = np.eye(4) * 1e-1

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.score = float(score) if score is not None else None

    def predict(self):
        # If s would go negative, damp velocity
        if self.x[2] + self.x[6] <= 0:
            self.x[6] = 0
        # x = F x
        self.x = self.F @ self.x
        # P = F P F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q

        self.age += 1
        self.time_since_update += 1
        self.history.append(x_to_bbox(self.x.flatten()))
        return self.history[-1]

    def update(self, bbox, score=None):
        z = bbox_to_z(bbox).reshape(4, 1)
        y = z - (self.H @ self.x)                          # innovation
        S = self.H @ self.P @ self.H.T + self.R            # innovation cov
        K = self.P @ self.H.T @ np.linalg.inv(S)           # Kalman gain
        self.x = self.x + (K @ y)
        I = np.eye(7)
        self.P = (I - K @ self.H) @ self.P

        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        if score is not None:
            self.score = float(score)

    def get_state(self):
        return x_to_bbox(self.x.flatten(), self.score)
