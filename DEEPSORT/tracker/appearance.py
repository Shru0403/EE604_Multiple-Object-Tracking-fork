import numpy as np
import cv2

class SimpleAppearance:
    """Compute a simple 3×16-bin HSV color histogram as a feature vector."""
    def __init__(self, bins=16):
        self.bins = bins

    def encode(self, crops):
        if len(crops) == 0:
            return np.zeros((0, self.bins * 3), dtype=np.float32)
        feats = []
        for img in crops:
            if img is None or img.size == 0:
                feats.append(np.zeros(self.bins * 3, dtype=np.float32))
                continue
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h = cv2.calcHist([hsv],[0],None,[self.bins],[0,180]).flatten()
            s = cv2.calcHist([hsv],[1],None,[self.bins],[0,256]).flatten()
            v = cv2.calcHist([hsv],[2],None,[self.bins],[0,256]).flatten()
            feat = np.concatenate([h, s, v]).astype(np.float32)
            feat /= (np.linalg.norm(feat) + 1e-6)
            feats.append(feat)
        return np.stack(feats).astype(np.float32)
