import numpy as np

def iou(tlwh_a, tlwh_b):
    t1, l1, w1, h1 = tlwh_a
    t2, l2, w2, h2 = tlwh_b
    r1, b1 = t1 + w1, l1 + h1
    r2, b2 = t2 + w2, l2 + h2
    inter_w = max(0.0, min(r1, r2) - max(t1, t2))
    inter_h = max(0.0, min(b1, b2) - max(l1, l2))
    inter = inter_w * inter_h
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0

def cosine_distance(A, B):
    if len(A) == 0 or len(B) == 0:
        return np.zeros((len(A), len(B)), dtype=np.float32)
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-6)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-6)
    return 1.0 - A @ B.T

CHI2INV_95_DF4 = 9.4877
