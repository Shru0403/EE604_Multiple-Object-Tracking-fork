import numpy as np

def xyxy_to_tlwh(box):
    x1, y1, x2, y2 = box
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return np.array([x1, y1, w, h], dtype=np.float32)

def tlwh_to_xyah(tlwh):
    t, l, w, h = tlwh
    cx = t + w / 2.0
    cy = l + h / 2.0
    a = w / max(1e-6, h)
    return np.array([cx, cy, a, h], dtype=np.float32)

def xyah_to_tlwh(xyah):
    cx, cy, a, h = xyah
    w = a * h
    t = cx - w / 2.0
    l = cy - h / 2.0
    return np.array([t, l, w, h], dtype=np.float32)
