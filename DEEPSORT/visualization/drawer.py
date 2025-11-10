import cv2

def draw_tracks(frame, tracks, fps_text=False):
    for tid, tlwh in tracks:
        x, y, w, h = map(int, tlwh)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)
        cv2.putText(frame, f"ID {tid}", (x, max(0, y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    return frame
