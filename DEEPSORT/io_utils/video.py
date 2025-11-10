# import cv2

# def open_video_writer(path, cap):
#     if path is None:
#         return None
#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     fps = cap.get(cv2.CAP_PROP_FPS) or 30
#     w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     return cv2.VideoWriter(path, fourcc, fps, (w, h))
import os
import cv2

def open_video_writer(path: str, fps: float, width: int, height: int, codec: str = "mp4v") -> cv2.VideoWriter:
    """
    Create a VideoWriter with explicit fps/size (do NOT rely on a capture object).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(path, fourcc, float(fps), (int(width), int(height)))
    if not writer.isOpened():
        raise RuntimeError(
            f"Failed to open VideoWriter for {path} (codec={codec}, fps={fps}, size=({width},{height}))"
        )
    return writer
