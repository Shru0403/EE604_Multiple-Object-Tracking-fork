"""
io package — handles reading/writing videos and loading detections.
"""

from .video import open_video_writer
from .detections_json import load_detections

__all__ = ["open_video_writer", "load_detections"]
