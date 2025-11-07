# Introduction
In this project we implemented Multiple Object Tracking(MOT) to develop a robust system to track football players consistenly across a match video using SORT, DeepSORT and ByteTrack algorithm. We implemented these three techniques from scratch and evaluated them on the video sequence with multiple moving objects. SORT provides a baseline with simple Kalman filtering, whereas DeepSORT leverages appearance features to reduce identity switches. ByteTrack further improves performance by associating low-confidence detections, handling occlusions more effectively while maintaining high frame rates. We demonstrate a use case in football analytics, combining YOLO-based detection with ByteTrack to track players across match footage.

# Dataset
https://motchallenge.net/data/MOT16/

# Requirements
Python 3.x
Ultralytics
Supervision
OpenCV
NumPy
Matplotlib
PyTorch
Pandas



