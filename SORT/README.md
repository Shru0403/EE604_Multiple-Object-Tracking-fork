# SORT: Simple Online and Realtime Tracking (Implemented from Scratch)

This project implements the **SORT (Simple Online and Realtime Tracking)** algorithm completely from scratch using Python.  
The tracker is based on the original paper by Bewley et al. (2016) and is applied to the **MOT16** dataset for multi-object tracking.

---

##  Overview

SORT is a lightweight tracking-by-detection algorithm that links detections across video frames using:
- **Kalman Filter** for motion prediction  
- **IoU (Intersection-over-Union)** for association  
- **Hungarian Algorithm** for optimal assignment  

It provides real-time tracking performance while maintaining reasonable accuracy.

---

##  Components

| File | Description |
|------|--------------|
| `sort/association.py` | Computes IoU between detections and trackers, and performs Hungarian matching. |
| `sort/kalman_filter.py` | Implements a 7D Kalman filter for bounding box prediction and update. |
| `sort/tracker.py` | Combines Kalman filter + IoU matching into the SORT pipeline. |
| `run_sort.py` | Runs SORT on a single MOT16 sequence with visualization and saves results/video. |
| `run_sort_all.py` | Runs SORT on all MOT16 training sequences automatically. |
| `evaluate_mot16_train.py` | Evaluates tracking results using standard MOT metrics (MOTA, IDF1, etc.). |
| `results/` | Folder where output text files and videos are stored. |

---

##  Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/sort-from-scratch.git
cd sort-from-scratch
---

## 📦 Dataset Setup

Download the **MOT16 dataset** from the official MOT Challenge website:  
🔗 [https://motchallenge.net/data/MOT16/]

After downloading, extract it into the `data/` folder in your project directory



# (Optional) Create a virtual environment
python -m venv .venv
.venv\Scripts\activate     # For Windows
source .venv/bin/activate  # For macOS/Linux

# Install required packages
pip install numpy scipy opencv-python pandas motmetrics tqdm
(or)
pip install -r requirements.txt

## ▶️ How to Run

### 1️⃣ Run SORT on a single MOT16 sequence
Edit the paths in `run_sort.py`:
```python
DET_PATH = "data/MOT16/train/MOT16-02/det/det.txt"
SEQ_PATH = "data/MOT16/train/MOT16-02/img1"


```bash
python run_sort.py


# To run the tracker on all MOT16 training sequences, use:
python run_sort_all.py

# To evaluate your tracking results using official MOT metrics (MOTA, IDF1, etc.), run:
python evaluate_mot16_train.py

This script compares  predictions with the ground truth annotations and saves the performance summary in:
results/mot16_train_metrics.csv