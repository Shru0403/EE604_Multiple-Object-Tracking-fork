import json

def load_detections(path):
    if not path:
        return None
    with open(path, "r") as f:
        return json.load(f)
