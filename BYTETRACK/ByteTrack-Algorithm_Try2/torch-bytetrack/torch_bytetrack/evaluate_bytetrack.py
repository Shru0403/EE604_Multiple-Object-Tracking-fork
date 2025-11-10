import motmetrics as mm
import os

gt_path = "/Users/radhikadarpe/Documents/IITK/EE604 Project/MOT16/train"
res_path = "/Users/radhikadarpe/Documents/IITK/EE604 Project/ByteTrack/bytetrack_results"

accs = []
names = []

for seq in sorted(os.listdir(gt_path)):
    seq_path = os.path.join(gt_path, seq)
    gt_file = os.path.join(seq_path, "gt/gt.txt")
    res_file = os.path.join(res_path, f"{seq}.txt")

    if not os.path.exists(res_file):
        print("Skipping", seq, "(no result found)")
        continue

    gt = mm.io.loadtxt(gt_file, fmt="mot15-2D", min_confidence=1)
    res = mm.io.loadtxt(res_file, fmt="mot15-2D")

    acc = mm.utils.compare_to_groundtruth(gt, res, "iou", distth=0.5)
    accs.append(acc)
    names.append(seq)

mh = mm.metrics.create()
summary = mh.compute_many(
    accs, names=names, metrics=mm.metrics.motchallenge_metrics, generate_overall=True
)

print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))