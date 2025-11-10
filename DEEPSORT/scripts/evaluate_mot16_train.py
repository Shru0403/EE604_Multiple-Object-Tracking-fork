# scripts/evaluate_mot16_train.py
import os
import pandas as pd
import motmetrics as mm

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_ROOT = os.path.join(ROOT, "datasets", "MOT16", "train")
RES_ROOT  = os.path.join(ROOT, "outputs", "MOT16-train")
os.makedirs(RES_ROOT, exist_ok=True)

def eval_one_sequence(seq_name, mh, iou_threshold=0.5, min_vis=0.0):
    """
    Evaluate one sequence and return (name, accumulator)
    """
    gt_path  = os.path.join(DATA_ROOT, seq_name, "gt", "gt.txt")
    res_path = os.path.join(RES_ROOT,  f"{seq_name}.txt")

    # Load GT
    cols_gt = ["frame","id","x","y","w","h","conf","class","vis"]
    gt = pd.read_csv(gt_path, header=None, names=cols_gt)
    # MOT16 eval: only consider conf==1 & class==1 (pedestrian)
    gt = gt[(gt["conf"] == 1) & (gt["class"] == 1)]
    if min_vis > 0:
        gt = gt[gt["vis"] >= min_vis]
    gt["x2"] = gt["x"] + gt["w"]
    gt["y2"] = gt["y"] + gt["h"]

    # Load HYP (tracker output)
    if os.path.exists(res_path) and os.path.getsize(res_path) > 0:
        cols_res = ["frame","id","x","y","w","h","score","_7","_8","_9"]
        res = pd.read_csv(res_path, header=None, names=cols_res)
        res["x2"] = res["x"] + res["w"]
        res["y2"] = res["y"] + res["h"]
    else:
        # empty hypothesis
        res = pd.DataFrame(columns=["frame","id","x","y","w","h","score","_7","_8","_9","x2","y2"])

    acc = mm.MOTAccumulator(auto_id=True)
    frames = sorted(set(gt["frame"].unique()).union(set(res["frame"].unique())))
    for f in frames:
        g = gt[gt["frame"] == f]
        h = res[res["frame"] == f]
        oids = g["id"].astype(int).tolist()
        hids = h["id"].astype(int).tolist()
        g_boxes = g[["x","y","x2","y2"]].to_numpy(copy=False)
        h_boxes = h[["x","y","x2","y2"]].to_numpy(copy=False)

        dists = mm.distances.iou_matrix(g_boxes, h_boxes, max_iou=iou_threshold)
        acc.update(oids, hids, dists)

    return seq_name, acc

def main():
    mm.lap.default_solver = "scipy"
    mh = mm.metrics.create()

    # sequences we have results for
    seqs = [f[:-4] for f in os.listdir(RES_ROOT) if f.endswith(".txt")]
    seqs = sorted(seqs)
    if not seqs:
        print("No result files found in", RES_ROOT)
        print("Run:  python scripts/run_deepsort_all.py  first.")
        return

    accs = {}
    for seq in seqs:
        print("Evaluating:", seq)
        name, acc = eval_one_sequence(seq, mh, iou_threshold=0.5, min_vis=0.0)
        accs[name] = acc

    summary = mh.compute_many(
        list(accs.values()),
        names=list(accs.keys()),
        metrics=mm.metrics.motchallenge_metrics,
        generate_overall=True
    )

    print("\n===== MOT16 Train Evaluation (IoU=0.5) =====")
    print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))


    out_csv = os.path.join(RES_ROOT, "metrics_summary.csv")
    summary.to_csv(out_csv)
    print("Saved summary CSV to:", out_csv)

if __name__ == "__main__":
    main()
