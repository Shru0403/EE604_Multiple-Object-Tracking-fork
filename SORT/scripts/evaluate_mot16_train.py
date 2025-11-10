import os
import glob
import pandas as pd
import motmetrics as mm

DATA_ROOT = r"data/MOT16/train"   # where MOT16 train sequences live
RES_ROOT  = r"results"             # where your <SEQ>_sort.txt files are

def load_gt(gt_path: str) -> pd.DataFrame:
    df = pd.read_csv(gt_path, header=None)
    df.columns = ['FrameId','Id','X','Y','Width','Height','Conf','ClassId','Vis']
    # Standard MOT16 filtering: evaluate only pedestrians (class=1) that are valid (Conf==1)
    # Many evaluations also discard fully invisible boxes (Vis==0). Keep Vis>0 to be safe.
    df = df[(df['ClassId'] == 1) & (df['Conf'] == 1) & (df['Vis'] > 0.0)]
    return df

def load_res(res_path: str) -> pd.DataFrame:
    df = pd.read_csv(res_path, header=None)
    df.columns = ['FrameId','Id','X','Y','Width','Height','Conf','ClassId','Vis']
    return df

def evaluate_seq(seq_name: str, gt_df: pd.DataFrame, res_df: pd.DataFrame):
    acc = mm.MOTAccumulator(auto_id=True)
    frames = sorted(gt_df['FrameId'].unique())
    for f in frames:
        gtf = gt_df[gt_df['FrameId'] == f]
        resf = res_df[res_df['FrameId'] == f]

        gt_boxes  = gtf[['X','Y','Width','Height']].to_numpy()
        res_boxes = resf[['X','Y','Width','Height']].to_numpy()
        gt_ids    = gtf['Id'].to_numpy()
        res_ids   = resf['Id'].to_numpy()

        # IOU-based distance with a 0.5 IoU match threshold (common for MOT16)
        distances = mm.distances.iou_matrix(gt_boxes, res_boxes, max_iou=0.5)
        acc.update(gt_ids, res_ids, distances)

    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=[
            'num_frames', 'mota', 'motp', 'idf1', 'num_switches',
            'num_false_positives', 'num_misses', 'num_objects', 'recall', 'precision'
        ],
        name=seq_name
    )
    return summary

def main():
    mm.lap.default_solver = 'scipy'  # use fast solver if available

    seq_paths = sorted(glob.glob(os.path.join(DATA_ROOT, "MOT16-*")))
    if not seq_paths:
        print(f"No sequences found under {DATA_ROOT}")
        return

    all_summaries = []
    for seq_path in seq_paths:
        seq_name = os.path.basename(seq_path)
        gt_path  = os.path.join(seq_path, "gt", "gt.txt")
        res_path = os.path.join(RES_ROOT, f"{seq_name}_sort.txt")

        if not os.path.exists(gt_path):
            print(f"[SKIP] GT missing: {gt_path}")
            continue
        if not os.path.exists(res_path):
            print(f"[SKIP] Results missing: {res_path} (run your tracker for {seq_name})")
            continue

        gt_df  = load_gt(gt_path)
        res_df = load_res(res_path)
        summary = evaluate_seq(seq_name, gt_df, res_df)
        all_summaries.append(summary)

    if not all_summaries:
        print("No sequences evaluated.")
        return

    final = pd.concat(all_summaries)
    # Pretty print
    mh = mm.metrics.create()
    print("\n=== Per-sequence metrics (MOT16 train) ===")
    print(mm.io.render_summary(
        final, formatters=mh.formatters, namemap={
            'mota': 'MOTA','motp': 'MOTP','idf1': 'IDF1','num_switches': 'ID Sw',
            'num_false_positives': 'FP','num_misses': 'FN','num_objects': 'GT',
            'recall': 'Recall','precision': 'Precision','num_frames':'Frames'
        }
    ))

    # Macro average across sequences
    macro = final.mean(numeric_only=True).to_frame().T
    macro.index = ['MOT16-train (macro)']
    print("\n=== Macro average (unweighted) ===")
    print(macro[['mota','motp','idf1','num_switches','num_false_positives','num_misses','recall','precision']])

    # Save CSV
    os.makedirs(RES_ROOT, exist_ok=True)
    out_csv = os.path.join(RES_ROOT, "mot16_train_metrics.csv")
    pd.concat([final, macro]).to_csv(out_csv)
    print(f"\nSaved metrics to {out_csv}")

if __name__ == "__main__":
    main()
