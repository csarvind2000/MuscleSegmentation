"""Aggregate all results/*.json into summary tables (CSV + LaTeX booktabs)."""
import os, json, glob
import numpy as np
import config as C


def load_all():
    out = []
    for f in sorted(glob.glob(os.path.join(C.RESULTS, "*.json"))):
        out.append((os.path.basename(f), json.load(open(f))))
    return out


def main():
    rows = load_all()
    lines_csv = ["task,model,params_M,dice_fg_mean,iou_fg_mean,minutes,per_class_dice"]
    for fn, r in rows:
        if "task" in r:  # single-task
            pc = "|".join(f"{c}:{d}" for c, d in zip(r["classes"], r["test_dice_per_class"]))
            lines_csv.append(f"{r['task']},{r['model']},{r.get('n_params_M','')},"
                             f"{r['test_dice_fg_mean']},{r.get('test_iou_fg_mean','')},"
                             f"{r.get('minutes','')},{pc}")
        else:  # multitask
            lines_csv.append(f"aattct,{r['model']},,{r['aattct']['test_dice_fg_mean']},,,{r['minutes']}")
            lines_csv.append(f"thigh,{r['model']},,{r['thigh']['test_dice_fg_mean']},,,")
    open(os.path.join(C.RESULTS, "summary.csv"), "w").write("\n".join(lines_csv))
    print("\n".join(lines_csv))
    print("\nwrote", os.path.join(C.RESULTS, "summary.csv"))


if __name__ == "__main__":
    main()
