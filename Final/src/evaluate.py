"""
評価スクリプト（このレポートの数値は全部ここから出る）

入力:
    dataset/annotations/gt_items.csv    ... image, wallet, key, ... (0/1) + condition
    results/<tag>/predictions.json      ... {image: {item: [{box, conf}, ...]}}

出力（results/<tag>/ 以下）:
    metrics_overall.csv     ... 全体の Precision / Recall / F1
    metrics_per_item.csv    ... アイテム別
    metrics_per_condition.csv ... 撮影条件別（← Discussion の主表）
    confusion_pairs.csv     ... どのアイテムを取りこぼしたか

使い方:
    python src/evaluate.py --tag sam3
    python src/evaluate.py --tag gdino_sam2
    python src/evaluate.py --compare gdino_sam2 sam3 oracle_sam2   # 横並び比較表
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from config import (GT_ITEMS_CSV, RESULTS_DIR, ITEM_IDS,
                    PRESENCE_CONF, CONDITION_CAUSE)


def load_gt():
    df = pd.read_csv(GT_ITEMS_CSV)
    if "image" not in df.columns:
        raise SystemExit("[!] gt_items.csv に 'image' 列が必要です")
    if "condition" not in df.columns:
        df["condition"] = "normal"
    return df


def load_pred(tag, conf_thr):
    path = RESULTS_DIR / tag / "predictions.json"
    if not path.exists():
        raise SystemExit(f"[!] {path} がありません。先に推論スクリプトを回してください。")
    raw = json.loads(path.read_text())

    rows = []
    for img, per_item in raw.items():
        row = {"image": img}
        for item in ITEM_IDS:
            dets = per_item.get(item, [])
            # そのアイテムが「検出された」= conf_thr 以上の検出が1つ以上ある
            hits = [d for d in dets if d.get("conf", 0.0) >= conf_thr]
            row[item] = 1 if hits else 0
            row[f"{item}__n"] = len(hits)
            row[f"{item}__maxconf"] = max([d["conf"] for d in hits], default=0.0)
        rows.append(row)
    return pd.DataFrame(rows)


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def evaluate(tag, conf_thr, quiet=False):
    gt = load_gt()
    pred = load_pred(tag, conf_thr)
    m = gt.merge(pred, on="image", suffixes=("_gt", "_pred"), how="inner")
    if len(m) == 0:
        raise SystemExit("[!] GTと予測で画像名が一致しません。ファイル名を確認。")

    out_dir = RESULTS_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- アイテム別 ----------
    per_item = []
    for item in ITEM_IDS:
        g = m[f"{item}_gt"].astype(int)
        p = m[f"{item}_pred"].astype(int)
        tp = int(((g == 1) & (p == 1)).sum())
        fp = int(((g == 0) & (p == 1)).sum())
        fn = int(((g == 1) & (p == 0)).sum())
        tn = int(((g == 0) & (p == 0)).sum())
        pr, rc, f1 = prf(tp, fp, fn)
        per_item.append(dict(item=item, TP=tp, FP=fp, FN=fn, TN=tn,
                             precision=round(pr, 3), recall=round(rc, 3),
                             f1=round(f1, 3)))
    per_item_df = pd.DataFrame(per_item).sort_values("f1")
    per_item_df.to_csv(out_dir / "metrics_per_item.csv", index=False)

    # ---------- 全体（micro） ----------
    TP = per_item_df.TP.sum(); FP = per_item_df.FP.sum(); FN = per_item_df.FN.sum()
    pr, rc, f1 = prf(TP, FP, FN)
    overall = pd.DataFrame([dict(tag=tag, conf_thr=conf_thr,
                                 n_images=len(m), TP=TP, FP=FP, FN=FN,
                                 precision=round(pr, 3), recall=round(rc, 3),
                                 f1=round(f1, 3))])
    overall.to_csv(out_dir / "metrics_overall.csv", index=False)

    # ---------- 撮影条件別（Discussion の主表） ----------
    per_cond = []
    for cond, sub in m.groupby("condition"):
        tp = fp = fn = 0
        for item in ITEM_IDS:
            g = sub[f"{item}_gt"].astype(int)
            p = sub[f"{item}_pred"].astype(int)
            tp += int(((g == 1) & (p == 1)).sum())
            fp += int(((g == 0) & (p == 1)).sum())
            fn += int(((g == 1) & (p == 0)).sum())
        pr_c, rc_c, f1_c = prf(tp, fp, fn)
        per_cond.append(dict(condition=cond,
                             cause=CONDITION_CAUSE.get(cond, "?"),
                             n_images=len(sub), TP=tp, FP=fp, FN=fn,
                             precision=round(pr_c, 3), recall=round(rc_c, 3),
                             f1=round(f1_c, 3)))
    per_cond_df = pd.DataFrame(per_cond).sort_values("f1")
    per_cond_df.to_csv(out_dir / "metrics_per_condition.csv", index=False)

    # ---------- 見落とし一覧（失敗例の図に使う画像を特定する） ----------
    misses = []
    for _, row in m.iterrows():
        for item in ITEM_IDS:
            if row[f"{item}_gt"] == 1 and row[f"{item}_pred"] == 0:
                misses.append(dict(image=row["image"], condition=row["condition"],
                                   item=item, type="FN(見落とし)"))
            if row[f"{item}_gt"] == 0 and row[f"{item}_pred"] == 1:
                misses.append(dict(image=row["image"], condition=row["condition"],
                                   item=item, type="FP(誤検出)",
                                   conf=round(row[f"{item}__maxconf"], 3)))
    pd.DataFrame(misses).to_csv(out_dir / "confusion_pairs.csv", index=False)

    if not quiet:
        print(f"\n===== {tag} (conf>={conf_thr}) =====")
        print(overall.to_string(index=False))
        print("\n--- 撮影条件別 ---")
        print(per_cond_df.to_string(index=False))
        print("\n--- アイテム別（F1の低い順）---")
        print(per_item_df.head(6).to_string(index=False))
        print(f"\n[OK] -> {out_dir}")

    return overall.iloc[0].to_dict(), per_cond_df


def compare(tags, conf_thr):
    rows = []
    for t in tags:
        o, _ = evaluate(t, conf_thr, quiet=True)
        rows.append(o)
    df = pd.DataFrame(rows)
    print("\n===== 手法比較 =====")
    print(df.to_string(index=False))

    # エラー伝播の定量化
    if "gdino_sam2" in tags and "oracle_sam2" in tags:
        a = df[df.tag == "gdino_sam2"].f1.iloc[0]
        b = df[df.tag == "oracle_sam2"].f1.iloc[0]
        if b > 0:
            print(f"\n★ エラー伝播による損失: F1 {b:.3f} (oracle) -> {a:.3f} (2段) "
                  f"= {(b - a) / b * 100:.1f}% の低下が検出段階に起因")

    out = RESULTS_DIR / "comparison.csv"
    df.to_csv(out, index=False)
    print(f"\n[OK] -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, default="sam3")
    ap.add_argument("--conf", type=float, default=PRESENCE_CONF)
    ap.add_argument("--compare", nargs="+", default=None)
    args = ap.parse_args()

    if args.compare:
        compare(args.compare, args.conf)
    else:
        evaluate(args.tag, args.conf)
