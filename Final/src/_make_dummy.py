"""
【動作確認専用】ダミーのGTと予測を生成して、evaluate.py が通ることを検証する。
実データが撮れたら不要。消してよい。

    python src/_make_dummy.py && python src/evaluate.py --compare gdino_sam2 sam3 oracle_sam2
"""
import json
import random
from pathlib import Path

import pandas as pd

from config import (ANNOT_DIR, RESULTS_DIR, ITEM_IDS, GT_ITEMS_CSV, CONDITIONS)

random.seed(0)

N = 20
# 条件の割り当て：normal 8枚 + 敵対的12枚
conds = ["normal"] * 8 + ["mirror", "mirror", "transparent", "transparent",
                          "dense", "polysemy", "occlusion", "viewpoint",
                          "lowlight", "blur", "clutter", "tiny"]

rows = []
gt_presence = {}
for i in range(N):
    img = f"image{i+1:02d}.jpg"
    row = {"image": img, "condition": conds[i]}
    present = {}
    for item in ITEM_IDS:
        v = 1 if random.random() < 0.55 else 0
        row[item] = v
        present[item] = v
    rows.append(row)
    gt_presence[img] = present

ANNOT_DIR.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(GT_ITEMS_CSV, index=False)
print(f"[dummy] GT -> {GT_ITEMS_CSV}")

# 条件ごとの「検出しやすさ」を変えて、現実っぽい予測を作る
DIFFICULTY = {
    "normal": 0.92, "clutter": 0.72, "viewpoint": 0.70, "occlusion": 0.58,
    "lowlight": 0.50, "blur": 0.48, "dense": 0.55, "tiny": 0.40,
    "transparent": 0.35, "polysemy": 0.45, "mirror": 0.60,
}
FP_RATE = {"normal": 0.02, "mirror": 0.30, "clutter": 0.15, "polysemy": 0.25}


def synth(tag, recall_scale=1.0, fp_scale=1.0, only_images=None):
    preds = {}
    for r in rows:
        img = r["image"]
        if only_images and img not in only_images:
            continue
        cond = r["condition"]
        base = DIFFICULTY[cond] * recall_scale
        fpr = FP_RATE.get(cond, 0.05) * fp_scale
        per_item = {}
        for item in ITEM_IDS:
            dets = []
            if r[item] == 1 and random.random() < min(base, 0.99):
                dets.append({"box": [10, 10, 100, 100],
                             "conf": round(random.uniform(0.3, 0.95), 3)})
            elif r[item] == 0 and random.random() < fpr:
                dets.append({"box": [10, 10, 60, 60],
                             "conf": round(random.uniform(0.26, 0.5), 3)})
            per_item[item] = dets
        preds[img] = per_item

    d = RESULTS_DIR / tag
    d.mkdir(parents=True, exist_ok=True)
    (d / "predictions.json").write_text(json.dumps(preds, indent=2))
    print(f"[dummy] pred -> {d/'predictions.json'}")


# ① 2段構成（検出誤りが乗る）
synth("gdino_sam2", recall_scale=1.0, fp_scale=1.0)
# ③ 1段構成（SAM3：やや強い）
synth("sam3", recall_scale=1.12, fp_scale=0.7)
# ② オラクル（検出は完璧 → 失敗画像8枚のみ）
oracle_imgs = [r["image"] for r in rows if r["condition"] != "normal"][:8]
synth("oracle_sam2", recall_scale=1.6, fp_scale=0.0, only_images=oracle_imgs)
