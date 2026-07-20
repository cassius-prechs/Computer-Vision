"""
SAM 3（1段構成：テキスト -> マスク）で全画像を推論する。

前提:
    pip install -U ultralytics            # 8.3.237 以降
    sam3.pt をプロジェクト直下に配置      # ← HuggingFace で access request が必要（README参照）

出力:
    results/sam3/predictions.json   ... 画像ごとの {item: [ {box, conf} ] }
    results/sam3/vis/*.jpg          ... 可視化画像

使い方:
    python src/run_sam3.py
    python src/run_sam3.py --conf 0.15          # 閾値を変える
    python src/run_sam3.py --tag sweep_c015     # 出力先にタグを付ける
"""
import argparse
import json
from pathlib import Path

import numpy as np

from config import (IMAGES_DIR, RESULTS_DIR, ITEMS, ITEM_IDS, PRESENCE_CONF)


def masks_to_boxes(masks):
    """(N,H,W) の bool/float マスクから xyxy box を作る（保険用）。"""
    boxes = []
    for m in masks:
        ys, xs = np.nonzero(m > 0.5)
        if len(xs) == 0:
            boxes.append([0, 0, 0, 0])
        else:
            boxes.append([float(xs.min()), float(ys.min()),
                          float(xs.max()), float(ys.max())])
    return boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=PRESENCE_CONF)
    ap.add_argument("--model", type=str, default="sam3.pt")
    ap.add_argument("--tag", type=str, default="sam3")
    ap.add_argument("--no-vis", action="store_true")
    args = ap.parse_args()

    from ultralytics.models.sam import SAM3SemanticPredictor

    out_dir = RESULTS_DIR / args.tag
    vis_dir = out_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    overrides = dict(
        conf=args.conf,
        task="segment",
        mode="predict",
        model=args.model,
        quantize=16,          # FP16。VRAMが厳しければ有効なまま
        save=not args.no_vis,
        project=str(vis_dir),
        verbose=False,
    )
    predictor = SAM3SemanticPredictor(overrides=overrides)

    images = sorted([p for p in IMAGES_DIR.iterdir()
                     if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not images:
        raise SystemExit(f"[!] 画像がありません: {IMAGES_DIR}")

    predictions = {}
    for i, img_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {img_path.name}")

        # 画像を1回セットして、複数のテキストクエリを投げる（公式推奨の使い方）
        predictor.set_image(str(img_path))

        per_item = {}
        for item_id in ITEM_IDS:
            prompt = ITEMS[item_id]
            try:
                results = predictor(text=[prompt])
            except Exception as e:      # 1アイテムのコケで全体を止めない
                print(f"    [warn] '{prompt}' failed: {e}")
                per_item[item_id] = []
                continue

            dets = []
            for r in results:
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                for b, c in zip(xyxy, confs):
                    dets.append({"box": [float(v) for v in b],
                                 "conf": float(c),
                                 "prompt": prompt})
            per_item[item_id] = dets

        predictions[img_path.name] = per_item

    out_json = out_dir / "predictions.json"
    out_json.write_text(json.dumps(predictions, indent=2, ensure_ascii=False))
    print(f"\n[OK] saved -> {out_json}")
    print(f"     visualizations -> {vis_dir}")


if __name__ == "__main__":
    main()
