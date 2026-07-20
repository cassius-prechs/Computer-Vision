"""
オラクル実験：GT box -> SAM 2

これが本レポートの「勝ち筋」。
  ① Grounding DINO -> SAM2 のスコア = 検出誤り + セグメント誤り
  ② GT box         -> SAM2 のスコア = セグメント誤り のみ
  ② - ① = エラー伝播による損失の定量値

失敗が起きた画像 8枚 だけに GT box を引けば十分（全25枚は不要）。

入力:
    dataset/annotations/gt_boxes.json
    {
      "image07.jpg": [
          {"item": "wallet", "box": [x1, y1, x2, y2]},
          {"item": "key",    "box": [x1, y1, x2, y2]}
      ],
      ...
    }

出力:
    results/oracle_sam2/masks/*.npz
    results/oracle_sam2/predictions.json   （conf は 1.0 固定 = 検出は完璧という仮定）
"""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from config import IMAGES_DIR, RESULTS_DIR, GT_BOXES_JSON, ITEM_IDS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    if not GT_BOXES_JSON.exists():
        raise SystemExit(f"[!] {GT_BOXES_JSON} がありません。CVAT等で8枚だけboxを引いてください。")

    gt = json.loads(GT_BOXES_JSON.read_text())

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    sam2_model = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml",
                            "checkpoints/sam2.1_hiera_large.pt",
                            device=DEVICE)
    predictor = SAM2ImagePredictor(sam2_model)

    out_dir = RESULTS_DIR / "oracle_sam2"
    mask_dir = out_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    predictions = {}
    for img_name, objs in gt.items():
        img_path = IMAGES_DIR / img_name
        if not img_path.exists():
            print(f"[warn] 画像なし: {img_name}")
            continue
        print(f"[oracle] {img_name}  ({len(objs)} objects)")

        image = np.array(Image.open(img_path).convert("RGB"))
        boxes = np.array([o["box"] for o in objs], dtype=np.float32)

        predictor.set_image(image)
        with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
            masks, scores, _ = predictor.predict(box=boxes, multimask_output=False)

        masks = np.asarray(masks)
        masks = masks.squeeze(1) if masks.ndim == 4 else masks

        np.savez_compressed(mask_dir / f"{Path(img_name).stem}.npz",
                            masks=masks.astype(np.uint8),
                            boxes=boxes,
                            items=np.array([o["item"] for o in objs]))

        per_item = {k: [] for k in ITEM_IDS}
        for o, s in zip(objs, np.asarray(scores).ravel()):
            per_item.setdefault(o["item"], []).append(
                {"box": o["box"], "conf": 1.0, "sam2_score": float(s)}
            )
        predictions[img_name] = per_item

    (out_dir / "predictions.json").write_text(
        json.dumps(predictions, indent=2, ensure_ascii=False))
    print(f"\n[OK] saved -> {out_dir / 'predictions.json'}")
    print("→ evaluate.py で gdino_sam2 と比較すれば、エラー伝播の定量値が出る。")


if __name__ == "__main__":
    main()
