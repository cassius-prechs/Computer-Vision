"""
2段構成：Grounding DINO（検出） -> SAM 2（セグメント）

HuggingFace transformers 版の Grounding DINO を使う（公式リポより導入が圧倒的に楽）。
    pip install transformers torch pillow
    pip install "sam2 @ git+https://github.com/facebookresearch/sam2.git"

出力:
    results/gdino_sam2/predictions.json   ... {image: {item: [{box, conf}]}}
    results/gdino_sam2/masks/*.npz        ... マスク
    results/gdino_sam2/vis/*.jpg

使い方:
    python src/run_gdino_sam2.py
    python src/run_gdino_sam2.py --box-thr 0.20 --tag sweep_b020
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from config import (IMAGES_DIR, RESULTS_DIR, ITEMS, ITEM_IDS,
                    GDINO_BOX_THRESHOLD, GDINO_TEXT_THRESHOLD)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_gdino():
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    model_id = "IDEA-Research/grounding-dino-base"   # tiny なら grounding-dino-tiny
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(DEVICE).eval()
    return processor, model


def build_sam2(ckpt: str, cfg: str):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    sam2_model = build_sam2(cfg, ckpt, device=DEVICE)
    return SAM2ImagePredictor(sam2_model)


def gdino_detect(processor, model, image: Image.Image, prompts, box_thr, text_thr):
    """
    Grounding DINO は 'a. b. c.' 形式のピリオド区切りプロンプトを取る。
    返り値: [(box_xyxy, score, label_text), ...]
    """
    text = ". ".join(prompts) + "."
    inputs = processor(images=image, text=text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)

    res = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_thr,
        text_threshold=text_thr,
        target_sizes=[image.size[::-1]],   # (h, w)
    )[0]

    out = []
    for box, score, label in zip(res["boxes"], res["scores"], res["text_labels"]):
        out.append(([float(v) for v in box.tolist()],
                    float(score),
                    str(label)))
    return out


def label_to_item(label_text: str):
    """Grounding DINO が返すラベル文字列を、こちらの item_id に対応づける。"""
    lt = label_text.lower().strip()
    best, best_len = None, 0
    for item_id, prompt in ITEMS.items():
        p = prompt.lower()
        # 部分一致（'plastic bottle' が 'bottle' として返ってくることがある）
        if p in lt or lt in p:
            if len(p) > best_len:
                best, best_len = item_id, len(p)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--box-thr", type=float, default=GDINO_BOX_THRESHOLD)
    ap.add_argument("--text-thr", type=float, default=GDINO_TEXT_THRESHOLD)
    ap.add_argument("--tag", type=str, default="gdino_sam2")
    ap.add_argument("--sam2-ckpt", type=str, default="checkpoints/sam2.1_hiera_large.pt")
    ap.add_argument("--sam2-cfg", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--skip-sam2", action="store_true",
                    help="検出だけ回す（閾値スイープ時に速い）")
    args = ap.parse_args()

    out_dir = RESULTS_DIR / args.tag
    mask_dir = out_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    processor, gdino = build_gdino()
    sam2 = None if args.skip_sam2 else build_sam2(args.sam2_ckpt, args.sam2_cfg)

    prompts = [ITEMS[k] for k in ITEM_IDS]
    images = sorted([p for p in IMAGES_DIR.iterdir()
                     if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not images:
        raise SystemExit(f"[!] 画像がありません: {IMAGES_DIR}")

    predictions = {}
    for i, img_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {img_path.name}")
        image = Image.open(img_path).convert("RGB")

        dets = gdino_detect(processor, gdino, image, prompts,
                            args.box_thr, args.text_thr)

        per_item = {k: [] for k in ITEM_IDS}
        boxes_for_sam2 = []
        for box, score, label in dets:
            item_id = label_to_item(label)
            if item_id is None:
                continue    # 対応づかないラベルは捨てる（要ログ：FPの原因分析に使う）
            per_item[item_id].append({"box": box, "conf": score, "label": label})
            boxes_for_sam2.append(box)

        # --- SAM2 でマスク化 ---
        if sam2 is not None and boxes_for_sam2:
            sam2.set_image(np.array(image))
            with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
                masks, scores, _ = sam2.predict(
                    box=np.array(boxes_for_sam2),
                    multimask_output=False,
                )
            masks = np.asarray(masks).squeeze(1) if masks.ndim == 4 else np.asarray(masks)
            np.savez_compressed(mask_dir / f"{img_path.stem}.npz",
                                masks=masks.astype(np.uint8),
                                boxes=np.array(boxes_for_sam2))

        predictions[img_path.name] = per_item

    out_json = out_dir / "predictions.json"
    out_json.write_text(json.dumps(predictions, indent=2, ensure_ascii=False))
    print(f"\n[OK] saved -> {out_json}")


if __name__ == "__main__":
    main()
