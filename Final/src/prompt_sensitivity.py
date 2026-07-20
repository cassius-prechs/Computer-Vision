"""
プロンプト感度分析（レポートで最も独創性が出る部分）

同じ物体に対してプロンプトの言い方だけを変え、検出率がどう変わるかを測る。
SAM3 公式も "tool" "bag" のような曖昧語ではノイジーになると Limitations で認めている。
それを自前データで再現・定量化する。

出力:
    results/prompt_sensitivity/prompt_results.csv
        prompt, target_item, n_detections, mean_conf, recall

使い方:
    python src/prompt_sensitivity.py --model sam3
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from config import (IMAGES_DIR, RESULTS_DIR, GT_ITEMS_CSV,
                    PROMPT_VARIANTS, PRESENCE_CONF)


def run_sam3_prompts(prompts, conf, model_path="sam3.pt"):
    from ultralytics.models.sam import SAM3SemanticPredictor
    overrides = dict(conf=conf, task="segment", mode="predict",
                     model=model_path, quantize=16, save=False, verbose=False)
    predictor = SAM3SemanticPredictor(overrides=overrides)

    images = sorted([p for p in IMAGES_DIR.iterdir()
                     if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])

    out = {}   # {prompt: {image: [conf,...]}}
    for prompt in prompts:
        out[prompt] = {}
        for img in images:
            predictor.set_image(str(img))
            try:
                results = predictor(text=[prompt])
            except Exception as e:
                print(f"  [warn] '{prompt}' on {img.name}: {e}")
                out[prompt][img.name] = []
                continue
            confs = []
            for r in results:
                if r.boxes is not None and len(r.boxes):
                    confs += [float(c) for c in r.boxes.conf.cpu().numpy()]
            out[prompt][img.name] = confs
        n = sum(len(v) for v in out[prompt].values())
        print(f"  '{prompt}': {n} detections across {len(images)} images")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["sam3", "gdino"], default="sam3")
    ap.add_argument("--conf", type=float, default=PRESENCE_CONF)
    args = ap.parse_args()

    gt = pd.read_csv(GT_ITEMS_CSV) if GT_ITEMS_CSV.exists() else None

    # 全プロンプト変種をフラット化
    plan = []          # (prompt, target_item)
    for target, variants in PROMPT_VARIANTS.items():
        for v in variants:
            plan.append((v, target))
    prompts = [p for p, _ in plan]

    print(f"[prompt sensitivity] {len(prompts)} prompts x images")
    if args.model == "sam3":
        raw = run_sam3_prompts(prompts, args.conf)
    else:
        raise SystemExit("gdino 版は run_gdino_sam2.py を --tag を変えて回すこと")

    rows = []
    for prompt, target in plan:
        per_img = raw[prompt]
        n_total = sum(len(v) for v in per_img.values())
        all_conf = [c for v in per_img.values() for c in v]
        n_imgs_hit = sum(1 for v in per_img.values() if len(v) > 0)

        # target が本物のアイテムなら recall を計算
        recall = None
        if gt is not None and target in gt.columns:
            pos = gt[gt[target] == 1]["image"].tolist()
            if pos:
                hit = sum(1 for im in pos if len(per_img.get(im, [])) > 0)
                recall = round(hit / len(pos), 3)

        rows.append(dict(
            prompt=prompt,
            target=target,
            n_detections=n_total,
            n_images_with_hit=n_imgs_hit,
            mean_conf=round(sum(all_conf) / len(all_conf), 3) if all_conf else 0.0,
            recall=recall,
        ))

    df = pd.DataFrame(rows)
    out_dir = RESULTS_DIR / "prompt_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "prompt_results.csv", index=False)
    (out_dir / "raw.json").write_text(json.dumps(raw, indent=2))

    print("\n===== プロンプト感度 =====")
    print(df.to_string(index=False))
    print(f"\n[OK] -> {out_dir}")
    print("\n※ '__ambiguous__' 行（tool/stuff/thing/bag）の n_detections が")
    print("   異常に多い/少ないことを確認 → 曖昧語での挙動不安定の証拠になる。")


if __name__ == "__main__":
    main()
