"""
CVATから書き出した COCO 1.0 形式のアノテーションを
run_oracle_sam2.py が読む gt_boxes.json に変換する。

前提:
    CVATのラベル名が config.py の ITEMS のキー（wallet, key, ...）と
    一致していること。一致しない場合は LABEL_MAP で読み替える。

使い方:
    python src/coco_to_gt_boxes.py path/to/instances_default.json
"""
import argparse
import json
from pathlib import Path

from config import GT_BOXES_JSON, ITEM_IDS

# CVAT側のラベル名がconfig.pyのITEM_IDSと違う場合はここで読み替える
# 例: {"student id": "student_id"}
LABEL_MAP = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("coco_json", type=str)
    args = ap.parse_args()

    coco = json.loads(Path(args.coco_json).read_text())

    cat_id_to_name = {c["id"]: LABEL_MAP.get(c["name"], c["name"]) for c in coco["categories"]}
    img_id_to_name = {im["id"]: im["file_name"] for im in coco["images"]}

    unknown = set(cat_id_to_name.values()) - set(ITEM_IDS)
    if unknown:
        print(f"[warn] config.py の ITEM_IDS にないラベル: {unknown}"
              f"\n       LABEL_MAP で読み替えるか、config.py 側を直してください。")

    gt_boxes = {}
    for ann in coco["annotations"]:
        img_name = Path(img_id_to_name[ann["image_id"]]).name
        item = cat_id_to_name[ann["category_id"]]
        x, y, w, h = ann["bbox"]
        box = [round(x, 1), round(y, 1), round(x + w, 1), round(y + h, 1)]
        gt_boxes.setdefault(img_name, []).append({"item": item, "box": box})

    GT_BOXES_JSON.parent.mkdir(parents=True, exist_ok=True)
    GT_BOXES_JSON.write_text(json.dumps(gt_boxes, indent=2, ensure_ascii=False))
    print(f"[OK] {len(gt_boxes)} images -> {GT_BOXES_JSON}")


if __name__ == "__main__":
    main()
