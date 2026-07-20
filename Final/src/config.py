"""
中央設定ファイル。ここだけ書き換えれば全スクリプトに反映される。
"""
from pathlib import Path

# ---------------------------------------------------------------
# パス
# ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "dataset" / "images"
ANNOT_DIR = ROOT / "dataset" / "annotations"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

GT_ITEMS_CSV = ANNOT_DIR / "gt_items.csv"        # 画像 x アイテムの 0/1 表（必須）
GT_BOXES_JSON = ANNOT_DIR / "gt_boxes.json"      # オラクル用（8枚だけ）
CAPTURE_LOG_CSV = ANNOT_DIR / "capture_log.csv"  # 撮影条件メモ

# ---------------------------------------------------------------
# タスク定義：持ち物チェッカーの対象アイテム
#   キー   = 内部ID（CSVの列名になる）
#   value  = デフォルトのテキストプロンプト
#   ※ 自分の持ち物に合わせて自由に書き換えること
# ---------------------------------------------------------------
ITEMS = {
    "wallet":        "wallet",
    "key":           "key",
    "earphones":     "earphones",
    "bottle":        "plastic bottle",
    "umbrella":      "folding umbrella",
    "notebook":      "notebook",
    "pen":           "pen",
    "cable":         "charging cable",
    "power_bank":    "power bank",
    "glasses_case":  "glasses case",
}

ITEM_IDS = list(ITEMS.keys())
DEFAULT_PROMPTS = [ITEMS[k] for k in ITEM_IDS]

# ---------------------------------------------------------------
# プロンプト感度分析用（Step 6-1）
#   同じ物体を指す複数の言い方を並べ、検出率がどう変わるかを測る
# ---------------------------------------------------------------
PROMPT_VARIANTS = {
    # 上位語 -> 下位語（specificity の階段）
    "bottle": [
        "container",              # 上位語
        "bottle",                 # 標準
        "plastic water bottle",   # 下位語
        "transparent bottle",     # 属性付与
    ],
    # 同義語
    "power_bank": [
        "battery",
        "power bank",
        "portable charger",
    ],
    # 曖昧な抽象語（SAM3公式が「noisyになる」と認めている領域）
    "__ambiguous__": [
        "tool",
        "stuff",
        "thing",
        "bag",
        "object",
    ],
}

# ---------------------------------------------------------------
# 閾値スイープ（Step 6-2）
# ---------------------------------------------------------------
CONF_THRESHOLDS = [0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

# Grounding DINO 用
GDINO_BOX_THRESHOLD = 0.15
GDINO_TEXT_THRESHOLD = 0.25

# 検出を「そのアイテムが存在する」とみなす最低信頼度
PRESENCE_CONF = 0.25

# ---------------------------------------------------------------
# 撮影条件のタグ（capture_log.csv の condition 列で使う）
# ---------------------------------------------------------------
CONDITIONS = [
    "normal",        # 明るい・整理済み（成功期待）
    "mirror",        # 鏡/モニタ映り込み      -> アルゴリズム起因
    "transparent",   # 透明・光沢物体          -> アルゴリズム起因
    "dense",         # 同種物体の密集          -> アルゴリズム起因
    "polysemy",      # 語義曖昧(mouse等)       -> アルゴリズム起因
    "occlusion",     # 遮蔽                    -> アルゴリズム起因
    "viewpoint",     # 極端な視点              -> アルゴリズム起因
    "lowlight",      # 低照度                  -> データ起因
    "blur",          # 手ブレ                  -> データ起因
    "clutter",       # 背景が雑然              -> 複合
    "tiny",          # 極小物体                -> 複合
]

# 失敗原因の帰属（Discussion の表に直結）
CONDITION_CAUSE = {
    "normal": "-",
    "mirror": "algorithmic",
    "transparent": "algorithmic",
    "dense": "algorithmic",
    "polysemy": "algorithmic",
    "occlusion": "algorithmic",
    "viewpoint": "algorithmic",
    "lowlight": "data",
    "blur": "data",
    "clutter": "mixed",
    "tiny": "mixed",
}
