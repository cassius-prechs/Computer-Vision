"""
レポート用の図を自動生成する。

    python src/make_figures.py

出力（figures/）:
    fig_method_comparison.png     ... ①2段 vs ③1段 vs ②オラクル の F1 比較
    fig_per_condition.png         ... 撮影条件別 Recall（データ起因/アルゴリズム起因で色分け）
    fig_threshold_sweep.png       ... 閾値 vs Precision/Recall のトレードオフ
    fig_prompt_sensitivity.png    ... プロンプト言い換えによる Recall 変動
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import RESULTS_DIR, FIGURES_DIR, PRESENCE_CONF, CONF_THRESHOLDS
from evaluate import evaluate

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "font.size": 10})

CAUSE_COLOR = {"data": "#E69F00", "algorithmic": "#0072B2",
               "mixed": "#999999", "-": "#009E73"}


def fig_method_comparison(tags=("gdino_sam2", "sam3", "oracle_sam2")):
    rows = []
    for t in tags:
        if not (RESULTS_DIR / t / "predictions.json").exists():
            continue
        o, _ = evaluate(t, PRESENCE_CONF, quiet=True)
        rows.append(o)
    if not rows:
        return
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    x = range(len(df))
    w = 0.26
    ax.bar([i - w for i in x], df.precision, w, label="Precision", color="#56B4E9")
    ax.bar(list(x), df.recall, w, label="Recall", color="#E69F00")
    ax.bar([i + w for i in x], df.f1, w, label="F1", color="#0072B2")
    ax.set_xticks(list(x))
    ax.set_xticklabels(["Grounding DINO\n+ SAM2\n(2-stage)",
                        "SAM 3\n(1-stage)",
                        "GT box\n+ SAM2\n(oracle)"][:len(df)])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Item-level score")
    ax.set_title("Method comparison (item-level, self-captured data)")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.0),
              ncol=3, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_method_comparison.png")
    plt.close(fig)
    print("[fig] fig_method_comparison.png")


def fig_per_condition(tag="sam3"):
    p = RESULTS_DIR / tag / "metrics_per_condition.csv"
    if not p.exists():
        evaluate(tag, PRESENCE_CONF, quiet=True)
    df = pd.read_csv(p).sort_values("recall")

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [CAUSE_COLOR.get(c, "#999999") for c in df.cause]
    ax.barh(df.condition, df.recall, color=colors)
    ax.set_xlabel("Recall")
    ax.set_xlim(0, 1.05)
    ax.set_title(f"Recall by capture condition ({tag})")
    ax.grid(axis="x", alpha=0.3)

    handles = [plt.Rectangle((0, 0), 1, 1, color=v) for v in
               ["#0072B2", "#E69F00", "#999999", "#009E73"]]
    ax.legend(handles, ["algorithmic assumption", "data quality",
                        "mixed", "baseline"], fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_per_condition.png")
    plt.close(fig)
    print("[fig] fig_per_condition.png")


def fig_threshold_sweep(tag="sam3"):
    """conf 閾値を変えて再評価（推論は回し直さない。predictions.json を再解釈するだけ）"""
    rows = []
    for thr in CONF_THRESHOLDS:
        try:
            o, _ = evaluate(tag, thr, quiet=True)
        except SystemExit:
            return
        o["conf_thr"] = thr
        rows.append(o)
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(df.conf_thr, df.precision, "o-", label="Precision", color="#56B4E9")
    ax.plot(df.conf_thr, df.recall, "s-", label="Recall", color="#E69F00")
    ax.plot(df.conf_thr, df.f1, "^-", label="F1", color="#0072B2")
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Threshold sensitivity ({tag})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_threshold_sweep.png")
    plt.close(fig)
    print("[fig] fig_threshold_sweep.png")


def fig_prompt_sensitivity():
    p = RESULTS_DIR / "prompt_sensitivity" / "prompt_results.csv"
    if not p.exists():
        print("[skip] prompt_results.csv なし")
        return
    df = pd.read_csv(p)
    df = df[df.target != "__ambiguous__"].dropna(subset=["recall"])
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    for target, sub in df.groupby("target"):
        ax.plot(sub.prompt, sub.recall, "o-", label=target)
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1.05)
    ax.set_title("Prompt sensitivity: same object, different wording")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_prompt_sensitivity.png")
    plt.close(fig)
    print("[fig] fig_prompt_sensitivity.png")


if __name__ == "__main__":
    fig_method_comparison()
    fig_per_condition("sam3")
    fig_threshold_sweep("sam3")
    fig_prompt_sensitivity()
    print(f"\n[OK] -> {FIGURES_DIR}")
