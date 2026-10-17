"""Figures of the revised manuscript (matplotlib, vector PDF + PNG). Palette: validated reference categorical slots."""
import json, os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "..", "TEX_paper", "Figures_revised"); os.makedirs(OUT, exist_ok=True)
R = json.load(open(os.path.join(HERE, "results", "paper_numbers.json")))
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "magenta": "#e87ba4", "green": "#008300", "violet": "#6250d6", "red": "#e34948"}
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.edgecolor": BASE, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight"); fig.savefig(os.path.join(OUT, name + ".png"), dpi=300, bbox_inches="tight"); plt.close(fig); print("saved", name)

# ---------- Figure: learning curve
LC, L = R["lc"], R["llm"]
ns = ["10", "50", "100", "500", "1000", "all"]; x_all = 4545; xs = [10, 50, 100, 500, 1000, x_all]
fig, ax = plt.subplots(figsize=(6.4, 4.0))
series = [("Dense head, 150 tokens", LC["dense150"], C["blue"], "o"), ("RBF-SVM, 150 tokens", LC["svm150"], C["orange"], "s"),
          ("Dense head, 512 tokens", LC["dense512"], C["aqua"], "D"), ("Fully fine-tuned CAMeLBERT", LC["finetune512"], C["violet"], "^")]
for name, d, col, mk in series:
    pts = [(x, d[n]["acc"] * 100, (d[n]["acc_std"] or 0) * 100) for x, n in zip(xs, ns) if d.get(n)]
    ax.errorbar([q[0] for q in pts], [q[1] for q in pts], yerr=[q[2] for q in pts], color=col, marker=mk, ms=5, lw=2, capsize=2, label=name, zorder=3)
lora = [(100, L["lora100"]["acc"] * 100), (1000, L["lora1000"]["acc"] * 100), (x_all, L["lora_all"]["acc"] * 100)]
ax.plot([q[0] for q in lora], [q[1] for q in lora], color=C["magenta"], marker="P", ms=7, lw=2, label="Qwen2.5-7B + QLoRA", zorder=3)
ax.axhline(L["gpt41"]["acc"] * 100, color=MUTED, lw=1.2, ls="--", zorder=1); ax.text(10.5, L["gpt41"]["acc"] * 100 + 0.25, "best zero-shot LLM (GPT-4.1, 92.8)", va="bottom", ha="left", fontsize=7.5, color=INK2)
ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels(["10", "50", "100", "500", "1,000", "all\n(4,545)"])
ax.set_xlabel("Labelled training articles per class"); ax.set_ylabel("Test accuracy (%)"); ax.set_ylim(80, 100); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=8, loc="lower right"); ax.set_xlim(8, x_all * 1.6)
save(fig, "fig_learning_curve")

# ---------- Figure: prompt robustness (dot + whisker over 5 prompts, with few-shot k=2 marker) on the 700 subset
pr = {r["model"]: r for r in R["tables_prompt_robustness"]}; fs = pd.DataFrame(R["tables_fewshot"])
order = [("openai/gpt-4.1", "GPT-4.1"), ("anthropic/claude-haiku-4.5", "Claude Haiku 4.5"), ("google/gemini-3.8-flash", "Gemini 3.8 Flash"), ("deepseek-ai/DeepSeek-V4-Flash-0731", "DeepSeek V4 Flash"), ("Qwen/Qwen2.5-7B-Instruct", "Qwen2.5-7B-Instruct")]
fig, ax = plt.subplots(figsize=(6.4, 3.0))
for i, (mid, nm) in enumerate(order):
    r = pr[mid]; y = len(order) - 1 - i
    ax.plot([r["acc_min"] * 100, r["acc_max"] * 100], [y, y], color=C["blue"], lw=2, solid_capstyle="butt", zorder=2)
    ax.plot(r["acc_mean"] * 100, y, "o", color=C["blue"], ms=7, zorder=3)
    ax.plot(r["acc_primary_sub"] * 100, y, "|", color=INK, ms=14, mew=1.6, zorder=4)
    k2 = fs[(fs.model == mid) & (fs.shots == (2 if "Qwen" not in mid else 1))]
    if len(k2): ax.plot(k2.iloc[0]["acc_sub"] * 100, y, "D", color=C["orange"], ms=6, zorder=3)
ax.set_yticks(range(len(order))); ax.set_yticklabels([nm for _, nm in order][::-1]); ax.set_xlabel("Accuracy on the 700-article evaluation subset (%)")
ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True); ax.set_xlim(84, 97)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], color=C["blue"], lw=2, marker="o", ms=6, label="five prompts: mean and range"), Line2D([], [], color=INK, marker="|", ms=12, mew=1.6, ls="", label="primary prompt"),
                   Line2D([], [], color=C["orange"], marker="D", ms=6, ls="", label="few-shot ($k$=2; $k$=1 for Qwen)")], frameon=False, fontsize=7.5, loc="upper left")
save(fig, "fig_prompt_robustness")

# ---------- Figure: confusion matrices (dense head vs GPT-4.1), row-normalised, sequential blue
CATS = ["Culture", "Finance", "Medical", "Politics", "Religion", "Sports", "Tech"]
import sys; sys.path.insert(0, HERE)
from analysis import load_preds
mats = [("Frozen CAMeLBERT + dense head (150 tokens)", load_preds(os.path.join(HERE, "results/supervised/dense__CAMeL-Lab_bert-base-arabic-camelbert-msa__len150__nall__s0__split__dl-none.pred.csv"))),
        ("GPT-4.1, zero-shot", load_preds(os.path.join(HERE, "results/openrouter__openai_gpt-4.1__en_v2__k0.jsonl")))]
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("seqblue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
for ax, (title, d) in zip(axes, mats):
    cm = pd.crosstab(d.gold, d.pred).reindex(index=CATS, columns=CATS, fill_value=0).values
    ax.imshow(cm, cmap=cmap, vmin=0, vmax=cm.max()); ax.set_xticks(range(7)); ax.set_yticks(range(7)); ax.set_xticklabels(CATS, rotation=45, ha="right", fontsize=7.5); ax.set_yticklabels(CATS, fontsize=7.5)
    for i in range(7):
        for j in range(7):
            v = cm[i, j]
            if v: ax.text(j, i, str(v), ha="center", va="center", fontsize=6.5, color=("white" if v > cm.max() * 0.55 else INK))
    ax.set_title(title, fontsize=8.5, color=INK); ax.set_xlabel("Predicted", fontsize=8); ax.set_ylabel("True", fontsize=8)
    for s in ax.spines.values(): s.set_visible(False)
fig.tight_layout(); save(fig, "fig_confusion")
