"""Generate the LaTeX tables of the revised manuscript from results/paper_numbers.json (single source of truth)."""
import json, os, pandas as pd, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "..", "TEX_paper", "elsarticle", "tables")
os.makedirs(OUT, exist_ok=True)
R = json.load(open(os.path.join(HERE, "results", "paper_numbers.json")))
S, L, LC = R["sup"], R["llm"], R["lc"]
CATS = ["Culture", "Finance", "Medical", "Politics", "Religion", "Sports", "Tech"]
def p(x, d=1): return f"{100*x:.{d}f}"
def ci(v): return f"[{p(v[0])}, {p(v[1])}]"
def w(name, s): open(os.path.join(OUT, name), "w").write(s); print("wrote", name)

# ---- main table
rows = [("Frozen CAMeLBERT + dense head (150 tok.)", "0.30\\,M", S["dense150"]), ("Frozen CAMeLBERT + dense head (150 tok., original recipe, 1 seed)", "0.30\\,M", S["dense150_paperlr"]),
        ("Frozen CAMeLBERT + RBF-SVM (150 tok.)", "--", S["svm150"]),
        ("Frozen CAMeLBERT + dense head (512 tok.)", "0.30\\,M", S["dense512"]), ("Frozen CAMeLBERT + RBF-SVM (512 tok.)", "--", S["svm512"]),
        ("Fully fine-tuned CAMeLBERT (512 tok.)", "109.1\\,M", S["ft512"]),
        ("Qwen2.5-7B-Instruct + QLoRA, all training data", "40.4\\,M", L["lora_all"]), ("Qwen2.5-7B-Instruct + QLoRA, 1{,}000 / class", "40.4\\,M", L["lora1000"]),
        ("Qwen2.5-7B-Instruct + QLoRA, 100 / class", "40.4\\,M", L["lora100"]),
        ("GPT-4.1, zero-shot", "0", L["gpt41"]), ("Claude Haiku 4.5, zero-shot", "0", L["claude45"]), ("Gemini 3.8 Flash, zero-shot", "0", L["gemini38"]),
        ("DeepSeek V4 Flash, zero-shot", "0", L["deepseekv4"]), ("Qwen2.5-7B-Instruct, zero-shot", "0", L["qwen"])]
body = ""
for i, (name, params, v) in enumerate(rows):
    acc = p(v["acc"]) + (f"$\\pm${p(v['acc_std'])}" if v.get("n_runs", 1) > 1 else "")
    body += f"{name} & {params} & {acc} & {ci(v['acc_ci'])} & {p(v['macro_p'])} & {p(v['macro_r'])} & {p(v['macro_f1'])} \\\\\n"
    if i in (5, 8): body += "\\addlinespace\n"
w("tab_main.tex", r"""\begin{table}[!ht]
\centering
\caption{Main results on the shared, deduplicated test split (6{,}822 articles). Accuracy is the mean over training seeds ($\pm$ standard deviation where more than one seed was run) with the 95\,\% bootstrap confidence interval of a single run; precision, recall and F1 are macro-averaged over the seven classes, in percent. ``Trainable'' is the number of parameters updated on the SANAD training split. All LLMs use the primary prompt with deterministic decoding.}
\label{tab:main}
\small
\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%
\begin{tabular}{@{}l r r l r r r@{}}
\toprule
\textbf{System} & \textbf{Trainable} & \textbf{Accuracy} & \textbf{95\,\% CI} & \textbf{Prec.} & \textbf{Rec.} & \textbf{F1} \\
\midrule
""" + body + r"""\bottomrule
\end{tabular}}
\end{table}
""")

# ---- per-class recall
cols = [("Dense 150", S["dense150"]), ("SVM 150", S["svm150"]), ("Fine-tuned", S["ft512"]), ("QLoRA all", L["lora_all"]), ("GPT-4.1", L["gpt41"]), ("Claude 4.5", L["claude45"]), ("Gemini 3.8", L["gemini38"]), ("DeepSeek V4", L["deepseekv4"])]
body = ""
for c in CATS:
    body += c + " & " + " & ".join(f"{v['per_class'][c]['recall']:.3f}" for _, v in cols) + " \\\\\n"
body += "\\midrule\nMacro F1 & " + " & ".join(f"{v['macro_f1']:.3f}" for _, v in cols) + " \\\\\n"
w("tab_perclass.tex", r"""\begin{table}[!ht]
\centering
\caption{Per-class recall on the shared test split (975 articles per class, 973--977 after deduplication). Supervised systems: mean over seeds; LLMs: zero-shot, primary prompt. Religion is the weakest class for every zero-shot LLM and the only class where the gap to the supervised systems exceeds 20 points.}
\label{tab:perclass}
\small
\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%
\begin{tabular}{@{}l rrrr rrrr@{}}
\toprule
 & \multicolumn{4}{c}{\textbf{Supervised / adapted}} & \multicolumn{4}{c}{\textbf{Zero-shot LLMs}} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}
\textbf{Class} & """ + " & ".join(n for n, _ in cols) + r""" \\
\midrule
""" + body + r"""\bottomrule
\end{tabular}}
\end{table}
""")

# ---- learning curve table
ns = ["10", "50", "100", "500", "1000", "all"]
def lcrow(name, d):
    return name + " & " + " & ".join((p(d[n]["acc"]) + (f"$\\pm${p(d[n]['acc_std'])}" if d[n]["n_runs"] > 1 else "")) if d.get(n) else "--" for n in ns) + " \\\\\n"
lora = {"100": L["lora100"], "1000": L["lora1000"], "all": L["lora_all"]}
body = lcrow("Dense head, 150 tokens", LC["dense150"]) + lcrow("RBF-SVM, 150 tokens", LC["svm150"]) + lcrow("Dense head, 512 tokens", LC["dense512"]) + lcrow("Fully fine-tuned CAMeLBERT", LC["finetune512"])
body += "Qwen2.5-7B + QLoRA & -- & -- & " + " & ".join(p(lora[n]["acc"]) for n in ["100"]) + " & -- & " + " & ".join(p(lora[n]["acc"]) for n in ["1000", "all"]) + " \\\\\n"
w("tab_lc.tex", r"""\begin{table}[!ht]
\centering
\caption{Learning curve: test accuracy (\%) as a function of the number of labelled training articles per class (mean $\pm$ standard deviation over three seeds; ``all'' = the full training split, about 4{,}545 per class). Nested subsets drawn with a fixed ranking of the training split. For reference, the best zero-shot LLM reaches 92.8\,\% and the best few-shot condition 95.3\,\% (Table~\ref{tab:prompts}).}
\label{tab:lc}
\small
\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%
\begin{tabular}{@{}l rrrrrr@{}}
\toprule
\textbf{System} & \textbf{10} & \textbf{50} & \textbf{100} & \textbf{500} & \textbf{1{,}000} & \textbf{all} \\
\midrule
""" + body + r"""\bottomrule
\end{tabular}}
\end{table}
""")

# ---- prompt robustness + few-shot + controls (subset 700)
pr = {r["model"]: r for r in R["tables_prompt_robustness"]}
fs = pd.DataFrame(R["tables_fewshot"]); ct = pd.DataFrame(R["tables_controls"])
names = [("openai/gpt-4.1", "GPT-4.1"), ("anthropic/claude-haiku-4.5", "Claude Haiku 4.5"), ("google/gemini-3.8-flash", "Gemini 3.8 Flash"), ("deepseek-ai/DeepSeek-V4-Flash-0731", "DeepSeek V4 Flash"), ("Qwen/Qwen2.5-7B-Instruct", "Qwen2.5-7B-Instruct")]
def get(df, model, **kw):
    d = df[df.model == model]
    for k, v in kw.items(): d = d[d[k] == v]
    return d.iloc[0] if len(d) else None
body = ""
for mid, nm in names:
    r = pr[mid]; k1 = get(fs, mid, shots=1); k2 = get(fs, mid, shots=2)
    rep = get(ct, mid, control="rep2"); nd = get(ct, mid, control="nodateline"); tr = get(ct, mid, control="trunc=100")
    f = lambda x, key="acc_sub": (p(x[key]) if x is not None else "--")
    body += f"{nm} & {p(r['acc_primary_sub'])} & {p(r['acc_mean'])}$\\pm${p(r['acc_std'])} & {p(r['acc_min'])}--{p(r['acc_max'])} & {f(k1)} & {f(k2)} & {f(rep)} & {f(nd)} & {f(tr)} \\\\\n"
w("tab_prompts.tex", r"""\begin{table}[!ht]
\centering
\caption{LLM robustness and in-context conditions on the fixed 700-article evaluation subset (100 per class; identical articles for every model; accuracy in \%). ``Primary'' is the main prompt; the spread is over five semantically equivalent prompts (three English, two Arabic); $k$ exemplars per class are drawn from the training split; ``Repeat'' is a second identical pass of the primary prompt at temperature 0; ``No dateline'' removes the newspaper dateline from every article; ``100 words'' truncates each article to its first 100 words. Qwen2.5-7B zero-/few-shot runs used the full test set and are reported on the same 700 articles.}
\label{tab:prompts}
\small
\setlength{\tabcolsep}{4pt}
\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%
\begin{tabular}{@{}l r r r rr r r r@{}}
\toprule
 & & \multicolumn{2}{c}{\textbf{5 prompts}} & \multicolumn{2}{c}{\textbf{Few-shot}} & \multicolumn{3}{c}{\textbf{Controls}} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-9}
\textbf{Model} & \textbf{Primary} & \textbf{mean$\pm$std} & \textbf{min--max} & $k=1$ & $k=2$ & \textbf{Repeat} & \textbf{No dateline} & \textbf{100 words} \\
\midrule
""" + body + r"""\bottomrule
\end{tabular}}
\end{table}
""")

# ---- mapping audit per model
mb = pd.DataFrame(R["mapping_by_model"]).set_index("model")
lab = {"openai_gpt-4.1": "GPT-4.1", "anthropic_claude-haiku-4.5": "Claude Haiku 4.5", "google_gemini-3.8-flash": "Gemini 3.8 Flash", "deepseek-ai_DeepSeek-V4-Flash-0731": "DeepSeek V4 Flash", "Qwen_Qwen2.5-7B-Instruct": "Qwen2.5-7B (zero-/few-shot)", "lora_all": "Qwen2.5-7B + QLoRA (all)"}
body = ""
for k, nm in lab.items():
    r = mb.loc[k]
    body += f"{nm} & {int(r['runs'])} & {int(r['n']):,} & {p(r['exact'])} & {p(1-r['exact'])} & {p(r['nonexact_max'])} & {100*r['unmapped']:.2f} & {100*r['unmapped_max']:.2f} \\\\\n"
w("tab_mapping.tex", r"""\begin{table}[!ht]
\centering
\caption{Audit of the label-mapping step over all runs of each model (zero-shot prompts, few-shot, controls and repeats). ``Exact'' outputs are the bare label string; ``resolved'' outputs contained extra formatting, an Arabic label name, or a sentence with exactly one label and were mapped by the fixed rules; unmapped (no label, two labels, or an API error) outputs are scored as errors. Percentages of all outputs.}
\label{tab:mapping}
\small
\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%
\begin{tabular}{@{}l r r r r r r r@{}}
\toprule
\textbf{Model} & \textbf{Runs} & \textbf{Outputs} & \textbf{Exact (\%)} & \textbf{Resolved (\%)} & \textbf{Resolved, worst run (\%)} & \textbf{Unmapped (\%)} & \textbf{Unmapped, worst run (\%)} \\
\midrule
""" + body + r"""\bottomrule
\end{tabular}}
\end{table}
""")

# ---- QLoRA budget + timing
B = R["lora_budget"]; order = sorted(B.values(), key=lambda b: b["n_train_examples"])
body = ""
for b, acc in zip(order, [L["lora100"], L["lora1000"], L["lora_all"]]):
    body += f"{b['n_per_class']} & {b['n_train_examples']:,} & {b['epochs']:.0f} & {b['training_tokens_total']/1e6:.1f}\\,M & {b['train_gpu_hours']:.2f} & {b['peak_vram_gb']:.1f} & {p(acc['acc'])} \\\\\n"
w("tab_lora.tex", r"""\begin{table}[!ht]
\centering
\caption{Adaptation budget of the QLoRA baseline (Qwen2.5-7B-Instruct, 4-bit NF4 base, LoRA rank 16, $\alpha=32$, all linear projections, 40.4\,M trainable parameters, 0.53\,\% of the 7.62\,B parameters of the base model; one NVIDIA A100-PCIE-40GB). Accuracy on the full test set.}
\label{tab:lora}
\small
\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%
\begin{tabular}{@{}l r r r r r r@{}}
\toprule
\textbf{Articles / class} & \textbf{Training articles} & \textbf{Epochs} & \textbf{Training tokens} & \textbf{GPU-hours} & \textbf{Peak VRAM (GB)} & \textbf{Accuracy (\%)} \\
\midrule
""" + body + r"""\bottomrule
\end{tabular}}
\end{table}
""")
d, s, f = S["dense150"], S["svm150"], S["ft512"]
w("tab_timing.tex", r"""\begin{table}[!ht]
\centering
\caption{Training and inference cost of the supervised systems on one A100-PCIE-40GB (full training split of 31{,}833 articles; test split of 6{,}822). The frozen-encoder heads share one encoder pass, which dominates their cost and is reported separately; the head numbers are the training and prediction time of the head alone. Wall-clock seconds, mean over seeds.}
\label{tab:timing}
\small
\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%
\begin{tabular}{@{}l r r r@{}}
\toprule
\textbf{Component} & \textbf{Dense head} & \textbf{RBF-SVM (cuML)} & \textbf{Full fine-tuning} \\
\midrule
Encoder forward pass, all 45{,}477 articles (150 tokens) & \multicolumn{2}{c}{""" + f"{d['emb_sec']:.0f}" + r"""} & -- \\
Encoder forward pass, all articles (512 tokens) & \multicolumn{2}{c}{""" + f"{S['dense512']['emb_sec']:.0f}" + r"""} & included \\
Training (150 tokens) & """ + f"{d['train_sec']:.1f}" + r""" & """ + f"{s['train_sec']:.1f}" + r""" & -- \\
Training (512 tokens) & """ + f"{S['dense512']['train_sec']:.1f}" + r""" & """ + f"{S['svm512']['train_sec']:.1f}" + r""" & """ + f"{f['train_sec']:.0f}" + r""" \\
Prediction on the test split (head only / full model) & """ + f"{d['infer_head']:.3f}" + r""" & """ + f"{s['infer_head']:.2f}" + r""" & """ + f"{f['infer_full']:.1f}" + r""" \\
\bottomrule
\end{tabular}}
\end{table}
""")

# ---- split table
sc = R["split"]["split_counts"]
body = "".join(f"{c} & {sc[c]['train']:,} & {sc[c]['val']:,} & {sc[c]['test']:,} \\\\\n" for c in CATS)
tot = {k: sum(sc[c][k] for c in CATS) for k in ("train", "val", "test")}
body += f"\\midrule\nTotal & {tot['train']:,} & {tot['val']:,} & {tot['test']:,} \\\\\n"
w("tab_split.tex", r"""\begin{table}[!ht]
\centering
\caption{The canonical split used by every system: 45{,}477 articles after removing 23 exact duplicates, partitioned 70/15/15 per class so that no near-duplicate cluster (TF--IDF cosine $\geq 0.8$) is divided between partitions.}
\label{tab:split}
\small
\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%
\begin{tabular}{@{}l r r r@{}}
\toprule
\textbf{Class} & \textbf{Train} & \textbf{Validation} & \textbf{Test} \\
\midrule
""" + body + r"""\bottomrule
\end{tabular}}
\end{table}
""")
print("done")
