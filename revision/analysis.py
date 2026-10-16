"""
Aggregate every run (supervised .pred.csv / runs.jsonl and LLM .jsonl / .summary.json) into the tables the
revised paper needs, with bootstrap CIs and paired significance tests on the shared test split.

  python analysis.py --results-dir results                      # writes results/tables/*.csv and prints them
  python analysis.py --mcnemar results/supervised/dense__...pred.csv results/openai__gpt-4.1-2025-04-14__en_v1__k0.jsonl
  python analysis.py --sim-bins results/supervised/dense__...pred.csv   # accuracy by max-cosine-to-train bin (leakage check)

Tables produced
  main_table.csv          one row per system/config: acc, macro-F1, 95% bootstrap CI, per-class recall, unmapped rate, served model
  learning_curve.csv      mean +- std over seeds of acc/macro-F1 per (head, n_per_class)
  prompt_robustness.csv   per LLM: zero-shot acc per prompt variant, mean +- std, min-max, and few-shot gains
  mapping_audit.csv       per LLM run: share of outputs by mapping method, unmapped/ambiguous/empty rates, finish reasons
  pairwise_mcnemar.csv    McNemar exact test + paired bootstrap CI of the accuracy difference for selected pairs
"""
import os, sys, json, glob, argparse, itertools, re
import numpy as np, pandas as pd
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data_split import CATS  # noqa: E402


def load_preds(path):
    if path.endswith(".jsonl"):
        d = pd.DataFrame([json.loads(l) for l in open(path)]).drop_duplicates("doc_id", keep="last")
    else:
        d = pd.read_csv(path)
    return d[["doc_id", "gold", "pred"]].set_index("doc_id")


def bootstrap(y, p, n=2000, seed=0):
    rng = np.random.RandomState(seed)
    from sklearn.metrics import f1_score
    y, p = np.asarray(y), np.asarray(p)
    acc, f1 = [], []
    for _ in range(n):
        ix = rng.randint(0, len(y), len(y))
        acc.append((y[ix] == p[ix]).mean()); f1.append(f1_score(y[ix], p[ix], labels=CATS, average="macro", zero_division=0))
    return np.percentile(acc, [2.5, 97.5]), np.percentile(f1, [2.5, 97.5])


def mcnemar(path_a, path_b, n_boot=2000):
    a, b = load_preds(path_a), load_preds(path_b)
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    ca, cb = (a.pred == a.gold).values, (b.pred == b.gold).values
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())  # a right/b wrong, a wrong/b right
    p = binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if (n01 + n10) else 1.0
    rng = np.random.RandomState(0)
    diffs = []
    for _ in range(n_boot):
        ix = rng.randint(0, len(common), len(common))
        diffs.append(ca[ix].mean() - cb[ix].mean())
    return dict(a=os.path.basename(path_a), b=os.path.basename(path_b), n_common=int(len(common)), acc_a=float(ca.mean()), acc_b=float(cb.mean()),
                diff=float(ca.mean() - cb.mean()), diff_ci95=[float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
                a_right_b_wrong=n01, a_wrong_b_right=n10, mcnemar_exact_p=float(p))


def sim_bins(path, split_csv=os.path.join(HERE, "splits", "sanad_alkhaleej_split_v1.csv")):
    sp = pd.read_csv(split_csv).set_index("doc_id")
    d = load_preds(path).join(sp[["max_cos_other", "dup_group_size", "category"]])
    d["correct"] = d.pred == d.gold
    bins = [0, .2, .3, .4, .5, .6, .7, .8, .9, 1.01]
    d["bin"] = pd.cut(d.max_cos_other, bins, right=False)
    t = d.groupby("bin", observed=True).agg(n=("correct", "size"), acc=("correct", "mean"))
    t["share"] = t.n / t.n.sum()
    return t


def main_table(results_dir):
    rows = []
    for l in open(os.path.join(results_dir, "supervised", "runs.jsonl")) if os.path.exists(os.path.join(results_dir, "supervised", "runs.jsonl")) else []:
        r = json.loads(l)
        rows.append(dict(system=f"BERT-{r['head']}", model=r["model_id"], config=f"len{r['max_len']} n={r['n_per_class']} {r['split_col']} dateline={r['strip_dateline']}",
                         prompt="", shots="", n_train=r["n_train_docs"], seed=r["seed"], acc=r["accuracy"], acc_lo=r["acc_ci95"][0], acc_hi=r["acc_ci95"][1],
                         macro_f1=r["macro_f1"], f1_lo=r["macro_f1_ci95"][0], f1_hi=r["macro_f1_ci95"][1],
                         **{f"recall_{c}": r["per_class"][c]["recall"] for c in CATS}, unmapped_rate=0.0, train_sec=r.get("train_sec"),
                         trainable_params=r.get("trainable_params"), served_model="", file=r["run"]))
    for f in glob.glob(os.path.join(results_dir, "*.summary.json")):
        r = json.load(open(f))
        meta_f = f.replace(".summary.json", ".meta.json")
        m = json.load(open(meta_f)) if os.path.exists(meta_f) else {}
        d = load_preds(f.replace(".summary.json", ".jsonl"))
        (alo, ahi), (flo, fhi) = bootstrap(d.gold, d.pred)
        rows.append(dict(system=f"LLM-{m.get('provider', '?')}", model=m.get("model", "?"), config=f"T={m.get('temperature')} max_tok={m.get('max_tokens')}" + (f" trunc={m['truncate_words']}" if m.get("truncate_words") else ""),
                         prompt=m.get("prompt"), shots=m.get("shots"), n_train=(m.get("shots", 0) or 0) * 7, seed=m.get("fewshot_seed"),
                         acc=r["strict"]["accuracy"], acc_lo=alo, acc_hi=ahi, macro_f1=r["strict"]["macro_f1"], f1_lo=flo, f1_hi=fhi,
                         **{f"recall_{c}": r["strict"]["per_class"][c]["recall"] for c in CATS}, unmapped_rate=r["unmapped_rate"], train_sec=None,
                         trainable_params=0, served_model=";".join(r["served_models"]), file=os.path.basename(f)))
    return pd.DataFrame(rows).sort_values(["system", "model", "config", "prompt", "shots", "seed"])


def learning_curve(main):
    s = main[main.system.str.startswith("BERT")].copy()
    s["n_per_class"] = s.config.str.extract(r"n=(\w+)")[0]
    g = s.groupby(["system", "config"]).agg(n_train=("n_train", "first"), acc_mean=("acc", "mean"), acc_std=("acc", "std"), f1_mean=("macro_f1", "mean"),
                                           f1_std=("macro_f1", "std"), seeds=("seed", "count"))
    return g.reset_index()


def prompt_robustness(main):
    l = main[main.system.str.startswith("LLM")].copy()
    zs = l[l.shots == 0].groupby(["system", "model"]).agg(n_prompts=("prompt", "nunique"), acc_mean=("acc", "mean"), acc_std=("acc", "std"), acc_min=("acc", "min"),
                                                          acc_max=("acc", "max"), f1_mean=("macro_f1", "mean"), f1_std=("macro_f1", "std"),
                                                          religion_recall_mean=("recall_Religion", "mean"), religion_recall_min=("recall_Religion", "min"))
    fs = l[l.shots > 0].groupby(["system", "model", "shots"]).agg(acc_mean=("acc", "mean"), acc_std=("acc", "std")).reset_index()
    return zs.reset_index(), fs


def mapping_audit(results_dir):
    rows = []
    for f in glob.glob(os.path.join(results_dir, "*.summary.json")):
        r = json.load(open(f)); mm = r["mapping_methods"]; n = r["n"]
        rows.append(dict(run=os.path.basename(f).replace(".summary.json", ""), n=n, **{f"share_{k}": v / n for k, v in mm.items()},
                         unmapped_rate=r["unmapped_rate"], non_exact_rate=r["non_exact_rate"], api_error_rate=r["error_rate_api"],
                         strict_acc=r["strict"]["accuracy"], lenient_acc=r["lenient"]["accuracy"], finish=json.dumps(r["finish_reasons"]),
                         examples=" | ".join(r["unmapped_examples"][:8])))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=os.path.join(HERE, "results"))
    ap.add_argument("--mcnemar", nargs=2)
    ap.add_argument("--sim-bins")
    ap.add_argument("--pairs", nargs="*", default=[], help="pairs 'fileA,fileB' for pairwise_mcnemar.csv")
    a = ap.parse_args()
    if a.mcnemar:
        print(json.dumps(mcnemar(*a.mcnemar), indent=2)); sys.exit()
    if a.sim_bins:
        print(sim_bins(a.sim_bins)); sys.exit()
    out = os.path.join(a.results_dir, "tables"); os.makedirs(out, exist_ok=True)
    main = main_table(a.results_dir); main.to_csv(os.path.join(out, "main_table.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(main[["system", "model", "config", "prompt", "shots", "seed", "acc", "acc_lo", "acc_hi", "macro_f1", "recall_Religion", "unmapped_rate"]].to_string(index=False))
        lc = learning_curve(main); lc.to_csv(os.path.join(out, "learning_curve.csv"), index=False); print("\n", lc.to_string(index=False))
        zs, fs = prompt_robustness(main); zs.to_csv(os.path.join(out, "prompt_robustness.csv"), index=False); fs.to_csv(os.path.join(out, "fewshot.csv"), index=False)
        print("\n", zs.to_string(index=False)); print("\n", fs.to_string(index=False))
        ma = mapping_audit(a.results_dir); ma.to_csv(os.path.join(out, "mapping_audit.csv"), index=False); print("\n", ma.drop(columns=["examples"]).to_string(index=False))
    if a.pairs:
        pd.DataFrame([mcnemar(*p.split(",")) for p in a.pairs]).to_csv(os.path.join(out, "pairwise_mcnemar.csv"), index=False)
