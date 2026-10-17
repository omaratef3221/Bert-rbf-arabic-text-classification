"""Residual near-duplicate exposure of the canonical test split.

For every test article, the maximum TF-IDF cosine (same vectoriser as data_split.py) to any
article in train or validation is computed; the script then reports, for several thresholds,
how many test articles have such a neighbour and how accurate each system is on them versus
the rest.  Output: results/tables/leakage_cos.csv and the per-article results/tables/test_max_cos_trainval.csv.

    python leakage_check.py [--root /path/to/kagglehub/.../sanad-dataset/versions/1]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from data_split import DEFAULT_SPLIT, default_root, load_corpus

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SYSTEMS = {
    "dense150_s0": os.path.join(RES, "supervised", "dense__CAMeL-Lab_bert-base-arabic-camelbert-msa__len150__nall__s0__split__dl-none.pred.csv"),
    "svm150_s0": os.path.join(RES, "supervised", "svm__CAMeL-Lab_bert-base-arabic-camelbert-msa__len150__nall__s0__split__dl-none.pred.csv"),
    "finetune512_s0": os.path.join(RES, "supervised", "finetune__CAMeL-Lab_bert-base-arabic-camelbert-msa__len512__nall__s0__split__dl-none.pred.csv"),
    "qlora_all": os.path.join(RES, "vllm__lora_all__en_v2__k0.jsonl"),
    "gpt41_zeroshot": os.path.join(RES, "openrouter__openai_gpt-4.1__en_v2__k0.jsonl"),
}


def load_preds(path):
    if path.endswith(".jsonl"):
        d = pd.DataFrame([json.loads(l) for l in open(path)]).drop_duplicates("doc_id", keep="last")
    else:
        d = pd.read_csv(path)
    return d[["doc_id", "gold", "pred"]].set_index("doc_id")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--split-csv", default=DEFAULT_SPLIT)
    ap.add_argument("--thresholds", default="0.5,0.6,0.7,0.8")
    a = ap.parse_args()
    sp = pd.read_csv(a.split_csv, dtype={"exact_dup_of": str}).fillna({"exact_dup_of": ""})
    sp = sp[sp.split.isin(["train", "val", "test"])].reset_index(drop=True)
    corpus = load_corpus(a.root or default_root())
    df = sp.merge(corpus[["doc_id", "text"]], on="doc_id", how="left")
    assert df.text.notna().all(), "corpus and split do not match"
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=500000, sublinear_tf=True,
                          token_pattern=r"[ء-ي]{2,}", dtype=np.float32)
    A = vec.fit_transform(df.text)
    is_test = (df.split == "test").values
    pools = {"trainval": df.split.isin(["train", "val"]).values, "train": (df.split == "train").values}
    T = A[is_test]
    out = pd.DataFrame({"doc_id": df.doc_id[is_test].values, "category": df.category[is_test].values})
    for name, mask in pools.items():
        P = A[mask].T.tocsc()
        mx = np.zeros(T.shape[0], dtype=np.float32)
        for i in range(0, T.shape[0], 400):
            mx[i:i + 400] = (T[i:i + 400] @ P).max(axis=1).toarray().ravel()
        out[f"max_cos_{name}"] = mx
    os.makedirs(os.path.join(RES, "tables"), exist_ok=True)
    out.to_csv(os.path.join(RES, "tables", "test_max_cos_trainval.csv"), index=False)
    out = out.set_index("doc_id")
    preds = {k: load_preds(p) for k, p in SYSTEMS.items() if os.path.exists(p)}
    rows = []
    for thr in [float(t) for t in a.thresholds.split(",")]:
        for pool in pools:
            near = out[f"max_cos_{pool}"] >= thr
            row = dict(threshold=thr, pool=pool, n_test=len(out), n_near=int(near.sum()), share_near=float(near.mean()))
            for k, d in preds.items():
                c = (d.pred == d.gold).reindex(out.index)
                row[f"{k}_acc_near"] = float(c[near].mean()) if near.any() else float("nan")
                row[f"{k}_acc_rest"] = float(c[~near].mean())
                row[f"{k}_acc_all"] = float(c.mean())
            rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(RES, "tables", "leakage_cos.csv"), index=False)
    pd.set_option("display.width", 250)
    print(t.round(4).to_string(index=False))
    for k, d in preds.items():
        print(k, "overall acc", round(float((d.pred == d.gold).mean()), 4), "n", len(d))


if __name__ == "__main__":
    main()
