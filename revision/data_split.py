"""
Canonical data preparation for the revision of
"Arabic News Classification: Lightweight Supervised Models vs. Zero-Shot LLMs".

Produces ONE split file that every system (BERT heads, LoRA LLM, API LLMs) must use:

  revision/splits/sanad_alkhaleej_split_v1.csv
      doc_id            "<Category>/<file>.txt" inside the Kaggle dump (sorted, deterministic)
      category          gold label (7 classes)
      split             train / val / test   (deduplicated, near-duplicate-group-aware, stratified, seed 42)
      split_legacy      train / val / test   (paper-style random stratified split, seed 42, no dedup) for comparison
      dup_group         id of the near-duplicate cluster (TF-IDF cosine >= --near-threshold, union-find)
      dup_group_size    cluster size (1 = no near duplicate anywhere in the corpus)
      max_cos_other     max TF-IDF cosine to any OTHER document in the corpus
      exact_dup_of      doc_id of an earlier exact duplicate (these rows are dropped from `split`, kept in file)
      n_words           whitespace token count
      text_sha1         sha1 of the raw text (integrity check on another machine)
      lc_rank_s0..s2    per-category random rank inside `train` for 3 seeds -> nested learning-curve
                        subsets ("first n per class") and few-shot exemplar pools ("first k per class")

Usage:
  python data_split.py --root /path/to/kagglehub/.../sanad-dataset/versions/1 [--near-threshold 0.8]
  from data_split import load_split; df = load_split(root)   # adds the text column back
"""
import os, re, sys, json, hashlib, argparse, time
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

CATS = ['Culture', 'Finance', 'Medical', 'Politics', 'Religion', 'Sports', 'Tech']
AR_WORD = re.compile(r"[ء-ي]+")
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SPLIT = os.path.join(HERE, "splits", "sanad_alkhaleej_split_v1.csv")
SEED = 42


def default_root():
    env = os.environ.get("SANAD_ROOT")
    if env:
        return env
    cand = os.path.expanduser("~/.cache/kagglehub/datasets/haithemhermessi/sanad-dataset/versions/1")
    if os.path.isdir(cand):
        return cand
    import kagglehub  # noqa
    return kagglehub.dataset_download("haithemhermessi/sanad-dataset")


def load_corpus(root):
    """Deterministic load: categories and files sorted (the paper's code used unsorted os.listdir)."""
    rows = []
    for cat in sorted(os.listdir(root)):
        p = os.path.join(root, cat)
        if not os.path.isdir(p) or cat not in CATS:
            continue
        files = sorted((f for f in os.listdir(p) if f.endswith(".txt")), key=lambda s: (len(s), s))
        for fn in files:
            with open(os.path.join(p, fn), encoding="utf-8") as f:
                rows.append((f"{cat}/{fn}", cat, f.read()))
    df = pd.DataFrame(rows, columns=["doc_id", "category", "text"])
    assert len(df) > 0, f"no data under {root}"
    return df


def dedup_key(text):
    return " ".join(AR_WORD.findall(text))


def near_dup_groups(texts, threshold, chunk=1000, log=print):
    """Union-find clusters over pairs with TF-IDF (word 1-2gram) cosine >= threshold."""
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=500000, sublinear_tf=True,
                          token_pattern=r"[ء-ي]{2,}", dtype=np.float32)
    A = vec.fit_transform(texts)
    AT = A.T.tocsc()
    n = A.shape[0]
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    maxsim = np.zeros(n, dtype=np.float32)
    n_pairs = 0
    t0 = time.time()
    for i in range(0, n, chunk):
        S = (A[i:i + chunk] @ AT).toarray()
        for r in range(S.shape[0]):
            S[r, i + r] = 0.0  # ignore self-similarity
        maxsim[i:i + chunk] = S.max(1)
        rows, cols = np.where(S >= threshold)
        for r, c in zip(rows, cols):
            a, b = i + r, int(c)
            if a < b:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
                n_pairs += 1
        if (i // chunk) % 10 == 0:
            log(f"  near-dup scan {i + S.shape[0]}/{n}  ({time.time() - t0:.0f}s)")
    groups = np.array([find(i) for i in range(n)])
    return groups, maxsim, n_pairs


def group_aware_split(df, seed=SEED, targets=(0.70, 0.15, 0.15)):
    """Stratified 70/15/15 split that never separates a near-duplicate cluster.

    Clusters (dup_group) are allocated greedily, largest first, to the split that is most below its
    per-category target (so the few very large templated series, e.g. daily market reports, do not
    unbalance the test set the way a plain stratified split over clusters would). Singletons fill
    the remainder, giving ~exact 70/15/15 per category. Ties are broken with the fixed seed.
    """
    rng = np.random.RandomState(seed)
    names = ["train", "val", "test"]
    assign = {}
    for cat in CATS:
        sub = df[df.category == cat]
        groups = sub.groupby("dup_group").size()
        order = groups.sample(frac=1.0, random_state=rng).sort_values(ascending=False, kind="stable")
        total = float(groups.sum())
        fill = np.zeros(3)
        for g, sz in order.items():
            if g in assign:  # mixed-category cluster already placed via another category
                fill[names.index(assign[g])] += sz
                continue
            deficit = np.array(targets) * total - fill
            j = int(np.argmax(deficit))
            assign[g] = names[j]
            fill[j] += sz
    return df["dup_group"].map(assign)


def legacy_split(df, seed=SEED):
    """Paper-style: random stratified 70/15/15 on documents, no dedup (for the comparison table)."""
    idx = np.arange(len(df))
    tr, tmp = train_test_split(idx, test_size=0.30, stratify=df["category"], random_state=seed)
    va, te = train_test_split(tmp, test_size=0.50, stratify=df["category"].iloc[tmp], random_state=seed)
    s = pd.Series("train", index=df.index)
    s.iloc[va] = "val"
    s.iloc[te] = "test"
    return s


def add_learning_curve_ranks(df, n_seeds=3):
    for s in range(n_seeds):
        rng = np.random.RandomState(SEED + s)
        col = np.full(len(df), -1, dtype=int)
        tr = df["split"] == "train"
        for cat in CATS:
            ix = np.where(tr & (df["category"] == cat))[0]
            col[ix] = rng.permutation(len(ix))
        df[f"lc_rank_s{s}"] = col
    return df


def build(root, near_threshold=0.8, out=DEFAULT_SPLIT, log=print):
    t0 = time.time()
    df = load_corpus(root)
    log(f"loaded {len(df)} docs from {root} ({time.time() - t0:.0f}s)")
    df["n_words"] = df["text"].str.split().str.len()
    df["text_sha1"] = [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in df["text"]]
    key = df["text"].map(dedup_key)
    first = {}
    exact_dup_of = []
    for d, k in zip(df["doc_id"], key):
        exact_dup_of.append(first.get(k, ""))
        first.setdefault(k, d)
    df["exact_dup_of"] = exact_dup_of
    keep = df["exact_dup_of"] == ""
    log(f"exact duplicates dropped: {(~keep).sum()}")

    groups, maxsim, n_pairs = near_dup_groups(df.loc[keep, "text"].values, near_threshold, log=log)
    df.loc[keep, "dup_group"] = groups
    df.loc[keep, "max_cos_other"] = maxsim
    df["dup_group"] = df["dup_group"].fillna(-1).astype(int)
    sizes = df.loc[keep].groupby("dup_group")["doc_id"].transform("size")
    df.loc[keep, "dup_group_size"] = sizes
    df["dup_group_size"] = df["dup_group_size"].fillna(0).astype(int)
    log(f"near-duplicate pairs (cos>={near_threshold}): {n_pairs}; docs in a cluster of size>1: {(df['dup_group_size'] > 1).sum()}")

    df["split"] = ""
    df.loc[keep, "split"] = group_aware_split(df.loc[keep]).values
    df["split_legacy"] = legacy_split(df).values
    df = add_learning_curve_ranks(df)

    cols = ["doc_id", "category", "split", "split_legacy", "dup_group", "dup_group_size", "max_cos_other",
            "exact_dup_of", "n_words", "text_sha1", "lc_rank_s0", "lc_rank_s1", "lc_rank_s2"]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df[cols].to_csv(out, index=False)
    summary = {
        "root": root, "n_docs": int(len(df)), "exact_dups_dropped": int((~keep).sum()),
        "near_threshold": near_threshold, "near_dup_pairs": int(n_pairs),
        "docs_in_clusters": int((df["dup_group_size"] > 1).sum()),
        "split_counts": df[df.split != ""].groupby(["split", "category"]).size().unstack().to_dict(),
        "legacy_split_counts": df.groupby(["split_legacy", "category"]).size().unstack().to_dict(),
        "test_docs_with_cos>=0.5_elsewhere": int(((df.split == "test") & (df.max_cos_other >= 0.5)).sum()),
        "seed": SEED, "sha1_of_split_file": hashlib.sha1(open(out, "rb").read()).hexdigest(),
    }
    with open(out.replace(".csv", "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log(json.dumps(summary, indent=2))
    log(f"wrote {out} ({time.time() - t0:.0f}s)")
    return df


def load_split(root=None, split_csv=DEFAULT_SPLIT, verify=True):
    """Return the split table joined with the text column (text is re-read from the Kaggle dump)."""
    root = root or default_root()
    sp = pd.read_csv(split_csv, dtype={"exact_dup_of": str}).fillna({"exact_dup_of": ""})
    corpus = load_corpus(root)
    df = sp.merge(corpus[["doc_id", "text"]], on="doc_id", how="left")
    assert df["text"].notna().all(), "some doc_ids from the split file are missing in the corpus"
    if verify:
        bad = (df["text"].map(lambda t: hashlib.sha1(t.encode("utf-8")).hexdigest()) != df["text_sha1"]).sum()
        assert bad == 0, f"{bad} documents differ from the split file's sha1 (different dataset version?)"
    return df


def learning_curve_subset(df, n_per_class, seed_id=0):
    """Nested subsets of `train`: the first n per class by lc_rank (n=None -> full train)."""
    tr = df[df.split == "train"]
    if n_per_class is None:
        return tr
    return tr[tr[f"lc_rank_s{seed_id}"] < n_per_class]


def fewshot_exemplars(df, k_per_class, seed_id=0, max_words=120):
    """k exemplars per class from train (disjoint from test by construction), truncated for prompt length."""
    ex = learning_curve_subset(df, k_per_class, seed_id).copy()
    ex["text"] = ex["text"].map(lambda t: " ".join(t.split()[:max_words]))
    return ex[["doc_id", "category", "text"]].reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--near-threshold", type=float, default=0.8)
    ap.add_argument("--out", default=DEFAULT_SPLIT)
    a = ap.parse_args()
    build(a.root or default_root(), a.near_threshold, a.out)
