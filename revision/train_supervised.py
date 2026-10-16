"""
Supervised baselines on the canonical split (revision/data_split.py), with everything the reviewers asked for:

  --head dense      frozen CAMeLBERT [CLS] embedding -> Linear(768,384)-ReLU-Dropout(.25)-Linear(384,7)   (paper's BERT-Dense)
  --head svm        frozen CAMeLBERT [CLS] embedding -> RBF-SVM (C=9, gamma=scale; cuML if available)      (paper's BERT-SVM)
  --head finetune   full fine-tuning of CAMeLBERT + linear classifier (all ~109M params trainable)         (new, task-adapted encoder)

  --n-per-class 10 50 100 500 1000 all   nested learning-curve subsets (data_split.lc_rank_s{seed})
  --seeds 0 1 2                          subset seed AND torch seed -> mean +- std
  --max-len 150 | 512                    input truncation in wordpieces (paper used 150, never stated)
  --strip-dateline none|test_only|train_and_test   lexical intervention: remove the "<city>: «الخليج»" dateline
  --split-col split | split_legacy       deduplicated group-aware split (default) vs paper-style random split

Every run writes: results/supervised/<run>.pred.csv (doc_id, gold, pred) for McNemar / bootstrap against LLM runs,
and appends one JSON line with metrics, 95% bootstrap CIs, timings, trainable-parameter counts to results/supervised/runs.jsonl.
Embeddings for the frozen heads are cached in results/emb_cache/ so the learning curve re-uses one encoder pass.

Examples (A100):
  python train_supervised.py --head dense --n-per-class all --seeds 0 1 2
  python train_supervised.py --head dense --n-per-class 10 50 100 500 1000 all --seeds 0 1 2
  python train_supervised.py --head svm   --n-per-class 10 50 100 500 1000 all --seeds 0 1 2
  python train_supervised.py --head finetune --n-per-class 100 1000 all --seeds 0 --max-len 512 --epochs 3 --lr 2e-5
  python train_supervised.py --head dense --n-per-class all --strip-dateline test_only
"""
import os, re, sys, json, time, argparse, hashlib, datetime, random
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data_split import load_split, learning_curve_subset, CATS  # noqa: E402

LAB2ID = {c: i for i, c in enumerate(CATS)}
DATELINE = re.compile(r"^\s*[ء-ي ]{2,20}?\s*[:\-–]?\s*[«\"']?\s*الخليج\s*[»\"']?\s*[:\-–]?\s*")


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def strip_dateline(t):
    return DATELINE.sub("", t, count=1)


def bootstrap_ci(y, p, n_boot=1000, seed=0):
    rng = np.random.RandomState(seed)
    y, p = np.asarray(y), np.asarray(p)
    accs, f1s = [], []
    for _ in range(n_boot):
        ix = rng.randint(0, len(y), len(y))
        accs.append((y[ix] == p[ix]).mean())
        f1s.append(f1_score(y[ix], p[ix], average="macro"))
    return dict(acc_ci95=[float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))],
                macro_f1_ci95=[float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))])


def metrics(y, p):
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=list(range(7)), zero_division=0)
    return dict(accuracy=float(accuracy_score(y, p)), macro_f1=float(f1_score(y, p, average="macro")),
                per_class={c: dict(precision=float(a), recall=float(b), f1=float(d), support=int(s)) for c, a, b, d, s in zip(CATS, pr, rc, f1, sup)},
                confusion=confusion_matrix(y, p, labels=list(range(7))).tolist(), **bootstrap_ci(y, p))


# ----------------------------------------------------------------------------- frozen embeddings
@torch.no_grad()
def embed(model_id, texts, max_len, batch_size, device, cache_key):
    os.makedirs(os.path.join(HERE, "results", "emb_cache"), exist_ok=True)
    path = os.path.join(HERE, "results", "emb_cache", f"{cache_key}.npy")
    if os.path.exists(path):
        return np.load(path), None
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(model_id)
    enc = AutoModel.from_pretrained(model_id).to(device).eval()
    out = []
    t0 = time.time()
    for i in range(0, len(texts), batch_size):
        b = tok(list(texts[i:i + batch_size]), max_length=max_len, truncation=True, padding=True, return_tensors="pt").to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
            h = enc(**b).last_hidden_state[:, 0, :]
        out.append(h.float().cpu().numpy())
    X = np.concatenate(out)
    sec = time.time() - t0
    np.save(path, X)
    return X, sec


class DenseHead(nn.Module):
    def __init__(self, n_classes=7):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(768, 384), nn.ReLU(), nn.Dropout(0.25), nn.Linear(384, n_classes))

    def forward(self, x):
        return self.net(x)


def train_dense(Xtr, ytr, Xva, yva, args, device):
    set_seed(args.torch_seed)
    m = DenseHead().to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lossf = nn.CrossEntropyLoss()
    tr = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)), batch_size=args.batch_size, shuffle=True)
    Xva_t, yva_t = torch.tensor(Xva).to(device), torch.tensor(yva).to(device)
    best, best_state, bad, hist = 1e9, None, 0, []
    t0 = time.time()
    for ep in range(args.epochs):
        m.train()
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = lossf(m(xb), yb); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vl = float(lossf(m(Xva_t), yva_t)); va = float((m(Xva_t).argmax(1) == yva_t).float().mean())
        hist.append(dict(epoch=ep + 1, val_loss=vl, val_acc=va))
        if vl < best - 1e-4:
            best, bad, best_state = vl, 0, {k: v.detach().clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    m.load_state_dict(best_state)
    train_sec = time.time() - t0
    n_params = sum(p.numel() for p in m.parameters())
    return (lambda X: m.eval()(torch.tensor(X).to(device)).argmax(1).cpu().numpy()), dict(train_sec=train_sec, epochs_run=len(hist), history=hist, trainable_params=n_params)


def train_svm(Xtr, ytr, args):
    try:
        from cuml.svm import SVC as SVC_  # GPU
        backend = "cuml"
    except Exception:
        from sklearn.svm import SVC as SVC_
        backend = "sklearn"
    clf = SVC_(kernel="rbf", C=args.svm_C, gamma="scale")
    t0 = time.time(); clf.fit(Xtr, ytr); train_sec = time.time() - t0
    n_sv = int(getattr(clf, "n_support_", np.array([0])).sum()) if hasattr(clf, "n_support_") else None
    return (lambda X: np.asarray(clf.predict(X)).astype(int)), dict(train_sec=train_sec, backend=backend, n_support_vectors=n_sv, trainable_params=None)


# ----------------------------------------------------------------------------- full fine-tuning
def train_finetune(tr_df, va_df, te_df, args, device):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    set_seed(args.torch_seed)
    tok = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_id, num_labels=7).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def loader(df, shuffle):
        enc = tok(df.text.tolist(), max_length=args.max_len, truncation=True, padding="max_length", return_tensors="pt")
        ds = TensorDataset(enc["input_ids"], enc["attention_mask"], torch.tensor(df.category.map(LAB2ID).values))
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle)

    tr, va, te = loader(tr_df, True), loader(va_df, False), loader(te_df, False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = len(tr) * args.epochs
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))
    lossf = nn.CrossEntropyLoss()

    @torch.no_grad()
    def evaluate(dl):
        model.eval(); preds, losses = [], []
        for ids, am, y in dl:
            ids, am, y = ids.to(device), am.to(device), y.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
                lg = model(input_ids=ids, attention_mask=am).logits
            losses.append(float(lossf(lg.float(), y))); preds.append(lg.argmax(1).cpu().numpy())
        return np.concatenate(preds), float(np.mean(losses))

    best, best_state, bad, hist = 1e9, None, 0, []
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        for ids, am, y in tr:
            ids, am, y = ids.to(device), am.to(device), y.to(device)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
                loss = lossf(model(input_ids=ids, attention_mask=am).logits.float(), y)
            scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step()
        pv, vl = evaluate(va)
        hist.append(dict(epoch=ep + 1, val_loss=vl, val_acc=float((pv == va_df.category.map(LAB2ID).values).mean())))
        print(f"    epoch {ep + 1}: val_loss={vl:.4f} val_acc={hist[-1]['val_acc']:.4f}")
        if vl < best - 1e-4:
            best, bad, best_state = vl, 0, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    train_sec = time.time() - t0
    model.load_state_dict(best_state)
    t0 = time.time(); pt, _ = evaluate(te); infer_sec = time.time() - t0
    return pt, dict(train_sec=train_sec, infer_sec_incl_encoder=infer_sec, epochs_run=len(hist), history=hist, trainable_params=n_params)


def wandb_log_run(args, run_name, rec):
    """Opt-in Weights & Biases logging (--wandb-project): one W&B run per (head, n, seed), named like the results files."""
    import wandb
    flat = {k: v for k, v in rec.items() if isinstance(v, (int, float, str, bool)) or v is None}
    cfg_keys = ("head", "model_id", "max_len", "n_per_class", "n_train_docs", "seed", "split_col", "strip_dateline", "truncate_words",
                "lr", "epochs", "patience", "batch_size", "device", "gpu", "trainable_params", "backend")
    w = wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=run_name, job_type="supervised", tags=["revision", args.head],
                   group=f"{args.head}_len{args.max_len}_{args.split_col}", config={k: flat[k] for k in cfg_keys if k in flat}, reinit=True)
    for h in rec.get("history") or []:
        w.log({"epoch": h["epoch"], "val_loss": h["val_loss"], "val_acc": h["val_acc"]})
    summary = {k: v for k, v in flat.items() if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in cfg_keys}
    summary.update({f"{c}/{m}": rec["per_class"][c][m] for c in CATS for m in ("precision", "recall", "f1")})
    summary.update(acc_ci_lo=rec["acc_ci95"][0], acc_ci_hi=rec["acc_ci95"][1], macro_f1_ci_lo=rec["macro_f1_ci95"][0], macro_f1_ci_hi=rec["macro_f1_ci95"][1])
    w.log(summary)
    w.log({"confusion": wandb.Table(columns=["true"] + CATS, data=[[c] + row for c, row in zip(CATS, rec["confusion"])])})
    w.finish()


# ----------------------------------------------------------------------------- main
def main(args):
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    df = load_split(args.root, args.split_csv)
    df = df[df[args.split_col].isin(["train", "val", "test"])].copy()
    df["split"] = df[args.split_col]  # learning_curve_subset uses .split
    if args.smoke:  # tiny stratified subset of every split for an end-to-end pipeline check
        df = df.groupby(["split", "category"], group_keys=False).head(args.smoke).copy()
        tr_mask = df.split == "train"
        for s in range(3):
            col = f"lc_rank_s{s}"
            df.loc[tr_mask, col] = (df[tr_mask].groupby("category")[col].rank(method="first") - 1).astype(int)
    if args.strip_dateline == "train_and_test":
        df["text"] = df["text"].map(strip_dateline)
    elif args.strip_dateline == "test_only":
        df.loc[df.split == "test", "text"] = df.loc[df.split == "test", "text"].map(strip_dateline)
    if args.truncate_words:
        df["text"] = df["text"].map(lambda t: " ".join(t.split()[: args.truncate_words]))
    df["_row"] = np.arange(len(df))
    va_df, te_df = df[df.split == "val"], df[df.split == "test"]
    os.makedirs(os.path.join(HERE, "results", "supervised"), exist_ok=True)
    runs_path = os.path.join(HERE, "results", "supervised", "runs.jsonl")

    emb_sec = None
    if args.head in ("dense", "svm"):
        key = hashlib.sha1(f"{args.model_id}|{args.max_len}|{args.split_col}|{args.strip_dateline}|{args.truncate_words}|{len(df)}".encode()).hexdigest()[:12]
        X_all, emb_sec = embed(args.model_id, df.text.values, args.max_len, args.batch_size, device, key)

    for n in args.n_per_class:
        n_pc = None if n == "all" else int(n)
        for seed in args.seeds:
            args.torch_seed = 1000 + seed
            tr_df = learning_curve_subset(df, n_pc, seed)
            run = f"{args.head}__{re.sub(r'[^A-Za-z0-9.-]+', '_', args.model_id)}__len{args.max_len}__n{n}__s{seed}__{args.split_col}__dl-{args.strip_dateline}" + (f"__trunc{args.truncate_words}" if args.truncate_words else "")
            print(f"\n=== {run}  (train docs: {len(tr_df)})")
            ytr, yva, yte = [d.category.map(LAB2ID).values for d in (tr_df, va_df, te_df)]
            if args.head == "finetune":
                pred, info = train_finetune(tr_df, va_df, te_df, args, device)
            else:
                Xtr, Xva, Xte = X_all[tr_df._row.values], X_all[va_df._row.values], X_all[te_df._row.values]
                if args.head == "dense":
                    predict, info = train_dense(Xtr, ytr, Xva, yva, args, device)
                else:
                    predict, info = train_svm(Xtr, ytr, args)
                t0 = time.time(); pred = predict(Xte); info["infer_sec_head_only"] = time.time() - t0
                info["encoder_embedding_sec_all_docs"] = emb_sec
            m = metrics(yte, pred)
            rec = dict(run=run, head=args.head, model_id=args.model_id, max_len=args.max_len, n_per_class=n, n_train_docs=int(len(tr_df)), seed=seed,
                       split_col=args.split_col, strip_dateline=args.strip_dateline, truncate_words=args.truncate_words, lr=args.lr, epochs=args.epochs,
                       patience=args.patience, batch_size=args.batch_size, device=device, gpu=(torch.cuda.get_device_name(0) if device.startswith("cuda") else None),
                       ts=datetime.datetime.now(datetime.timezone.utc).isoformat(), **info, **m)
            pd.DataFrame(dict(doc_id=te_df.doc_id.values, gold=te_df.category.values, pred=[CATS[i] for i in pred])).to_csv(
                os.path.join(HERE, "results", "supervised", f"{run}.pred.csv"), index=False)
            with open(runs_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"    acc={m['accuracy']:.4f} CI{m['acc_ci95']}  macroF1={m['macro_f1']:.4f}  train={info.get('train_sec', 0):.1f}s  "
                  f"Religion recall={m['per_class']['Religion']['recall']:.3f}")
            if args.wandb_project:
                wandb_log_run(args, run, rec)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", choices=["dense", "svm", "finetune"], required=True)
    ap.add_argument("--model-id", default="CAMeL-Lab/bert-base-arabic-camelbert-msa")
    ap.add_argument("--n-per-class", nargs="+", default=["all"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--max-len", type=int, default=150)
    ap.add_argument("--split-col", default="split", choices=["split", "split_legacy"])
    ap.add_argument("--split-csv", default=os.path.join(HERE, "splits", "sanad_alkhaleej_split_v1.csv"))
    ap.add_argument("--root", default=None)
    ap.add_argument("--strip-dateline", default="none", choices=["none", "test_only", "train_and_test"])
    ap.add_argument("--truncate-words", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=50, help="max epochs (early stopping on val loss); use 3 for --head finetune")
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3, help="dense head default 1e-3; paper used 5e-5 x 10 epochs; finetune 2e-5")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--svm-C", type=float, default=9.0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", type=int, default=0, help="use only N docs per class per split (pipeline test)")
    ap.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT"), help="log every run to this W&B project (off when unset)")
    ap.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    main(ap.parse_args())
