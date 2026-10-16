"""
Minimal-resource task adaptation of an open-weight LLM with LoRA / QLoRA (Reviewer 1, major point 2(i)).

Trains the model to emit the category name given the SAME prompt used for zero-shot evaluation (llm_eval.PROMPTS[en_v2]),
on the SAME train split (or a learning-curve subset), then serves it for evaluation with llm_eval.py via vLLM
(or evaluates directly with --eval-after-train using HF generate).

Everything the reviewers asked to be reported is written to results/lora/<run>.budget.json:
  trainable parameters (and % of total), training tokens seen, wall-clock GPU hours, GPU type, peak VRAM,
  LoRA rank/alpha/targets, learning rate, epochs, quantization.

Default model: Qwen/Qwen2.5-7B-Instruct (strong Arabic, fits an A100-40GB in 4-bit QLoRA with batch 8 x 1024 tokens).
Alternatives worth considering: meta-llama/Llama-3.1-8B-Instruct, google/gemma-2-9b-it, ALLaM-AI/ALLaM-7B-Instruct-preview,
inceptionai/jais-family-6p7b-chat (Arabic-centric).

Examples (A100 40GB):
  pip install "unsloth[cu121-torch240]" trl peft bitsandbytes   # or plain peft+trl if unsloth is unavailable
  python lora_finetune.py --model Qwen/Qwen2.5-7B-Instruct --n-per-class all --epochs 1
  python lora_finetune.py --model Qwen/Qwen2.5-7B-Instruct --n-per-class 100 --epochs 3
  # serve the adapter for llm_eval.py:
  vllm serve Qwen/Qwen2.5-7B-Instruct --enable-lora --lora-modules sanad=results/lora/<run>/adapter --max-lora-rank 32 --port 8000
  python llm_eval.py --provider vllm --model sanad --prompt en_v2
"""
import os, sys, json, time, argparse, datetime, re
import numpy as np, pandas as pd, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data_split import load_split, learning_curve_subset, CATS  # noqa: E402
from llm_eval import build_messages, map_output  # noqa: E402


def make_examples(df, prompt_name, max_words):
    rows = []
    for r in df.itertuples(index=False):
        text = " ".join(r.text.split()[:max_words]) if max_words else r.text
        system, user = build_messages(prompt_name, text, None)
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user},
                                                                              {"role": "assistant", "content": r.category}]
        rows.append({"messages": msgs, "doc_id": r.doc_id, "category": r.category})
    return rows


def main(args):
    df = load_split(args.root, args.split_csv)
    tr = learning_curve_subset(df, None if args.n_per_class == "all" else int(args.n_per_class), args.seed)
    va = df[df.split == "val"].groupby("category").head(args.val_per_class)
    run = f"lora__{re.sub(r'[^A-Za-z0-9.-]+', '_', args.model)}__n{args.n_per_class}__s{args.seed}__r{args.rank}__ep{args.epochs}"
    out_dir = os.path.join(HERE, "results", "lora", run)
    os.makedirs(out_dir, exist_ok=True)
    train_rows, val_rows = make_examples(tr, args.prompt, args.max_words), make_examples(va, args.prompt, args.max_words)
    print(f"[{run}] train examples={len(train_rows)} val examples={len(val_rows)}")

    # ---- model + tokenizer (Unsloth if available, else transformers + peft)
    use_unsloth = False
    try:
        from unsloth import FastLanguageModel
        model, tok = FastLanguageModel.from_pretrained(args.model, max_seq_length=args.max_seq_len, load_in_4bit=args.load_in_4bit, dtype=None)
        model = FastLanguageModel.get_peft_model(model, r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0, bias="none",
                                                 target_modules=args.targets.split(","), use_gradient_checkpointing="unsloth", random_state=args.seed)
        use_unsloth = True
    except Exception as e:
        print("unsloth unavailable (", str(e)[:80], ") -> transformers + peft")
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        tok = AutoTokenizer.from_pretrained(args.model)
        qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True) if args.load_in_4bit else None
        model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=qcfg, torch_dtype=torch.bfloat16, device_map={"": 0})
        if args.load_in_4bit:
            model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                                                 target_modules=args.targets.split(",")))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {n_trainable:,} / {n_total:,} ({100 * n_trainable / n_total:.3f}%)")

    # ---- data -> chat template strings; train only on the assistant answer (completion-only loss)
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    def to_text(ex):
        return {"text": tok.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)}

    ds_tr = Dataset.from_list(train_rows).map(to_text, remove_columns=["messages", "doc_id", "category"])
    ds_va = Dataset.from_list(val_rows).map(to_text, remove_columns=["messages", "doc_id", "category"])
    n_tokens = int(sum(len(tok(t["text"]).input_ids) for t in ds_tr))

    wb = None
    if args.wandb_project:  # opt-in W&B: the trainer streams loss/eval curves into the same run, budget and test metrics are added at the end
        import wandb
        wb = wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=run, job_type="lora", tags=["revision", "lora"],
                        config=dict(vars(args), n_train_examples=len(train_rows), trainable_params=n_trainable, total_params=n_total))
    cfg = SFTConfig(output_dir=out_dir, per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
                    num_train_epochs=args.epochs, learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.03, logging_steps=20,
                    eval_strategy="epoch", save_strategy="no", bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(),
                    max_seq_length=args.max_seq_len, dataset_text_field="text", packing=False, seed=args.seed,
                    report_to=("wandb" if wb else "none"), run_name=run,
                    optim="adamw_8bit" if args.load_in_4bit else "adamw_torch")
    trainer = SFTTrainer(model=model, tokenizer=tok, train_dataset=ds_tr, eval_dataset=ds_va, args=cfg)
    try:  # completion-only loss: mask everything before the assistant turn (Qwen/Llama chat templates)
        from trl import DataCollatorForCompletionOnlyLM
        resp = "<|im_start|>assistant\n" if "qwen" in args.model.lower() else "<|start_header_id|>assistant<|end_header_id|>\n\n"
        trainer.data_collator = DataCollatorForCompletionOnlyLM(resp, tokenizer=tok)
    except Exception as e:
        print("completion-only collator unavailable:", str(e)[:80])

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    train_sec = time.time() - t0
    model.save_pretrained(os.path.join(out_dir, "adapter"))
    tok.save_pretrained(os.path.join(out_dir, "adapter"))
    budget = dict(run=run, model=args.model, n_train_examples=len(train_rows), n_per_class=args.n_per_class, seed=args.seed,
                  trainable_params=n_trainable, total_params=n_total, trainable_pct=100 * n_trainable / n_total,
                  training_tokens_per_epoch=n_tokens, epochs=args.epochs, training_tokens_total=n_tokens * args.epochs,
                  train_wall_sec=train_sec, train_gpu_hours=train_sec / 3600, gpu=torch.cuda.get_device_name(0),
                  peak_vram_gb=torch.cuda.max_memory_allocated() / 1e9, lora=dict(rank=args.rank, alpha=args.alpha, targets=args.targets),
                  lr=args.lr, batch_size=args.batch_size * args.grad_accum, max_seq_len=args.max_seq_len, load_in_4bit=args.load_in_4bit,
                  unsloth=use_unsloth, prompt=args.prompt, max_words=args.max_words, ts=datetime.datetime.now(datetime.timezone.utc).isoformat())
    with open(os.path.join(out_dir, "budget.json"), "w") as f:
        json.dump(budget, f, indent=2)
    print(json.dumps(budget, indent=2))
    if wb:
        wb.log({f"budget/{k}": v for k, v in budget.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})

    if args.eval_after_train:  # direct greedy decoding on the test split (slower than vLLM but dependency-free)
        if use_unsloth:
            from unsloth import FastLanguageModel
            FastLanguageModel.for_inference(model)
        model.eval()
        te = df[df.split == "test"]
        out = os.path.join(HERE, "results", f"vllm__{run}__{args.prompt}__k0.jsonl")
        with open(out, "w") as f:
            for i in range(0, len(te), args.eval_batch):
                chunk = te.iloc[i:i + args.eval_batch]
                prompts = []
                for r in chunk.itertuples(index=False):
                    text = " ".join(r.text.split()[:args.max_words]) if args.max_words else r.text
                    s, u = build_messages(args.prompt, text, None)
                    msgs = ([{"role": "system", "content": s}] if s else []) + [{"role": "user", "content": u}]
                    prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
                tok.padding_side = "left"
                enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_seq_len).to(model.device)
                with torch.no_grad():
                    gen = model.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id)
                dec = tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
                for r, raw in zip(chunk.itertuples(index=False), dec):
                    lab, meth = map_output(raw, True); lab_l, meth_l = map_output(raw, False)
                    f.write(json.dumps(dict(doc_id=r.doc_id, gold=r.category, raw=raw, pred=lab, map_method=meth, pred_lenient=lab_l, map_method_lenient=meth_l,
                                            finish_reason="stop", in_tokens=None, out_tokens=None, served_model=run, latency_s=None, attempts=1, error=None), ensure_ascii=False) + "\n")
                print(f"  eval {min(i + args.eval_batch, len(te))}/{len(te)}")
        from llm_eval import summarize
        res = summarize(out)
        if wb:
            wb.log({"test/accuracy": res["strict"]["accuracy"], "test/macro_f1": res["strict"]["macro_f1"], "test/unmapped_rate": res["unmapped_rate"],
                    **{f"test/{c}_recall": res["strict"]["per_class"][c]["recall"] for c in CATS}})
    if wb:
        wb.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--n-per-class", default="all")
    ap.add_argument("--seed", type=int, default=0, choices=[0, 1, 2])
    ap.add_argument("--val-per-class", type=int, default=100)
    ap.add_argument("--prompt", default="en_v2")
    ap.add_argument("--max-words", type=int, default=350, help="truncate articles to keep sequences <= max-seq-len (0 = full)")
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--targets", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--load-in-4bit", action="store_true", default=True)
    ap.add_argument("--no-4bit", dest="load_in_4bit", action="store_false")
    ap.add_argument("--eval-after-train", action="store_true")
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--split-csv", default=os.path.join(HERE, "splits", "sanad_alkhaleej_split_v1.csv"))
    ap.add_argument("--root", default=None)
    ap.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT"), help="log this run to W&B (off when unset)")
    ap.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    main(ap.parse_args())
