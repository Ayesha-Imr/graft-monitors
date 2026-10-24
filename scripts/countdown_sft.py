"""SFT-prime Qwen3-8B on Countdown-Code (seed doc §7.6 step 1), on a RunPod H200.

A line-for-line port of GRAFT's tools/modal_countdown_sft.py::sft without Modal: same
completion-masked loss, LoRA r128/alpha128 all-linear, lr 1e-4, 5 epochs, cosine,
warmup 3%, per-device batch 4 x grad-accum 8, max_len 2560, bf16, flash-attn 2.
Writes the merged model to --out and a meta.json with timing and peak memory.

    python scripts/countdown_sft.py --data /workspace/data/countdown_sft --out /workspace/ckpt/sft/qwen3-8b
"""
from __future__ import annotations

import argparse
import json
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora_rank", type=int, default=128)
    ap.add_argument("--lora_alpha", type=int, default=128)
    ap.add_argument("--max_len", type=int, default=2560)
    ap.add_argument("--per_device_bs", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_epochs", action="store_true")
    a = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

    set_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    eos = tok.eos_token_id

    def rows(name):
        r = [json.loads(line) for line in open(os.path.join(a.data, f"{name}.jsonl"))]
        return r[:a.limit] if a.limit else r

    def encode(row):
        prompt_ids = tok.apply_chat_template(row["prompt"], add_generation_prompt=True, tokenize=True,
                                             return_dict=False)
        if not isinstance(prompt_ids, list):
            prompt_ids = list(prompt_ids["input_ids"]) if hasattr(prompt_ids, "keys") else list(prompt_ids)
        resp_ids = tok(row["response"], add_special_tokens=False)["input_ids"] + [eos]
        return prompt_ids + resp_ids, [-100] * len(prompt_ids) + resp_ids

    def build(name):
        kept, dropped = [], 0
        for r in rows(name):
            ids, lab = encode(r)
            if len(ids) > a.max_len:
                dropped += 1
                continue
            kept.append({"input_ids": ids, "labels": lab})
        print(f"[{name}] {len(kept)} examples ({dropped} dropped > {a.max_len} tok)", flush=True)
        return kept, dropped

    (train_ds, d_tr), (val_ds, d_va) = build("train"), build("val")

    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        pad = tok.pad_token_id if tok.pad_token_id is not None else eos
        ii, lab, am = [], [], []
        for b in batch:
            n = maxlen - len(b["input_ids"])
            ii.append(b["input_ids"] + [pad] * n)
            lab.append(b["labels"] + [-100] * n)
            am.append([1] * len(b["input_ids"]) + [0] * n)
        return {"input_ids": torch.tensor(ii), "labels": torch.tensor(lab), "attention_mask": torch.tensor(am)}

    base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    base.config.use_cache = False
    model = get_peft_model(base, LoraConfig(r=a.lora_rank, lora_alpha=a.lora_alpha, lora_dropout=0.0, bias="none",
                                            task_type="CAUSAL_LM", target_modules="all-linear"))
    model.print_trainable_parameters()
    os.makedirs(a.out, exist_ok=True)
    args = TrainingArguments(
        output_dir=os.path.join(a.out, "epochs") if a.save_epochs else "/tmp/cc_sft_ckpt",
        num_train_epochs=a.epochs, learning_rate=a.lr, max_steps=a.max_steps if a.max_steps > 0 else -1,
        per_device_train_batch_size=a.per_device_bs, gradient_accumulation_steps=a.grad_accum, bf16=True,
        gradient_checkpointing=False, lr_scheduler_type="cosine", warmup_ratio=0.03, logging_steps=10,
        save_strategy="epoch" if a.save_epochs else "no", save_only_model=True, report_to=[],
        optim="adamw_torch", dataloader_num_workers=2, seed=a.seed)
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds, data_collator=collate)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = trainer.train()
    train_s = time.time() - t0
    peak_gb = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    ev = trainer.evaluate()
    meta = {"model": a.model, "epochs": a.epochs, "lr": a.lr, "lora_rank": a.lora_rank, "lora_alpha": a.lora_alpha,
            "max_len": a.max_len, "eff_batch": a.per_device_bs * a.grad_accum, "seed": a.seed,
            "n_train": len(train_ds), "n_val": len(val_ds), "dropped": [d_tr, d_va],
            "train_seconds": round(train_s, 1), "peak_gb": peak_gb, "train_metrics": out.metrics,
            "eval_metrics": ev, "gpu": torch.cuda.get_device_name(0)}
    print(json.dumps(meta, indent=2), flush=True)
    if a.max_steps > 0:
        return
    merged = model.merge_and_unload()
    merged.save_pretrained(a.out, safe_serialization=True)
    tok.save_pretrained(a.out)
    json.dump(meta, open(os.path.join(a.out, "sft_meta.json"), "w"), indent=2)
    print(f"saved merged model to {a.out}", flush=True)


if __name__ == "__main__":
    main()
