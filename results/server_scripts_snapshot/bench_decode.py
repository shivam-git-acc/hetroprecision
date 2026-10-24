"""Step 2: FP16/BF16 baseline cost of prefill and decode over context length x batch size.

For each (context, batch): chunked prefill of a synthetic prompt, then timed decode steps.
Prompt content does not affect timing, so random token ids are used.
One CSV row is appended per setting, so a disconnect loses at most the current setting.

Usage: python bench_decode.py --model Qwen/Qwen2.5-7B-Instruct --out ../results/decode_fp16.csv
"""
import argparse
import csv
import os
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, DynamicCache, QuantizedCache, StaticCache


def kv_bytes(cfg, batch, tokens, bytes_per_val=2):
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    return 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * head_dim * bytes_per_val * batch * tokens


@torch.inference_mode()
def run_one(model, ctx, batch, warmup, steps, chunk, profile=False, cache_kind="dynamic"):
    cfg = model.config
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    weights = torch.cuda.memory_allocated()

    ids = torch.randint(100, cfg.vocab_size - 100, (batch, ctx), device="cuda")
    if cache_kind.startswith("quanto"):
        # HF QuantizedCache: stores int4/int2 (groups of 64), keeps the last 128 tokens in BF16, and
        # dequantizes the whole cache back to BF16 before every attention call.
        cache = QuantizedCache(backend="quanto", config=cfg, nbits=int(cache_kind[-1]))
    elif cache_kind == "static":
        # Pre-allocated once: no per-step copy when appending a token.
        cache = StaticCache(config=cfg, max_cache_len=ctx + warmup + steps)
    else:
        # Grows with torch.cat on every step, copying the whole cache each time.
        cache = DynamicCache()

    # Chunked prefill: full-prompt logits and MLP activations at 8 x 32K would not fit.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for s in range(0, ctx, chunk):
        out = model(input_ids=ids[:, s:s + chunk], past_key_values=cache, use_cache=True, logits_to_keep=1)
    nxt = out.logits[:, -1:].argmax(-1)
    torch.cuda.synchronize()
    ttft = time.perf_counter() - t0

    step_times = []
    for i in range(warmup + steps):
        if profile and i == warmup:
            torch.cuda.profiler.start()  # nsys --capture-range=cudaProfilerApi records only timed decode steps
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        out = model(input_ids=nxt, past_key_values=cache, use_cache=True)
        nxt = out.logits[:, -1:].argmax(-1)
        end.record()
        torch.cuda.synchronize()
        if i >= warmup:
            step_times.append(start.elapsed_time(end) / 1e3)

    if profile:
        torch.cuda.profiler.stop()
    tpot = statistics.median(step_times)
    q1, _, q3 = statistics.quantiles(step_times, n=4)
    total_tokens = ctx + warmup + steps
    return {
        "context": ctx,
        "batch": batch,
        "cache": cache_kind,
        "status": "ok",
        "ttft_s": round(ttft, 4),
        "tpot_ms_median": round(tpot * 1e3, 3),
        "tpot_ms_q1": round(q1 * 1e3, 3),
        "tpot_ms_q3": round(q3 * 1e3, 3),
        "throughput_tok_s": round(batch / tpot, 2),
        "kv_cache_gb": round(kv_bytes(cfg, batch, total_tokens) / 1e9, 3),
        "weights_gb": round(weights / 1e9, 3),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        # bytes each decode step must read at minimum: all weights + this batch's KV cache
        "min_bytes_per_step_gb": round((weights + kv_bytes(cfg, batch, total_tokens)) / 1e9, 3),
        "effective_bw_GBps": round((weights + kv_bytes(cfg, batch, total_tokens)) / tpot / 1e9, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--contexts", default="4096,8192,16384,32768")
    ap.add_argument("--batches", default="1,4,8")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--out", default="decode_fp16.csv")
    ap.add_argument("--cache", choices=["dynamic", "static", "quanto4", "quanto2"], default="dynamic")
    ap.add_argument("--profile", action="store_true", help="bracket timed decode steps for nsys")
    args = ap.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    print(f"model={args.model} attn={model.config._attn_implementation}", flush=True)

    fields = ["model", "cache", "context", "batch", "status", "ttft_s", "tpot_ms_median", "tpot_ms_q1", "tpot_ms_q3",
              "throughput_tok_s", "kv_cache_gb", "weights_gb", "peak_mem_gb", "min_bytes_per_step_gb",
              "effective_bw_GBps"]
    new_file = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
        for ctx in map(int, args.contexts.split(",")):
            for batch in map(int, args.batches.split(",")):
                try:
                    row = run_one(model, ctx, batch, args.warmup, args.steps, args.chunk, args.profile, args.cache)
                except torch.OutOfMemoryError:
                    row = {"context": ctx, "batch": batch, "cache": args.cache, "status": "OOM"}
                    torch.cuda.empty_cache()
                row["model"] = args.model
                w.writerow(row)
                f.flush()
                print(row, flush=True)


if __name__ == "__main__":
    main()
