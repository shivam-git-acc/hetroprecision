"""vLLM decode cost: FP16 vs FP8 KV cache, over context length x batch size.

TTFT = time to generate 1 token for the whole batch (prefill + first step).
TPOT = (time for 1 + N tokens - time for 1 token) / N, i.e. the decode step time.
Prompts are random token ids (content does not affect timing); ignore_eos keeps every request decoding.

Attention backend is chosen with the VLLM_ATTENTION_BACKEND env var (e.g. FLASH_ATTN, FLASHINFER) and recorded.

Usage: python bench_vllm.py --kv auto --out ../results/vllm_qwen7b.csv
       python bench_vllm.py --kv fp8  --out ../results/vllm_qwen7b.csv
"""
import argparse
import csv
import os
import random
import statistics
import time

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def timed_generate(llm, prompts, max_tokens, repeats):
    sp = SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=True)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        llm.generate(prompts, sp, use_tqdm=False)
        times.append(time.perf_counter() - t0)
    return times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--kv", default="auto", help="kv_cache_dtype: auto (= model dtype, BF16) or fp8")
    ap.add_argument("--contexts", default="4096,8192,16384,32640")
    ap.add_argument("--batches", default="1,4,8")
    ap.add_argument("--gen", type=int, default=64, help="decode tokens timed per request")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--out", default="vllm.csv")
    args = ap.parse_args()

    llm = LLM(args.model, dtype="bfloat16", kv_cache_dtype=args.kv, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_util, enable_prefix_caching=False)
    try:
        cc = llm.llm_engine.vllm_config.cache_config
        kv_capacity_tokens = cc.num_gpu_blocks * cc.block_size
    except Exception:
        kv_capacity_tokens = ""
    vocab = llm.get_tokenizer().vocab_size
    rng = random.Random(0)

    backend = os.environ.get("VLLM_ATTENTION_BACKEND", "default")
    fields = ["model", "kv_dtype", "attn_backend", "context", "batch", "ttft_s_median", "tpot_ms_median", "tpot_ms_q1",
              "tpot_ms_q3", "throughput_tok_s", "kv_capacity_tokens", "max_concurrent_at_ctx"]
    new_file = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
        for ctx in map(int, args.contexts.split(",")):
            for batch in map(int, args.batches.split(",")):
                prompts = [TokensPrompt(prompt_token_ids=[rng.randrange(100, vocab - 100) for _ in range(ctx)])
                           for _ in range(batch)]
                timed_generate(llm, prompts, 1 + args.gen, 1)  # warmup
                t1 = timed_generate(llm, prompts, 1, args.repeats)
                tn = timed_generate(llm, prompts, 1 + args.gen, args.repeats)
                # pair runs to cancel noise in prefill time; one TPOT sample per repeat
                tpots = [(b - a) / args.gen for a, b in zip(sorted(t1), sorted(tn))]
                tpot = statistics.median(tpots)
                q1, _, q3 = statistics.quantiles(tpots, n=4) if len(tpots) > 1 else (tpot, tpot, tpot)
                row = {
                    "model": args.model, "kv_dtype": args.kv, "attn_backend": backend, "context": ctx, "batch": batch,
                    "ttft_s_median": round(statistics.median(t1), 4),
                    "tpot_ms_median": round(tpot * 1e3, 3), "tpot_ms_q1": round(q1 * 1e3, 3),
                    "tpot_ms_q3": round(q3 * 1e3, 3), "throughput_tok_s": round(batch / tpot, 2),
                    "kv_capacity_tokens": kv_capacity_tokens,
                    "max_concurrent_at_ctx": (kv_capacity_tokens // (ctx + 1 + args.gen)) if kv_capacity_tokens else "",
                }
                w.writerow(row)
                f.flush()
                print(row, flush=True)


if __name__ == "__main__":
    main()
