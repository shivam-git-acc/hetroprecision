"""Per-head attention statistics on FP16 (no quantization): what each head does.

For each task sample, inside every attention layer, attention probabilities are recomputed for a set of
query rows (the scored positions) and summarised per query head:
  entropy       entropy of the attention distribution (nats), averaged over query rows
  sink          attention mass on the first token
  sink4/early16 attention mass on the first 4 / first 16 tokens (with a chat template the sink may not be token 0)
  local         attention mass on the 64 most recent tokens
  needle_mass   (needle task) attention mass on the passkey digits, at rows that predict the answer
  retrieval     (needle task) fraction of those rows whose top-1 key is the digit being copied (Wu et al., 2024)
Per KV head also:
  key_outlier   max channel |K| magnitude / median channel magnitude (how hard the keys are to quantize)

Usage: python head_stats.py --out ../results/stats_qwen7b.csv
"""
import argparse
import csv
import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

import head_sensitivity as hs  # reuses the task builders (its quantization hook stays inactive)


class CTX:
    rows = None        # absolute query positions to analyse
    needle_pos = None  # absolute positions of the passkey digits in the prompt
    copy_pos = None    # for each answer-predicting row, the needle position of the digit it should copy
    acc = None         # dict: (layer, q_head) -> dict of running sums
    kacc = None        # dict: (layer, kv_head) -> list of key outlier ratios


_inner = ALL_ATTENTION_FUNCTIONS["sdpa"]


def sdpa_with_stats(module, query, key, value, attention_mask, scaling=None, **kwargs):
    if CTX.rows is not None:
        record(module.layer_idx, query, key, scaling or 1 / math.sqrt(query.shape[-1]))
    return _inner(module, query, key, value, attention_mask, scaling=scaling, **kwargs)


ALL_ATTENTION_FUNCTIONS["sdpa"] = sdpa_with_stats


@torch.no_grad()
def record(layer, query, key, scaling):
    # query [1, Hq, S, D], key [1, Hkv, S, D] (post-RoPE)
    Hq, S = query.shape[1], query.shape[2]
    Hkv = key.shape[1]
    group = Hq // Hkv
    rows = torch.tensor(CTX.rows, device=query.device)
    q = query[0, :, rows].float()                                   # [Hq, R, D]
    k = key[0].float().repeat_interleave(group, dim=0)              # [Hq, S, D]
    scores = torch.einsum("hrd,hsd->hrs", q, k) * scaling
    causal = torch.arange(S, device=q.device)[None, :] > rows[:, None]  # [R, S]
    p = scores.masked_fill(causal, float("-inf")).softmax(-1)      # [Hq, R, S]

    ent = -(p * p.clamp_min(1e-12).log()).sum(-1).mean(-1)          # [Hq]
    sink = p[..., 0].mean(-1)
    sink4 = p[..., :4].sum(-1).mean(-1)
    early16 = p[..., :16].sum(-1).mean(-1)
    local_mask = (torch.arange(S, device=q.device)[None, :] > rows[:, None] - 64) & ~causal
    local = (p * local_mask).sum(-1).mean(-1)
    stats = {"entropy": ent, "sink": sink, "sink4": sink4, "early16": early16, "local": local}
    if CTX.needle_pos is not None:
        a_rows = len(CTX.copy_pos)                                  # answer-predicting rows are the last ones
        pa = p[:, -a_rows:]                                         # [Hq, A, S]
        stats["needle_mass"] = pa[..., CTX.needle_pos].sum(-1).mean(-1)
        top = pa.argmax(-1)                                         # [Hq, A]
        target = torch.tensor(CTX.copy_pos, device=q.device)[None, :]
        stats["retrieval"] = (top == target).float().mean(-1)
    for h in range(Hq):
        d = CTX.acc.setdefault((layer, h), {})
        for name, v in stats.items():
            d.setdefault(name, []).append(v[h].item())

    kmag = key[0].float().abs().mean(1)                             # [Hkv, D] mean |K| per channel
    ratio = kmag.amax(-1) / kmag.median(-1).values
    for h in range(Hkv):
        CTX.kacc.setdefault((layer, h), []).append(ratio[h].item())


def find_sub(seq, sub):
    seq, sub = seq.tolist(), sub.tolist()
    for i in range(len(seq) - len(sub), -1, -1):
        if seq[i:i + len(sub)] == sub:
            return i
    raise ValueError("needle not found")


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--tasks", default="lm,needle,gsm8k")
    ap.add_argument("--n", type=int, default=8, help="samples per task")
    ap.add_argument("--lm-ctx", type=int, default=4096)
    ap.add_argument("--lm-rows", type=int, default=128)
    ap.add_argument("--needle-ctx", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=1)  # different from the sensitivity calibration seed
    ap.add_argument("--out", default="stats.csv")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda",
                                                 attn_implementation="sdpa").eval()
    kv_group = model.config.num_attention_heads // model.config.num_key_value_heads

    rows_out, krows_out = [], []
    for task in args.tasks.split(","):
        if task == "lm":
            samples = hs.lm_samples(tok, args.n, args.lm_ctx, args.lm_rows, args.seed)
        elif task == "needle":
            samples = hs.needle_samples(tok, args.n, args.needle_ctx, args.seed)
        else:
            samples = hs.gsm8k_samples(tok, args.n, args.seed)
        CTX.acc, CTX.kacc = {}, {}
        for ids, n_score, ans in samples:
            T = ids.shape[1]
            CTX.rows = list(range(T - n_score - 1, T - 1))  # rows that predict the scored tokens
            if ans is not None:
                start = find_sub(ids[0, :T - n_score], ans)
                CTX.needle_pos = list(range(start, start + len(ans)))
                CTX.copy_pos = CTX.needle_pos  # row j predicts answer digit j, copied from needle digit j
            else:
                CTX.needle_pos = CTX.copy_pos = None
            model(input_ids=ids.cuda(), logits_to_keep=1, use_cache=False)
        CTX.rows = None
        for (layer, h), d in sorted(CTX.acc.items()):
            row = {"task": task, "layer": layer, "q_head": h, "kv_head": h // kv_group}
            row.update({k: sum(v) / len(v) for k, v in d.items()})
            rows_out.append(row)
        for (layer, h), v in sorted(CTX.kacc.items()):
            krows_out.append({"task": task, "layer": layer, "kv_head": h, "key_outlier": sum(v) / len(v)})
        print(f"task {task}: {len(samples)} samples done", flush=True)

    fields = ["task", "layer", "q_head", "kv_head", "entropy", "sink", "sink4", "early16", "local", "needle_mass",
              "retrieval"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)
    kout = args.out.replace(".csv", "_keys.csv")
    with open(kout, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "layer", "kv_head", "key_outlier"])
        w.writeheader()
        w.writerows(krows_out)
    print(f"wrote {args.out} and {kout}")


if __name__ == "__main__":
    main()
