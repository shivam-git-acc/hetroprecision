"""Per-KV-head quantization sensitivity.

For every (layer, kv_head): fake-quantize (quantize -> dequantize) only that head's K and V,
run the calibration data, and measure KL(FP16 || perturbed) on the next-token distributions.
Quantization happens inside attention after RoPE, so prefill also sees the quantized cache.

Tasks (each scored separately, so rankings can be compared):
  lm      WikiText-2 train, 4K-token windows, KL on the last `--lm-score` positions
  needle  passkey hidden in WikiText filler at several depths; KL + exact-match on the answer tokens
  gsm8k   GSM8K train question + reference solution (teacher-forced); KL on the solution tokens

Usage:
  python head_sensitivity.py --bits 2 --out ../results/sens_qwen7b_2bit.csv
  python head_sensitivity.py --self-check          # quick sanity test only
"""
import argparse
import csv
import os
import random
import time

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

# ---------------------------------------------------------------- fake quantization
# QCFG.target: None (no quantization), "all" (every head), (layer, kv_head), or "alloc" (per-head bits in QCFG.alloc)
class QCFG:
    target = None
    bits = 2
    alloc = {}        # (layer, kv_head) -> bits; >= 16 means leave unquantized
    scheme = "token"  # "token": K and V per-token over head_dim; "kivi": K per-channel, V per-token
    group = 128       # token group size for per-channel keys


def fake_quant(x, bits, dim):
    """Asymmetric min-max quantization along `dim`, returned dequantized in x's dtype."""
    xf = x.float()
    lo = xf.amin(dim=dim, keepdim=True)
    hi = xf.amax(dim=dim, keepdim=True)
    levels = 2 ** bits - 1
    scale = (hi - lo).clamp(min=1e-8) / levels
    q = ((xf - lo) / scale).round().clamp(0, levels)
    return (q * scale + lo).to(x.dtype)


def quant_k(k, bits=None):  # k: [B, S, D] for one head
    bits = bits or QCFG.bits
    if QCFG.scheme == "kivi":
        B, S, D = k.shape
        pad = (-S) % QCFG.group
        kp = F.pad(k, (0, 0, 0, pad)) if pad else k
        kg = kp.view(B, -1, QCFG.group, D)
        out = fake_quant(kg, bits, dim=2).view(B, -1, D)
        return out[:, :S]
    return fake_quant(k, bits, dim=-1)


def quant_v(v, bits=None):
    return fake_quant(v, bits or QCFG.bits, dim=-1)


_orig_sdpa = ALL_ATTENTION_FUNCTIONS["sdpa"]


def sdpa_with_kv_quant(module, query, key, value, attention_mask, **kwargs):
    # key/value: [B, n_kv_heads, S, D], post-RoPE, before GQA expansion
    t = QCFG.target
    if t == "alloc":
        key, value = key.clone(), value.clone()
        for h in range(key.shape[1]):
            b = QCFG.alloc[(module.layer_idx, h)]
            if b < 16:
                key[:, h] = quant_k(key[:, h], b)
                value[:, h] = quant_v(value[:, h], b)
    elif t is not None:
        key, value = key.clone(), value.clone()
        heads = range(key.shape[1]) if t == "all" else ([t[1]] if t[0] == module.layer_idx else [])
        for h in heads:
            key[:, h] = quant_k(key[:, h])
            value[:, h] = quant_v(value[:, h])
    return _orig_sdpa(module, query, key, value, attention_mask, **kwargs)


ALL_ATTENTION_FUNCTIONS["sdpa"] = sdpa_with_kv_quant

# ---------------------------------------------------------------- data
# Each sample: (input_ids [1, T], n_score, answer_ids or None). Scored positions are the last n_score
# tokens; logits_to_keep = n_score + 1 returns exactly the predictions for them.


def wikitext_text(split):
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    return "\n".join(t for t in ds["text"] if t.strip())


def lm_samples(tok, n, ctx, n_score, seed):
    ids = tok(wikitext_text("train"), return_tensors="pt").input_ids[0]
    rng = random.Random(seed)
    starts = [rng.randrange(0, len(ids) - ctx) for _ in range(n)]
    return [(ids[s:s + ctx].unsqueeze(0), n_score, None) for s in starts]


def needle_samples(tok, n, ctx, seed):
    filler = tok(wikitext_text("train"), return_tensors="pt").input_ids[0]
    rng = random.Random(seed)
    out = []
    for i in range(n):
        key = str(rng.randrange(10000, 99999))
        depth = (i + 0.5) / n  # spread needle depths over the context
        needle = tok(f" The secret passkey is {key}. Remember it. ", add_special_tokens=False).input_ids
        q = [{"role": "user", "content": "PLACEHOLDER"}]
        # build: chat(user = haystack-with-needle + question) + answer
        question = "\n\nWhat is the secret passkey mentioned in the text above? Answer with the number only."
        q_ids = tok(question, add_special_tokens=False).input_ids
        start = rng.randrange(0, len(filler) - ctx)
        body = filler[start:start + ctx - len(needle) - len(q_ids) - 64].tolist()
        pos = int(len(body) * depth)
        body = body[:pos] + needle + body[pos:]
        text = tok.decode(body + q_ids)
        q[0]["content"] = text
        prompt = tok.apply_chat_template(q, add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"]
        ans = tok(key, add_special_tokens=False, return_tensors="pt").input_ids
        out.append((torch.cat([prompt, ans], dim=1), ans.shape[1], ans[0]))
    return out


def gsm8k_samples(tok, n, seed):
    ds = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=seed).select(range(n))
    out = []
    for ex in ds:
        msgs = [{"role": "user", "content": ex["question"]}]
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"]
        ans = tok(ex["answer"], add_special_tokens=False, return_tensors="pt").input_ids
        out.append((torch.cat([prompt, ans], dim=1), ans.shape[1], None))
    return out


# ---------------------------------------------------------------- scoring
@torch.inference_mode()
def score_logits(model, ids, n_score):
    out = model(input_ids=ids.cuda(), logits_to_keep=n_score + 1, use_cache=False)
    return out.logits[0, :-1].float().log_softmax(-1)  # [n_score, V] log-probs for the scored tokens


@torch.inference_mode()
def evaluate(model, samples, refs):
    """Mean KL(ref || current) over scored tokens, and answer exact-match (needle only)."""
    kls, hits, n_ans = [], 0, 0
    for (ids, n_score, ans), ref in zip(samples, refs):
        lp = score_logits(model, ids, n_score)
        kl = (ref.exp() * (ref - lp)).sum(-1)  # per position
        kls.append(kl.mean().item())
        if ans is not None:
            n_ans += 1
            hits += int(torch.equal(lp.argmax(-1).cpu(), ans))
    return sum(kls) / len(kls), (hits / n_ans if n_ans else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--bits", type=int, default=2)
    ap.add_argument("--scheme", choices=["token", "kivi"], default="token")
    ap.add_argument("--tasks", default="lm,needle,gsm8k")
    ap.add_argument("--n-lm", type=int, default=8)
    ap.add_argument("--lm-ctx", type=int, default=4096)
    ap.add_argument("--lm-score", type=int, default=512)
    ap.add_argument("--n-needle", type=int, default=8)
    ap.add_argument("--needle-ctx", type=int, default=8192)
    ap.add_argument("--n-gsm8k", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="sensitivity.csv")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda",
                                                 attn_implementation="sdpa").eval()
    cfg = model.config
    L, H = cfg.num_hidden_layers, cfg.num_key_value_heads
    QCFG.bits, QCFG.scheme = args.bits, args.scheme
    print(f"model={args.model} layers={L} kv_heads={H} bits={args.bits} scheme={args.scheme}", flush=True)

    tasks = {}
    for name in args.tasks.split(","):
        if name == "lm":
            tasks[name] = lm_samples(tok, args.n_lm, args.lm_ctx, args.lm_score, args.seed)
        elif name == "needle":
            tasks[name] = needle_samples(tok, args.n_needle, args.needle_ctx, args.seed)
        elif name == "gsm8k":
            tasks[name] = gsm8k_samples(tok, args.n_gsm8k, args.seed)
    for name, s in tasks.items():
        print(f"task {name}: {len(s)} samples, {sum(x[0].shape[1] for x in s)} tokens", flush=True)

    # FP16 reference log-probs, kept on GPU
    QCFG.target = None
    refs = {name: [score_logits(model, ids, n) for ids, n, _ in s] for name, s in tasks.items()}

    def run(target):
        QCFG.target = target
        res = {name: evaluate(model, s, refs[name]) for name, s in tasks.items()}
        QCFG.target = None
        return res

    # Sanity: no quantization must give KL ~ 0; quantizing every head gives the uniform reference point
    base = run(None)
    allq = run("all")
    for name in tasks:
        print(f"[check] {name}: no-quant KL={base[name][0]:.2e} acc={base[name][1]:.2f} | "
              f"all heads {args.bits}-bit KL={allq[name][0]:.4f} acc={allq[name][1]:.2f}", flush=True)
    if args.self_check:
        t0 = time.time(); run((0, 0)); print(f"[check] one head took {time.time() - t0:.1f}s", flush=True)
        return

    fields = ["model", "bits", "scheme", "task", "layer", "kv_head", "kl", "acc"]
    new_file = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
        common = {"model": args.model, "bits": args.bits, "scheme": args.scheme}
        for name in tasks:
            w.writerow({**common, "task": name, "layer": -1, "kv_head": -1, "kl": allq[name][0], "acc": allq[name][1]})
        t0 = time.time()
        for layer in range(L):
            for h in range(H):
                res = run((layer, h))
                for name, (kl, acc) in res.items():
                    w.writerow({**common, "task": name, "layer": layer, "kv_head": h, "kl": kl, "acc": acc})
                f.flush()
            done = (layer + 1) * H
            eta = (time.time() - t0) / done * (L * H - done)
            print(f"layer {layer} done ({done}/{L * H} heads), eta {eta / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
