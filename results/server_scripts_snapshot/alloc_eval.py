"""Evaluate per-KV-head bit allocations on held-out data (test splits, new needle seed).

Each allocation maps (layer, kv_head) -> bits (16 = leave in BF16). Reported per task:
KL vs BF16, perplexity (lm), needle exact-match, and the average bits per KV value
(including one FP16 scale + zero-point per quantized 128-value vector: +32/128 bits).

Usage: python alloc_eval.py --sens ../results/sens_qwen7b.csv --out ../results/alloc_qwen7b.csv
"""
import argparse
import csv
import random

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import head_sensitivity as hs


def lm_test_samples(tok, n, ctx, n_score, seed):
    ids = tok(hs.wikitext_text("test"), return_tensors="pt").input_ids[0]
    rng = random.Random(seed)
    return [(ids[s:s + ctx].unsqueeze(0), n_score, None)
            for s in (rng.randrange(0, len(ids) - ctx) for _ in range(n))]


def gsm8k_test_samples(tok, n, seed):
    ds = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=seed).select(range(n))
    out = []
    for ex in ds:
        p = tok.apply_chat_template([{"role": "user", "content": ex["question"]}], add_generation_prompt=True,
                                    return_tensors="pt", return_dict=True)["input_ids"]
        a = tok(ex["answer"], add_special_tokens=False, return_tensors="pt").input_ids
        out.append((torch.cat([p, a], 1), a.shape[1], None))
    return out


def avg_bits(alloc, n_heads):
    total = 0
    for b in alloc.values():
        total += 16 if b >= 16 else b + 32 / 128
    return total / n_heads


@torch.inference_mode()
def evaluate(model, samples, refs):
    kls, nll, ntok, hits, nans = [], 0.0, 0, 0, 0
    for (ids, n, ans), ref in zip(samples, refs):
        lp = hs.score_logits(model, ids, n)
        kls.append((ref.exp() * (ref - lp)).sum(-1).mean().item())
        tgt = ids[0, -n:].cuda()
        nll += -lp.gather(-1, tgt[:, None]).sum().item()
        ntok += n
        if ans is not None:
            nans += 1
            hits += int(torch.equal(lp.argmax(-1).cpu(), ans))
    return sum(kls) / len(kls), float(torch.tensor(nll / ntok).exp()), (hits / nans if nans else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--sens", default="../results/sens_qwen7b.csv")
    ap.add_argument("--scheme", choices=["token", "kivi"], default="token")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", default="alloc.csv")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda",
                                                 attn_implementation="sdpa").eval()
    L, H = model.config.num_hidden_layers, model.config.num_key_value_heads
    heads = [(l, h) for l in range(L) for h in range(H)]

    # combined sensitivity from the 2-bit sweep of the same scheme:
    # per task, divide by that task's max, then take the max over tasks
    sens = {}
    for r in csv.DictReader(open(args.sens)):
        if r["bits"] == "2" and r["scheme"] == args.scheme and r["layer"] != "-1":  # rank with the same scheme
            sens.setdefault(r["task"], {})[(int(r["layer"]), int(r["kv_head"]))] = float(r["kl"])
    combined = {k: max(s[k] / max(s.values()) for s in sens.values()) for k in heads}
    ranked = sorted(heads, key=lambda k: -combined[k])
    rng = random.Random(0)
    random5 = rng.sample(heads, 5)

    def protect(top, keep_bits, rest_bits, chosen=None):
        chosen = set(chosen if chosen is not None else ranked[:top])
        return {k: (keep_bits if k in chosen else rest_bits) for k in heads}

    allocs = {
        "uniform-2": protect(0, 16, 2),
        "uniform-3": protect(0, 16, 3),
        "uniform-4": protect(0, 16, 4),
        "uniform-8": protect(0, 16, 8),
        "top1-bf16 rest-2": protect(1, 16, 2),
        "top5-bf16 rest-2": protect(5, 16, 2),
        "top5-8bit rest-2": protect(5, 8, 2),
        "top10-bf16 rest-2": protect(10, 16, 2),
        "random5-bf16 rest-2 (control)": protect(0, 16, 2, random5),
        "top5-bf16 rest-3": protect(5, 16, 3),
        "top5-bf16 rest-4": protect(5, 16, 4),
    }

    tasks = {
        "lm": lm_test_samples(tok, args.n, 4096, 512, args.seed),
        "needle": hs.needle_samples(tok, args.n, 8192, args.seed),
        "gsm8k": gsm8k_test_samples(tok, args.n, args.seed),
    }
    hs.QCFG.scheme = args.scheme
    hs.QCFG.target = None
    refs = {t: [hs.score_logits(model, ids, n) for ids, n, _ in s] for t, s in tasks.items()}
    base = {t: evaluate(model, s, refs[t]) for t, s in tasks.items()}

    # allocation mode: the hook quantizes each head at its own bit-width
    orig_target = hs.QCFG.target

    def run(alloc):
        hs.QCFG.alloc = alloc
        hs.QCFG.target = "alloc"
        res = {t: evaluate(model, s, refs[t]) for t, s in tasks.items()}
        hs.QCFG.target = orig_target
        return res

    fields = ["model", "scheme", "allocation", "avg_bits", "task", "kl", "ppl", "acc"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for t, (kl, ppl, acc) in base.items():
            w.writerow({"model": args.model, "scheme": args.scheme, "allocation": "bf16", "avg_bits": 16,
                        "task": t, "kl": kl, "ppl": ppl, "acc": acc})
        print(f"{'allocation':32s} {'bits':>5s} | " + " | ".join(f"{t}: KL / ppl / acc" for t in tasks), flush=True)
        print(f"{'bf16':32s} {16:5.2f} | " + " | ".join(f"{a:.3f} / {b:.2f} / {c:.2f}" for a, b, c in base.values()), flush=True)
        for name, alloc in allocs.items():
            res = run(alloc)
            bits = avg_bits(alloc, len(heads))
            for t, (kl, ppl, acc) in res.items():
                w.writerow({"model": args.model, "scheme": args.scheme, "allocation": name, "avg_bits": round(bits, 3),
                            "task": t, "kl": kl, "ppl": ppl, "acc": acc})
            f.flush()
            print(f"{name:32s} {bits:5.2f} | " + " | ".join(f"{a:.3f} / {b:.2f} / {c:.2f}" for a, b, c in res.values()), flush=True)


if __name__ == "__main__":
    main()
