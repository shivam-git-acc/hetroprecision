"""Diagnose the non-monotonic KIVI result (uniform 4-bit worse than 3-bit on needle/GSM8K).

1. Capture real post-RoPE K and V of every layer on one needle prompt.
2. Relative reconstruction error ||Q(x) - x|| / ||x|| for bits 2, 3, 4, 8, schemes token / kivi.
   Error must fall as bits rise; a rise at 4 bits points to a quantizer bug.
3. End-to-end check on the same prompt: answer exact-match + KL for uniform KIVI 3 vs 4 bits.

Usage: python diag_kivi.py
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

import head_sensitivity as hs

CAPTURE = {}
_inner = ALL_ATTENTION_FUNCTIONS["sdpa"]


def capturing(module, query, key, value, attention_mask, **kw):
    if hs.QCFG.target is None and module.layer_idx not in CAPTURE:
        CAPTURE[module.layer_idx] = (key.detach().clone(), value.detach().clone())
    return _inner(module, query, key, value, attention_mask, **kw)


ALL_ATTENTION_FUNCTIONS["sdpa"] = capturing


def rel_err(x, xq):
    return ((xq.float() - x.float()).norm() / x.float().norm()).item()


@torch.inference_mode()
def main():
    m = "Qwen/Qwen2.5-7B-Instruct"
    tok = AutoTokenizer.from_pretrained(m)
    model = AutoModelForCausalLM.from_pretrained(m, dtype=torch.bfloat16, device_map="cuda",
                                                 attn_implementation="sdpa").eval()
    ids, n, ans = hs.needle_samples(tok, 1, 8192, 123)[0]
    print(f"prompt tokens: {ids.shape[1]} (= {ids.shape[1] // 128} full groups of 128 + {ids.shape[1] % 128})")
    hs.QCFG.target = None
    ref = hs.score_logits(model, ids, n)

    print("\nRelative reconstruction error, mean over layers (and worst layer):")
    print(f"{'scheme':6s} {'tensor':6s} " + " ".join(f"{b:>16d}b" for b in (2, 3, 4, 8)))
    for scheme in ("token", "kivi"):
        hs.QCFG.scheme = scheme
        for name, idx, fn in (("K", 0, hs.quant_k), ("V", 1, hs.quant_v)):
            cells = []
            for b in (2, 3, 4, 8):
                errs = []
                for layer, kv in sorted(CAPTURE.items()):
                    x = kv[idx][0]  # [Hkv, S, D]
                    errs.append(max(rel_err(x[h:h + 1], fn(x[h:h + 1], b)) for h in range(x.shape[0])))
                worst = max(range(len(errs)), key=lambda i: errs[i])
                cells.append(f"{sum(errs) / len(errs):.4f} (L{worst} {errs[worst]:.3f})")
            print(f"{scheme:6s} {name:6s} " + " ".join(f"{c:>17s}" for c in cells))

    print("\nEnd-to-end on this prompt, uniform allocation:")
    heads = [(l, h) for l in range(model.config.num_hidden_layers) for h in range(model.config.num_key_value_heads)]
    for scheme in ("token", "kivi"):
        hs.QCFG.scheme = scheme
        for b in (3, 4):
            hs.QCFG.alloc = {k: b for k in heads}
            hs.QCFG.target = "alloc"
            lp = hs.score_logits(model, ids, n)
            hs.QCFG.target = None
            kl = (ref.exp() * (ref - lp)).sum(-1).mean().item()
            print(f"  {scheme:5s} {b}-bit: KL={kl:.4f}  answer correct={torch.equal(lp.argmax(-1).cpu(), ans)}  "
                  f"pred={tok.decode(lp.argmax(-1))!r} true={tok.decode(ans)!r}")


if __name__ == "__main__":
    main()
