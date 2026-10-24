# Results notes — session 1 (2026-10-17, 02:00–04:00)

Model: Qwen2.5-7B-Instruct, BF16, HF transformers 5.20.1, torch 2.14.1+cu130, RTX PRO 5000 Blackwell (110 SMs, 96 MB L2).
Decode timing: 10 warmup + 50 timed steps, median with IQR. Clocks NOT locked (no permission) — IQR is reported instead.

## Platform (`platform.json`)
- Measured DRAM bandwidth: **read 1232 GB/s**, copy 1103 GB/s (4 GiB buffer). Use 1232 GB/s as practical peak.
- `ncu` counters work without admin. Blackwell metric names: `dram__bytes_op_read.sum` (not `dram__bytes_read.sum`), `dram__throughput.avg.pct_of_peak_sustained_elapsed`.
- `nvidia-smi -lgc` (clock locking): not permitted for our user.
- nsys needs `TMPDIR=~/tmp` (no write access to /tmp/nvidia).

## FP16 decode sweep, DynamicCache (`decode_fp16_qwen7b.csv`)
| ctx | batch | TPOT ms | tok/s | KV GB | peak mem GB | effective BW GB/s |
|---|---|---|---|---|---|---|
| 4K | 1 | 14.4 | 70 | 0.24 | 16.1 | 1078 (87% of peak) |
| 32K | 1 | 17.6 | 57 | 1.9 | 18.2 | 972 |
| 4K | 8 | 18.8 | 425 | 1.9 | 21.8 | 912 |
| 32K | 8 | 53.8 | 149 | 15.1 | 37.5 | 563 (46% of peak) |

- Batch 1: weight-bandwidth bound (GEMV = 88% of GPU time at 4K). KV is small for Qwen (56 KB/token, 4 KV heads).
- Large batch × long context: KV (15 GB) ≈ weights (15 GB); step time 2.9× the 4K value; effective bandwidth falls to 46%.
- Ideal at 32K×8 = 30.3 GB / 1232 GB/s = 24.6 ms vs 53.8 ms measured.

## Where the time goes at 32K × batch 8 (nsys, `nsys/decode_c32768_b8*`)
- **48.6% `CatArrayBatchedCopy`** — DynamicCache appends with `torch.cat`, copying the whole KV cache every step (56 calls/step = 28 layers × K,V; 463 µs each).
- 26.1% GEMM (weights); 22.5% `flash_fwd_splitkv_kernel` (attention).
- The attention kernel itself runs at ~1.26 TB/s (538 MB KV per layer in 428 µs) = at peak. The waste is cache management, not attention.

## StaticCache experiment (`decode_static_qwen7b.csv`, `nsys/decode_static_c32768_b8*`)
- Expected to remove the copy. Result: **300 ms/step, 5.6× slower**.
- Cause: the static cache adds an attention mask → SDPA drops FlashAttention for the mem-efficient kernel (`fmha_cutlassF`, 5.3 ms/layer), which needs `repeat_kv` to expand 4 KV heads to 28 (7× traffic; copies = 46.6% of time).

## Takeaways for Evaluation 1
1. Measure first: the obvious "fix" made it worse. Cache layout and attention-kernel selection are coupled.
2. A decode kernel that reads a paged, pre-allocated cache directly with native GQA (no cat, no repeat_kv) is the right baseline — this is what the proposal's FP16 paged split-K kernel is, and what vLLM does.
3. HF DynamicCache numbers overstate the KV cost by ~2×; compare quantization against vLLM / our own kernel, not only HF.

## Next
- vLLM FP16 vs FP8 KV sweep (install: `bash ~/install_vllm.sh`).
- HF QuantizedCache (quanto int4/int2) sweep.
- ncu on the decode attention kernel; head statistics and sensitivity scripts.

---

# Results notes — session 2 (2026-10-17, 10:00–14:00)

Laptop was on a phone hotspot; the server (10.4.25.63) is only reachable on the college network or **college VPN**.

## Environment fixes
- vLLM 0.31.0 failed: `fatal error: Python.h: No such file or directory` (Triton/Inductor compiles a C helper; system Python has no dev headers, no sudo). Same issue would block our own `torch.utils.cpp_extension` kernels.
- Fix: `uv python install 3.12` (uv-managed CPython 3.12.15 ships headers) and rebuilt both envs on it (`server_logs/rebuild_envs.sh`, `rebuild_vllm.sh`). Main env: torch 2.14.1+cu130, transformers 5.20.1. vLLM env: vllm 0.31.0, torch 2.13.0+cu132.
- vLLM FP8 KV on Blackwell needs the FlashInfer backend: `flashinfer-python` 0.7.0 (skipped the 437 MB `flashinfer-cubin`; kernels JIT-compile with `CUDA_HOME=/usr/local/cuda-13.3`).
- Server internet dropped to ~91 KB/s in the morning (was 11–25 MB/s at night). Plan large downloads for night windows.
- Two `setsid nohup` jobs died silently this morning; long jobs now run inside `tmux`.
- vLLM smoke test (GPU shared, timings not valid): same 40% memory budget holds 51,360 tokens BF16 KV vs 88,240 FP8 KV (1.72x). FP8 used FLASHINFER, BF16 used FLASH_ATTN → must compare at the same backend.

## Head activation statistics (`stats_qwen7b.csv`, `stats_qwen7b_keys.csv`; script `head_stats.py`)
FP16, 8 samples per task, seed 1. Query-head level (28 layers x 28 heads = 784).
- Retrieval heads are rare: 8/784 with retrieval score > 0.5, 42 > 0.1. Clustered in L22 kv0 (q-heads 1,3,4,5), L14 kv0 (q 0,2,6), L23 kv1 (q 11), L5 kv2 (q 19).
  Correction (2026-10-19, rechecked from the CSV): the 8 heads > 0.5 are L22.q3 (1.00), L23.q11 (1.00), L14.q0 (0.80), L22.q4 (0.80), L14.q6 (0.78), L5.q19 (0.60), L14.q2 (0.53), L22.q25 (kv3, 0.53). L22.q1 and L22.q5 score 0.45 (just below). Score = fraction of the 40 answer-digit rows (8 samples x 5 digits) whose argmax attention lands on the digit being copied.
- Key outliers (max/median channel |K|): L27.kv1 = 295, all 4 KV heads of L0 = 109–152, L3.kv0 = 72; median head 10.7.
- LM task: mean attention on first token 0.30; none in layers 0–3.
- ~~GSM8K sink = 0.00~~ was a metric artifact. Rerun (10:58) with mass on first 4 / 16 tokens: with a chat template the sink sits at positions 1–3, not token 0.
  Mean attention on token 0 / first 4 / first 16 / last 64: lm 0.30/0.33/0.34/0.36, needle 0.00/0.34/0.35/0.39, gsm8k 0.00/0.38/0.39/0.56.
  ~1/3 of all attention goes to the first few tokens in every task → their KV entries matter disproportionately (cf. KIVI / StreamingLLM keeping them full precision).

## Per-KV-head sensitivity (`sens_qwen7b.csv`, `sens_qwen7b.log`; script `head_sensitivity.py`)
Fake-quantize one KV head (K and V, post-RoPE, inside attention so prefill also sees it), KL(BF16 || perturbed).
Calibration: WikiText-2 train (8 x 4K, KL on last 512 tokens), needle (8 x 8K, passkey at spread depths, KL + exact match on answer), GSM8K train (16, teacher-forced reference solutions). 9 s per head; 112 heads ≈ 17 min per config.
Configs run: 2-bit per-token (done), 4-bit per-token, 2-bit KIVI (keys per-channel, groups of 128 tokens) — the last two finishing ~11:20.

2-bit per-token results:
- One head breaks the model: quantizing only **L0.kv0** gives LM KL 6.14 (all 112 heads: 8.01) and needle accuracy 8/8 → 0/8.
- Top 5 heads = 83–100% of summed single-head KL; max/median ≈ 500–600x. Top 5 = L0.kv0–kv3 + L27.kv1 = the key-outlier heads.
- Needle accuracy drops only for L0.kv0 (0.00), L0.kv1 (0.75), L0.kv2 (0.88), L2.kv1 (0.88).
- Spearman between task rankings: lm–needle 0.47, lm–gsm8k 0.60, needle–gsm8k 0.53 (layer 0 agrees, mid-ranks differ).
- Activation ≠ sensitivity: retrieval score vs needle sensitivity Spearman 0.17; key_outlier vs LM sensitivity 0.37 (but the top heads coincide).
- Caveat: needle KL has low resolution (few answer tokens, confident model); median single-head needle KL ≈ 0.

## Bit allocation on held-out data (`alloc_qwen7b.csv`; script `alloc_eval.py`)
WikiText-2 test (16 x 4K, last 512 tokens), needle seed 123 (16 x 8K), GSM8K test (16). Per-token scheme. Ranking = max over tasks of (head KL / task max KL) from the 2-bit sweep. Avg bits include FP16 scale + zero per 128 values (+0.25 bit).

| allocation | avg bits | WikiText ppl | needle acc | GSM8K ppl |
|---|---|---|---|---|
| BF16 | 16 | 6.31 | 1.00 | 3.51 |
| uniform 8 | 8.25 | 6.34 | 1.00 | 3.63 |
| uniform 4 | 4.25 | 8204 | 0.00 | 15209 |
| uniform 3 | 3.25 | 9012 | 0.00 | 3505 |
| uniform 2 | 2.25 | 12950 | 0.00 | 3308 |
| top1 BF16, rest 2 | 2.37 | 1968 | 0.00 | 240 |
| top5 BF16, rest 2 | 2.86 | 225 | 0.00 | 29.9 |
| top5 8-bit, rest 2 | 2.52 | 249 | 0.00 | 27.7 |
| top10 BF16, rest 2 | 3.48 | 137 | 0.00 | 21.3 |
| random5 BF16, rest 2 (control) | 2.86 | 11716 | 0.00 | 2995 |
| top5 BF16, rest 3 | 3.82 | 8.34 | 0.81 | 3.81 |
| **top5 BF16, rest 4** | **4.77** | **6.73** | **1.00** | **3.72** |

Takeaways:
1. Uniform per-token 4-bit KV destroys Qwen2.5-7B (ppl 6.3 → 8204). Cause: per-token scale is dominated by outlier key channels (up to ~300x median) in a few heads.
2. Keeping 5/112 KV heads (4.5%) in BF16 restores quality at 4.77 avg bits (~3.4x smaller KV than BF16).
3. Random-5 control does nothing → the sensitivity ranking, not the extra bits, is what helps.
4. So per-head mixed precision is necessary for this model; the systems cost of it is the project question.

### Which heads were protected (answer to: same heads for every task?)
`alloc_eval.py` uses ONE combined ranking for all tasks: per head, max over tasks of (head KL / that task's max KL), from the 2-bit per-token sweep. "top5" = the same 5 heads for every task; the other 107 KV heads (all 28 layers) are quantized.

| combined rank | head | LM rank (KL) | needle rank (KL) | GSM8K rank (KL) |
|---|---|---|---|---|
| 1 | L0.kv0 | #1 (6.14) | #1 (7.15) | #1 (6.71) |
| 2 | L0.kv1 | #2 (0.39) | #2 (0.97) | #2 (0.29) |
| 3 | L0.kv2 | #3 (0.25) | #3 (0.31) | #3 (0.27) |
| 4 | L27.kv1 | #4 (0.25) | #5 (≈0, tie) | #27 (0.02) |
| 5 | L0.kv3 | #5 (0.17) | #22 (≈0) | #37 (0.02) |
| 6–10 | L3.kv2, L15.kv3, L4.kv1, L19.kv0, L12.kv0 | | | |

Per-task top 10:
- LM: L0.kv0, L0.kv1, L0.kv2, L27.kv1, L0.kv3, L4.kv1, L19.kv0, L6.kv0, L3.kv0, L21.kv2
- needle: L0.kv0, L0.kv1, L0.kv2, L2.kv1, L27.kv1, L13.kv1, L21.kv2, L16.kv2, L14.kv2, L4.kv1 (below #3 mostly ties at KL≈0)
- GSM8K: L0.kv0, L0.kv1, L0.kv2, L3.kv2, L15.kv3, L19.kv0, L12.kv0, L9.kv2, L16.kv0, L18.kv1

- Top 3 identical for all tasks (L0.kv0–kv2); top-5 and top-10 intersections are only those 3 heads.
- Share of summed single-head KL in layer 0: LM 83%, needle 99%, GSM8K 80%. Next layer ≤3% (LM: L27, needle: L2, GSM8K: L3).
- random5 control heads: L27.kv0, L12.kv1, L24.kv1, L13.kv1, L1.kv1.
- Not yet tested: task-specific allocations (e.g. GSM8K's own top 5 incl. L3.kv2, L15.kv3).

## 4-bit per-token sensitivity sweep (same file `sens_qwen7b.csv`, bits=4)
- All 112 heads at 4-bit: KL lm 7.40, needle 4.76 (acc 0.00), gsm8k 8.70 — consistent with alloc_eval uniform-4.
- Median single-head KL ~10x smaller than at 2-bit (lm 0.00097 vs 0.0097), but L0.kv0 alone still gives lm KL 3.76, gsm8k 3.90, needle acc 8/8 → 0/8.
- Top 5 = 97–100% of summed KL (2-bit: 83–100%) → at 4-bit sensitivity is almost purely the outlier heads.
- Top heads at 4-bit: LM L0.kv0, L0.kv3, L0.kv1, L27.kv1, L0.kv2 | needle L0.kv0 (acc 0), L0.kv3 (acc 0.375), rest ≈0 | GSM8K L0.kv0, L27.kv1 (1.48; was #27 at 2-bit), L0.kv1, L0.kv2, L13.kv0.
- = layer 0's 4 KV heads + L27.kv1 = the 5 highest key-outlier heads = the protected set in alloc_eval.
- Spearman 2-bit vs 4-bit ranking: lm 0.66, gsm8k 0.76, needle 0.36 (needle mostly zero ties).
- Interpretation: at 4-bit, which heads matter is set by key outliers, not by the task; task-specific heads appear only at 2-bit.

## 2-bit KIVI-style sensitivity sweep (`sens_qwen7b.csv`, scheme=kivi; finished 11:23)
KIVI-style = keys quantized per channel (over groups of 128 tokens), values per token.
- All 112 heads at 2-bit KIVI: KL lm **0.74**, needle 1.89 (acc 0.12), gsm8k 0.91.
  Compare per-token: 2-bit lm 8.01, **4-bit lm 7.40** → 2-bit KIVI is ~10x better than 4-bit per-token on LM.
- The outlier heads stop being special: L0.kv0 single-head KL 0.0197 (per-token 2-bit: 6.14, ~300x lower).
  Layer 0 share of summed KL falls from 83% to 13.6% (LM). No single head drops needle accuracy.
- KIVI top 10 (LM): L0.kv0 0.0197, L4.kv1 0.0166, L0.kv1 0.0155, L0.kv2 0.0131, L27.kv1 0.0104, L6.kv0, L26.kv1, L15.kv1, L8.kv3, L8.kv0 — sensitivity is spread much more evenly.
- Interpretation: Qwen's extreme head sensitivity under per-token quantization is caused by key outlier channels; quantizing keys per channel removes most of it. The quantizer's axis matters more than the bit allocation.
- Open: whether mixed precision still helps on top of KIVI at 2-bit (all-heads needle acc 0.12 is still bad) → run alloc_eval with --scheme kivi once the GPU is free (after timing runs).

Full per-layer/per-head tables for every config: `results/HEAD_RESULTS.md` (generated by `scripts/make_results_md.py`).

## Why background jobs die: the booking guard (found 11:25)
- `/usr/local/sbin/sshquota tick` runs every 15 s. If a team has **no active login session**, it runs `pkill -KILL -u <team>` ("stop leftovers (nohup, screen, tmux, ...)"). At slot end it also runs `loginctl terminate-user`.
- So nohup/tmux do NOT keep jobs alive after logout. This killed both rebuild jobs this morning and the vLLM/quanto/alloc tmux jobs at 11:23:45, when the last two SSH connections closed together.
- Rule we follow: keep one SSH session open for the whole slot while jobs run (a keepalive `ssh rtx5000 'while ...; do sleep 60; done'` until 13:58). The laptop must stay on and connected (VPN) during the slot. All jobs must finish before the slot ends.
- The login profile also sets `TMPDIR=$HOME/.tmp` for team accounts.
- Restarted at 11:25: vLLM sweeps → quanto sweeps → KIVI allocation (`alloc_eval.py --scheme kivi`), chained in tmux. First killed vLLM log kept as `vllm_qwen7b_killed_1123.log`.

## vLLM timing (`vllm_qwen7b.csv`, `vllm_qwen7b_fp8_quick.csv`; script `bench_vllm.py`)
vLLM 0.31.0, BF16 weights, CUDA graphs on, prefix caching off, gpu_memory_utilization 0.90, max_model_len 32768.
TPOT = (time for 1+64 tokens − time for 1 token) / 64, 5 repeats (FP8 quick run: 2 repeats). "32K" = 32,640-token prompts (model max 32,768).
**The slot was cut short to 12:00** (planned 14:00): BF16/FLASHINFER stopped after 10/12 rows (no 32K×4, 32K×8); FP8 ran only as a quick 4-row sweep at 11:53; quanto timing and KIVI allocation did not run.

| ctx | batch | HF BF16 (DynamicCache) | vLLM BF16 FLASH_ATTN | vLLM BF16 FLASHINFER | vLLM FP8 FLASHINFER |
|---|---|---|---|---|---|
| 4K | 1 | 14.36 ms | 12.81 | 12.90 | **12.55** (−3%) |
| 4K | 4 | 17.31 | 14.43 | 14.35 | **13.69** (−5%) |
| 4K | 8 | 18.82 | 15.52 | 15.40 | — |
| 8K | 8 | 23.81 | 17.49 | 17.26 | — |
| 16K | 8 | 33.89 | 22.08 | 22.49 | — |
| 32K | 1 | 17.65 | 14.16 | 14.14 | **13.11** (−7%) |
| 32K | 4 | 33.79 | 22.34 | (killed) | **17.29** (−23% vs FLASH_ATTN BF16) |
| 32K | 8 | 53.84 | 41.09 | (killed) | — |

KV cache capacity at gpu_util 0.90: BF16 493,872 tokens (FLASH_ATTN) / 494,752 (FLASHINFER) → 15 concurrent 32K requests; **FP8 973,328 tokens → 29 concurrent 32K requests (1.97x)**.

Observations:
1. vLLM vs HF: 1.12x faster at 4K×1, ~1.5x at 16K–32K (paged cache, no per-step cache copy).
2. BF16 FLASH_ATTN vs FLASHINFER within ~2% → backend is not a confound; FP8 differences come from precision.
3. FP8 KV (fused kernel, no separate dequant pass) is faster everywhere measured, and the gain grows with KV share: −3% at 4K×1 (weight-bound) → −23% at 32K×4 (KV ≈ half the bytes). Quantization done *in the kernel* reduces both memory and latency.
4. Unexplained: BF16 32K×8 = 41.1 ms vs 22.3 ms at 32K×4 (ideal ~30 ms; IQR ±0.2 ms, so real). Candidate for Nsight Compute next session.

## Not done this session (carry over)
- vLLM BF16 FLASHINFER 32K×4/×8; full FP8 sweep (all 12 settings, 5 repeats).
- HF quanto int4/int2 timing (`run_quanto.sh` ready).
- KIVI allocation test (`run_alloc_kivi.sh` ready).
- ncu on decode attention (32K×8 anomaly); Llama-3.1-8B head analysis.

---

# Results notes — session 3 (2026-10-18, slot until 22:00)
- 20:18 connected; keepalive SSH session opened first (booking guard).
- 20:18 started `scripts/run_session3.sh` in tmux (log `results/session3.log`): vLLM BF16 FLASHINFER 32K×4/×8 + full FP8 sweep → HF quanto int4/int2 timing → KIVI allocation test.
- 20:20 queued `scripts/profile_vllm_32k.sh` (nsys, vLLM BF16 FLASH_ATTN decode at 32K×4 vs ×8, CUDA-graph node tracing) to run after the session-3 chain.

## Session 3, step 1: full vLLM BF16 vs FP8 (`vllm_qwen7b.csv`, finished 20:44)
TPOT ms (median of 5). FA = FLASH_ATTN, FI = FLASHINFER. FP8 only runs on FI, so the fair comparison is FP8-FI vs BF16-FI.

| ctx | batch | BF16 FA | BF16 FI | FP8 FI | FP8 vs BF16-FI |
|---|---|---|---|---|---|
| 4K | 1 | 12.81 | 12.90 | 12.58 | −2% |
| 4K | 4 | 14.43 | 14.35 | 13.78 | −4% |
| 4K | 8 | 15.52 | 15.40 | 14.27 | −7% |
| 8K | 1 | 13.12 | 13.14 | 12.73 | −3% |
| 8K | 4 | 15.37 | 15.31 | 14.15 | −8% |
| 8K | 8 | 17.49 | 17.26 | 14.96 | −13% |
| 16K | 1 | 13.45 | 13.42 | 12.82 | −4% |
| 16K | 4 | 17.25 | 17.20 | 14.67 | −15% |
| 16K | 8 | 22.08 | 22.48 | 16.54 | −26% |
| 32K | 1 | 14.16 | 14.13 | 13.13 | −7% |
| 32K | 4 | 22.34 | **29.65** | 16.52 | −44% |
| 32K | 8 | 41.09 | 41.56 | **20.24** | **−51%** |

KV capacity (gpu_util 0.90): BF16 494,752 tokens → FP8 975,088 tokens (1.97x); concurrent 32K requests 15 → 29.

Observations:
1. FP8 gain grows with the KV share of bytes per step: −2% (4K×1, weight-bound) → −51% (32K×8).
2. **Correction to session 2:** the quick "−23% at 32K×4" compared FP8-FI with BF16-FA. Same-backend comparison is −44%.
3. FLASHINFER BF16 is 33% slower than FLASH_ATTN at 32K×4 (29.65 vs 22.34) but equal elsewhere (within 2%).
4. FP8 at 32K×8 = 20.24 ms vs ideal (15.27 GB weights + 7.5 GB KV)/1232 GB/s = 18.5 ms → ~91% of bandwidth bound. BF16 at 32K×8 = 41 ms vs ideal 24.6 ms → only 60%. The BF16 anomaly does not appear in FP8 → profile (step 4) to find out why.

## Session 3, step 2: HF quanto timing (`decode_quanto_qwen7b.csv`, done ~20:55)
HF QuantizedCache (quanto backend, groups of 64, last 128 tokens kept BF16), same harness as the BF16 DynamicCache sweep.

| ctx | batch | HF BF16 TPOT | quanto int4 TPOT | quanto int2 TPOT | HF BF16 peak mem | int4 peak mem |
|---|---|---|---|---|---|---|
| 4K | 1 | 14.36 | 15.48 (+8%) | 15.39 | 16.09 GB | 15.92 GB |
| 4K | 8 | 18.82 | 26.95 (+43%) | 26.32 | 21.81 | 20.46 |
| 16K | 8 | 33.89 | 75.55 (+123%) | 72.88 | 27.52 | 22.13 |
| 32K | 1 | 17.65 | 25.56 (+45%) | 24.43 | 18.17 | 16.89 |
| 32K | 8 | 53.84 | 139.42 (+159%) | (see CSV) | 37.52 | 27.26 |

- Naive quantization: memory down (−10 GB at 32K×8) but decode up to 2.6x slower; int2 ≈ int4 in time → cost is the per-step dequantize-to-BF16 of the whole cache, not the bit count.
- Contrast: fused vLLM FP8 at 32K×8 is 2x FASTER than BF16 with 2x capacity. Implementation decides whether quantization helps.

## Session 3, step 3: KIVI-scheme allocations (`alloc_kivi_qwen7b.csv`) — SUSPECT, do not report yet
Uniform KIVI: 2-bit ppl 11.93 / needle 0.06; 3-bit ppl 6.53 / needle 1.00 / GSM8K KL 0.147; **4-bit ppl 6.36 / needle 0.50 / GSM8K KL 0.408**; 8-bit ≈ BF16.
4-bit worse than 3-bit on needle and GSM8K is non-physical (LM column is monotonic). Suspected quantizer bug in the KIVI path (partial-group padding / sink tokens?). Diagnostic `scripts/diag_kivi.py` queued after the profile (reconstruction error vs bits per scheme + end-to-end 3 vs 4 bit on one needle prompt).

## Session 3, step 4: vLLM profile
- Attempt 1 captured no kernels ("does not contain CUDA kernel data"): the vLLM V1 engine runs in a child process. Log kept as `profile_vllm_32k_attempt1.log`.
- 20:59 rerun with `VLLM_ENABLE_V1_MULTIPROCESSING=0`.

## IMPORTANT correction: attention backend labels (found 21:06)
- vLLM 0.31 **ignores `VLLM_ATTENTION_BACKEND`** (log: "Unknown vLLM environment variable detected"). vLLM's own log lines show: every BF16 run (session 2 and 3) used **FLASH_ATTN**; FP8 runs used **FLASHINFER** (auto-selected, FP8 KV needs it).
- So in `vllm_qwen7b.csv` the rows `kv_dtype=auto, attn_backend=FLASHINFER` actually ran FLASH_ATTN. They are a second BF16 FLASH_ATTN measurement on a different day. 32K×4: 22.34 ms (17 Oct) vs 29.65 ms (18 Oct) → **day-to-day variation of up to 33% at that setting** (clocks not lockable; other rows agree within ~2%).
- So the "FP8 vs BF16" table above compares FP8-FLASHINFER with BF16-FLASH_ATTN (two kernel libraries). Not a pure precision comparison.
- Fix: `bench_vllm.py --backend X` passes `attention_backend=X` to the engine (verified: "Warming up FlashInfer attention"). 21:07 started the real BF16 FLASHINFER sweep → `vllm_qwen7b_bf16_flashinfer.csv`.

## vLLM profile, 32K×4 vs 32K×8, BF16 FLASH_ATTN (`nsys/vllm_bf16_c32640_b{4,8}*`)
- Needed `VLLM_ENABLE_V1_MULTIPROCESSING=0` (engine in-process) and `nsys stats --force-export=true` (a stale empty .sqlite from attempt 1 was being reused).
- Totals are dominated by prefill GEMMs. Decode attention = `flash_fwd_splitkv_kernel`, 3808 calls (28 layers × all decode steps): **193 µs/layer at b4, 335 µs/layer at b8** → ~5.4 ms vs ~9.4 ms of attention per step.
- So the decode attention kernel is NOT the cause of the 41 ms step at 32K×8 (vs 22 ms at b4, ideal ~30 ms): ~20 ms per step is spent elsewhere. Open: per-step nsys timeline restricted to the decode window (NVTX ranges), and check vLLM scheduling (chunked prefill / max_num_batched_tokens) at 8×32K.

## KIVI diagnostic (`diag_kivi.log`)
Relative reconstruction error (mean over layers, worst layer):

| | 2b | 3b | 4b | 8b |
|---|---|---|---|---|
| K per-token | 0.941 (L0 1.57) | 0.428 | 0.193 | 0.011 |
| K KIVI per-channel | 0.304 (L4 0.40) | 0.129 | 0.060 | 0.004 |
| V per-token (both schemes) | 0.661 (L4 1.01) | 0.283 | 0.132 | 0.008 |

- Error falls monotonically with bits in both schemes → the quantizer is correct.
- KIVI cuts key error ~3x; with KIVI, **values (per-token) are now the larger error source**.
- On the diagnostic needle prompt (8162 tokens = 63 full groups + 98), KIVI 3-bit and 4-bit both answer correctly (KL ≈ 0). Per-token 3-bit KL 4.77 vs 4-bit 5.12 — end-to-end KL is non-monotonic once the model is broken.
- The KIVI-allocation anomaly (uniform 4-bit needle 0.50 < 3-bit 1.00) did not reproduce on this prompt → sample-specific. Prime suspect: the last partial group (S mod 128 tokens, which contains the scored answer tokens) is zero-padded and quantized; real KIVI keeps this residual window in full precision.
- **Status: `alloc_kivi_qwen7b.csv` not reportable.** Next: keep the last partial group in BF16 (KIVI residual), rerun the KIVI sweep + allocation with more samples (32).
- 21:15 fixed `head_sensitivity.py` KIVI path: last S % 128 tokens (residual window) kept in full precision for K and V, as in KIVI. Queued `alloc_eval.py --scheme kivi --n 32` → `alloc_kivi_v2_qwen7b.csv` after the BF16 FLASHINFER sweep. Caveats: ranking still comes from the pre-fix KIVI sweep; avg_bits ignores the BF16 residual (GSM8K prompts are short, ~1–2 groups, so a large fraction stays BF16 there).
- 21:22 queued `scripts/gen_examples.py` (real generated text for needle / GSM8K / free-text under BF16, uniform 4-bit, top-5 BF16 + rest 4-bit, random-5 BF16 + rest 2-bit) after the KIVI v2 run → `results/GENERATION_EXAMPLES.md`.

## Clean FP8 vs BF16, both FLASHINFER (`vllm_qwen7b_bf16_flashinfer.csv` + FP8 rows of `vllm_qwen7b.csv`; 21:31)
BF16 run with `--backend FLASHINFER` (verified in vLLM log). Only the KV precision differs. TPOT ms, median of 5.

| ctx | batch | BF16 FA | BF16 FI | FP8 FI | FP8 vs BF16-FI |
|---|---|---|---|---|---|
| 4K | 1 | 12.81 | 12.55 | 12.58 | 0% |
| 4K | 4 | 14.43 | 13.99 | 13.78 | −2% |
| 4K | 8 | 15.52 | 15.11 | 14.27 | −6% |
| 8K | 1 | 13.12 | 12.83 | 12.73 | −1% |
| 8K | 4 | 15.37 | 14.92 | 14.15 | −5% |
| 8K | 8 | 17.49 | 16.66 | 14.96 | −10% |
| 16K | 1 | 13.45 | 13.18 | 12.82 | −3% |
| 16K | 4 | 17.25 | 16.37 | 14.67 | −10% |
| 16K | 8 | 22.08 | 19.57 | 16.54 | −15% |
| 32K | 1 | 14.16 | 13.82 | 13.13 | −5% |
| 32K | 4 | 22.34 | 19.55 | 16.52 | −16% |
| 32K | 8 | 41.09 | 26.07 | 20.24 | **−22%** |

**These replace the earlier "−44% / −51%" figures, which mixed FP8-FLASHINFER with BF16-FLASH_ATTN.**
- Clean FP8 effect: 0% when weight-bound (4K×1) → −22% at 32K×8. Capacity 1.97x at every setting.
- The 32K×8 "anomaly" is FLASH_ATTN-specific: FLASHINFER does BF16 32K×8 in 26.07 ms (ideal 24.6 ms ≈ 94% of bandwidth bound) vs 41.09 ms on FLASH_ATTN. Profile showed FA's decode attention kernel itself is only ~9.4 ms/step, so the extra time is elsewhere in vLLM's FLASH_ATTN path (metadata/scheduling) — open.
- FLASHINFER BF16 ≤ FLASH_ATTN BF16 at every setting; gap grows with context.

## KIVI allocations, fixed residual window (`alloc_kivi_v2_qwen7b.csv`, 32 samples/task; 21:33)
| allocation | avg bits | WikiText ppl | needle acc | GSM8K KL |
|---|---|---|---|---|
| BF16 | 16 | 6.05 | 1.00 | 0 |
| uniform 2 | 2.25 | 11.75 | 0.38 | 0.579 |
| uniform 3 | 3.25 | 6.29 | 1.00 | 0.077 |
| uniform 4 | 4.25 | 6.09 | 1.00 | 0.020 |
| uniform 8 | 8.25 | 6.05 | 1.00 | 0.001 |
| top1 BF16, rest 2 | 2.37 | 10.30 | 0.62 | 0.547 |
| top5 BF16, rest 2 | 2.86 | 8.84 | 0.53 | 0.476 |
| top5 8-bit, rest 2 | 2.52 | 8.86 | 0.81 | 0.444 |
| top10 BF16, rest 2 | 3.48 | 7.69 | 0.81 | 0.350 |
| random5 BF16, rest 2 (control) | 2.86 | 11.26 | 0.66 | 0.480 |
| top5 BF16, rest 3 | 3.82 | 6.27 | 1.00 | 0.070 |
| top5 BF16, rest 4 | 4.77 | 6.09 | 1.00 | 0.015 |

(BF16 ppl 6.05 vs 6.31 in the per-token table: different held-out sample, n=32 vs 16.)
- The non-monotonic 4-bit result is gone → it was the zero-padded partial group. `alloc_kivi_qwen7b.csv` (v1) is superseded.
- With KIVI, **uniform 4-bit is near-lossless** (ppl 6.09 vs 6.05, needle 100%) vs per-token uniform 4-bit ppl 8204.
- At 2-bit, sensitivity-ranked protection helps LM (top5 8.84 vs random5 11.26; top10 7.69) but not needle reliably (top5 0.53 vs random5 0.66; needle has 32 binary outcomes, noisy). With a good quantizer the head ranking matters much less.
- Caveat: ranking taken from the pre-fix KIVI sensitivity sweep; avg bits ignore the BF16 residual window.

## Overall picture for Evaluation 1
1. Quantizer axis first: per-channel keys (KIVI) fix Qwen's outlier problem; per-token 4-bit is unusable, KIVI 4-bit is near-lossless.
2. Head-level mixed precision matters a lot for a bad quantizer (5 heads rescue per-token 4-bit) and moderately at 2-bit with KIVI.
3. Systems: implementation decides cost. HF quanto int4: −10 GB, up to 2.6x slower. vLLM FP8 fused: 1.97x capacity, 0–22% faster. Kernel choice also matters (FLASHINFER vs FLASH_ATTN up to 1.6x at 32K×8).
- 21:35 generation examples done → `results/GENERATION_EXAMPLES.md`: BF16 and top-5-BF16+rest-4-bit correct on needle (16863) and GSM8K; uniform 4-bit and random-5 control produce gibberish on all three prompts.
- 21:35 KIVI generation examples → `results/GENERATION_EXAMPLES_KIVI.md` (BF16, KIVI uniform 4-bit, KIVI uniform 2-bit, KIVI top-5 BF16 + rest 2-bit):
  - KIVI 4-bit: needle 16863 ✅, GSM8K 160 ✅ with near-identical wording to BF16; free text fluent.
  - KIVI 2-bit: needle "16883" (one digit off), top-5 variant "1689"; GSM8K fluent but wrong reasoning ("Initial 40 GB: 20 minutes" — took 40% of 200 GB as 40 GB). Caveat: 2-bit GSM8K answers also hit the 320-token limit, so the ✅/❌ check on "160" is partly truncation.
  - Contrast: per-token 4-bit fails catastrophically (gibberish); KIVI 2-bit fails subtly (fluent but wrong). Perplexity alone does not show this.

## Session 3 summary (ended ~21:40, slot until 22:00)
Done: full vLLM FP8 sweep; HF quanto timing; vLLM profile (32K×4 vs ×8); found + fixed vLLM backend-selection bug and re-measured BF16 on FLASHINFER (clean FP8 comparison: 0% → −22%); found + fixed KIVI residual-window bug (KIVI 4-bit near-lossless); generation examples (per-token and KIVI).
Open for next session: why vLLM FLASH_ATTN path is 41 ms at 32K×8 (attention kernel only ~9.4 ms of it); rerun KIVI sensitivity sweep with the fix; Llama-3.1-8B comparison; Evaluation 1 slides.
- 2026-10-18 Evaluation 1 slide deck (15 slides) published: https://claude.ai/artifact/JUQfBtR1pC4j2WX6mA3jQd · source copy in slides_eval1/
