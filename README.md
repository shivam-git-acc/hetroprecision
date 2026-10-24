# hetroprecision: the systems cost of per-head KV-cache quantization

Course project for CS3.406 *Programming AI Accelerators* (IIIT Hyderabad), by Shivam Patel, Suyash Pande and Divyanshu Jain.

**Question.** Research papers say different attention heads can use different KV-cache bit-widths at no inference cost. Is that true on real hardware, or do dispatch, load imbalance and unpacking eat the savings?

**What this repository contains.** The Evaluation 1 work: baselines, head analysis and profiling on Qwen2.5-7B-Instruct. It answers two questions before any custom kernel is written:

1. Which KV heads are sensitive to quantization, and does it depend on the task?
2. Does KV quantization reduce or increase decoding cost, depending on how it is implemented?

The custom paged mixed-precision decode kernel is the next stage (see [Next steps](#next-steps)) and is **not in this repository yet**.

## Setup

| | |
|---|---|
| Model | Qwen2.5-7B-Instruct, BF16: 28 layers, 28 query heads, 4 KV heads per layer (112 KV heads to rank), 56 KB of KV per token |
| GPU | NVIDIA RTX PRO 5000 Blackwell (sm_120, 110 SMs, 48 GB); measured DRAM read bandwidth 1,232 GB/s, copy 1,103 GB/s (`results/platform.json`) |
| Software | transformers 5.20.1, torch 2.14.1+cu130, vLLM 0.31.0, optimum-quanto, Nsight Systems / Compute |
| Tasks | WikiText-2 (perplexity), needle-in-a-haystack passkey retrieval (generated, 8K), GSM8K with chain-of-thought (KL on teacher-forced solutions) |
| Protocol | Rankings calibrated on train splits and scored on held-out data; timings are medians with IQR (10 warmup + 50 timed decode steps for HF, 5 repeats for vLLM) |

## Key findings

1. **The quantizer matters more than the bit budget.** Uniform per-token 4-bit KV breaks the model (WikiText perplexity 6.31 to 8,204; needle accuracy 100% to 0%). The cause is a few heads with huge key outlier channels (up to ~300x the median channel). A KIVI-style quantizer (keys per channel, values per token) makes uniform 4-bit nearly lossless (perplexity 6.05 to 6.09, needle accuracy 100%).
2. **A handful of heads dominate sensitivity.** Quantizing only L0.kv0 to 2-bit drops needle accuracy from 8/8 to 0/8. The top 5 of 112 heads carry 83 to 100% of the summed single-head damage. Keeping those 5 heads in BF16 and quantizing the rest to 4-bit gives perplexity 6.73 and 100% needle accuracy at 4.77 average bits (about 3.4x smaller than BF16). Protecting 5 random heads does nothing, so the sensitivity ranking is what helps.
3. **Task dependence shows up only at low bits.** The top 3 heads (all in layer 0) are identical across the three tasks. Rankings agree only moderately (Spearman 0.47 to 0.60). At 4-bit the ranking is set almost entirely by key outliers; task-specific heads appear at 2-bit.
4. **Naive quantization saves memory but costs latency.** HF `QuantizedCache` (quanto) at 32K context, batch 8 saves about 10 GB but decodes 2.6x slower (139 ms vs 54 ms per token). INT2 is no faster than INT4, so the cost is dequantizing the whole cache every step, not the bit-width.
5. **Fused quantization helps.** vLLM with an FP8 KV cache (same FlashInfer backend as the BF16 comparison) is equal at 4K context, batch 1 and 22% faster at 32K, batch 8, with about 2x the KV capacity (975K vs about 490K tokens).
6. **Measure before optimizing.** At 32K, batch 8, HF's default `DynamicCache` spends 48.6% of each step in `torch.cat` copies, while the attention kernel runs at about peak bandwidth. Switching to `StaticCache` made decoding 5.6x slower (300 ms), because the added mask pushes SDPA off FlashAttention onto a kernel that expands 4 KV heads to 28.

## Results

### Decode cost (time per output token, ms)

| Context | Batch | HF BF16 | HF quanto int4 | vLLM BF16 (FlashInfer) | vLLM FP8 (FlashInfer) | FP8 vs BF16 |
|---|---|---|---|---|---|---|
| 4K | 1 | 14.4 | 15.5 | 12.5 | 12.6 | 0% |
| 4K | 8 | 18.8 | 27.0 | 15.1 | 14.3 | -6% |
| 16K | 8 | 33.9 | 75.5 | 19.6 | 16.5 | -15% |
| 32K | 1 | 17.6 | 25.6 | 13.8 | 13.1 | -5% |
| 32K | 4 | 33.8 | 75.4 | 19.6 | 16.5 | -16% |
| 32K | 8 | 53.8 | 139.4 | 26.1 | 20.2 | -22% |

HF runs use 32,768-token contexts and vLLM runs use 32,640-token prompts (the model maximum is 32,768). At 32K, batch 8 the FP8 step is about 91% of the bandwidth bound, and the FlashInfer BF16 step about 94%. Full 12-setting sweeps are in `results/vllm_qwen7b*.csv` and `results/decode_*.csv`.

### Bit allocation on held-out data, per-token scheme

| Allocation | Avg bits | WikiText ppl | Needle acc | GSM8K ppl |
|---|---|---|---|---|
| BF16 | 16 | 6.31 | 1.00 | 3.51 |
| uniform 8-bit | 8.25 | 6.34 | 1.00 | 3.63 |
| uniform 4-bit | 4.25 | 8,204 | 0.00 | 15,209 |
| uniform 2-bit | 2.25 | 12,950 | 0.00 | 3,308 |
| top-5 heads BF16, rest 2-bit | 2.86 | 225 | 0.00 | 29.9 |
| random-5 heads BF16, rest 2-bit (control) | 2.86 | 11,716 | 0.00 | 2,995 |
| top-5 heads BF16, rest 3-bit | 3.82 | 8.34 | 0.81 | 3.81 |
| **top-5 heads BF16, rest 4-bit** | **4.77** | **6.73** | **1.00** | **3.72** |

### Bit allocation, KIVI-style scheme (32 samples per task)

| Allocation | Avg bits | WikiText ppl | Needle acc |
|---|---|---|---|
| BF16 | 16 | 6.05 | 1.00 |
| uniform 2-bit | 2.25 | 11.75 | 0.38 |
| uniform 3-bit | 3.25 | 6.29 | 1.00 |
| uniform 4-bit | 4.25 | 6.09 | 1.00 |
| top-5 BF16, rest 2-bit | 2.86 | 8.84 | 0.53 |
| random-5 BF16, rest 2-bit (control) | 2.86 | 11.26 | 0.66 |

With a good quantizer the head ranking matters much less. It still helps perplexity at 2-bit, but not reliably for needle retrieval (noisy: 32 binary outcomes). Generated text under each setting is in `results/GENERATION_EXAMPLES.md` and `results/GENERATION_EXAMPLES_KIVI.md`; per-layer, per-head tables are in `results/HEAD_RESULTS.md`.

## Repository layout

| Path | Contents |
|---|---|
| `scripts/` | Experiment code (below) |
| `results/` | CSVs, logs, `NOTES.md` (running lab notes, with corrections), `HEAD_RESULTS.md`, generation examples, Nsight traces in `nsys/` |
| `results/server_scripts_snapshot/`, `results/server_logs/` | Copies of the scripts and environment-setup logs as run on the lab server |
| `slides_eval1/` | Source of the Evaluation 1 slide deck |
| `*.pdf`, `*.docx` | Project proposal, Evaluation 1 plan and course brief |

| Script | Purpose |
|---|---|
| `platform_check.py` | Measure real DRAM bandwidth and record device properties |
| `bench_decode.py` | HF decode sweep (context x batch) for the dynamic, static and quanto caches |
| `bench_vllm.py` | vLLM decode sweep for BF16 and FP8 KV, with capacity and TPOT |
| `head_stats.py` | Per-query-head attention statistics (entropy, sink, local fraction, retrieval score, key outliers) |
| `head_sensitivity.py` | Per-KV-head sensitivity: fake-quantize one head's post-RoPE K and V inside attention, score KL divergence from BF16 |
| `alloc_eval.py` | Evaluate per-head bit allocations on held-out data |
| `gen_examples.py` | Generate real text under each allocation |
| `diag_kivi.py` | Diagnostic that found the KIVI residual-window bug |
| `make_results_md.py` | Build `results/HEAD_RESULTS.md` from the CSVs |

## Reproducing

```bash
cd scripts
python platform_check.py --out ../results/platform.json
python bench_decode.py --cache dynamic --out ../results/decode_fp16_qwen7b.csv   # also: static, quanto4, quanto2
python head_stats.py --out ../results/stats_qwen7b.csv
python head_sensitivity.py --bits 2 --scheme token --out ../results/sens_qwen7b.csv  # also: --bits 4, --scheme kivi
python alloc_eval.py --scheme token --out ../results/alloc_qwen7b.csv                # also: --scheme kivi
python bench_vllm.py --kv auto --backend FLASHINFER --out ../results/vllm_qwen7b.csv
python bench_vllm.py --kv fp8  --backend FLASHINFER --out ../results/vllm_qwen7b.csv
```

Use `--backend` to pick vLLM's attention backend. vLLM 0.31 ignores the `VLLM_ATTENTION_BACKEND` variable that `run_vllm.sh` and `run_session3.sh` still set. The scripts need a Blackwell-capable PyTorch build (cu128 or newer) and Python headers (`Python.h`) for Triton and vLLM compilation.

## Caveats and corrections

These are recorded in detail in `results/NOTES.md`.

- **Timing conditions.** GPU clocks could not be locked and the GPU was shared, so medians come with IQR. The same BF16 setting (32K, batch 4) varied by up to 33% between two days; other settings agreed within about 2%.
- **Backend labels were wrong at first.** Early "BF16 FlashInfer" rows actually ran FlashAttention because vLLM ignored the environment variable. The BF16 FlashInfer rows were re-measured and the FP8 comparison above uses matched backends. Earlier "-44% / -51%" figures in the notes were superseded.
- **KIVI bug fixed.** The first KIVI allocation run (`alloc_kivi_qwen7b.csv`) zero-padded the last partial group and gave a non-monotonic 4-bit result; it is superseded by `alloc_kivi_v2_qwen7b.csv`, which keeps the residual window in full precision.
- **Remaining limits.** The KIVI head ranking still comes from the pre-fix sensitivity sweep, and average bits ignore the BF16 residual window. Needle accuracy is noisy (16 to 32 samples). Only Qwen2.5-7B has been analysed; Llama-3.1-8B is pending.
- **Open question.** At 32K, batch 8, vLLM's FlashAttention path takes 41 ms per step against 26 ms for FlashInfer, but its attention kernel accounts for only about 9 ms of it. The remaining time has not been explained.

## Next steps

Per the plan in the Evaluation 1 document:

- A paged split-K FP16 decode kernel validated against PyTorch SDPA, plus INT4/INT8 `quantize_pack` with fused unpacking.
- End-to-end generation with that kernel, compared against the baselines above.
- Per-head mixed precision from this repository's allocation, comparing dispatch strategies (templated kernels, runtime branching, grouped launch) with Nsight evidence.
- Llama-3.1-8B as a second model family.
