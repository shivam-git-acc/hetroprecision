# Session 3 (2026-10-18, slot until 22:00): carry-over queue, run strictly one after another.
set -x
cd ~/project/scripts
export TMPDIR=~/tmp HF_HUB_OFFLINE=1 CUDA_HOME=/usr/local/cuda-13.3 PATH=/usr/local/cuda-13.3/bin:$PATH

# 1. vLLM: missing BF16 FLASHINFER rows, then the full FP8 sweep (same CSV as session 2)
source ~/venv-vllm/bin/activate
VLLM_ATTENTION_BACKEND=FLASHINFER python bench_vllm.py --kv auto --contexts 32640 --batches 4,8 --out ../results/vllm_qwen7b.csv
VLLM_ATTENTION_BACKEND=FLASHINFER python bench_vllm.py --kv fp8 --out ../results/vllm_qwen7b.csv
echo STEP1_VLLM_DONE
deactivate

# 2. HF QuantizedCache (quanto) timing
source ~/venv/bin/activate
python bench_decode.py --cache quanto4 --out ../results/decode_quanto_qwen7b.csv
python bench_decode.py --cache quanto2 --out ../results/decode_quanto_qwen7b.csv
echo STEP2_QUANTO_DONE

# 3. Bit allocations with the KIVI-style scheme
python alloc_eval.py --scheme kivi --out ../results/alloc_kivi_qwen7b.csv
echo STEP3_ALLOC_KIVI_DONE
echo SESSION3_DONE
