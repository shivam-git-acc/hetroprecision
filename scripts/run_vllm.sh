# Waits for the sensitivity sweeps to free the GPU, then runs the three vLLM decode sweeps.
until grep -q SENS_DONE ~/project/results/sens_qwen7b.log; do sleep 15; done
export TMPDIR=~/tmp CUDA_HOME=/usr/local/cuda-13.3 PATH=/usr/local/cuda-13.3/bin:$PATH HF_HUB_OFFLINE=1
cd ~/project/scripts && source ~/venv-vllm/bin/activate
OUT=../results/vllm_qwen7b.csv
VLLM_ATTENTION_BACKEND=FLASH_ATTN python bench_vllm.py --kv auto --out $OUT
VLLM_ATTENTION_BACKEND=FLASHINFER python bench_vllm.py --kv auto --out $OUT
VLLM_ATTENTION_BACKEND=FLASHINFER python bench_vllm.py --kv fp8  --out $OUT
echo VLLM_SWEEP_DONE
