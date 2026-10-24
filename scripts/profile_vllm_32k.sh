# Nsight Systems profile of vLLM BF16 decode at 32K context, batch 4 vs batch 8, to explain why
# batch 8 takes 41 ms/step (ideal ~30 ms) while batch 4 takes 22 ms.
# Kernels inside CUDA graphs are traced per node. Prefill kernels are in the trace too; the
# summary script separates them by kernel name and instance count.
cd ~/project/scripts
export TMPDIR=~/tmp HF_HUB_OFFLINE=1 CUDA_HOME=/usr/local/cuda-13.3 PATH=/usr/local/cuda-13.3/bin:$PATH
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
# run the engine in this process so nsys sees its kernels (default V1 engine runs in a child process)
export VLLM_ENABLE_V1_MULTIPROCESSING=0
source ~/venv-vllm/bin/activate
mkdir -p ../results/nsys
for b in 4 8; do
  /usr/local/cuda-13.3/bin/nsys profile -t cuda,nvtx --cuda-graph-trace=node -f true \
    -o ../results/nsys/vllm_bf16_c32640_b$b \
    python bench_vllm.py --kv auto --contexts 32640 --batches $b --repeats 1 --gen 64 \
      --out ~/tmp/nsys_vllm_dummy.csv
  /usr/local/cuda-13.3/bin/nsys stats -r cuda_gpu_kern_sum -f csv \
    -o ../results/nsys/vllm_bf16_c32640_b$b ../results/nsys/vllm_bf16_c32640_b$b.nsys-rep
done
echo PROFILE_DONE
