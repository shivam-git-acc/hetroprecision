# Waits for the vLLM sweeps to finish, then runs the HF QuantizedCache (quanto) decode sweeps.
until grep -q VLLM_SWEEP_DONE ~/project/results/vllm_qwen7b.log 2>/dev/null; do sleep 15; done
export HF_HUB_OFFLINE=1
cd ~/project/scripts && source ~/venv/bin/activate
python bench_decode.py --cache quanto4 --out ../results/decode_quanto_qwen7b.csv
python bench_decode.py --cache quanto2 --out ../results/decode_quanto_qwen7b.csv
echo QUANTO_DONE
