# Waits for the quanto timing sweeps, then evaluates bit allocations with the KIVI-style scheme.
until grep -q QUANTO_DONE ~/project/results/decode_quanto_qwen7b.log 2>/dev/null; do sleep 15; done
export HF_HUB_OFFLINE=1
cd ~/project/scripts && source ~/venv/bin/activate
python alloc_eval.py --scheme kivi --out ../results/alloc_kivi_qwen7b.csv
echo ALLOC_KIVI_DONE
