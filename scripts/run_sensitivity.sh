cd ~/project/scripts
source ~/venv/bin/activate
export HF_HUB_DISABLE_XET=1
python head_sensitivity.py --bits 2 --scheme token --out ../results/sens_qwen7b.csv
python head_sensitivity.py --bits 4 --scheme token --out ../results/sens_qwen7b.csv
python head_sensitivity.py --bits 2 --scheme kivi  --out ../results/sens_qwen7b.csv
echo SENS_DONE
