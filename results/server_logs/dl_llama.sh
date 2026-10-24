export PATH="$HOME/.local/bin:$PATH"
export HF_HUB_DISABLE_XET=1
until grep -q DL_DONE ~/dl_qwen.log || ! pgrep -f "[d]l_qwen.sh" >/dev/null; do sleep 10; done
echo "starting llama at $(date +%H:%M)"
uvx --from huggingface_hub hf download meta-llama/Llama-3.1-8B-Instruct --exclude "original/*"
echo DL_DONE
