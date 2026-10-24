export PATH="$HOME/.local/bin:$PATH"
export HF_HUB_DISABLE_XET=1
hf_bin=$(ls ~/.cache/uv/archive-v0/*/bin/hf 2>/dev/null | head -1)
uvx --from huggingface_hub hf download Qwen/Qwen2.5-7B-Instruct
echo DL_DONE
