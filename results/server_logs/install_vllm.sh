set -e
export PATH="$HOME/.local/bin:$PATH"
uv venv ~/venv-vllm --python 3.12 --allow-existing
source ~/venv-vllm/bin/activate
uv pip install vllm --torch-backend=auto
python -c "import vllm, torch; print(\"vllm\", vllm.__version__, \"torch\", torch.__version__, torch.version.cuda)"
echo VLLM_DONE
