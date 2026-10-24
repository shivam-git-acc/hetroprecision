export PATH="$HOME/.local/bin:$PATH"
PY=$(uv python find --managed-python 3.12)
rm -rf ~/venv-vllm && uv venv ~/venv-vllm --python "$PY" && uv pip install --python ~/venv-vllm/bin/python vllm --torch-backend=auto && ~/venv-vllm/bin/python -c "import vllm, torch; print(\"VLLM\", vllm.__version__, torch.__version__)" && echo VLLM_REBUILD_DONE
