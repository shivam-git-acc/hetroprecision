set -e
export PATH="$HOME/.local/bin:$PATH"
PY=$(uv python find --managed-python 3.12)
# main env: same torch build + same libraries
~/venv/bin/python -m pip --version >/dev/null 2>&1 || true
uv pip freeze --python ~/venv/bin/python | grep -viE "^(torch|torchvision|triton|nvidia-|cuda-)" > ~/venv_reqs.txt
rm -rf ~/venv && uv venv ~/venv --python "$PY"
uv pip install --python ~/venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cu130
uv pip install --python ~/venv/bin/python -r ~/venv_reqs.txt
~/venv/bin/python -c "import torch, transformers, sysconfig, os; print(\"MAIN torch\", torch.__version__, torch.cuda.is_available(), \"Python.h:\", os.path.exists(sysconfig.get_paths()[\"include\"]+\"/Python.h\"))"
# vllm env
rm -rf ~/venv-vllm && uv venv ~/venv-vllm --python "$PY"
uv pip install --python ~/venv-vllm/bin/python vllm --torch-backend=auto
~/venv-vllm/bin/python -c "import vllm, torch; print(\"VLLM\", vllm.__version__, torch.__version__)"
echo REBUILD_DONE
