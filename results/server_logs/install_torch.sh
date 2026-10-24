set -e
cd ~
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv ~/venv --python 3.12
source ~/venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130 \
  || uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(\"torch\", torch.__version__, \"cuda\", torch.version.cuda, \"available\", torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)); x=torch.randn(4096,4096,device=\"cuda\",dtype=torch.float16); print(\"matmul ok\", (x@x).float().abs().mean().item())"
echo INSTALL_DONE
