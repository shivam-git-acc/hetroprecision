export PATH="$HOME/.local/bin:$PATH"
until grep -q INSTALL_DONE ~/install_torch.log; do
  if ! pgrep -f install_torch.sh >/dev/null; then echo "torch install stopped without INSTALL_DONE"; exit 1; fi
  sleep 10
done
set -e
source ~/venv/bin/activate
uv pip install transformers accelerate datasets optimum-quanto huggingface_hub sentencepiece \
  numpy scipy pandas matplotlib seaborn nvidia-ml-py tqdm ninja jupyter
python -c "import torch, transformers, optimum.quanto; print(\"torch\", torch.__version__, \"transformers\", transformers.__version__)"
echo LIBS_DONE
