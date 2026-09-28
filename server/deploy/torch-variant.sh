#!/usr/bin/env sh
# Reinstala torch (y torchvision/torchaudio si están) con la variante pedida, en las mismas
# versiones que fijó uv.lock (CPU). Uso: torch-variant.sh cu128 | rocm6.4 | xpu | cpu
set -eu
variant=${1:-cpu}
[ "$variant" = "cpu" ] && exit 0
py=/app/.venv/bin/python
pkgs=$($py - <<'PY'
import importlib.metadata as m
out = []
for name in ("torch", "torchvision", "torchaudio"):
    try:
        out.append(f"{name}=={m.version(name).split('+')[0]}")
    except m.PackageNotFoundError:
        pass
print(" ".join(out))
PY
)
uv pip install --python "$py" --reinstall --no-deps \
  --index-url "https://download.pytorch.org/whl/$variant" $pkgs
$py -c "import torch; print('torch', torch.__version__)"
