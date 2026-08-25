#!/usr/bin/env bash
# Download MolCA Stage-1 pretrained weights (Graph encoder + Q-Former initialization).
# Source: Hugging Face `acharkq/MolCA`. Saved to checkpoints/molca/stage1.ckpt,
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="$HERE/../checkpoints/molca"
mkdir -p "$DEST"

python - "$DEST" <<'PY'
import os, sys, shutil
from huggingface_hub import hf_hub_download
dest = sys.argv[1]
# NOTE: confirm the exact filename in the acharkq/MolCA repo if this changes.
src = hf_hub_download(repo_id="acharkq/MolCA", filename="stage1.ckpt")
out = os.path.join(dest, "stage1.ckpt")
shutil.copy(src, out)
print("Saved:", out)
PY
