#!/usr/bin/env bash
# Create an isolated Python 3.10 venv for Octo (JAX/Flax), separate from the
# repo's PyTorch .venv so the two dependency stacks never collide.
#
# Octo + JAX 0.4.20 + TF 2.15 is a brittle dependency set — these pins are the
# exact combination verified to import and run (octo @ 241fb35, octo-*-1.5).
#
#   bash policies/octo/setup_env.sh           # Linux + NVIDIA GPU (CUDA 12) [default]
#   bash policies/octo/setup_env.sh cpu       # CPU-only (any platform; slow finetune)
#
# Then:  source .venv-octo/bin/activate
set -euo pipefail

PYBIN="${PYBIN:-python3.10}"
VENV="${VENV:-.venv-octo}"
MODE="${1:-gpu}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

command -v "$PYBIN" >/dev/null || { echo "need $PYBIN (Octo targets Python 3.10)"; exit 1; }

"$PYBIN" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip wheel setuptools

# --- JAX (must come before octo so the right jaxlib wins) ---
if [ "$MODE" = "cpu" ]; then
  pip install "jax[cpu]==0.4.20"
else
  # Linux + CUDA 12. (CUDA 11: use jax[cuda11_pip].)
  pip install "jax[cuda12_pip]==0.4.20" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
fi

# --- Octo + dlimp (its data helper), then the pinned compatible stack ---
pip install "git+https://github.com/octo-models/octo.git"
pip install "git+https://github.com/kvablack/dlimp.git"

# Pins that resolve the JAX 0.4.20 <-> TF 2.15 <-> tensorstore/orbax/scipy conflicts:
pip install \
  "numpy==1.26.4" "scipy==1.11.4" \
  "flax==0.7.5" "optax==0.1.7" "distrax==0.1.5" "chex==0.1.85" \
  "orbax-checkpoint==0.4.0" "tensorstore==0.1.45" \
  "tensorflow==2.15.0" "tensorflow-probability==0.23.0" \
  "transformers==4.34.1" "einops" "ml_collections"

# --- FR5 adapter extras (read the LeRobot dataset without torch/lerobot) ---
pip install -r "$HERE/requirements-octo.txt"

echo
echo "== verifying octo import =="
python -c "from octo.model.octo_model import OctoModel; print('octo import OK')"
echo
echo "done. activate with:  source $VENV/bin/activate"
echo "then:  python policies/octo/inference_pretrained.py"
