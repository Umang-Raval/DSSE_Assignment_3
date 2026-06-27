#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR="$ROOT/vendor/mining-design-decisions"
DEEP_LEARNING="$VENDOR/deep_learning"

if [[ ! -d "$DEEP_LEARNING/dl_manager/accelerator" ]]; then
  echo "Cloning mining-design-decisions (deep_learning only)..."
  git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/mining-design-decisions/mining-design-decisions.git \
    "$VENDOR"
  git -C "$VENDOR" sparse-checkout set deep_learning
fi

cd "$ROOT"
if [[ -d .venv ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

pip install setuptools_rust nltk contractions gensim
cd "$DEEP_LEARNING"
python setup.py build_ext --inplace
echo "Rust accelerator built at $DEEP_LEARNING/dl_manager/accelerator*.so"
