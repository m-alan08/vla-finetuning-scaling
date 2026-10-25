#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$REPO/slurm/env.sh"

LEROBOT_TAG="${LEROBOT_TAG:-v0.6.1}"
LEROBOT_COMMIT="${LEROBOT_COMMIT:-7e241bd630a3719a56157a497ce5d08f244784f1}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
LEROBOT_DIR="$REPO/third_party/lerobot"
EXTRAS="libero,smolvla,pi,dataset"

say() { printf '\n== %s\n' "$*"; }

say "repo: $REPO"
say "venv: $VLA_VENV, cache: $VLA_CACHE, HF_HOME: $HF_HOME"
mkdir -p "$VLA_CACHE" "$HF_HOME" "$LIBERO_CONFIG_PATH" "$REPO/third_party"

if ! command -v uv >/dev/null 2>&1; then
  say "installing uv"
  if command -v python3 >/dev/null 2>&1 && python3 -m pip --version >/dev/null 2>&1; then
    python3 -m pip install --user --quiet uv
  else
    curl -LsSf https://astral.sh/uv/install.sh | sh
  fi
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
say "uv: $(uv --version)"

if [ ! -f "$LEROBOT_DIR/pyproject.toml" ]; then
  say "cloning lerobot $LEROBOT_TAG (Git LFS files skipped, they are not needed to install)"
  TMP_CLONE="$REPO/third_party/.lerobot-clone-$$"
  rm -rf "$TMP_CLONE"
  GIT_LFS_SKIP_SMUDGE=1 git -c filter.lfs.process= -c filter.lfs.smudge=cat -c filter.lfs.required=false \
    clone --depth 1 --branch "$LEROBOT_TAG" https://github.com/huggingface/lerobot.git "$TMP_CLONE"
  if [ ! -f "$TMP_CLONE/pyproject.toml" ]; then
    echo "lerobot clone is incomplete; see the git output above." >&2
    exit 1
  fi
  if [ -e "$LEROBOT_DIR" ]; then
    echo "$LEROBOT_DIR exists but has no pyproject.toml. Move it away and rerun." >&2
    exit 1
  fi
  mv "$TMP_CLONE" "$LEROBOT_DIR"
fi
ACTUAL_COMMIT="$(git -C "$LEROBOT_DIR" rev-parse HEAD)"
if [ "$ACTUAL_COMMIT" != "$LEROBOT_COMMIT" ]; then
  echo "lerobot clone is at $ACTUAL_COMMIT, expected $LEROBOT_COMMIT ($LEROBOT_TAG). Remove $LEROBOT_DIR and rerun." >&2
  exit 1
fi
say "lerobot $LEROBOT_TAG at $ACTUAL_COMMIT"

if [ ! -x "$VLA_VENV/bin/python" ]; then
  say "creating Python $PYTHON_VERSION venv"
  uv venv --python "$PYTHON_VERSION" "$VLA_VENV"
fi

say "installing lerobot[$EXTRAS] and test tools"
INDEX_ARGS=()
if [ -n "${TORCH_INDEX_URL:-}" ]; then
  INDEX_ARGS=(--extra-index-url "$TORCH_INDEX_URL" --index-strategy unsafe-best-match)
fi
uv pip install --python "$VLA_VENV/bin/python" "cmake>=3.20,<4"
PATH="$VLA_VENV/bin:$PATH" uv pip install --python "$VLA_VENV/bin/python" ${INDEX_ARGS[@]+"${INDEX_ARGS[@]}"} -e "$LEROBOT_DIR[$EXTRAS]" "pytest>=8,<9"

source "$VLA_VENV/bin/activate"

if [ "$(uname -s)" = "Linux" ] && [ ! -f "$LIBERO_CONFIG_PATH/config.yaml" ]; then
  say "writing the default LIBERO config to $LIBERO_CONFIG_PATH"
  echo n | python -c "import libero.libero" || echo "LIBERO config step failed; the probe will report details" >&2
fi

say "installed versions"
python - <<'EOF'
import importlib.metadata as m
for name in ["torch", "torchvision", "lerobot", "transformers", "torchcodec", "av", "mujoco", "robosuite", "hf-libero", "huggingface-hub"]:
    try:
        print(f"{name:16} {m.version(name)}")
    except m.PackageNotFoundError:
        print(f"{name:16} not installed")
EOF

say "done. Next: sbatch probe/run_probe.sbatch (cluster) or python probe/probe.py --laptop (Mac)"
