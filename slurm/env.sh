#!/usr/bin/env bash

VLA_REPO="${VLA_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export VLA_REPO
export VLA_VENV="${VLA_VENV:-$VLA_REPO/.venv}"
export VLA_CACHE="${VLA_CACHE:-$VLA_REPO/.cache}"

export HF_HOME="${HF_HOME:-$VLA_CACHE/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$VLA_CACHE/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$VLA_CACHE/uv-python}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export TORCH_HOME="${TORCH_HOME:-$VLA_CACHE/torch}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$VLA_CACHE/libero}"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export TOKENIZERS_PARALLELISM=false

if [ "$(uname -s)" = "Linux" ]; then
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
  export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-$MUJOCO_GL}"
fi

case ":$PATH:" in
  *":$HOME/.local/bin:"*) ;;
  *) export PATH="$HOME/.local/bin:$PATH" ;;
esac

if [ -f "$VLA_VENV/bin/activate" ]; then
  source "$VLA_VENV/bin/activate"
fi
