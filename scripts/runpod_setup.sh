#!/usr/bin/env bash
# once per fresh pod: bash /workspace/setup.sh   (see runpod.md)
# the volume keeps code, data and the uv download cache; this restores the
# tools and env vars the container disk lost
set -euo pipefail

command -v rsync >/dev/null || { apt-get update -qq && apt-get install -y -qq rsync; }
[ -x "$HOME/.local/bin/uv" ] || curl -LsSf https://astral.sh/uv/install.sh | sh

# login shells (ssh, bash -lc) pick this up
cat > /etc/profile.d/workspace.sh <<'EOF'
export PATH="$HOME/.local/bin:$PATH"
export LC_ALL=C.UTF-8
# only the download cache lives on the volume. python + venv go on the local
# disk: on the network volume import torch took 52s, locally 2-4s
export UV_CACHE_DIR=/workspace/.uv-cache
export UV_PYTHON_INSTALL_DIR=/root/.uv-python
export UV_PROJECT_ENVIRONMENT=/root/venv  # one project per pod
export UV_LINK_MODE=copy  # cache and venv are on different filesystems
# secrets live on the volume, since the container disk is wiped on terminate
[ -f /workspace/.secrets ] && . /workspace/.secrets
EOF

echo "setup done -- then: cd /workspace/<project> && uv sync   (~2.5 min from warm cache)"
