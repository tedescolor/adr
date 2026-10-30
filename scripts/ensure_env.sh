#!/usr/bin/env bash
# Create (once) a Python environment with the adr package and print its python executable.
# Usage: scripts/ensure_env.sh [ENV_DIR]   (default: $CLAUDE_PLUGIN_DATA/venv or ~/.adr/venv)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${1:-${CLAUDE_PLUGIN_DATA:-$HOME/.adr}/venv}"
# Reinstall whenever the package code changes (fingerprint of pyproject.toml and adr/*.py).
VERSION="$(cat "$ROOT/pyproject.toml" "$ROOT"/adr/*.py | cksum | cut -d" " -f1)"
if [ ! -x "$ENV_DIR/bin/python" ] || [ "$(cat "$ENV_DIR/.adr_version" 2>/dev/null)" != "$VERSION" ]; then
  python3 -m venv "$ENV_DIR" >&2
  "$ENV_DIR/bin/pip" install --quiet --upgrade pip >&2
  "$ENV_DIR/bin/pip" install --quiet "$ROOT" >&2
  "$ENV_DIR/bin/pip" install --quiet --force-reinstall --no-deps "$ROOT" >&2
  echo "$VERSION" > "$ENV_DIR/.adr_version"
fi
echo "$ENV_DIR/bin/python"
