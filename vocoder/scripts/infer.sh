#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${1:-}"
case "$mode" in
  wav) entry_point="inference.py" ;;
  mel) entry_point="inference_e2e.py" ;;
  *)
    echo "Usage: bash scripts/infer.sh {wav|mel} [inference arguments]" >&2
    exit 2
    ;;
esac
shift
exec "${BIGVGAN_PYTHON:-python3}" "$repo_root/$entry_point" "$@"
