#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${ADVERSARIAL_TTS_PYTHON:-python3}" "$root/scripts/run_component.py" acoustic "$@"
