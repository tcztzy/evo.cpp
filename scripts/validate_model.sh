#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

if (( $# < 2 || $# > 3 )); then
  echo "usage: $0 MODEL.safetensors[.index.json] GPU_LIST [CONTEXT]" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$(cd -- "${script_dir}/.." && pwd)"
model="$1"
gpu_list="$2"
context="${3:-8192}"
binary="${EVO_BINARY:-${source_dir}/build/evo}"
inspector="${EVO_INSPECT:-${source_dir}/build/evo-inspect}"
artifact_dir="$(mktemp -d "${TMPDIR:-/tmp}/evo.cpp-smoke.XXXXXX")"
trap 'rm -rf -- "$artifact_dir"' EXIT HUP INT TERM

"$inspector" "$model" --tensor norm.scale
"$binary" \
  --model "$model" \
  --prompt ACGTACGT \
  --tokens 2 \
  --ctx "$context" \
  --gpu "$gpu_list" \
  --top-k 1 \
  --seed 0 \
  --dump-logits "$artifact_dir/logits.npy"

if [[ -n "${EVO_EXPECTED_LOGITS:-}" ]]; then
  python3 "$source_dir/tools/compare_logits.py" \
    --reference "$EVO_EXPECTED_LOGITS" \
    --candidate "$artifact_dir/logits.npy"
fi
