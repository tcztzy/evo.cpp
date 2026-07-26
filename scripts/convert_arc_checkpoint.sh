#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

if (( $# != 3 )); then
  echo "usage: $0 MODEL_ID CHECKPOINT.pt OUTPUT.evo2" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$(cd -- "${script_dir}/.." && pwd)"
model_id="$1"
checkpoint="$2"
output="$3"
python_bin="${EVO2C_PYTHON:-python3}"
inspector="${EVO2C_INSPECT:-${source_dir}/build/evo2c-inspect}"

config_name="$(
  "$python_bin" - "$source_dir/configs/model-registry.json" "$model_id" <<'PY'
import json
import pathlib
import sys

registry = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
entry = registry["models"].get(sys.argv[2])
if entry is None or sys.argv[2] == "evo2_40b_bionemo_bf16":
    raise SystemExit(f"unsupported Arc model ID: {sys.argv[2]}")
print(entry["config"])
PY
)"
config="${source_dir}/configs/${config_name}"

arguments=(
  "$source_dir/tools/convert_checkpoint.py"
  --input "$checkpoint"
  --config "$config"
  --output "$output"
)
if [[ -n "${EVO2C_SOURCE_SHA256:-}" ]]; then
  arguments+=(--source-sha256 "$EVO2C_SOURCE_SHA256")
fi
if [[ "${EVO2C_DRY_RUN:-0}" == 1 ]]; then
  arguments+=(--dry-run)
fi

PYTHONPATH="${source_dir}/tools${PYTHONPATH:+:${PYTHONPATH}}" \
  "$python_bin" "${arguments[@]}"
if [[ "${EVO2C_DRY_RUN:-0}" != 1 ]]; then
  "$inspector" "$output" --tensor norm.scale
fi
