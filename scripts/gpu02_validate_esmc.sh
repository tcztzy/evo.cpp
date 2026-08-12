#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

source_dir="${EVO_ESMC_SOURCE_DIR:-$HOME/evo.cpp}"
binary="${EVO_ESMC_BINARY:-$source_dir/build-gpu/evo}"
registry="$source_dir/configs/model-registry.json"
cache_dir="${EVO_ESMC_CACHE_DIR:-$HOME/esmc-oracle/hf-cache}"
reference_image="${EVO_ESMC_REFERENCE_IMAGE:-$HOME/bionemo-pytorch-26.06-py3.sif}"
reference_pythonpath="${EVO_ESMC_REFERENCE_PYTHONPATH:-$HOME/esmc-oracle/pydeps}"
gpu_id="${EVO_ESMC_GPU_ID:-0}"
models="${EVO_ESMC_MODELS:-esmc_300m esmc_600m esmc_6b}"
revision_tag="$(git -C "$source_dir" rev-parse --short=12 HEAD)"
artifact_dir="${EVO_ESMC_ACCEPTANCE_DIR:-$HOME/evo.cpp-artifacts/t22-esmc-$revision_tag}"

test -x "$binary"
test -f "$registry"
test -f "$reference_image"
test -f "$reference_pythonpath/transformers/models/esmc/modeling_esmc.py"
if [[ ! "$gpu_id" =~ ^[0-9]+$ ]]; then
  echo "gpu02_validate_esmc: EVO_ESMC_GPU_ID must be one numeric GPU ID" >&2
  exit 2
fi
for model_id in $models; do
  case "$model_id" in
    esmc_300m|esmc_600m|esmc_6b) ;;
    *)
      echo "gpu02_validate_esmc: unknown canonical model $model_id" >&2
      exit 2
      ;;
  esac
done

mkdir -p "$artifact_dir"
if test -f "$artifact_dir/complete"; then
  echo "gpu02_validate_esmc: acceptance already complete: $artifact_dir" >&2
  exit 2
fi
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used \
  --format=csv,noheader >"$artifact_dir/gpu-before.csv"
sha256sum "$binary" >"$artifact_dir/runtime-sha256.txt"
git -C "$source_dir" rev-parse HEAD >"$artifact_dir/source-commit.txt"
git -C "$source_dir" status --short >"$artifact_dir/source-status.txt"

container_python() {
  /usr/bin/apptainer exec \
    --env "PYTHONPATH=$reference_pythonpath" \
    "$reference_image" python3 "$@"
}

for model_id in $models; do
  case "$model_id" in
    esmc_300m) layer=30 ;;
    esmc_600m) layer=36 ;;
    esmc_6b) layer=80 ;;
  esac
  model_dir="$artifact_dir/$model_id"
  fetch_json="$model_dir/fetch.json"
  runtime_output="$model_dir/runtime.safetensors"
  runtime_load="$runtime_output"
  oracle_dir="$model_dir/oracle"
  input_path="$model_dir/input.fasta"
  native_logits_dir="$model_dir/native-logits"
  native_hidden_dir="$model_dir/native-hidden"
  mkdir -p "$model_dir"

  fetch_arguments=(
    "$source_dir/tools/evo_fetch.py"
    --cache-dir "$cache_dir"
  )
  if test "${EVO_ESMC_FETCH_ONLINE:-0}" != "1"; then
    fetch_arguments+=(--local-files-only)
  fi
  fetch_arguments+=(source "$model_id" --registry "$registry")
  container_python "${fetch_arguments[@]}" >"$fetch_json"
  receipt="$({ container_python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["receipt"])' \
    "$fetch_json"; })"
  upstream_dir="$({ container_python -c \
    'import json,os,sys; print(os.path.dirname(json.load(open(sys.argv[1]))["files"][0]["path"]))' \
    "$fetch_json"; })"

  if ! test -f "$runtime_output" && ! test -f "$runtime_output.index.json"; then
    /usr/bin/apptainer exec \
      --env "PYTHONPATH=$source_dir/tools:$reference_pythonpath" \
      "$reference_image" python3 "$source_dir/tools/convert_esmc_checkpoint.py" \
      --receipt "$receipt" --registry "$registry" --output "$runtime_output" \
      >"$model_dir/conversion.log" 2>&1
  fi
  if test -f "$runtime_output.index.json"; then
    runtime_load="$runtime_output.index.json"
  fi
  test -f "$runtime_load"

  if ! test -f "$oracle_dir/oracle.json"; then
    CUDA_VISIBLE_DEVICES="$gpu_id" /usr/bin/apptainer exec --nv \
      --env "PYTHONPATH=$reference_pythonpath" \
      --env "CUDA_VISIBLE_DEVICES=$gpu_id" \
      "$reference_image" python3 \
      "$source_dir/tools/generate_esmc_official_oracle.py" \
      --model-dir "$upstream_dir" --model-id "$model_id" \
      --registry "$registry" --output-dir "$oracle_dir" --device cuda:0 \
      >"$model_dir/oracle.log" 2>&1
  fi

  printf '>acceptance\nLAGV<mask>ERT\n' >"$input_path"
  test ! -e "$native_logits_dir"
  test ! -e "$native_hidden_dir"
  "$binary" logits -m "$runtime_load" --input "$input_path" \
    --output "$native_logits_dir" --ctx 16 --gpu "$gpu_id" --profile exact \
    >"$model_dir/native-logits.stdout" 2>"$model_dir/native-logits.stderr"
  "$binary" embed -m "$runtime_load" --input "$input_path" \
    --output "$native_hidden_dir" --layer "$layer" --pooling none \
    --ctx 16 --gpu "$gpu_id" --profile exact \
    >"$model_dir/native-hidden.stdout" 2>"$model_dir/native-hidden.stderr"
  container_python "$source_dir/tools/compare_esmc_oracle.py" \
    --oracle-dir "$oracle_dir" \
    --native-logits "$native_logits_dir/000000.npy" \
    --native-hidden "$native_hidden_dir/000000.npy" \
    --model-id "$model_id" --output "$model_dir/comparison.json" \
    >"$model_dir/comparison.stdout"
done

nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used \
  --format=csv,noheader >"$artifact_dir/gpu-after.csv"
find "$artifact_dir" -type f \
  ! -name artifact-sha256.txt -print0 | sort -z | xargs -0 sha256sum \
  >"$artifact_dir/artifact-sha256.txt"
printf 'all three canonical ESMC official-oracle gates passed\n' \
  >"$artifact_dir/complete"
echo "artifact_dir=$artifact_dir"
for model_id in $models; do
  cat "$artifact_dir/$model_id/comparison.json"
done
