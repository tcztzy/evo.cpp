#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

if test "${EVO_REMOTE_ESMC_BENCHMARK_WORKER:-0}" != "1"; then
  remote_host="${EVO_GPU02_HOST:-gpu02}"
  requested_gpu="${EVO_ESMC_BENCHMARK_GPU:-1}"
  ssh -o ServerAliveInterval=5 -o ServerAliveCountMax=720 \
    "$remote_host" 'bash -s' -- "$requested_gpu" <<'REMOTE'
set -euo pipefail
EVO_REMOTE_ESMC_BENCHMARK_WORKER=1 EVO_ESMC_BENCHMARK_GPU="$1" \
  bash "$HOME/evo.cpp/scripts/gpu02_benchmark_esmc.sh"
REMOTE
  exit 0
fi

source_dir="${EVO_ESMC_SOURCE_DIR:-$HOME/evo.cpp}"
binary="${EVO_ESMC_BINARY:-$source_dir/build-gpu/evo}"
runtime_image="${EVO_ESMC_RUNTIME_IMAGE:-$HOME/evo.cpp-cuda12.8-rocky8.sif}"
reference_image="${EVO_ESMC_REFERENCE_IMAGE:-$HOME/bionemo-pytorch-26.06-py3.sif}"
reference_pythonpath="${EVO_ESMC_REFERENCE_PYTHONPATH:-$HOME/esmc-oracle/pydeps}"
hf_home="${EVO_ESMC_HF_HOME:-${HF_HOME:-/build/grp_icg/users/tang/.cache/huggingface}}"
gpu="${EVO_ESMC_BENCHMARK_GPU:-1}"
models="${EVO_ESMC_BENCHMARK_MODELS:-esmc_300m esmc_600m esmc_6b}"
lengths="${EVO_ESMC_BENCHMARK_LENGTHS:-128 512 2048}"
warmups="${EVO_ESMC_BENCHMARK_WARMUPS:-2}"
repeats="${EVO_ESMC_BENCHMARK_REPEATS:-5}"
fingerprint_file="$(dirname "$binary")/.evo-source-fingerprint"
fingerprint="$(sed -n '1p' "$fingerprint_file")"
artifact_dir="${EVO_ESMC_BENCHMARK_DIR:-$HOME/evo.cpp-artifacts/esmc-performance-${fingerprint:0:12}-gpu${gpu}}"

test -x "$binary"
test -f "$runtime_image"
test -f "$reference_image"
test -d "$reference_pythonpath/transformers/models/esmc"
test -d "$hf_home/hub"
if [[ ! "$gpu" =~ ^[0-9]+$ ]] || [[ ! "$warmups" =~ ^[0-9]+$ ]] || \
   [[ ! "$repeats" =~ ^[0-9]+$ ]] || ((warmups < 1 || repeats < 2)); then
  echo "gpu02_benchmark_esmc: invalid GPU, warmup, or repeat count" >&2
  exit 2
fi
for length in $lengths; do
  if [[ ! "$length" =~ ^[0-9]+$ ]] || ((length < 2 || length > 2048)); then
    echo "gpu02_benchmark_esmc: length must be in [2, 2048]" >&2
    exit 2
  fi
done
gpu_uuid="$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader,nounits | tr -d '[:space:]')"
applications="$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits)"
if [[ "$applications" == *"$gpu_uuid"* ]]; then
  echo "gpu02_benchmark_esmc: CUDA device $gpu is not idle" >&2
  exit 2
fi
test ! -e "$artifact_dir"
mkdir -p "$artifact_dir"
sha256sum "$binary" "$runtime_image" "$reference_image" >"$artifact_dir/runtime-sha256.txt"
cp -- "$fingerprint_file" "$artifact_dir/source-fingerprint.txt"
nvidia-smi -i "$gpu" \
  --query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader >"$artifact_dir/gpu-before.csv"

length_arguments=()
for length in $lengths; do
  length_arguments+=(--length "$length")
done

sequence_for_length() {
  local encoded_length="$1"
  local residues=$((encoded_length - 2))
  local alphabet="LAGVSERTIDPKQNFYMHWC"
  local sequence=""
  while ((${#sequence} < residues)); do
    sequence+="$alphabet"
  done
  printf '%s' "${sequence:0:residues}"
}

for model_id in $models; do
  case "$model_id" in
    esmc_300m)
      repo="ESMC-300M"; revision="a59b831785f907e96e6a246b1d142bfb76df31ee"
      runtime="$HOME/evo.cpp-artifacts/t22-esmc-300m-58eb96f/esmc_300m/runtime.safetensors"
      ;;
    esmc_600m)
      repo="ESMC-600M"; revision="a7e82012c83126b9eedb055fea9fa84b6c02f094"
      runtime="$HOME/evo.cpp-artifacts/t22-esmc-600m-58eb96f/esmc_600m/runtime.safetensors"
      ;;
    esmc_6b)
      repo="ESMC-6B"; revision="45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a"
      runtime="$HOME/evo.cpp-artifacts/t22-esmc-6b-6aa9e42/esmc_6b/runtime.safetensors.index.json"
      ;;
    *) echo "gpu02_benchmark_esmc: unknown model $model_id" >&2; exit 2 ;;
  esac
  model_dir="$artifact_dir/$model_id"
  snapshot="$hf_home/hub/models--biohub--$repo/snapshots/$revision"
  test -f "$runtime"
  test -f "$snapshot/config.json"
  mkdir -p "$model_dir"
  input="$model_dir/input.fasta"
  : >"$input"
  for length in $lengths; do
    sequence="$(sequence_for_length "$length")"
    for ((index = 0; index < warmups; ++index)); do
      printf '>length_%d_warmup_%d\n%s\n' "$length" "$index" "$sequence" >>"$input"
    done
    for ((index = 0; index < repeats; ++index)); do
      printf '>length_%d_repeat_%d\n%s\n' "$length" "$index" "$sequence" >>"$input"
    done
  done

  CUDA_VISIBLE_DEVICES="$gpu" /usr/bin/apptainer exec --nv \
    -B "$hf_home:$hf_home:ro" \
    --env "PYTHONPATH=$source_dir/tools:$reference_pythonpath" \
    --env "CUDA_VISIBLE_DEVICES=$gpu" \
    "$reference_image" python3 "$source_dir/tools/benchmark_esmc_official.py" \
    --model-dir "$snapshot" --model-id "$model_id" \
    --registry "$source_dir/configs/model-registry.json" \
    --output "$model_dir/official.json" --device cuda:0 \
    --warmups "$warmups" --repeats "$repeats" "${length_arguments[@]}" \
    >"$model_dir/official.stdout" 2>"$model_dir/official.stderr"

  /usr/bin/apptainer exec --nv "$runtime_image" "$binary" logits \
    -m "$runtime" --input "$input" --output "$model_dir/native-output" \
    --ctx 2048 --gpu "$gpu" --profile exact \
    >"$model_dir/native.stdout" 2>"$model_dir/native.stderr"

  /usr/bin/apptainer exec \
    --env "PYTHONPATH=$source_dir/tools:$reference_pythonpath" \
    "$reference_image" python3 "$source_dir/tools/summarize_esmc_benchmark.py" \
    --official "$model_dir/official.json" \
    --native-log "$model_dir/native.stderr" \
    --output "$model_dir/comparison.json" \
    --warmups "$warmups" --repeats "$repeats" "${length_arguments[@]}" \
    >"$model_dir/comparison.stdout"
done

nvidia-smi -i "$gpu" \
  --query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader >"$artifact_dir/gpu-after.csv"
find "$artifact_dir" -type f ! -name artifact-sha256.txt -print0 |
  sort -z | xargs -0 sha256sum >"$artifact_dir/artifact-sha256.txt"
printf 'official versus native ESMC performance benchmark passed: %s\n' "$models" \
  >"$artifact_dir/complete"
echo "artifact_dir=$artifact_dir"
for model_id in $models; do
  cat "$artifact_dir/$model_id/comparison.json"
done
