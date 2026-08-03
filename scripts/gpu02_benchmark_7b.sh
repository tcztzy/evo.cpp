#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

if test "${EVO2C_REMOTE_7B_BENCHMARK_WORKER:-0}" != "1"; then
  remote_host="${EVO2C_GPU02_HOST:-gpu02}"
  requested_gpu="${EVO2C_7B_GPU:-3}"
  ssh -o ServerAliveInterval=5 -o ServerAliveCountMax=120 \
    "$remote_host" 'bash -s' -- "$requested_gpu" <<'REMOTE'
set -euo pipefail
gpu="$1"
EVO2C_REMOTE_7B_BENCHMARK_WORKER=1 EVO2C_7B_GPU="$gpu" \
  bash "$HOME/evo2c/scripts/gpu02_benchmark_7b.sh"
REMOTE
  exit 0
fi

source_dir="$HOME/evo2c"
image="$HOME/evo2c-cuda12.8-rocky8.sif"
nix_root="$HOME/.local/share/nix-root"
binary="${EVO2C_7B_BINARY:-$source_dir/build-gpu/evo2c}"
model="${EVO2C_7B_MODEL:-$HOME/evo2c-models/evo2-7b.safetensors.index.json}"
gpu="${EVO2C_7B_GPU:-3}"
input_dir="${EVO2C_7B_INPUT_DIR:-/data/grp_icg/users/tang/evo2c_7b_base_benchmark/inputs}"
official_dir="${EVO2C_7B_OFFICIAL_DIR:-/data/grp_icg/users/tang/evo2c_7b_1m_runtime_compare/official}"
comparison_python="/nix/store/dnsh5jd817k0zddr0k6x3zmyl146bbs6-profile/bin/python3"
expected_checkpoint_sha256="c66645929dc1b9c631f5be656da8726f38946315dc9167000a615dd626fcecf4"
repeat_16="${EVO2C_7B_REPEAT_16:-20}"
repeat_128="${EVO2C_7B_REPEAT_128:-20}"
repeat_1024="${EVO2C_7B_REPEAT_1024:-10}"
minimum_rate_16="${EVO2C_7B_MIN_RATE_16:-1000}"
minimum_rate_128="${EVO2C_7B_MIN_RATE_128:-7000}"
minimum_rate_1024="${EVO2C_7B_MIN_RATE_1024:-9300}"
maximum_load_seconds="${EVO2C_7B_MAX_LOAD_SECONDS:-5}"

if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
  echo "gpu02_benchmark_7b: GPU must be one numeric device ID" >&2
  exit 2
fi
for value in "$repeat_16" "$repeat_128" "$repeat_1024"; do
  if [[ ! "$value" =~ ^[0-9]+$ ]] || ((value < 2)); then
    echo "gpu02_benchmark_7b: repeat counts must be integers >= 2" >&2
    exit 2
  fi
done

test -f "$image"
test -d "$nix_root/store"
test -x "$binary"
test -f "$model"
test -f "$official_dir/official_metrics.json"
test -f "$official_dir/official_prefill_1024.npy"
test -f "$official_dir/official_generation_logits.npy"
for tokens in 16 128 1024; do
  test -f "$input_dir/prompt_${tokens}.fa"
done
grep -q "$expected_checkpoint_sha256" "$official_dir/official_metrics.json"

gpu_uuid="$(
  nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader,nounits |
    tr -d '[:space:]'
)"
applications="$(
  nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits
)"
if [[ "$applications" == *"$gpu_uuid"* ]]; then
  echo "gpu02_benchmark_7b: CUDA device $gpu is not idle" >&2
  exit 2
fi

binary_sha256="$(sha256sum "$binary" | cut -d' ' -f1)"
model_index_sha256="$(sha256sum "$model" | cut -d' ' -f1)"
artifact_dir="${EVO2C_7B_ARTIFACT_DIR:-$HOME/evo2c-artifacts/t19-evo2-7b-prefill-${binary_sha256:0:12}-gpu${gpu}}"
mkdir -p "$artifact_dir"
printf '%s  %s\n%s  %s\n' \
  "$binary_sha256" "$binary" \
  "$model_index_sha256" "$model" >"$artifact_dir/runtime-sha256.txt"
nvidia-smi -i "$gpu" \
  --query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader >"$artifact_dir/gpu-before.csv"

run_native() {
  apptainer exec --nv -B "$nix_root:/nix:ro" -B /data:/data "$image" "$@"
}

for tokens in 16 128 1024; do
  dump_args=()
  if test "$tokens" = "1024"; then
    dump_args=(
      --dump-logits "$artifact_dir/prefill_1024.npy"
    )
  fi
  run_native "$binary" -m "$model" \
    --score "$input_dir/prompt_${tokens}.fa" --ctx 8192 --gpu "$gpu" \
    "${dump_args[@]}" \
    >"$artifact_dir/score_${tokens}.jsonl" \
    2>"$artifact_dir/metrics_${tokens}.log"

  repeat_variable="repeat_${tokens}"
  repeats="${!repeat_variable}"
  sequence="$(
    tail -n +2 "$input_dir/prompt_${tokens}.fa" |
      tr -d '\n\r'
  )"
  repeat_input="$artifact_dir/repeat_${tokens}.fa"
  : >"$repeat_input"
  for ((index = 0; index < repeats; ++index)); do
    printf '>prompt_%d\n%s\n' "$index" "$sequence" >>"$repeat_input"
  done
  run_native "$binary" -m "$model" --score "$repeat_input" \
    --ctx 8192 --gpu "$gpu" \
    >"$artifact_dir/repeat_${tokens}.jsonl" \
    2>"$artifact_dir/repeat_${tokens}.log"
done

prompt="$(
  tail -n +2 "$input_dir/prompt_128.fa" |
    tr -d '\n\r'
)"
printf 'ACGTGCAATGCCGTTAACGTGCAATGCCGTTA' \
  >"$artifact_dir/official-generation.bin"
run_native "$binary" -m "$model" -p "$prompt" -n 32 --ctx 8192 --gpu "$gpu" \
  --top-k 1 --seed 1 \
  --dump-logits "$artifact_dir/generation_logits.npy" \
  >"$artifact_dir/generation.bin" 2>"$artifact_dir/generation.log"

run_native "$comparison_python" "$source_dir/tools/compare_logits.py" \
  --reference "$official_dir/official_prefill_1024.npy" \
  --candidate "$artifact_dir/prefill_1024.npy" \
  --minimum-cosine 0.99999 \
  --minimum-top1-agreement 0.999 \
  --output "$artifact_dir/prefill-comparison.json" >/dev/null
run_native "$comparison_python" "$source_dir/tools/compare_logits.py" \
  --reference "$official_dir/official_generation_logits.npy" \
  --candidate "$artifact_dir/generation_logits.npy" \
  --minimum-cosine 0.99999 \
  --minimum-top1-agreement 1 \
  --reference-bytes "$artifact_dir/official-generation.bin" \
  --candidate-bytes "$artifact_dir/generation.bin" \
  --output "$artifact_dir/generation-comparison.json" >/dev/null
run_native "$comparison_python" \
  "$source_dir/tools/evo2c/prefill_benchmark_gate.py" \
  --artifact-dir "$artifact_dir" \
  --official-metrics "$official_dir/official_metrics.json" \
  --repeat-count "16=$repeat_16" \
  --repeat-count "128=$repeat_128" \
  --repeat-count "1024=$repeat_1024" \
  --minimum-rate "16=$minimum_rate_16" \
  --minimum-rate "128=$minimum_rate_128" \
  --minimum-rate "1024=$minimum_rate_1024" \
  --maximum-load-seconds "$maximum_load_seconds" \
  --output "$artifact_dir/performance-gate.json" >/dev/null

nvidia-smi -i "$gpu" \
  --query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader >"$artifact_dir/gpu-after.csv"
find "$artifact_dir" -maxdepth 1 -type f \
  ! -name artifact-sha256.txt -print0 |
  sort -z |
  xargs -0 sha256sum >"$artifact_dir/artifact-sha256.txt"
echo "artifact_dir=$artifact_dir"
cat "$artifact_dir/performance-gate.json"
