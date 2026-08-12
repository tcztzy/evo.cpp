#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

run_remote_worker() {
  local source_dir="$HOME/evo.cpp"
  local image="$HOME/evo.cpp-cuda12.8-rocky8.sif"
  local nix_root="$HOME/.local/share/nix-root"
  local binary="$source_dir/build-gpu/evo"
  local model="$HOME/evo.cpp-models/evo2-40b-bionemo-bf16.safetensors.index.json"
  local expected_greedy_sha256="b28b7e7e6b70661dfee15d5290c4bca097ca145f721c4fbc4de73ad1d1660b8b"
  local gpu_list="${EVO_BIONEMO_GPU_LIST:-0,1,2,3}"
  local gpu_tag="${gpu_list//,/}"
  local oracle_dir="${EVO_BIONEMO_ORACLE_DIR:-$HOME/evo.cpp-artifacts/t22-bionemo-oracle}"
  local comparison_python="/nix/store/dnsh5jd817k0zddr0k6x3zmyl146bbs6-profile/bin/python3"
  local model_sha256
  local greedy_sha256
  local artifact_dir
  local score_input
  local prompt="ACGTACGTACGTACGT"
  local applications
  local unavailable

  if [[ ! "$gpu_list" =~ ^[0-9]+(,[0-9]+){1,3}$ ]]; then
    echo "gpu02_validate_bionemo_40b: GPU list must contain two to four numeric IDs" >&2
    exit 2
  fi

  wait_for_exclusive_devices() {
    while true; do
      applications="$(
        nvidia-smi --query-compute-apps=gpu_uuid \
          --format=csv,noheader,nounits
      )"
      unavailable=""
      while IFS=, read -r index uuid mode; do
        index="${index//[[:space:]]/}"
        uuid="${uuid//[[:space:]]/}"
        mode="${mode//[[:space:]]/}"
        if [[ ",$gpu_list," != *",$index,"* ]]; then
          continue
        fi
        if test "$mode" = "Exclusive_Process" &&
           [[ "$applications" == *"$uuid"* ]]; then
          unavailable="${unavailable}${unavailable:+,}$index"
        fi
      done < <(
        nvidia-smi --query-gpu=index,uuid,compute_mode \
          --format=csv,noheader,nounits
      )
      if test -z "$unavailable"; then
        return
      fi
      echo "gpu02_validate_bionemo_40b: waiting for exclusive GPU(s) $unavailable" >&2
      sleep 30
    done
  }

  test -f "$image"
  test -x "$binary"
  test -f "$model"
  test -f "$source_dir/tests/vectors/t13_short.fasta"
  model_sha256="$(sha256sum "$model" | cut -d' ' -f1)"
  artifact_dir="$HOME/evo.cpp-artifacts/t22-bionemo-bf16-${model_sha256:0:8}-gpu${gpu_tag}"
  score_input="$artifact_dir/score-short.fasta"
  mkdir -p "$artifact_dir"
  cp -- "$source_dir/tests/vectors/t13_short.fasta" "$score_input"
  wait_for_exclusive_devices
  nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used \
    --format=csv,noheader >"$artifact_dir/gpu-before.csv"
  sha256sum "$binary" >"$artifact_dir/runtime-sha256.txt"
  printf '%s  %s\n' "$model_sha256" "$model" >>"$artifact_dir/runtime-sha256.txt"

  apptainer exec --nv -B "$nix_root:/nix:ro" "$image" \
    "$binary" -m "$model" --score "$score_input" --ctx 8192 \
    --gpu "$gpu_list" --profile fast-q8-kv \
    --dump-logits "$artifact_dir/native-score-logits.npy" \
    >"$artifact_dir/native-score.jsonl" \
    2>"$artifact_dir/native-score-metrics.log"

  apptainer exec --nv -B "$nix_root:/nix:ro" "$image" \
    "$binary" -m "$model" -p "$prompt" -n 8 --ctx 8192 \
    --gpu "$gpu_list" --profile fast-q8-kv --top-k 1 --seed 1 \
    --dump-logits "$artifact_dir/native-greedy-logits.npy" \
    >"$artifact_dir/native-greedy.bin" \
    2>"$artifact_dir/native-greedy-metrics.log"

  test "$(stat --printf='%s' "$artifact_dir/native-greedy.bin")" = "8"
  greedy_sha256="$(
    sha256sum "$artifact_dir/native-greedy.bin" | cut -d' ' -f1
  )"
  if test "$greedy_sha256" != "$expected_greedy_sha256"; then
    echo "gpu02_validate_bionemo_40b: greedy output differs from BioNeMo" >&2
    exit 2
  fi
  grep -q '^evo_metrics ' "$artifact_dir/native-score-metrics.log"
  grep -q '^evo_metrics ' "$artifact_dir/native-greedy-metrics.log"
  if test -f "$oracle_dir/bionemo-score-logits.npy" &&
     test -f "$oracle_dir/bionemo-greedy.bin"; then
    apptainer exec -B "$nix_root:/nix:ro" "$image" \
      "$comparison_python" "$source_dir/tools/compare_logits.py" \
      --reference "$oracle_dir/bionemo-score-logits.npy" \
      --candidate "$artifact_dir/native-score-logits.npy" \
      --minimum-cosine 0.999 \
      --reference-bytes "$oracle_dir/bionemo-greedy.bin" \
      --candidate-bytes "$artifact_dir/native-greedy.bin" \
      --output "$artifact_dir/bionemo-comparison.json" >/dev/null
  else
    echo "gpu02_validate_bionemo_40b: oracle artifacts absent; logits comparison skipped" >&2
  fi
  nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used \
    --format=csv,noheader >"$artifact_dir/gpu-after.csv"
  find "$artifact_dir" -maxdepth 1 -type f \
    ! -name artifact-sha256.txt -print0 |
    sort -z |
    xargs -0 sha256sum >"$artifact_dir/artifact-sha256.txt"
  echo "artifact_dir=$artifact_dir"
  cat "$artifact_dir/native-score.jsonl"
  cat "$artifact_dir/native-score-metrics.log"
  echo "greedy_sha256=$greedy_sha256"
  cat "$artifact_dir/native-greedy-metrics.log"
}

if test "${EVO_REMOTE_BIONEMO_VALIDATION_WORKER:-0}" = "1"; then
  run_remote_worker
  exit 0
fi

remote_host="${EVO_GPU02_HOST:-gpu02}"
gpu_list="${EVO_BIONEMO_GPU_LIST:-0,1,2,3}"
ssh_options=(
  -o ConnectTimeout=10
  -o ServerAliveInterval=5
  -o ServerAliveCountMax=3
)
launch_output=""
until launch_output="$(
  ssh "${ssh_options[@]}" "${remote_host}" 'bash -s' -- "$gpu_list" <<'REMOTE_LAUNCH'
set -euo pipefail
gpu_list="$1"
worker="$HOME/evo.cpp/scripts/gpu02_validate_bionemo_40b.sh"
binary="$HOME/evo.cpp/build-gpu/evo"
worker_sha256="$(sha256sum "$worker" | cut -d' ' -f1)"
binary_sha256="$(sha256sum "$binary" | cut -d' ' -f1)"
job_sha256="$(
  printf '%s\n%s\n%s\n' "$worker_sha256" "$binary_sha256" "$gpu_list" |
    sha256sum |
    cut -d' ' -f1
)"
job_dir="$HOME/evo.cpp-artifacts/.bionemo-validation-jobs/$job_sha256"
pid_file="$job_dir/pid"
status_file="$job_dir/status"
log_file="$job_dir/output.log"
mkdir -p "$job_dir"

running=0
pid=""
if test -f "$pid_file"; then
  pid="$(cat "$pid_file")"
  if test -n "$pid" && kill -0 "$pid" 2>/dev/null; then
    command_line="$(tr '\0' ' ' <"/proc/$pid/cmdline")"
    if [[ "$command_line" == *evo-bionemo-validation-runner* ]]; then
      running=1
    fi
  fi
fi
if test "$running" = "0" && test -f "$status_file" &&
   test "$(cat "$status_file")" != "0"; then
  rm -f -- "$status_file"
fi
if test "$running" = "0" && ! test -f "$status_file"; then
  : >"$log_file"
  nohup setsid bash -c '
    worker="$1"
    log_file="$2"
    status_file="$3"
    gpu_list="$4"
    EVO_REMOTE_BIONEMO_VALIDATION_WORKER=1 \
      EVO_BIONEMO_GPU_LIST="$gpu_list" \
      bash "$worker" >"$log_file" 2>&1
    code=$?
    status_partial="${status_file}.$$.partial"
    printf "%s\n" "$code" >"$status_partial"
    mv -- "$status_partial" "$status_file"
  ' evo-bionemo-validation-runner "$worker" "$log_file" "$status_file" \
    "$gpu_list" \
    </dev/null >/dev/null 2>&1 &
  pid="$!"
  printf '%s\n' "$pid" >"$pid_file"
fi
printf 'job_dir=%s\npid=%s\n' "$job_dir" "$pid"
REMOTE_LAUNCH
)"; do
  echo "gpu02_validate_bionemo_40b: SSH unavailable while launching; retrying" >&2
  sleep 5
done

job_dir="$(printf '%s\n' "$launch_output" | sed -n 's/^job_dir=//p')"
worker_pid="$(printf '%s\n' "$launch_output" | sed -n 's/^pid=//p')"
if test -z "$job_dir"; then
  echo "gpu02_validate_bionemo_40b: remote launcher returned invalid state" >&2
  exit 2
fi
echo "gpu02_validate_bionemo_40b: remote worker PID ${worker_pid:-completed}"

while true; do
  probe=""
  if probe="$(
    ssh "${ssh_options[@]}" "${remote_host}" 'bash -s' -- "$job_dir" <<'REMOTE_PROBE'
set -euo pipefail
job_dir="$1"
pid="$(cat "$job_dir/pid" 2>/dev/null || true)"
if test -f "$job_dir/status"; then
  printf 'done=%s\n' "$(cat "$job_dir/status")"
elif test -n "$pid" && kill -0 "$pid" 2>/dev/null; then
  echo "running=1"
else
  echo "lost=1"
fi
REMOTE_PROBE
  )"; then
    if [[ "$probe" == done=* ]]; then
      worker_status="${probe#done=}"
      if [[ -z "$worker_status" || "$worker_status" == *[!0-9]* ]]; then
        worker_status=2
      fi
      break
    fi
    if test "$probe" = "lost=1"; then
      worker_status=2
      break
    fi
  fi
  sleep 10
done

until ssh "${ssh_options[@]}" "${remote_host}" 'bash -s' -- \
  "$job_dir" "$worker_status" <<'REMOTE_RESULT'
set -euo pipefail
job_dir="$1"
worker_status="$2"
log_file="$job_dir/output.log"
if test "$worker_status" = "0"; then
  cat "$log_file"
else
  tail -n 100 "$log_file"
fi
REMOTE_RESULT
do
  echo "gpu02_validate_bionemo_40b: SSH unavailable while collecting result; retrying" >&2
  sleep 5
done

if test "$worker_status" != "0"; then
  echo "gpu02_validate_bionemo_40b: remote worker failed with status $worker_status" >&2
  exit "$worker_status"
fi
