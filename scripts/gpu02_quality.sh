#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

run_remote_worker() {
  local start_index="${EVO_QUALITY_START_INDEX:-0}"
  local limit="${EVO_QUALITY_LIMIT:-}"
  local source_dir="$HOME/evo.cpp"
  local image="$HOME/evo.cpp-cuda12.8-rocky8.sif"
  local nix_root="$HOME/.local/share/nix-root"
  local python_bin="/nix/store/flbw79qdmvzbdrafd93avy5a7d29m2vb-python3-3.12.12/bin/python3"
  local artifact_root="$HOME/evo.cpp-artifacts"
  local complete
  local generated
  local model
  local model_sha256
  local output
  local -a quality_args

  export HF_HOME="${EVO_HF_HOME:-/build/grp_icg/users/tang/.cache}"
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  model="$HOME/evo.cpp-models/evo2-40b-e4m3sw.safetensors.index.json"
  test -f "$model"
  model_sha256="$(sha256sum "$model" | cut -d' ' -f1)"
  output="$artifact_root/t13-native-quality-4x500.json"
  quality_args=(
    --binary "$source_dir/build-gpu/evo"
    --model "$model"
    --model-sha256 "$model_sha256"
    --prompts "$HOME/evo.cpp-vortex-reference/test/data/prompts.csv"
    --output "$output"
    --num-tokens 500
    --ctx 8192
    --gpu 0,1,2,3
    --force-prompt-threshold 3000
    --start-index "$start_index"
  )
  if test -n "$limit"; then
    quality_args+=(--limit "$limit")
  fi
  if test "$start_index" != "0" || test -n "$limit"; then
    quality_args+=(--skip-mean-gate)
  fi

  mkdir -p "$artifact_root"
  test -f "$image"
  test -x "$source_dir/build-gpu/evo"
  test -f "$HOME/evo.cpp-vortex-reference/test/data/prompts.csv"
  apptainer exec --nv -B "$nix_root:/nix:ro" "$image" \
    "$python_bin" "$source_dir/tools/evo2_quality_gate.py" \
    "${quality_args[@]}"
  complete=1
  for index in 0 1 2 3; do
    generated="$artifact_root/t13-native-quality-4x500-artifacts/generated-${index}.bin"
    if ! test -f "$generated" || test "$(stat --printf='%s' "$generated")" != "500"; then
      complete=0
    fi
  done
  if test "$complete" = "1"; then
    apptainer exec --nv -B "$nix_root:/nix:ro" "$image" \
      "$python_bin" "$source_dir/tools/evo2_quality_gate.py" \
      --binary "$source_dir/build-gpu/evo" \
      --model "$model" \
      --model-sha256 "$model_sha256" \
      --prompts "$HOME/evo.cpp-vortex-reference/test/data/prompts.csv" \
      --output "$output" --num-tokens 500 --ctx 8192 --gpu 0,1,2,3 \
      --force-prompt-threshold 3000 --summarize-existing
  fi
  sha256sum "$output"
}

if test "${EVO_REMOTE_QUALITY_WORKER:-0}" = "1"; then
  run_remote_worker
  exit 0
fi

remote_host="${EVO_GPU02_HOST:-gpu02}"
start_index="${EVO_QUALITY_START_INDEX:-0}"
limit="${EVO_QUALITY_LIMIT:-}"
if [[ ! "$start_index" =~ ^[0-3]$ ]]; then
  echo "gpu02_quality: EVO_QUALITY_START_INDEX must be in [0,3]" >&2
  exit 2
fi
if test -n "$limit" &&
   { [[ ! "$limit" =~ ^[1-4]$ ]] ||
     test "$((start_index + limit))" -gt 4; }; then
  echo "gpu02_quality: EVO_QUALITY_LIMIT selects outside four prompts" >&2
  exit 2
fi

ssh_options=(
  -o ConnectTimeout=10
  -o ServerAliveInterval=5
  -o ServerAliveCountMax=3
)
launch_output=""
until launch_output="$(
  ssh "${ssh_options[@]}" "${remote_host}" 'bash -s' -- \
    "$start_index" "$limit" <<'REMOTE_LAUNCH'
set -euo pipefail
start_index="$1"
limit="$2"
worker="$HOME/evo.cpp/scripts/gpu02_quality.sh"
worker_sha256="$(sha256sum "$worker" | cut -d' ' -f1)"
job_key="${worker_sha256}-start${start_index}-limit${limit:-all}"
job_dir="$HOME/evo.cpp-artifacts/.quality-jobs/$job_key"
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
    if [[ "$command_line" == *evo-quality-runner* ]]; then
      running=1
    fi
  fi
fi
if test "$running" = "0" && ! test -f "$status_file"; then
  : >"$log_file"
  nohup setsid bash -c '
    worker="$1"
    log_file="$2"
    status_file="$3"
    start_index="$4"
    limit="$5"
    EVO_REMOTE_QUALITY_WORKER=1 \
      EVO_QUALITY_START_INDEX="$start_index" \
      EVO_QUALITY_LIMIT="$limit" \
      bash "$worker" >"$log_file" 2>&1
    code=$?
    status_partial="${status_file}.$$.partial"
    printf "%s\n" "$code" >"$status_partial"
    mv -- "$status_partial" "$status_file"
  ' evo-quality-runner "$worker" "$log_file" "$status_file" \
    "$start_index" "$limit" </dev/null >/dev/null 2>&1 &
  pid="$!"
  printf '%s\n' "$pid" >"$pid_file"
fi
printf 'job_dir=%s\npid=%s\n' "$job_dir" "$pid"
REMOTE_LAUNCH
)"; do
  echo "gpu02_quality: SSH unavailable while launching; retrying" >&2
  sleep 5
done

job_dir="$(printf '%s\n' "$launch_output" | sed -n 's/^job_dir=//p')"
worker_pid="$(printf '%s\n' "$launch_output" | sed -n 's/^pid=//p')"
if test -z "$job_dir"; then
  echo "gpu02_quality: remote launcher returned invalid state" >&2
  exit 2
fi
echo "gpu02_quality: remote worker PID ${worker_pid:-completed}"

while true; do
  probe=""
  if probe="$(
    ssh "${ssh_options[@]}" "${remote_host}" 'bash -s' -- \
      "$job_dir" <<'REMOTE_PROBE'
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
  echo "gpu02_quality: SSH unavailable while collecting results; retrying" >&2
  sleep 5
done
exit "$worker_status"
