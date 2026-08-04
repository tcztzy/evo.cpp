#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

run_remote_worker() {
archive_name="evo2_40b_bf16_finetune_wandb_Ji2IRcrz_step_119.tar.gz"
archive_size="63680606710"
archive_sha256="544b47e033d1fb0261b686a53f7c4fe240cd290253187d31e8c99dea9e35a680"
ngc_url="https://api.ngc.nvidia.com/v2/models/nvidia/clara/evo2-40b-1m-fp8-bf16-nemo2/versions/1.0/files/$archive_name"
source_dir="$HOME/evo.cpp"
model_dir="$HOME/evo.cpp-models"
output_base="$model_dir/evo2-40b-bionemo-bf16.safetensors"
output="$output_base.index.json"
output_receipt="$output_base.sha256"
image="$HOME/evo.cpp-cuda12.8-rocky8.sif"
nix_root="$HOME/.local/share/nix-root"
venv="$HOME/.venv-evo.cpp-convert"
nix_python="/nix/store/flbw79qdmvzbdrafd93avy5a7d29m2vb-python3-3.12.12/bin/python3"
torch_pythonpath="/nix/store/zram8zr9aikvw9igsyikdxmn5af9915g-python3.12-torch-2.10.0/lib/python3.12/site-packages"
aria2c="/nix/store/qc3vxqi6a1vzqrafgz0dnjrad6lg1xib-aria2-1.37.0-bin/bin/aria2c"
python_runtime_library_path="/nix/store/0p8b2lqk47fvxm9hc6c8mnln5l8x51q1-gcc-14.3.0-lib/lib:/nix/store/p98zvq4nb98krxcv7ss2zr1qngfmi0f5-gcc-14.3.0-libgcc/lib:/nix/store/2kdz3m7ic8w226pcvkz1dlg169v91p6a-zlib-1.3.2/lib"
bionemo_cache="${EVO_BIONEMO_CACHE:-/build/grp_icg/users/tang/.cache/bionemo}"
resource_dir="$bionemo_cache/evo2/40b-1m-fp8-bf16/1.0"
archive="$resource_dir/$archive_name"
archive_incomplete="$archive.incomplete"
extract_dir="$resource_dir/checkpoint"
extract_partial="$resource_dir/.checkpoint.partial"

mkdir -p "$model_dir" "$resource_dir"
exec 9>"$model_dir/.prepare-bionemo-40b.lock"
flock 9
test -f "$image"
test -f "$source_dir/configs/evo2-40b-1m-bionemo-bf16.yml"
test -f "$source_dir/tools/convert_bionemo_checkpoint.py"
test -x "$source_dir/build-gpu/evo-inspect"
apptainer exec -B "$nix_root:/nix:ro" "$image" test -x "$aria2c"

if ! apptainer exec -B "$nix_root:/nix:ro" "$image" \
    test -x "$venv/bin/python"; then
  apptainer exec -B "$nix_root:/nix:ro" "$image" \
    "$nix_python" -m venv "$venv"
fi
if ! apptainer exec -B "$nix_root:/nix:ro" "$image" \
    env LD_LIBRARY_PATH="$python_runtime_library_path" \
    "$venv/bin/python" -c 'import numpy'; then
  apptainer exec -B "$nix_root:/nix:ro" "$image" \
    env LD_LIBRARY_PATH="$python_runtime_library_path" \
    "$venv/bin/pip" install 'numpy>=1.26,<3'
fi
apptainer exec -B "$nix_root:/nix:ro" "$image" \
  env LD_LIBRARY_PATH="$python_runtime_library_path" \
  PYTHONPATH="$torch_pythonpath" "$venv/bin/python" -c \
  'import numpy, torch; print(f"conversion torch={torch.__version__} numpy={numpy.__version__}")'

if ! test -f "$archive"; then
  apptainer exec \
    -B "$nix_root:/nix:ro" \
    -B "$bionemo_cache:$bionemo_cache" \
    "$image" \
    "$aria2c" \
    --ca-certificate=/etc/pki/tls/certs/ca-bundle.crt \
    --continue=true \
    --max-connection-per-server=16 \
    --split=16 \
    --min-split-size=8M \
    --file-allocation=none \
    --auto-file-renaming=false \
    --allow-overwrite=true \
    --check-integrity=false \
    --retry-wait=5 \
    --max-tries=0 \
    --summary-interval=30 \
    --dir="$resource_dir" \
    --out="$(basename -- "$archive_incomplete")" \
    "$ngc_url"
  if test -e "$archive_incomplete.aria2"; then
    echo "gpu02_prepare_bionemo_40b: aria2 left an incomplete download" >&2
    exit 2
  fi
  actual_archive_size="$(stat --printf='%s' "$archive_incomplete")"
  if test "$actual_archive_size" != "$archive_size"; then
    echo "gpu02_prepare_bionemo_40b: archive has $actual_archive_size bytes; expected $archive_size" >&2
    exit 2
  fi
  actual_archive_sha256="$(sha256sum "$archive_incomplete" | cut -d' ' -f1)"
  if test "$actual_archive_sha256" != "$archive_sha256"; then
    echo "gpu02_prepare_bionemo_40b: archive SHA256 mismatch" >&2
    exit 2
  fi
  mv -- "$archive_incomplete" "$archive"
fi

actual_archive_size="$(stat --printf='%s' "$archive")"
if test "$actual_archive_size" != "$archive_size"; then
  echo "gpu02_prepare_bionemo_40b: archive has $actual_archive_size bytes; expected $archive_size" >&2
  exit 2
fi
actual_archive_sha256="$(sha256sum "$archive" | cut -d' ' -f1)"
if test "$actual_archive_sha256" != "$archive_sha256"; then
  echo "gpu02_prepare_bionemo_40b: archive SHA256 mismatch" >&2
  exit 2
fi
echo "archive_sha256=$actual_archive_sha256"

if ! test -d "$extract_dir"; then
  if test -e "$extract_partial"; then
    echo "gpu02_prepare_bionemo_40b: replacing interrupted temporary extraction $extract_partial" >&2
    rm -rf -- "$extract_partial"
  fi
  mkdir -p "$extract_partial"
  trap 'rm -rf -- "$extract_partial"' EXIT HUP INT TERM
  apptainer exec \
    -B "$bionemo_cache:$bionemo_cache" \
    "$image" tar -xzf "$archive" -C "$extract_partial"
  mapfile -t partial_metadata < <(
    find "$extract_partial" -type f -name .metadata -print
  )
  if test "${#partial_metadata[@]}" != "1"; then
    echo "gpu02_prepare_bionemo_40b: extracted archive has ${#partial_metadata[@]} DCP metadata files; expected 1" >&2
    exit 2
  fi
  printf '%s\n' "$archive_sha256" >"$extract_partial/.evo.cpp-archive-sha256"
  mv -- "$extract_partial" "$extract_dir"
  trap - EXIT HUP INT TERM
fi

if ! test -f "$extract_dir/.evo.cpp-archive-sha256" ||
   test "$(cat "$extract_dir/.evo.cpp-archive-sha256")" != "$archive_sha256"; then
  echo "gpu02_prepare_bionemo_40b: extraction provenance is missing or stale" >&2
  exit 2
fi
mapfile -t metadata_files < <(
  find "$extract_dir" -type f -name .metadata -print
)
if test "${#metadata_files[@]}" != "1"; then
  echo "gpu02_prepare_bionemo_40b: checkpoint has ${#metadata_files[@]} DCP metadata files; expected 1" >&2
  exit 2
fi
checkpoint_dir="$(dirname -- "${metadata_files[0]}")"
echo "checkpoint_dir=$checkpoint_dir"

converter=(
  "$venv/bin/python"
  "$source_dir/tools/convert_bionemo_checkpoint.py"
  --input "$checkpoint_dir"
  --config "$source_dir/configs/evo2-40b-1m-bionemo-bf16.yml"
  --source-sha256 "$archive_sha256"
)
apptainer exec \
  -B "$nix_root:/nix:ro" \
  -B "$bionemo_cache:$bionemo_cache" \
  "$image" \
  env LD_LIBRARY_PATH="$python_runtime_library_path" \
  PYTHONPATH="$source_dir/tools:$torch_pythonpath" \
  "${converter[@]}" --output "$output_base" --dry-run

if ! test -f "$output"; then
  apptainer exec \
    -B "$nix_root:/nix:ro" \
    -B "$bionemo_cache:$bionemo_cache" \
    "$image" \
    env LD_LIBRARY_PATH="$python_runtime_library_path" \
    PYTHONPATH="$source_dir/tools:$torch_pythonpath" \
    "${converter[@]}" --output "$output_base"
fi

apptainer exec -B "$nix_root:/nix:ro" "$image" \
  "$source_dir/build-gpu/evo-inspect" "$output" \
  --tensor blocks.0.projections.weight
actual_output_sha256="$(sha256sum "$output" | cut -d' ' -f1)"
if test -f "$output_receipt"; then
  recorded_output_sha256="$(cut -d' ' -f1 <"$output_receipt")"
  if test "$actual_output_sha256" != "$recorded_output_sha256"; then
    echo "gpu02_prepare_bionemo_40b: Safetensors output SHA256 disagrees with receipt" >&2
    exit 2
  fi
else
  printf '%s  %s\n' "$actual_output_sha256" "$output" >"$output_receipt"
fi
echo "output_size=$(stat --printf='%s' "$output")"
echo "$actual_output_sha256  $output"
}

if test "${EVO_REMOTE_BIONEMO_WORKER:-0}" = "1"; then
  run_remote_worker
  exit 0
fi

remote_host="${EVO_GPU02_HOST:-gpu02}"
ssh_options=(
  -o ConnectTimeout=10
  -o ServerAliveInterval=5
  -o ServerAliveCountMax=3
)

launch_output=""
until launch_output="$(
  ssh "${ssh_options[@]}" "${remote_host}" 'bash -s' <<'REMOTE_LAUNCH'
set -euo pipefail

worker="$HOME/evo.cpp/scripts/gpu02_prepare_bionemo_40b.sh"
source_dir="$HOME/evo.cpp"
job_manifest="$(
  sha256sum \
    "$worker" \
    "$source_dir/tools/convert_bionemo_checkpoint.py" \
    "$source_dir/tools/evo/bionemo_checkpoint.py" \
    "$source_dir/tools/evo/model_config.py" \
    "$source_dir/configs/evo2-40b-1m-bionemo-bf16.yml"
)"
job_sha256="$(printf '%s\n' "$job_manifest" | sha256sum | cut -d' ' -f1)"
job_dir="$HOME/evo.cpp-models/.prepare-bionemo-40b-jobs/$job_sha256"
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
    if [[ "$command_line" == *evo-bionemo-prepare-runner* ]]; then
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
    EVO_REMOTE_BIONEMO_WORKER=1 bash "$worker" >"$log_file" 2>&1
    code=$?
    status_partial="${status_file}.$$.partial"
    printf "%s\n" "$code" >"$status_partial"
    mv -- "$status_partial" "$status_file"
  ' evo-bionemo-prepare-runner "$worker" "$log_file" "$status_file" \
    </dev/null >/dev/null 2>&1 &
  pid="$!"
  printf '%s\n' "$pid" >"$pid_file"
fi
printf 'job_dir=%s\npid=%s\n' "$job_dir" "$pid"
REMOTE_LAUNCH
)"; do
  echo "gpu02_prepare_bionemo_40b: SSH unavailable while launching; retrying" >&2
  sleep 5
done

job_dir="$(printf '%s\n' "$launch_output" | sed -n 's/^job_dir=//p')"
worker_pid="$(printf '%s\n' "$launch_output" | sed -n 's/^pid=//p')"
if test -z "$job_dir"; then
  echo "gpu02_prepare_bionemo_40b: remote launcher returned invalid state" >&2
  exit 2
fi
echo "gpu02_prepare_bionemo_40b: remote worker PID ${worker_pid:-completed}"

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
  grep -E \
    '^(conversion torch=|archive_sha256=|checkpoint_dir=|validated |BioNeMo |wrote |model |tensor |output_size=|[0-9a-f]{64}  )' \
    "$log_file" || true
else
  tail -n 100 "$log_file" |
    sed -E 's#https?://[^ ]+#<redirected-url>#g'
fi
REMOTE_RESULT
do
  echo "gpu02_prepare_bionemo_40b: SSH unavailable while collecting result; retrying" >&2
  sleep 5
done

if test "$worker_status" != "0"; then
  echo "gpu02_prepare_bionemo_40b: remote worker failed with status $worker_status" >&2
  exit "$worker_status"
fi
