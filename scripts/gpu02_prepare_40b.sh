#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

run_remote_worker() {
revision="d529aa57c30771814217ad89baaeaf6e2315c7d7"
part_size="41126745847"
merged_size="82253491694"
merged_sha256="dd299612b1c1cdded0dfdcaf4d16f98fc97458261d80f4d662429f0ccb316bc3"
part0_sha256="3b74fa4e6158d49265e3e270ba8869390d064358f8bf3d2af0b3e1772728f485"
part1_sha256="bdc4a76e0f23f8295e7061c2f0deff24f723bd916dc4cdc4d9216cac9c2d49d5"
mirror="${HF_ENDPOINT:-https://hf-mirror.com}"
source_dir="$HOME/evo.cpp"
model_dir="$HOME/evo.cpp-models"
merged="$model_dir/evo2_40b.pt"
output_base="$model_dir/evo2-40b-e4m3sw.safetensors"
output="$output_base.index.json"
output_receipt="$output_base.sha256"
image="$HOME/evo.cpp-cuda12.8-rocky8.sif"
nix_root="$HOME/.local/share/nix-root"
venv="$HOME/.venv-evo.cpp-convert"
nix_python="/nix/store/flbw79qdmvzbdrafd93avy5a7d29m2vb-python3-3.12.12/bin/python3"
torch_pythonpath="/nix/store/zram8zr9aikvw9igsyikdxmn5af9915g-python3.12-torch-2.10.0/lib/python3.12/site-packages"
aria2c="/nix/store/qc3vxqi6a1vzqrafgz0dnjrad6lg1xib-aria2-1.37.0-bin/bin/aria2c"
python_runtime_library_path="/nix/store/0p8b2lqk47fvxm9hc6c8mnln5l8x51q1-gcc-14.3.0-lib/lib:/nix/store/p98zvq4nb98krxcv7ss2zr1qngfmi0f5-gcc-14.3.0-libgcc/lib:/nix/store/2kdz3m7ic8w226pcvkz1dlg169v91p6a-zlib-1.3.2/lib"
inspector=""
for candidate in \
  "$source_dir/build-gpu01-cu128/evo-inspect" \
  "$source_dir/build-gpu/evo-inspect" \
  "$source_dir/build-gpu01/evo-inspect"; do
  if test -x "$candidate"; then
    inspector="$candidate"
    break
  fi
done
hf_home="${EVO_HF_HOME:-/build/grp_icg/users/tang/.cache/huggingface}"
export HF_HOME="$hf_home"
export HF_ENDPOINT="$mirror"
cache_dir="$hf_home/hub"
repo_cache="$cache_dir/models--arcinstitute--evo2_40b"
blob_dir="$repo_cache/blobs"
snapshot="$repo_cache/snapshots/$revision"

mkdir -p "$model_dir" "$blob_dir" "$snapshot"
exec 9>"$model_dir/.prepare-40b.lock"
flock 9
test -f "$image"
test -f "$source_dir/configs/evo2-40b-1m.yml"
test -n "$inspector"
apptainer exec -B "$nix_root:/nix:ro" "$image" test -x "$aria2c"
if ! apptainer exec -B "$nix_root:/nix:ro" "$image" \
    test -x "$venv/bin/python"; then
  apptainer exec -B "$nix_root:/nix:ro" "$image" \
    "$nix_python" -m venv "$venv"
fi
if ! apptainer exec -B "$nix_root:/nix:ro" "$image" \
    env LD_LIBRARY_PATH="$python_runtime_library_path" \
    "$venv/bin/python" -c 'import numpy, typing_extensions'; then
  apptainer exec -B "$nix_root:/nix:ro" "$image" \
    env LD_LIBRARY_PATH="$python_runtime_library_path" \
    "$venv/bin/pip" install 'numpy>=1.26,<3' 'typing-extensions>=4.10'
fi
apptainer exec -B "$nix_root:/nix:ro" "$image" \
  env LD_LIBRARY_PATH="$python_runtime_library_path" \
  PYTHONPATH="$torch_pythonpath" "$venv/bin/python" -c \
  'import numpy, torch; print(f"conversion torch={torch.__version__} numpy={numpy.__version__}")'

download_part() {
  local name="$1"
  local expected_sha256="$2"
  local output_variable="$3"
  local blob="$blob_dir/$expected_sha256"
  local incomplete="$blob.incomplete"
  local source="$blob"
  local actual_size
  local actual_sha256
  local link="$snapshot/$name"
  local expected_link="../../blobs/$expected_sha256"

  if ! test -f "$blob"; then
    if ! test -f "$incomplete" ||
        test -e "$incomplete.aria2" ||
        test "$(stat --printf='%s' "$incomplete")" != "$part_size"; then
      apptainer exec \
        -B "$nix_root:/nix:ro" \
        -B "$hf_home:$hf_home" \
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
        --max-tries=20 \
        --summary-interval=30 \
        --dir="$blob_dir" \
        --out="$(basename -- "$incomplete")" \
        "$mirror/arcinstitute/evo2_40b/resolve/$revision/$name?download=true"
    fi
    source="$incomplete"
  fi

  actual_size="$(stat --printf='%s' "$source")"
  if test "$actual_size" != "$part_size"; then
    echo "gpu02_prepare_40b: $source has $actual_size bytes; expected $part_size" >&2
    exit 2
  fi
  actual_sha256="$(sha256sum "$source" | cut -d' ' -f1)"
  if test "$actual_sha256" != "$expected_sha256"; then
    echo "gpu02_prepare_40b: $source SHA256 mismatch" >&2
    exit 2
  fi
  if test "$source" = "$incomplete"; then
    mv -- "$incomplete" "$blob"
  fi

  if test -L "$link"; then
    if test "$(readlink -- "$link")" != "$expected_link"; then
      echo "gpu02_prepare_40b: $link points to an unexpected blob" >&2
      exit 2
    fi
  elif test -e "$link"; then
    echo "gpu02_prepare_40b: $link exists and is not a symbolic link" >&2
    exit 2
  else
    ln -s -- "$expected_link" "$link"
  fi
  test -f "$link"
  printf -v "$output_variable" '%s' "$link"
  echo "${name}_sha256=$actual_sha256"
}

download_part evo2_40b.pt.part0 "$part0_sha256" part0
download_part evo2_40b.pt.part1 "$part1_sha256" part1

if test -e "$merged"; then
  actual_size="$(stat --printf='%s' "$merged")"
  if test "$actual_size" != "$merged_size"; then
    echo "gpu02_prepare_40b: existing $merged has unexpected size $actual_size" >&2
    exit 2
  fi
else
  merged_partial="$model_dir/.safetensors_40b.pt.partial"
  rm -f -- "$merged_partial"
  trap 'rm -f -- "$merged_partial"' EXIT HUP INT TERM
  cp --reflink=auto "$part0" "$merged_partial"
  dd if="$part1" of="$merged_partial" bs=16M oflag=append conv=notrunc status=progress
  actual_size="$(stat --printf='%s' "$merged_partial")"
  if test "$actual_size" != "$merged_size"; then
    echo "gpu02_prepare_40b: merged checkpoint has unexpected size $actual_size" >&2
    exit 2
  fi
  mv -- "$merged_partial" "$merged"
  trap - EXIT HUP INT TERM
fi
source_sha256="$(sha256sum "$merged" | cut -d' ' -f1)"
if test "$source_sha256" != "$merged_sha256"; then
  echo "gpu02_prepare_40b: merged checkpoint SHA256 mismatch" >&2
  exit 2
fi
echo "checkpoint_sha256=$source_sha256"

if ! test -f "$output"; then
  apptainer exec \
    -B "$nix_root:/nix:ro" \
    "$image" \
    env LD_LIBRARY_PATH="$python_runtime_library_path" \
    PYTHONPATH="$source_dir/tools:$torch_pythonpath" \
    "$venv/bin/python" "$source_dir/tools/convert_checkpoint.py" \
    --input "$merged" \
    --config "$source_dir/configs/evo2-40b-1m.yml" \
    --output "$output_base" \
    --source-sha256 "$source_sha256"
fi
apptainer exec -B "$nix_root:/nix:ro" "$image" \
  "$inspector" "$output" \
  --tensor embedding_layer.weight
actual_output_sha256="$(sha256sum "$output" | cut -d' ' -f1)"
if test -f "$output_receipt"; then
  recorded_output_sha256="$(cut -d' ' -f1 <"$output_receipt")"
  if test "$actual_output_sha256" != "$recorded_output_sha256"; then
    echo "gpu02_prepare_40b: Safetensors output SHA256 disagrees with receipt" >&2
    exit 2
  fi
else
  printf '%s  %s\n' "$actual_output_sha256" "$output" >"$output_receipt"
fi
echo "output_size=$(stat --printf='%s' "$output")"
echo "$actual_output_sha256  $output"
}

if test "${EVO_REMOTE_WORKER:-0}" = "1"; then
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

worker="$HOME/evo.cpp/scripts/gpu02_prepare_40b.sh"
worker_sha256="$(sha256sum "$worker" | cut -d' ' -f1)"
job_dir="$HOME/evo.cpp-models/.prepare-40b-jobs/$worker_sha256"
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
    if [[ "$command_line" == *evo-prepare-runner* ]]; then
      running=1
    fi
  fi
fi
if test "$running" = "0"; then
  rm -f -- "$status_file"
  : >"$log_file"
  nohup setsid bash -c '
    worker="$1"
    log_file="$2"
    status_file="$3"
    EVO_REMOTE_WORKER=1 bash "$worker" >"$log_file" 2>&1
    code=$?
    status_partial="${status_file}.$$.partial"
    printf "%s\n" "$code" >"$status_partial"
    mv -- "$status_partial" "$status_file"
  ' evo-prepare-runner "$worker" "$log_file" "$status_file" \
    </dev/null >/dev/null 2>&1 &
  pid="$!"
  printf '%s\n' "$pid" >"$pid_file"
fi
printf 'job_dir=%s\npid=%s\n' "$job_dir" "$pid"
REMOTE_LAUNCH
)"; do
  echo "gpu02_prepare_40b: SSH unavailable while launching; retrying" >&2
  sleep 5
done

job_dir="$(printf '%s\n' "$launch_output" | sed -n 's/^job_dir=//p')"
worker_pid="$(printf '%s\n' "$launch_output" | sed -n 's/^pid=//p')"
if test -z "$job_dir" || test -z "$worker_pid"; then
  echo "gpu02_prepare_40b: remote launcher returned invalid state" >&2
  exit 2
fi
echo "gpu02_prepare_40b: remote worker PID $worker_pid"

while true; do
  probe=""
  if probe="$(
    ssh "${ssh_options[@]}" "${remote_host}" 'bash -s' -- "$job_dir" <<'REMOTE_PROBE'
set -euo pipefail
job_dir="$1"
pid="$(cat "$job_dir/pid")"
if test -f "$job_dir/status"; then
  printf 'done=%s\n' "$(cat "$job_dir/status")"
elif kill -0 "$pid" 2>/dev/null; then
  echo "running=1"
else
  echo "lost=1"
fi
REMOTE_PROBE
  )"; then
    if [[ "$probe" == done=* ]]; then
      worker_status="${probe#done=}"
      if [[ -z "$worker_status" || "$worker_status" == *[!0-9]* ]]; then
        worker_status="2"
      fi
      break
    fi
    if test "$probe" = "lost=1"; then
      worker_status="2"
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
    '^(conversion torch=|evo2_40b\.pt\.part[01]_sha256=|checkpoint_sha256=|validated |wrote |model |tensor |[0-9a-f]{64}  )' \
    "$log_file" || true
else
  tail -n 100 "$log_file" |
    sed -E 's#https?://[^ ]+#<redirected-url>#g'
fi
REMOTE_RESULT
do
  echo "gpu02_prepare_40b: SSH unavailable while collecting result; retrying" >&2
  sleep 5
done

if test "$worker_status" != "0"; then
  echo "gpu02_prepare_40b: remote worker failed with status $worker_status" >&2
  exit "$worker_status"
fi
