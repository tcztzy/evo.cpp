#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

remote_host="${EVO2C_GPU02_HOST:-gpu02}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
rsync_retries="${EVO2C_RSYNC_RETRIES:-12}"
rsync_retry_delay="${EVO2C_RSYNC_RETRY_DELAY:-5}"

if [[ ! "$rsync_retries" =~ ^[1-9][0-9]*$ ]]; then
  echo "gpu02_build: EVO2C_RSYNC_RETRIES must be a positive integer" >&2
  exit 2
fi
if [[ ! "$rsync_retry_delay" =~ ^[0-9]+$ ]]; then
  echo "gpu02_build: EVO2C_RSYNC_RETRY_DELAY must be a nonnegative integer" >&2
  exit 2
fi

rsync_attempt=1
until rsync -az --delete --delay-updates \
    -e "ssh -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=3" \
    --exclude .git \
    --exclude '/build*/' \
    --exclude __pycache__ \
    "${repo_root}/" "${remote_host}:evo2c/"; do
  rsync_exit=$?
  if (( rsync_attempt >= rsync_retries )); then
    echo "gpu02_build: source sync failed after $rsync_attempt attempts" >&2
    exit "$rsync_exit"
  fi
  echo "gpu02_build: source sync attempt $rsync_attempt failed; retrying" >&2
  sleep "$rsync_retry_delay"
  ((rsync_attempt += 1))
done

ssh -o ServerAliveInterval=5 -o ServerAliveCountMax=720 \
  "${remote_host}" 'bash -s' <<'REMOTE'
set -euo pipefail

source_dir="$HOME/evo2c"
build_dir="$source_dir/build-gpu"
image="$HOME/evo2c-cuda12.8-rocky8.sif"
definition="$source_dir/containers/evo2c-cuda12.8-rocky8.def"
nix_root="$HOME/.local/share/nix-root"
cmake_bin="/nix/store/dnsh5jd817k0zddr0k6x3zmyl146bbs6-profile/bin/cmake"

test -d "$source_dir"
test -f "$definition"
test -d "$nix_root/store"
if ! test -f "$image"; then
  image_partial="$HOME/.evo2c-cuda12.8-rocky8.sif.partial"
  rm -f -- "$image_partial"
  trap 'rm -f -- "$image_partial"' EXIT HUP INT TERM
  apptainer build "$image_partial" "$definition"
  mv -- "$image_partial" "$image"
  trap - EXIT HUP INT TERM
fi
gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if test "$gpu_count" -lt 4; then
  echo "gpu02_build: four CUDA GPUs are required, found $gpu_count" >&2
  exit 2
fi

nvidia-smi --query-gpu=index,name,driver_version,memory.total \
  --format=csv,noheader
apptainer exec -B "$nix_root:/nix:ro" "$image" nvcc --version
apptainer exec -B "$nix_root:/nix:ro" "$image" "$cmake_bin" \
  -S "$source_dir" \
  -B "$build_dir" \
  -DEVO2C_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DEVO2C_WARNINGS_AS_ERRORS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDAToolkit_rt_LIBRARY=/usr/lib64/librt.so
apptainer exec -B "$nix_root:/nix:ro" "$image" "$cmake_bin" \
  --build "$build_dir" -j4
REMOTE
