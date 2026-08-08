#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

remote_host="${EVO_GPU02_HOST:-gpu02}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
rsync_retries="${EVO_RSYNC_RETRIES:-12}"
rsync_retry_delay="${EVO_RSYNC_RETRY_DELAY:-5}"

if [[ ! "$rsync_retries" =~ ^[1-9][0-9]*$ ]]; then
  echo "gpu02_build: EVO_RSYNC_RETRIES must be a positive integer" >&2
  exit 2
fi
if [[ ! "$rsync_retry_delay" =~ ^[0-9]+$ ]]; then
  echo "gpu02_build: EVO_RSYNC_RETRY_DELAY must be a nonnegative integer" >&2
  exit 2
fi

rsync_attempt=1
until rsync -az --delay-updates \
    -e "ssh -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=3" \
    --exclude .git \
    --exclude '/build*/' \
    --exclude '/.cache/' \
    --exclude '/.venv/' \
    --exclude '/CMakeUserPresets.json' \
    --exclude '/.pytest_cache/' \
    --exclude '/.ruff_cache/' \
    --exclude __pycache__ \
    --exclude '*.pt' \
    --exclude '*.pt.part*' \
    --exclude '*.safetensors' \
    --exclude '*.safetensors.index.json' \
    "${repo_root}/" "${remote_host}:evo.cpp/"; do
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

source_dir="$HOME/evo.cpp"
build_dir="$source_dir/build-gpu"
image="$HOME/evo.cpp-cuda12.8-rocky8.sif"
definition="$source_dir/containers/evo.cpp-cuda12.8-rocky8.def"
nix_root="$HOME/.local/share/nix-root"
deps_dir="$HOME/evo.cpp-deps"

cmake_host=""
for candidate in "$nix_root"/store/*-cmake-[3-9]*/bin/cmake; do
  if test -x "$candidate"; then
    cmake_host="$candidate"
    break
  fi
done
test -n "$cmake_host" || {
  echo "gpu02_build: no CMake >=3 was found in $nix_root/store" >&2
  exit 2
}
cmake_bin="/nix/store/${cmake_host#"$nix_root/store/"}"

test -d "$source_dir"
test -f "$definition"
test -d "$nix_root/store"
test -f "$deps_dir/flash-attention-628452c73a4fab560189a7caa8702642c6a38235/csrc/flash_attn/src/flash_fwd_kernel.h"
test -f "$deps_dir/cutlass-7127592069c2fe01b041e174ba4345ef9b279671/include/cute/tensor.hpp"
test -f "$deps_dir/libnpy-890ea4fcda302a580e633c624c6a63e2a5d422f6/include/npy.hpp"
if ! test -f "$image"; then
  image_partial="$HOME/.evo.cpp-cuda12.8-rocky8.sif.partial"
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
  -DEVO_CUDA=ON \
  -DEVO_NPY=ON \
  -DBUILD_TESTING=ON \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DEVO_WARNINGS_AS_ERRORS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDAToolkit_rt_LIBRARY=/usr/lib64/librt.so \
  -DEVO_LIBNPY_SOURCE_DIR="$deps_dir/libnpy-890ea4fcda302a580e633c624c6a63e2a5d422f6" \
  -DEVO_FLASH_ATTENTION_SOURCE_DIR="$deps_dir/flash-attention-628452c73a4fab560189a7caa8702642c6a38235" \
  -DEVO_CUTLASS_SOURCE_DIR="$deps_dir/cutlass-7127592069c2fe01b041e174ba4345ef9b279671"
apptainer exec -B "$nix_root:/nix:ro" "$image" "$cmake_bin" \
  --build "$build_dir" -j4
REMOTE
