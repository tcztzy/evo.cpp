#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

remote_host="${EVO_GPU01_HOST:-gpu01}"
cuda_release="${EVO_GPU01_CUDA_RELEASE:-12.8}"
image_name="${EVO_GPU01_IMAGE_NAME:-evo.cpp-cuda${cuda_release}-rocky8.sif}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
rsync_retries="${EVO_RSYNC_RETRIES:-12}"
rsync_retry_delay="${EVO_RSYNC_RETRY_DELAY:-5}"

if [[ ! "$rsync_retries" =~ ^[1-9][0-9]*$ ]]; then
  echo "gpu01_build: EVO_RSYNC_RETRIES must be a positive integer" >&2
  exit 2
fi
if [[ ! "$rsync_retry_delay" =~ ^[0-9]+$ ]]; then
  echo "gpu01_build: EVO_RSYNC_RETRY_DELAY must be a nonnegative integer" >&2
  exit 2
fi
if [[ ! "$cuda_release" =~ ^12\.[0-9]+$ ]]; then
  echo "gpu01_build: EVO_GPU01_CUDA_RELEASE must be a CUDA 12.x release" >&2
  exit 2
fi
if [[ ! "$image_name" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "gpu01_build: EVO_GPU01_IMAGE_NAME must be a basename" >&2
  exit 2
fi

rsync_attempt=1
until rsync -az --delay-updates \
    -e "ssh -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=3" \
    --exclude .git \
    --exclude '/build*/' \
    --exclude '/.cache/' \
    --exclude '/.venv/' \
    --exclude '/.evo2-7b-hf-upload/' \
    --exclude __pycache__ \
    --exclude '*.pt' \
    --exclude '*.pt.part*' \
    --exclude '*.safetensors' \
    --exclude '*.safetensors.index.json' \
    "${repo_root}/" "${remote_host}:evo.cpp/"; do
  rsync_exit=$?
  if (( rsync_attempt >= rsync_retries )); then
    echo "gpu01_build: source sync failed after $rsync_attempt attempts" >&2
    exit "$rsync_exit"
  fi
  echo "gpu01_build: source sync attempt $rsync_attempt failed; retrying" >&2
  sleep "$rsync_retry_delay"
  ((rsync_attempt += 1))
done

ssh -o ServerAliveInterval=5 -o ServerAliveCountMax=720 \
  "${remote_host}" bash -s -- "$cuda_release" "$image_name" <<'REMOTE'
set -euo pipefail
export LANG=C
export LC_ALL=C

expected_cuda_release="$1"
image_name="$2"
source_dir="$HOME/evo.cpp"
build_tag="${expected_cuda_release//./}"
build_dir="$source_dir/build-gpu01-cu${build_tag}"
image="$HOME/$image_name"
deps_dir="$HOME/evo.cpp-deps"
nix_root="$HOME/.local/share/nix-root"
apptainer="${EVO_APPTAINER:-$HOME/.local/apptainer/bin/apptainer}"

test -x "$apptainer" || {
  echo "gpu01_build: user Apptainer is missing at $apptainer" >&2
  echo "gpu01_build: install a relocatable non-setuid build under ~/.local/apptainer" >&2
  exit 2
}
test -f "$image" || {
  echo "gpu01_build: $image is missing" >&2
  echo "gpu01_build: provide the requested CUDA $expected_cuda_release SIF" >&2
  exit 2
}
test -d "$nix_root/store"
test -f "$deps_dir/flash-attention-628452c73a4fab560189a7caa8702642c6a38235/csrc/flash_attn/src/flash_fwd_kernel.h"
test -f "$deps_dir/cutlass-7127592069c2fe01b041e174ba4345ef9b279671/include/cute/tensor.hpp"

cmake_host=""
for candidate in "$nix_root"/store/*-cmake-[3-9]*/bin/cmake; do
  if test -x "$candidate"; then
    cmake_host="$candidate"
    break
  fi
done
test -n "$cmake_host" || {
  echo "gpu01_build: no CMake >=3 was found in $nix_root/store" >&2
  exit 2
}
cmake_bin="/nix/store/${cmake_host#"$nix_root/store/"}"
ctest_bin="${cmake_bin%/cmake}/ctest"

export APPTAINER_CACHEDIR="$HOME/.cache/evo.cpp/apptainer"
export APPTAINER_TMPDIR="/tmp/evo.cpp-apptainer-$UID"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

nvidia-smi --query-gpu=index,name,driver_version,memory.total \
  --format=csv,noheader
cuda_release="$($apptainer exec "$image" nvcc --version | \
  sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p')"
if test "$cuda_release" != "$expected_cuda_release"; then
  echo "gpu01_build: expected CUDA $expected_cuda_release image, found ${cuda_release:-unknown}" >&2
  exit 2
fi

"$apptainer" exec -B "$nix_root:/nix:ro" "$image" "$cmake_bin" \
  -S "$source_dir" \
  -B "$build_dir" \
  -DEVO_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DEVO_WARNINGS_AS_ERRORS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDAToolkit_rt_LIBRARY=/usr/lib64/librt.so \
  -DEVO_FLASH_ATTENTION_SOURCE_DIR="$deps_dir/flash-attention-628452c73a4fab560189a7caa8702642c6a38235" \
  -DEVO_CUTLASS_SOURCE_DIR="$deps_dir/cutlass-7127592069c2fe01b041e174ba4345ef9b279671"
"$apptainer" exec -B "$nix_root:/nix:ro" "$image" "$cmake_bin" \
  --build "$build_dir" -j4
"$apptainer" exec --nv -B "$nix_root:/nix:ro" "$image" "$ctest_bin" \
  --test-dir "$build_dir" --output-on-failure -R '^(cuda_smoke|cuda_ops)$'
REMOTE
