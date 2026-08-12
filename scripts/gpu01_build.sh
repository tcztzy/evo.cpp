#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

remote_host="${EVO_GPU01_HOST:-gpu01}"
cuda_release="${EVO_GPU01_CUDA_RELEASE:-12.8}"
image_name="${EVO_GPU01_IMAGE_NAME:-evo.cpp-cuda${cuda_release}-rocky8.sif}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
. "${script_dir}/remote_build_common.sh"
rsync_retries="${EVO_RSYNC_RETRIES:-12}"
rsync_retry_delay="${EVO_RSYNC_RETRY_DELAY:-5}"
build_jobs="${EVO_BUILD_JOBS:-4}"
remote_root="${EVO_REMOTE_ROOT:-}"
remote_source_dir="${EVO_REMOTE_SOURCE_DIR:-}"
remote_build_dir="${EVO_REMOTE_BUILD_DIR:-}"
remote_deps_dir="${EVO_REMOTE_DEPS_DIR:-}"
remote_container_path="${EVO_REMOTE_CONTAINER_PATH:-}"
remote_nix_root="${EVO_REMOTE_NIX_ROOT:-}"
remote_cache_dir="${EVO_REMOTE_CACHE_DIR:-}"
apptainer_override="${EVO_APPTAINER:-}"
cmake_override="${EVO_CMAKE_BIN:-}"
python_override="${EVO_PYTHON_BIN:-}"
cuda_visible_devices="${EVO_CUDA_VISIBLE_DEVICES:-0}"

if [[ ! "$rsync_retries" =~ ^[1-9][0-9]*$ ]]; then
  echo "gpu01_build: EVO_RSYNC_RETRIES must be a positive integer" >&2
  exit 2
fi
if [[ ! "$rsync_retry_delay" =~ ^[0-9]+$ ]]; then
  echo "gpu01_build: EVO_RSYNC_RETRY_DELAY must be a nonnegative integer" >&2
  exit 2
fi
if [[ ! "$build_jobs" =~ ^[1-9][0-9]*$ ]]; then
  echo "gpu01_build: EVO_BUILD_JOBS must be a positive integer" >&2
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
evo_validate_remote_overrides gpu01_build \
  EVO_REMOTE_ROOT "$remote_root" \
  EVO_REMOTE_SOURCE_DIR "$remote_source_dir" \
  EVO_REMOTE_BUILD_DIR "$remote_build_dir" \
  EVO_REMOTE_DEPS_DIR "$remote_deps_dir" \
  EVO_REMOTE_CONTAINER_PATH "$remote_container_path" \
  EVO_REMOTE_NIX_ROOT "$remote_nix_root" \
  EVO_REMOTE_CACHE_DIR "$remote_cache_dir" \
  EVO_APPTAINER "$apptainer_override" \
  EVO_CMAKE_BIN "$cmake_override" \
  EVO_PYTHON_BIN "$python_override"
evo_validate_cuda_visible_devices gpu01_build "$cuda_visible_devices"
evo_remote_source_for_sync "$remote_root" "$remote_source_dir"
remote_sync_source="$EVO_REMOTE_SYNC_SOURCE"
ssh -o ConnectTimeout=10 "$remote_host" mkdir -p -- "$remote_sync_source"

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
    --exclude '/.evo2-7b-hf-upload/' \
    --exclude __pycache__ \
    --exclude '*.pt' \
    --exclude '*.pt.part*' \
    --exclude '*.safetensors' \
    --exclude '*.safetensors.index.json' \
    "${repo_root}/" "${remote_host}:${remote_sync_source}/"; do
  rsync_exit=$?
  if (( rsync_attempt >= rsync_retries )); then
    echo "gpu01_build: source sync failed after $rsync_attempt attempts" >&2
    exit "$rsync_exit"
  fi
  echo "gpu01_build: source sync attempt $rsync_attempt failed; retrying" >&2
  sleep "$rsync_retry_delay"
  ((rsync_attempt += 1))
done

evo_quote_remote_command bash \
  "$cuda_release" "$image_name" "$remote_root" "$remote_source_dir" \
  "$remote_build_dir" "$remote_deps_dir" "$remote_container_path" \
  "$remote_nix_root" "$remote_cache_dir" "$apptainer_override" \
  "$cmake_override" "$python_override" "$cuda_visible_devices" "$build_jobs"
ssh -o ServerAliveInterval=5 -o ServerAliveCountMax=720 \
  "${remote_host}" "$EVO_REMOTE_COMMAND" <<'REMOTE'
set -euo pipefail
export LANG=C
export LC_ALL=C

expected_cuda_release="$1"
image_name="$2"
remote_root_arg="$3"
source_dir_arg="$4"
build_dir_arg="$5"
deps_dir_arg="$6"
container_path_arg="$7"
nix_root_arg="$8"
cache_dir_arg="$9"
apptainer_override="${10}"
cmake_override="${11}"
python_override="${12}"
cuda_visible_devices="${13}"
build_jobs="${14}"
root_for_source="${remote_root_arg:-$HOME}"
source_for_helpers="${source_dir_arg:-${root_for_source}/evo.cpp}"
. "$source_for_helpers/scripts/remote_build_common.sh"
build_tag="${expected_cuda_release//./}"
evo_configure_remote_paths gpu01_build \
  "$remote_root_arg" "$source_dir_arg" "$build_dir_arg" "$deps_dir_arg" \
  "$container_path_arg" "$nix_root_arg" "$cache_dir_arg" \
  "build-gpu01-cu${build_tag}" "$image_name"
evo_validate_cuda_visible_devices gpu01_build "$cuda_visible_devices"
source_dir="$EVO_REMOTE_SOURCE_DIR_RESOLVED"
build_dir="$EVO_REMOTE_BUILD_DIR_RESOLVED"
image="$EVO_REMOTE_CONTAINER_PATH_RESOLVED"
deps_dir="$EVO_REMOTE_DEPS_DIR_RESOLVED"
nix_root="$EVO_REMOTE_NIX_ROOT_RESOLVED"
cache_dir="$EVO_REMOTE_CACHE_DIR_RESOLVED"
apptainer="${apptainer_override:-$HOME/.local/apptainer/bin/apptainer}"

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
test -f "$deps_dir/flash-attention-628452c73a4fab560189a7caa8702642c6a38235/csrc/flash_attn/src/flash_fwd_kernel.h"
test -f "$deps_dir/cutlass-7127592069c2fe01b041e174ba4345ef9b279671/include/cute/tensor.hpp"
test -f "$deps_dir/libnpy-890ea4fcda302a580e633c624c6a63e2a5d422f6/include/npy.hpp"

export APPTAINER_CACHEDIR="$cache_dir/apptainer"
export APPTAINER_TMPDIR="/tmp/evo.cpp-apptainer-$UID"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "${build_dir%/*}"

cuda_release="$("$apptainer" exec "$image" nvcc --version | \
  sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p')"
if test "$cuda_release" != "$expected_cuda_release"; then
  echo "gpu01_build: expected CUDA $expected_cuda_release image, found ${cuda_release:-unknown}" >&2
  exit 2
fi
evo_select_cmake gpu01_build "$apptainer" "$image" "$nix_root" \
  "$cmake_override"
cmake_bin="$EVO_CMAKE_BIN_SELECTED"
ctest_bin="$EVO_CTEST_BIN_SELECTED"
echo "gpu01_build: using CMake $EVO_CMAKE_VERSION_SELECTED at $cmake_bin"
evo_select_python gpu01_build "$apptainer" "$image" "$nix_root" \
  "$python_override"
python_bin="$EVO_PYTHON_BIN_SELECTED"
echo "gpu01_build: using Python $EVO_PYTHON_VERSION_SELECTED at $python_bin"
container_binds=(
  -B "$nix_root:/nix:ro"
  -B "$source_dir:$source_dir"
  -B "$deps_dir:$deps_dir:ro"
)
if [[ "$build_dir" != "$source_dir"/* ]]; then
  container_binds+=(-B "${build_dir%/*}:${build_dir%/*}")
fi

"$apptainer" exec "${container_binds[@]}" "$image" "$cmake_bin" \
  -S "$source_dir" \
  -B "$build_dir" \
  -DEVO_CUDA=ON \
  -DEVO_NPY=ON \
  -DBUILD_TESTING=ON \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DEVO_WARNINGS_AS_ERRORS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE="$python_bin" \
  -DCUDAToolkit_rt_LIBRARY=/usr/lib64/librt.so \
  -DEVO_LIBNPY_SOURCE_DIR="$deps_dir/libnpy-890ea4fcda302a580e633c624c6a63e2a5d422f6" \
  -DEVO_FLASH_ATTENTION_SOURCE_DIR="$deps_dir/flash-attention-628452c73a4fab560189a7caa8702642c6a38235" \
  -DEVO_CUTLASS_SOURCE_DIR="$deps_dir/cutlass-7127592069c2fe01b041e174ba4345ef9b279671"
"$apptainer" exec "${container_binds[@]}" "$image" "$cmake_bin" \
  --build "$build_dir" -j"$build_jobs"
CUDA_VISIBLE_DEVICES="$cuda_visible_devices" \
  "$apptainer" exec --nv "${container_binds[@]}" "$image" env \
  CUDA_VISIBLE_DEVICES="$cuda_visible_devices" "$ctest_bin" \
  --test-dir "$build_dir" --output-on-failure \
  -R '^(npy|cuda_smoke|cuda_ops)$'
REMOTE
