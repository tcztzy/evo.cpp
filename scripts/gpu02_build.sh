#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

remote_host="${EVO_GPU02_HOST:-gpu02}"
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

if [[ ! "$rsync_retries" =~ ^[1-9][0-9]*$ ]]; then
  echo "gpu02_build: EVO_RSYNC_RETRIES must be a positive integer" >&2
  exit 2
fi
if [[ ! "$rsync_retry_delay" =~ ^[0-9]+$ ]]; then
  echo "gpu02_build: EVO_RSYNC_RETRY_DELAY must be a nonnegative integer" >&2
  exit 2
fi
if [[ ! "$build_jobs" =~ ^[1-9][0-9]*$ ]]; then
  echo "gpu02_build: EVO_BUILD_JOBS must be a positive integer" >&2
  exit 2
fi
evo_validate_remote_overrides gpu02_build \
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
    --exclude __pycache__ \
    --exclude '*.pt' \
    --exclude '*.pt.part*' \
    --exclude '*.safetensors' \
    --exclude '*.safetensors.index.json' \
    "${repo_root}/" "${remote_host}:${remote_sync_source}/"; do
  rsync_exit=$?
  if (( rsync_attempt >= rsync_retries )); then
    echo "gpu02_build: source sync failed after $rsync_attempt attempts" >&2
    exit "$rsync_exit"
  fi
  echo "gpu02_build: source sync attempt $rsync_attempt failed; retrying" >&2
  sleep "$rsync_retry_delay"
  ((rsync_attempt += 1))
done

evo_quote_remote_command bash \
  "$remote_root" "$remote_source_dir" "$remote_build_dir" \
  "$remote_deps_dir" "$remote_container_path" "$remote_nix_root" \
  "$remote_cache_dir" "$apptainer_override" "$cmake_override" \
  "$python_override" "$build_jobs"
ssh -o ServerAliveInterval=5 -o ServerAliveCountMax=720 \
  "${remote_host}" "$EVO_REMOTE_COMMAND" <<'REMOTE'
set -euo pipefail
export LANG=C
export LC_ALL=C

remote_root_arg="$1"
source_dir_arg="$2"
build_dir_arg="$3"
deps_dir_arg="$4"
container_path_arg="$5"
nix_root_arg="$6"
cache_dir_arg="$7"
apptainer_override="$8"
cmake_override="$9"
python_override="${10}"
build_jobs="${11}"
root_for_source="${remote_root_arg:-$HOME}"
source_for_helpers="${source_dir_arg:-${root_for_source}/evo.cpp}"
. "$source_for_helpers/scripts/remote_build_common.sh"
evo_configure_remote_paths gpu02_build \
  "$remote_root_arg" "$source_dir_arg" "$build_dir_arg" "$deps_dir_arg" \
  "$container_path_arg" "$nix_root_arg" "$cache_dir_arg" \
  build-gpu evo.cpp-cuda12.8-rocky8.sif
source_dir="$EVO_REMOTE_SOURCE_DIR_RESOLVED"
build_dir="$EVO_REMOTE_BUILD_DIR_RESOLVED"
image="$EVO_REMOTE_CONTAINER_PATH_RESOLVED"
deps_dir="$EVO_REMOTE_DEPS_DIR_RESOLVED"
nix_root="$EVO_REMOTE_NIX_ROOT_RESOLVED"
cache_dir="$EVO_REMOTE_CACHE_DIR_RESOLVED"
definition="$source_dir/containers/evo.cpp-cuda12.8-rocky8.def"
apptainer="${apptainer_override:-$(command -v apptainer || true)}"

test -d "$source_dir"
test -f "$definition"
test -f "$deps_dir/flash-attention-628452c73a4fab560189a7caa8702642c6a38235/csrc/flash_attn/src/flash_fwd_kernel.h"
test -f "$deps_dir/cutlass-7127592069c2fe01b041e174ba4345ef9b279671/include/cute/tensor.hpp"
test -f "$deps_dir/libnpy-890ea4fcda302a580e633c624c6a63e2a5d422f6/include/npy.hpp"
test -n "$apptainer" && test -x "$apptainer" || {
  echo "gpu02_build: Apptainer is not available at ${apptainer:-<unset>}" >&2
  exit 2
}
export APPTAINER_CACHEDIR="$cache_dir/apptainer"
export APPTAINER_TMPDIR="/tmp/evo.cpp-apptainer-$UID"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "${image%/*}" \
  "${build_dir%/*}"
if ! test -f "$image"; then
  image_partial="${image%/*}/.${image##*/}.$$.partial"
  trap 'rm -f -- "$image_partial"' EXIT HUP INT TERM
  "$apptainer" build "$image_partial" "$definition"
  mv -- "$image_partial" "$image"
  trap - EXIT HUP INT TERM
fi
evo_select_cmake gpu02_build "$apptainer" "$image" "$nix_root" \
  "$cmake_override"
cmake_bin="$EVO_CMAKE_BIN_SELECTED"
echo "gpu02_build: using CMake $EVO_CMAKE_VERSION_SELECTED at $cmake_bin"
evo_select_python gpu02_build "$apptainer" "$image" "$nix_root" \
  "$python_override"
python_bin="$EVO_PYTHON_BIN_SELECTED"
echo "gpu02_build: using Python $EVO_PYTHON_VERSION_SELECTED at $python_bin"
container_binds=(
  -B "$nix_root:/nix:ro"
  -B "$source_dir:$source_dir"
  -B "$deps_dir:$deps_dir:ro"
)
if [[ "$build_dir" != "$source_dir"/* ]]; then
  container_binds+=(-B "${build_dir%/*}:${build_dir%/*}")
fi

"$apptainer" exec "${container_binds[@]}" "$image" nvcc --version
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
REMOTE
