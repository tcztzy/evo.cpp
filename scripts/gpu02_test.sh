#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

remote_host="${EVO_GPU02_HOST:-gpu02}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
. "${script_dir}/remote_build_common.sh"
test_regex="${EVO_CTEST_REGEX:-}"
remote_root="${EVO_REMOTE_ROOT:-}"
remote_source_dir="${EVO_REMOTE_SOURCE_DIR:-}"
remote_build_dir="${EVO_REMOTE_BUILD_DIR:-}"
remote_deps_dir="${EVO_REMOTE_DEPS_DIR:-}"
remote_container_path="${EVO_REMOTE_CONTAINER_PATH:-}"
remote_nix_root="${EVO_REMOTE_NIX_ROOT:-}"
remote_cache_dir="${EVO_REMOTE_CACHE_DIR:-}"
apptainer_override="${EVO_APPTAINER:-}"
cmake_override="${EVO_CMAKE_BIN:-}"
cuda_visible_devices="${EVO_CUDA_VISIBLE_DEVICES:-0}"
required_gpus="${EVO_CTEST_REQUIRED_GPUS:-1}"

evo_validate_remote_overrides gpu02_test \
  EVO_REMOTE_ROOT "$remote_root" \
  EVO_REMOTE_SOURCE_DIR "$remote_source_dir" \
  EVO_REMOTE_BUILD_DIR "$remote_build_dir" \
  EVO_REMOTE_DEPS_DIR "$remote_deps_dir" \
  EVO_REMOTE_CONTAINER_PATH "$remote_container_path" \
  EVO_REMOTE_NIX_ROOT "$remote_nix_root" \
  EVO_REMOTE_CACHE_DIR "$remote_cache_dir" \
  EVO_APPTAINER "$apptainer_override" \
  EVO_CMAKE_BIN "$cmake_override"
evo_validate_cuda_visible_devices gpu02_test "$cuda_visible_devices"
if [[ ! "$required_gpus" =~ ^[1-4]$ ]]; then
  echo "gpu02_test: EVO_CTEST_REQUIRED_GPUS must be an integer in [1,4]" >&2
  exit 2
fi
evo_quote_remote_command bash \
  "$test_regex" "$remote_root" "$remote_source_dir" "$remote_build_dir" \
  "$remote_deps_dir" "$remote_container_path" "$remote_nix_root" \
  "$remote_cache_dir" "$apptainer_override" "$cmake_override" \
  "$cuda_visible_devices" "$required_gpus"

ssh -o ServerAliveInterval=5 -o ServerAliveCountMax=720 \
  "${remote_host}" "$EVO_REMOTE_COMMAND" <<'REMOTE'
set -euo pipefail
export LANG=C
export LC_ALL=C

test_regex="$1"
remote_root_arg="$2"
source_dir_arg="$3"
build_dir_arg="$4"
deps_dir_arg="$5"
container_path_arg="$6"
nix_root_arg="$7"
cache_dir_arg="$8"
apptainer_override="$9"
cmake_override="${10}"
cuda_visible_devices="${11}"
required_gpus="${12}"
root_for_source="${remote_root_arg:-$HOME}"
source_for_helpers="${source_dir_arg:-${root_for_source}/evo.cpp}"
. "$source_for_helpers/scripts/remote_build_common.sh"
evo_configure_remote_paths gpu02_test \
  "$remote_root_arg" "$source_dir_arg" "$build_dir_arg" "$deps_dir_arg" \
  "$container_path_arg" "$nix_root_arg" "$cache_dir_arg" \
  build-gpu evo.cpp-cuda12.8-rocky8.sif
evo_validate_cuda_visible_devices gpu02_test "$cuda_visible_devices"
source_dir="$EVO_REMOTE_SOURCE_DIR_RESOLVED"
build_dir="$EVO_REMOTE_BUILD_DIR_RESOLVED"
image="$EVO_REMOTE_CONTAINER_PATH_RESOLVED"
nix_root="$EVO_REMOTE_NIX_ROOT_RESOLVED"
apptainer="${apptainer_override:-$(command -v apptainer || true)}"

test -d "$build_dir"
test -f "$image"
test -n "$apptainer" && test -x "$apptainer" || {
  echo "gpu02_test: Apptainer is not available at ${apptainer:-<unset>}" >&2
  exit 2
}
fingerprint_file="$build_dir/.evo-source-fingerprint"
evo_require_file gpu02_test "successful build fingerprint" "$fingerprint_file"
expected_fingerprint="$(evo_source_fingerprint "$source_dir")"
actual_fingerprint="$(sed -n '1p' "$fingerprint_file")"
if test "$actual_fingerprint" != "$expected_fingerprint"; then
  echo "gpu02_test: source fingerprint does not match build; run gpu02_build.sh first" >&2
  exit 2
fi
evo_require_idle_multi_gpu_list gpu02_test "$cuda_visible_devices" \
  "$required_gpus"
evo_select_cmake gpu02_test "$apptainer" "$image" "$nix_root" \
  "$cmake_override"
ctest_bin="$EVO_CTEST_BIN_SELECTED"
echo "gpu02_test: using CTest $EVO_CMAKE_VERSION_SELECTED at $ctest_bin"
ctest_args=(--test-dir "$build_dir" --output-on-failure)
if test -n "$test_regex"; then
  ctest_args+=(-R "$test_regex")
fi
container_binds=(-B "$nix_root:/nix:ro" -B "$source_dir:$source_dir")
if [[ "$build_dir" != "$source_dir"/* ]]; then
  container_binds+=(-B "${build_dir%/*}:${build_dir%/*}")
fi
CUDA_VISIBLE_DEVICES="$cuda_visible_devices" \
  "$apptainer" exec --nv "${container_binds[@]}" "$image" env \
  CUDA_VISIBLE_DEVICES="$cuda_visible_devices" \
  "$ctest_bin" "${ctest_args[@]}"
REMOTE
