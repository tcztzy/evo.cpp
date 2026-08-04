#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$(cd -- "${script_dir}/.." && pwd)"
build_dir="${1:-${source_dir}/build-cpu}"
cmake_bin="${CMAKE:-cmake}"
build_jobs="${EVO_BUILD_JOBS:-4}"
sanitize="${EVO_SANITIZE:-OFF}"

"$cmake_bin" \
  -S "$source_dir" \
  -B "$build_dir" \
  -DEVO_CUDA=OFF \
  -DEVO_WARNINGS_AS_ERRORS=ON \
  -DEVO_SANITIZE="$sanitize" \
  -DCMAKE_BUILD_TYPE=Release
"$cmake_bin" --build "$build_dir" -j"$build_jobs"
ctest --test-dir "$build_dir" --output-on-failure
