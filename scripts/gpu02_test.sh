#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

remote_host="${EVO_GPU02_HOST:-gpu02}"
test_regex="${EVO_CTEST_REGEX:-}"
printf -v quoted_regex '%q' "$test_regex"

ssh -o ServerAliveInterval=5 -o ServerAliveCountMax=720 \
  "${remote_host}" "bash -s -- $quoted_regex" <<'REMOTE'
set -euo pipefail

test_regex="$1"
source_dir="$HOME/evo.cpp"
build_dir="$source_dir/build-gpu"
image="$HOME/evo.cpp-cuda12.8-rocky8.sif"
nix_root="$HOME/.local/share/nix-root"
ctest_bin="/nix/store/dnsh5jd817k0zddr0k6x3zmyl146bbs6-profile/bin/ctest"

test -d "$build_dir"
test -f "$image"
ctest_args=(--test-dir "$build_dir" --output-on-failure)
if test -n "$test_regex"; then
  ctest_args+=(-R "$test_regex")
fi
apptainer exec --nv -B "$nix_root:/nix:ro" "$image" \
  "$ctest_bin" "${ctest_args[@]}"
REMOTE
