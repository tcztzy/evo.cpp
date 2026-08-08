#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

remote_host="${EVO_GPU02_HOST:-gpu02}"

ssh -o ServerAliveInterval=5 -o ServerAliveCountMax=720 \
  "${remote_host}" 'bash -s' <<'REMOTE'
set -euo pipefail

source_dir="$HOME/evo.cpp"
build_dir="$source_dir/build-gpu"
image="$HOME/evo.cpp-cuda12.8-rocky8.sif"
nix_root="$HOME/.local/share/nix-root"

ctest_host=""
for candidate in "$nix_root"/store/*-cmake-[3-9]*/bin/ctest; do
  if test -x "$candidate"; then
    ctest_host="$candidate"
    break
  fi
done
test -n "$ctest_host" || {
  echo "gpu02_smoke: no CTest >=3 was found in $nix_root/store" >&2
  exit 2
}
ctest_bin="/nix/store/${ctest_host#"$nix_root/store/"}"

test -x "$build_dir/evo-cuda-pipeline-tests"
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used \
  --format=csv,noheader
apptainer exec --nv -B "$nix_root:/nix:ro" "$image" "$ctest_bin" \
  --test-dir "$build_dir" --output-on-failure

for binary in "$build_dir/evo" "$build_dir/evo-inspect" \
              "$build_dir/evo-cuda-pipeline-tests"; do
  if ldd "$binary" | grep -Eiq 'torch|vortex|transformer.engine'; then
    echo "gpu02_smoke: $binary has a forbidden framework dependency" >&2
    exit 2
  fi
done
apptainer exec --nv -B "$nix_root:/nix:ro" "$image" cuobjdump \
  --list-elf "$build_dir/evo-cuda-pipeline-tests" |
  tee "$build_dir/cuda-elf-targets.txt"
if grep -Evq '(^$|sm_80)' "$build_dir/cuda-elf-targets.txt"; then
  echo "gpu02_smoke: CUDA binary contains a non-sm_80 ELF image" >&2
  exit 2
fi
REMOTE
