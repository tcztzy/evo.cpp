#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

image_host="${EVO_GPU01_IMAGE_HOST:-gpu02}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
definition="containers/evo.cpp-cuda12.4-rocky8.def"

ssh -o ConnectTimeout=10 "$image_host" \
  'mkdir -p "$HOME/evo.cpp/containers"'
rsync -az --delay-updates -e "ssh -o ConnectTimeout=10" \
  "$repo_root/$definition" "$image_host:evo.cpp/containers/"

ssh -o ServerAliveInterval=5 -o ServerAliveCountMax=720 \
  "$image_host" 'bash -s' <<'REMOTE'
set -euo pipefail
export LANG=C
export LC_ALL=C

definition="$HOME/evo.cpp/containers/evo.cpp-cuda12.4-rocky8.def"
image="$HOME/evo.cpp-cuda12.4-rocky8.sif"
partial="$HOME/.evo.cpp-cuda12.4-rocky8.sif.partial"
archive="$HOME/.cache/evo.cpp/cuda12.4.1-devel-rockylinux8-amd64.docker.tar"
apptainer="${EVO_APPTAINER:-$(command -v apptainer || true)}"

test -n "$apptainer" && test -x "$apptainer" || {
  echo "gpu01_build_image: Apptainer is not available on the image host" >&2
  exit 2
}
export APPTAINER_CACHEDIR="$HOME/.cache/evo.cpp/apptainer"
export APPTAINER_TMPDIR="/tmp/evo.cpp-apptainer-$UID"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

if ! test -f "$image"; then
  trap 'rm -f -- "$partial"' EXIT HUP INT TERM
  build_source="$definition"
  if test -f "$archive"; then
    build_source="docker-archive://$archive"
    echo "gpu01_build_image: using offline Docker archive $archive"
  fi
  "$apptainer" build "$partial" "$build_source"
  mv -- "$partial" "$image"
  trap - EXIT HUP INT TERM
fi
"$apptainer" exec "$image" nvcc --version
REMOTE
