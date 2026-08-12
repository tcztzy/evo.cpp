#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

image_host="${EVO_GPU01_IMAGE_HOST:-gpu02}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
. "${script_dir}/remote_build_common.sh"
definition="containers/evo.cpp-cuda12.4-rocky8.def"
remote_root="${EVO_REMOTE_ROOT:-}"
remote_source_dir="${EVO_REMOTE_SOURCE_DIR:-}"
remote_container_path="${EVO_REMOTE_CONTAINER_PATH:-}"
remote_cache_dir="${EVO_REMOTE_CACHE_DIR:-}"
apptainer_override="${EVO_APPTAINER:-}"

evo_validate_remote_overrides gpu01_build_image \
  EVO_REMOTE_ROOT "$remote_root" \
  EVO_REMOTE_SOURCE_DIR "$remote_source_dir" \
  EVO_REMOTE_CONTAINER_PATH "$remote_container_path" \
  EVO_REMOTE_CACHE_DIR "$remote_cache_dir" \
  EVO_APPTAINER "$apptainer_override"
evo_remote_source_for_sync "$remote_root" "$remote_source_dir"
remote_sync_source="$EVO_REMOTE_SYNC_SOURCE"

ssh -o ConnectTimeout=10 "$image_host" \
  mkdir -p -- "$remote_sync_source/containers" "$remote_sync_source/scripts"
rsync -az --delay-updates -e "ssh -o ConnectTimeout=10" \
  "$repo_root/$definition" "$image_host:$remote_sync_source/containers/"
rsync -az --delay-updates -e "ssh -o ConnectTimeout=10" \
  "$script_dir/remote_build_common.sh" "$image_host:$remote_sync_source/scripts/"

evo_quote_remote_command bash \
  "$remote_root" "$remote_source_dir" "$remote_container_path" \
  "$remote_cache_dir" "$apptainer_override"
ssh -o ServerAliveInterval=5 -o ServerAliveCountMax=720 \
  "$image_host" "$EVO_REMOTE_COMMAND" <<'REMOTE'
set -euo pipefail
export LANG=C
export LC_ALL=C

remote_root_arg="$1"
source_dir_arg="$2"
container_path_arg="$3"
cache_dir_arg="$4"
apptainer_override="$5"
root_for_source="${remote_root_arg:-$HOME}"
source_for_helpers="${source_dir_arg:-${root_for_source}/evo.cpp}"
. "$source_for_helpers/scripts/remote_build_common.sh"
evo_configure_remote_paths gpu01_build_image \
  "$remote_root_arg" "$source_dir_arg" "" "" "$container_path_arg" "" \
  "$cache_dir_arg" unused-build evo.cpp-cuda12.4-rocky8.sif
source_dir="$EVO_REMOTE_SOURCE_DIR_RESOLVED"
image="$EVO_REMOTE_CONTAINER_PATH_RESOLVED"
cache_dir="$EVO_REMOTE_CACHE_DIR_RESOLVED"
definition="$source_dir/containers/evo.cpp-cuda12.4-rocky8.def"
partial="${image%/*}/.${image##*/}.$$.partial"
archive="$cache_dir/cuda12.4.1-devel-rockylinux8-amd64.docker.tar"
apptainer="${apptainer_override:-$(command -v apptainer || true)}"

test -n "$apptainer" && test -x "$apptainer" || {
  echo "gpu01_build_image: Apptainer is not available on the image host" >&2
  exit 2
}
test -f "$definition"
export APPTAINER_CACHEDIR="$cache_dir/apptainer"
export APPTAINER_TMPDIR="/tmp/evo.cpp-apptainer-$UID"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "${image%/*}"

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
