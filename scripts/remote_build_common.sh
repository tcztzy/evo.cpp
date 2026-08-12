#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

# Shared path, SSH argument, GPU, and CMake discovery helpers for the remote
# build entrypoints. This file is sourced on both the local and remote hosts.

evo_remote_error() {
  local script_name="$1"
  shift
  printf '%s: %s\n' "$script_name" "$*" >&2
}

evo_require_file() {
  local script_name="$1"
  local label="$2"
  local path="$3"
  if ! test -f "$path"; then
    evo_remote_error "$script_name" "$label is missing at $path"
    return 2
  fi
}

evo_sha256_file() {
  local path="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$path"
  else
    shasum -a 256 "$path"
  fi
}

evo_sha256_stream() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum
  else
    shasum -a 256
  fi
}

evo_source_fingerprint() {
  local source_dir="$1"
  local entry=""
  local path=""
  test -d "$source_dir" || {
    evo_remote_error source_fingerprint "source directory is missing at $source_dir"
    return 2
  }
  (
    cd -- "$source_dir"
    while IFS= read -r -d '' path; do
      evo_sha256_file "$path"
    done < <(
      for entry in CMakeLists.txt cmake conanfile.py configs include scripts \
          src tests third_party/flash-attention-shim tools; do
        if test -f "$entry"; then
          printf '%s\0' "$entry"
        elif test -d "$entry"; then
          find "$entry" -type f -print0
        fi
      done | LC_ALL=C sort -z
    )
  ) | evo_sha256_stream | awk '{print $1}'
}

evo_validate_optional_remote_path() {
  local script_name="$1"
  local variable_name="$2"
  local value="$3"
  if test -z "$value"; then
    return 0
  fi
  case "$value" in
    /*) ;;
    *)
      evo_remote_error "$script_name" "$variable_name must be an absolute path"
      return 2
      ;;
  esac
  case "$value" in
    /)
      evo_remote_error "$script_name" "$variable_name may not be /"
      return 2
      ;;
    *[!A-Za-z0-9_./+@%=-]*)
      evo_remote_error "$script_name" \
        "$variable_name contains unsupported or shell-unsafe characters"
      return 2
      ;;
    */../*|*/..|*/./*|*/.)
      evo_remote_error "$script_name" \
        "$variable_name may not contain . or .. path components"
      return 2
      ;;
  esac
}

evo_validate_remote_overrides() {
  local script_name="$1"
  shift
  while test "$#" -ge 2; do
    evo_validate_optional_remote_path "$script_name" "$1" "$2" || return
    shift 2
  done
}

evo_configure_remote_paths() {
  local script_name="$1"
  local remote_root_arg="$2"
  local source_dir_arg="$3"
  local build_dir_arg="$4"
  local deps_dir_arg="$5"
  local container_path_arg="$6"
  local nix_root_arg="$7"
  local cache_dir_arg="$8"
  local default_build_name="$9"
  local default_container_name="${10}"

  EVO_REMOTE_ROOT_RESOLVED="${remote_root_arg:-$HOME}"
  EVO_REMOTE_SOURCE_DIR_RESOLVED="${source_dir_arg:-${EVO_REMOTE_ROOT_RESOLVED}/evo.cpp}"
  EVO_REMOTE_BUILD_DIR_RESOLVED="${build_dir_arg:-${EVO_REMOTE_SOURCE_DIR_RESOLVED}/${default_build_name}}"
  EVO_REMOTE_DEPS_DIR_RESOLVED="${deps_dir_arg:-${EVO_REMOTE_ROOT_RESOLVED}/evo.cpp-deps}"
  EVO_REMOTE_CONTAINER_PATH_RESOLVED="${container_path_arg:-${EVO_REMOTE_ROOT_RESOLVED}/${default_container_name}}"
  EVO_REMOTE_NIX_ROOT_RESOLVED="${nix_root_arg:-${EVO_REMOTE_ROOT_RESOLVED}/.local/share/nix-root}"
  EVO_REMOTE_CACHE_DIR_RESOLVED="${cache_dir_arg:-${EVO_REMOTE_ROOT_RESOLVED}/.cache/evo.cpp}"

  evo_validate_remote_overrides "$script_name" \
    EVO_REMOTE_ROOT "$EVO_REMOTE_ROOT_RESOLVED" \
    EVO_REMOTE_SOURCE_DIR "$EVO_REMOTE_SOURCE_DIR_RESOLVED" \
    EVO_REMOTE_BUILD_DIR "$EVO_REMOTE_BUILD_DIR_RESOLVED" \
    EVO_REMOTE_DEPS_DIR "$EVO_REMOTE_DEPS_DIR_RESOLVED" \
    EVO_REMOTE_CONTAINER_PATH "$EVO_REMOTE_CONTAINER_PATH_RESOLVED" \
    EVO_REMOTE_NIX_ROOT "$EVO_REMOTE_NIX_ROOT_RESOLVED" \
    EVO_REMOTE_CACHE_DIR "$EVO_REMOTE_CACHE_DIR_RESOLVED"
}

evo_remote_source_for_sync() {
  local remote_root_arg="$1"
  local source_dir_arg="$2"
  if test -n "$source_dir_arg"; then
    EVO_REMOTE_SYNC_SOURCE="$source_dir_arg"
  elif test -n "$remote_root_arg"; then
    EVO_REMOTE_SYNC_SOURCE="${remote_root_arg}/evo.cpp"
  else
    EVO_REMOTE_SYNC_SOURCE="evo.cpp"
  fi
}

evo_quote_remote_command() {
  local remote_shell="$1"
  shift
  local remote_command="${remote_shell} -s --"
  local value=""
  local quoted_value=""
  for value in "$@"; do
    printf -v quoted_value '%q' "$value"
    remote_command+=" ${quoted_value}"
  done
  EVO_REMOTE_COMMAND="$remote_command"
}

evo_version_components() {
  local version="$1"
  if [[ ! "$version" =~ ^([0-9]+)\.([0-9]+)(\.([0-9]+))?$ ]]; then
    return 1
  fi
  EVO_VERSION_MAJOR=$((10#${BASH_REMATCH[1]}))
  EVO_VERSION_MINOR=$((10#${BASH_REMATCH[2]}))
  EVO_VERSION_PATCH=$((10#${BASH_REMATCH[4]:-0}))
}

evo_version_at_least_3_25() {
  evo_version_components "$1" || return 1
  (( EVO_VERSION_MAJOR > 3 || \
     (EVO_VERSION_MAJOR == 3 && EVO_VERSION_MINOR >= 25) ))
}

evo_version_at_least_3_9() {
  evo_version_components "$1" || return 1
  (( EVO_VERSION_MAJOR > 3 || \
     (EVO_VERSION_MAJOR == 3 && EVO_VERSION_MINOR >= 9) ))
}

evo_version_greater() {
  local left="$1"
  local right="$2"
  evo_version_components "$left" || return 1
  local left_major="$EVO_VERSION_MAJOR"
  local left_minor="$EVO_VERSION_MINOR"
  local left_patch="$EVO_VERSION_PATCH"
  evo_version_components "$right" || return 1
  (( left_major > EVO_VERSION_MAJOR ||
     (left_major == EVO_VERSION_MAJOR && left_minor > EVO_VERSION_MINOR) ||
     (left_major == EVO_VERSION_MAJOR && left_minor == EVO_VERSION_MINOR &&
      left_patch > EVO_VERSION_PATCH) ))
}

evo_probe_cmake_pair() {
  local apptainer="$1"
  local image="$2"
  local nix_root="$3"
  local cmake_bin="$4"
  local ctest_bin="${cmake_bin%/cmake}/ctest"
  local cmake_output=""
  local ctest_output=""

  cmake_output="$("$apptainer" exec -B "$nix_root:/nix:ro" "$image" \
    "$cmake_bin" --version 2>/dev/null)" || return 1
  ctest_output="$("$apptainer" exec -B "$nix_root:/nix:ro" "$image" \
    "$ctest_bin" --version 2>/dev/null)" || return 1
  EVO_PROBED_CMAKE_VERSION="$(printf '%s\n' "$cmake_output" | \
    sed -n '1s/^cmake version \([0-9][0-9.]*\).*$/\1/p')"
  EVO_PROBED_CTEST_VERSION="$(printf '%s\n' "$ctest_output" | \
    sed -n '1s/^ctest version \([0-9][0-9.]*\).*$/\1/p')"
  test -n "$EVO_PROBED_CMAKE_VERSION" || return 1
  test "$EVO_PROBED_CTEST_VERSION" = "$EVO_PROBED_CMAKE_VERSION" || return 1
  evo_version_at_least_3_25 "$EVO_PROBED_CMAKE_VERSION"
}

evo_select_cmake() {
  local script_name="$1"
  local apptainer="$2"
  local image="$3"
  local nix_root="$4"
  local cmake_override="$5"
  local candidate_host=""
  local candidate_container=""
  local store_entry=""
  local package_name=""
  local candidate_rank=1
  local best_bin=""
  local best_version=""
  local best_rank=2

  if test -n "$cmake_override"; then
    evo_validate_optional_remote_path "$script_name" EVO_CMAKE_BIN \
      "$cmake_override" || return
    if ! evo_probe_cmake_pair "$apptainer" "$image" "$nix_root" \
        "$cmake_override"; then
      evo_remote_error "$script_name" \
        "EVO_CMAKE_BIN must be CMake >=3.25 with matching CTest in the same bin directory"
      return 2
    fi
    EVO_CMAKE_BIN_SELECTED="$cmake_override"
    EVO_CTEST_BIN_SELECTED="${cmake_override%/cmake}/ctest"
    EVO_CMAKE_VERSION_SELECTED="$EVO_PROBED_CMAKE_VERSION"
    return 0
  fi

  test -d "$nix_root/store" || {
    evo_remote_error "$script_name" "Nix store is missing at $nix_root/store"
    return 2
  }
  shopt -s nullglob
  for candidate_host in "$nix_root"/store/*-cmake-*/bin/cmake; do
    test -x "$candidate_host" || continue
    test -x "${candidate_host%/cmake}/ctest" || continue
    candidate_container="/nix/store/${candidate_host#"$nix_root/store/"}"
    store_entry="${candidate_host#"$nix_root/store/"}"
    store_entry="${store_entry%%/*}"
    package_name="${store_entry#*-}"
    candidate_rank=1
    if [[ "$package_name" == cmake-* ]]; then
      candidate_rank=0
    fi
    evo_probe_cmake_pair "$apptainer" "$image" "$nix_root" \
      "$candidate_container" || continue
    if test -z "$best_bin" || \
        evo_version_greater "$EVO_PROBED_CMAKE_VERSION" "$best_version" || \
        { test "$EVO_PROBED_CMAKE_VERSION" = "$best_version" && \
          (( candidate_rank < best_rank )); } || \
        { test "$EVO_PROBED_CMAKE_VERSION" = "$best_version" && \
          (( candidate_rank == best_rank )) && \
          [[ "$candidate_container" < "$best_bin" ]]; }; then
      best_bin="$candidate_container"
      best_version="$EVO_PROBED_CMAKE_VERSION"
      best_rank="$candidate_rank"
    fi
  done
  shopt -u nullglob
  test -n "$best_bin" || {
    evo_remote_error "$script_name" \
      "no verified CMake >=3.25 with matching CTest was found in $nix_root/store"
    return 2
  }
  EVO_CMAKE_BIN_SELECTED="$best_bin"
  EVO_CTEST_BIN_SELECTED="${best_bin%/cmake}/ctest"
  EVO_CMAKE_VERSION_SELECTED="$best_version"
}

evo_probe_python() {
  local apptainer="$1"
  local image="$2"
  local nix_root="$3"
  local python_bin="$4"
  local python_output=""

  python_output="$("$apptainer" exec -B "$nix_root:/nix:ro" "$image" \
    "$python_bin" --version 2>&1)" || return 1
  EVO_PROBED_PYTHON_VERSION="$(printf '%s\n' "$python_output" | \
    sed -n '1s/^Python \([0-9][0-9.]*\).*$/\1/p')"
  test -n "$EVO_PROBED_PYTHON_VERSION" || return 1
  evo_version_at_least_3_9 "$EVO_PROBED_PYTHON_VERSION"
}

evo_select_python() {
  local script_name="$1"
  local apptainer="$2"
  local image="$3"
  local nix_root="$4"
  local python_override="$5"
  local candidate_host=""
  local candidate_container=""
  local store_entry=""
  local package_name=""
  local candidate_rank=1
  local best_bin=""
  local best_version=""
  local best_rank=2

  if test -n "$python_override"; then
    evo_validate_optional_remote_path "$script_name" EVO_PYTHON_BIN \
      "$python_override" || return
    if ! evo_probe_python "$apptainer" "$image" "$nix_root" \
        "$python_override"; then
      evo_remote_error "$script_name" \
        "EVO_PYTHON_BIN must be a working Python >=3.9 interpreter inside the container"
      return 2
    fi
    EVO_PYTHON_BIN_SELECTED="$python_override"
    EVO_PYTHON_VERSION_SELECTED="$EVO_PROBED_PYTHON_VERSION"
    return 0
  fi

  test -d "$nix_root/store" || {
    evo_remote_error "$script_name" "Nix store is missing at $nix_root/store"
    return 2
  }
  shopt -s nullglob
  for candidate_host in "$nix_root"/store/*-python3-*/bin/python3; do
    test -x "$candidate_host" || continue
    candidate_container="/nix/store/${candidate_host#"$nix_root/store/"}"
    evo_probe_python "$apptainer" "$image" "$nix_root" \
      "$candidate_container" || continue
    store_entry="${candidate_host#"$nix_root/store/"}"
    store_entry="${store_entry%%/*}"
    package_name="${store_entry#*-}"
    candidate_rank=1
    if test "$package_name" = "python3-$EVO_PROBED_PYTHON_VERSION"; then
      candidate_rank=0
    fi
    if test -z "$best_bin" || \
        evo_version_greater "$EVO_PROBED_PYTHON_VERSION" "$best_version" || \
        { test "$EVO_PROBED_PYTHON_VERSION" = "$best_version" && \
          (( candidate_rank < best_rank )); } || \
        { test "$EVO_PROBED_PYTHON_VERSION" = "$best_version" && \
          (( candidate_rank == best_rank )) && \
          [[ "$candidate_container" < "$best_bin" ]]; }; then
      best_bin="$candidate_container"
      best_version="$EVO_PROBED_PYTHON_VERSION"
      best_rank="$candidate_rank"
    fi
  done
  shopt -u nullglob
  test -n "$best_bin" || {
    evo_remote_error "$script_name" \
      "no verified Python >=3.9 was found in $nix_root/store"
    return 2
  }
  EVO_PYTHON_BIN_SELECTED="$best_bin"
  EVO_PYTHON_VERSION_SELECTED="$best_version"
}

evo_validate_cuda_visible_devices() {
  local script_name="$1"
  local gpu_list="$2"
  if [[ ! "$gpu_list" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    evo_remote_error "$script_name" \
      "EVO_CUDA_VISIBLE_DEVICES must be a comma-separated list of numeric GPU IDs"
    return 2
  fi
  local seen=","
  local gpu=""
  local gpu_items=()
  IFS=',' read -r -a gpu_items <<<"$gpu_list"
  for gpu in "${gpu_items[@]}"; do
    if (( 10#$gpu > 63 )); then
      evo_remote_error "$script_name" "CUDA GPU ID $gpu is outside [0,63]"
      return 2
    fi
    if [[ "$seen" == *",$gpu,"* ]]; then
      evo_remote_error "$script_name" "CUDA GPU ID $gpu is duplicated"
      return 2
    fi
    seen+="$gpu,"
  done
}

evo_require_idle_multi_gpu_list() {
  local script_name="$1"
  local gpu_list="$2"
  local required_count="$3"
  if [[ ! "$required_count" =~ ^[1-9][0-9]*$ ]] || \
      (( required_count > 4 )); then
    evo_remote_error "$script_name" \
      "EVO_CTEST_REQUIRED_GPUS must be an integer in [1,4]"
    return 2
  fi
  if (( required_count <= 1 )); then
    return 0
  fi

  local gpu_items=()
  IFS=',' read -r -a gpu_items <<<"$gpu_list"
  if (( ${#gpu_items[@]} < required_count )); then
    evo_remote_error "$script_name" \
      "$required_count GPUs were requested but EVO_CUDA_VISIBLE_DEVICES lists ${#gpu_items[@]}"
    return 2
  fi

  local inventory=""
  local compute_uuids=""
  inventory="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits)"
  compute_uuids="$(nvidia-smi --query-compute-apps=gpu_uuid \
    --format=csv,noheader,nounits 2>/dev/null || true)"
  local gpu=""
  local uuid=""
  local checked=0
  for gpu in "${gpu_items[@]}"; do
    uuid="$(printf '%s\n' "$inventory" | \
      sed -n "s/^${gpu},[[:space:]]*//p")"
    test -n "$uuid" || {
      evo_remote_error "$script_name" "CUDA GPU $gpu is not present on the host"
      return 2
    }
    if printf '%s\n' "$compute_uuids" | grep -Fqx "$uuid"; then
      evo_remote_error "$script_name" \
        "CUDA GPU $gpu is already running another compute process"
      return 2
    fi
    ((checked += 1))
    if (( checked == required_count )); then
      break
    fi
  done
}
