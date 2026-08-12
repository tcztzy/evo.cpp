#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import argparse
import re
import subprocess
import tempfile
from pathlib import Path


REMOTE_PATH_VARIABLES = (
    "EVO_REMOTE_ROOT",
    "EVO_REMOTE_SOURCE_DIR",
    "EVO_REMOTE_BUILD_DIR",
    "EVO_REMOTE_DEPS_DIR",
    "EVO_REMOTE_CONTAINER_PATH",
    "EVO_REMOTE_NIX_ROOT",
    "EVO_REMOTE_CACHE_DIR",
)


def run_bash(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def check_remote_helper(source_dir: Path) -> None:
    helper = source_dir / "scripts" / "remote_build_common.sh"
    helper_text = helper.read_text(encoding="utf-8")
    assert "evo_version_at_least_3_25" in helper_text
    assert 'ctest_bin="${cmake_bin%/cmake}/ctest"' in helper_text
    assert 'test "$EVO_PROBED_CTEST_VERSION" = ' in helper_text
    assert '"$nix_root"/store/*-cmake-*/bin/cmake' in helper_text
    assert "evo_version_greater" in helper_text
    assert 'package_name="${store_entry#*-}"' in helper_text
    assert "evo_select_python" in helper_text
    assert '"$nix_root"/store/*-python3-*/bin/python3' in helper_text
    assert "evo_require_file" in helper_text
    assert "evo_source_fingerprint" in helper_text

    missing = run_bash(
        "-c",
        '. "$1"; evo_require_file contract "pinned dependency" "$2"',
        "remote-helper-test",
        str(helper),
        "/missing/pinned-dependency.hpp",
    )
    assert missing.returncode != 0
    assert "/missing/pinned-dependency.hpp" in missing.stderr

    with tempfile.TemporaryDirectory(prefix="evo-source-fingerprint-") as temporary:
        fingerprint_root = Path(temporary)
        (fingerprint_root / "src").mkdir()
        source = fingerprint_root / "src" / "sample.cpp"
        source.write_text("first\n", encoding="utf-8")
        first_fingerprint = run_bash(
            "-c",
            '. "$1"; evo_source_fingerprint "$2"',
            "remote-helper-test",
            str(helper),
            str(fingerprint_root),
        )
        assert first_fingerprint.returncode == 0, first_fingerprint.stderr
        bytecode = fingerprint_root / "src" / "__pycache__"
        bytecode.mkdir()
        (bytecode / "sample.cpython-313.pyc").write_bytes(b"test artifact")
        artifact_fingerprint = run_bash(
            "-c",
            '. "$1"; evo_source_fingerprint "$2"',
            "remote-helper-test",
            str(helper),
            str(fingerprint_root),
        )
        assert artifact_fingerprint.returncode == 0, artifact_fingerprint.stderr
        assert artifact_fingerprint.stdout == first_fingerprint.stdout, (
            "runtime Python cache artifacts must not stale the source fingerprint"
        )
        source.write_text("second\n", encoding="utf-8")
        second_fingerprint = run_bash(
            "-c",
            '. "$1"; evo_source_fingerprint "$2"',
            "remote-helper-test",
            str(helper),
            str(fingerprint_root),
        )
        assert second_fingerprint.returncode == 0, second_fingerprint.stderr
        assert first_fingerprint.stdout != second_fingerprint.stdout

    accepted = run_bash(
        "-c",
        '. "$1"; evo_validate_optional_remote_path contract ROOT "$2"',
        "remote-helper-test",
        str(helper),
        "/build/grp_icg/users/tang",
    )
    assert accepted.returncode == 0, accepted.stderr
    for unsafe in (
        "relative/path",
        "/",
        "/build/grp_icg/users/tang/../other",
        "/build/grp_icg/users/tang/with space",
    ):
        rejected = run_bash(
            "-c",
            '. "$1"; evo_validate_optional_remote_path contract ROOT "$2"',
            "remote-helper-test",
            str(helper),
            unsafe,
        )
        assert rejected.returncode != 0, f"unsafe remote path accepted: {unsafe}"

    resolved = run_bash(
        "-c",
        (
            '. "$1"; evo_configure_remote_paths contract "$2" "" "$3" '
            '"" "" "" "" build-default image-default.sif; '
            'printf "%s\\n" "$EVO_REMOTE_ROOT_RESOLVED" '
            '"$EVO_REMOTE_SOURCE_DIR_RESOLVED" '
            '"$EVO_REMOTE_BUILD_DIR_RESOLVED" '
            '"$EVO_REMOTE_DEPS_DIR_RESOLVED" '
            '"$EVO_REMOTE_CONTAINER_PATH_RESOLVED" '
            '"$EVO_REMOTE_NIX_ROOT_RESOLVED" '
            '"$EVO_REMOTE_CACHE_DIR_RESOLVED"'
        ),
        "remote-helper-test",
        str(helper),
        "/build/grp_icg/users/tang",
        "/build/grp_icg/users/tang/custom-build",
    )
    assert resolved.returncode == 0, resolved.stderr
    assert resolved.stdout.splitlines() == [
        "/build/grp_icg/users/tang",
        "/build/grp_icg/users/tang/evo.cpp",
        "/build/grp_icg/users/tang/custom-build",
        "/build/grp_icg/users/tang/evo.cpp-deps",
        "/build/grp_icg/users/tang/image-default.sif",
        "/build/grp_icg/users/tang/.local/share/nix-root",
        "/build/grp_icg/users/tang/.cache/evo.cpp",
    ]

    quoted = run_bash(
        "-c",
        (
            '. "$1"; evo_quote_remote_command bash "" "$2" "$3"; '
            'printf "%s\\n" "$EVO_REMOTE_COMMAND"'
        ),
        "remote-helper-test",
        str(helper),
        "/build/grp_icg/users/tang",
        "^(cuda_ops|npy)$",
    )
    assert quoted.returncode == 0, quoted.stderr
    assert quoted.stdout.strip() == (
        "bash -s -- '' /build/grp_icg/users/tang "
        r"\^\(cuda_ops\|npy\)\$"
    )

    with tempfile.TemporaryDirectory(prefix="evo-remote-cmake-") as temporary:
        temp_dir = Path(temporary)
        nix_root = temp_dir / "nix-root"
        versions = (
            "000-python3.12-cmake-3.31.2",
            "aaa-cmake-3.24.9",
            "bbb-cmake-3.30.0",
            "ccc-cmake-3.29.4",
            "ddd-cmake-3.31.2",
        )
        for version in versions:
            bin_dir = nix_root / "store" / version / "bin"
            bin_dir.mkdir(parents=True)
            for tool in ("cmake", "ctest"):
                path = bin_dir / tool
                path.write_text("", encoding="utf-8")
                path.chmod(0o755)
        python_versions = (
            "111-python3-3.8.19",
            "222-python3-3.12.5-env",
            "333-python3-3.12.5",
            "444-python3-3.13.1",
        )
        for version in python_versions:
            bin_dir = nix_root / "store" / version / "bin"
            bin_dir.mkdir(parents=True)
            python = bin_dir / "python3"
            python.write_text("", encoding="utf-8")
            python.chmod(0o755)
        fake_apptainer = temp_dir / "apptainer"
        fake_apptainer.write_text(
            """#!/usr/bin/env bash
set -euo pipefail
previous=""
current=""
for argument in "$@"; do
  previous="$current"
  current="$argument"
done
tool="$previous"
kind="${tool##*/}"
if test "$kind" = python3; then
  case "$tool" in
    *111-python3*) version=3.8.19 ;;
    *222-python3*) version=3.12.5 ;;
    *333-python3*) version=3.12.5 ;;
    *444-python3*) version=3.13.1 ;;
    *) exit 1 ;;
  esac
  printf 'Python %s\\n' "$version"
  exit 0
fi
case "$tool" in
  *000-python3.12-cmake*) version=3.31.2 ;;
  *aaa-cmake*) version=3.24.9 ;;
  *bbb-cmake*/bin/cmake) version=3.30.0 ;;
  *bbb-cmake*/bin/ctest) version=3.29.9 ;;
  *ccc-cmake*) version=3.29.4 ;;
  *ddd-cmake*) version=3.31.2 ;;
  *) exit 1 ;;
esac
printf '%s version %s\\n' "$kind" "$version"
""",
            encoding="utf-8",
        )
        fake_apptainer.chmod(0o755)
        selection_script = (
            'set -euo pipefail; . "$1"; '
            'evo_select_cmake contract "$2" /image.sif "$3" "$4"; '
            'printf "%s\\n%s\\n%s\\n" "$EVO_CMAKE_BIN_SELECTED" '
            '"$EVO_CTEST_BIN_SELECTED" "$EVO_CMAKE_VERSION_SELECTED"'
        )
        selected = run_bash(
            "-c",
            selection_script,
            "remote-helper-test",
            str(helper),
            str(fake_apptainer),
            str(nix_root),
            "",
        )
        assert selected.returncode == 0, selected.stderr
        assert selected.stdout.splitlines() == [
            "/nix/store/ddd-cmake-3.31.2/bin/cmake",
            "/nix/store/ddd-cmake-3.31.2/bin/ctest",
            "3.31.2",
        ]
        too_old = run_bash(
            "-c",
            selection_script,
            "remote-helper-test",
            str(helper),
            str(fake_apptainer),
            str(nix_root),
            "/nix/store/aaa-cmake-3.24.9/bin/cmake",
        )
        assert too_old.returncode != 0
        mismatched_ctest = run_bash(
            "-c",
            selection_script,
            "remote-helper-test",
            str(helper),
            str(fake_apptainer),
            str(nix_root),
            "/nix/store/bbb-cmake-3.30.0/bin/cmake",
        )
        assert mismatched_ctest.returncode != 0

        python_selection_script = (
            'set -euo pipefail; . "$1"; '
            'evo_select_python contract "$2" /image.sif "$3" "$4"; '
            'printf "%s\\n%s\\n" "$EVO_PYTHON_BIN_SELECTED" '
            '"$EVO_PYTHON_VERSION_SELECTED"'
        )
        selected_python = run_bash(
            "-c",
            python_selection_script,
            "remote-helper-test",
            str(helper),
            str(fake_apptainer),
            str(nix_root),
            "",
        )
        assert selected_python.returncode == 0, selected_python.stderr
        assert selected_python.stdout.splitlines() == [
            "/nix/store/444-python3-3.13.1/bin/python3",
            "3.13.1",
        ]
        too_old_python = run_bash(
            "-c",
            python_selection_script,
            "remote-helper-test",
            str(helper),
            str(fake_apptainer),
            str(nix_root),
            "/nix/store/111-python3-3.8.19/bin/python3",
        )
        assert too_old_python.returncode != 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    args = parser.parse_args()
    check_remote_helper(args.source_dir)

    test_script = args.source_dir / "scripts" / "gpu02_test.sh"
    text = test_script.read_text(encoding="utf-8")
    assert '"$apptainer" exec --nv' in text, (
        "V18: the gpu02 CUDA test runner must expose the host NVIDIA driver "
        "with Apptainer's 'exec --nv' mode"
    )
    assert 'ctest_bin="$EVO_CTEST_BIN_SELECTED"' in text
    assert "evo_select_cmake gpu02_test" in text
    assert 'fingerprint_file="$build_dir/.evo-source-fingerprint"' in text
    assert "source fingerprint does not match" in text
    assert 'test -x "$ctest_bin"' not in text, (
        "V18: /nix is mounted only inside Apptainer, so container-only "
        "executables must not be preflighted on the gpu02 host"
    )
    assert "evo_quote_remote_command bash" in text
    assert 'cuda_visible_devices="${EVO_CUDA_VISIBLE_DEVICES:-0}"' in text
    assert 'required_gpus="${EVO_CTEST_REQUIRED_GPUS:-1}"' in text
    assert "evo_require_idle_multi_gpu_list gpu02_test" in text
    for script in sorted((args.source_dir / "scripts").glob("gpu02_*.sh")):
        script_text = script.read_text(encoding="utf-8")
        assert re.search(r"(?m)^[ \t]*python3(?:[ \t]|$)", script_text) is None, (
            "V18: gpu02 scripts must use the pinned container Python instead "
            f"of the older host interpreter: {script.name}"
        )
        assert (
            re.search(r"(?m)(?:^|[|;&(])[ \t]*rg(?:[ \t]|$)", script_text)
            is None
        ), (
            "V18: gpu02 host scripts may not assume optional ripgrep is "
            f"installed: {script.name}"
        )

    gpu01_build = (
        args.source_dir / "scripts" / "gpu01_build.sh"
    ).read_text(encoding="utf-8")
    assert 'remote_host="${EVO_GPU01_HOST:-gpu01}"' in gpu01_build
    assert 'cuda_release="${EVO_GPU01_CUDA_RELEASE:-12.8}"' in gpu01_build
    assert "--delete" not in gpu01_build
    assert 'image_name="${EVO_GPU01_IMAGE_NAME:-' in gpu01_build
    assert 'apptainer="${apptainer_override:-$HOME/.local/apptainer/' in gpu01_build
    assert 'export APPTAINER_TMPDIR="/tmp/evo.cpp-apptainer-$UID"' in gpu01_build
    assert 'if test "$cuda_release" != "$expected_cuda_release"' in gpu01_build
    assert 'ctest_bin="$EVO_CTEST_BIN_SELECTED"' in gpu01_build
    assert "evo_select_cmake gpu01_build" in gpu01_build
    assert 'cuda_visible_devices="${EVO_CUDA_VISIBLE_DEVICES:-0}"' in gpu01_build
    assert 'build_jobs="${EVO_BUILD_JOBS:-4}"' in gpu01_build
    assert '  --build "$build_dir" -j"$build_jobs"' in gpu01_build
    assert "evo_select_python gpu01_build" in gpu01_build
    assert '-DPython3_EXECUTABLE="$python_bin"' in gpu01_build
    assert "EVO_CTEST_REQUIRED_GPUS" not in gpu01_build
    assert "evo_quote_remote_command bash" in gpu01_build
    for variable in REMOTE_PATH_VARIABLES:
        assert variable in gpu01_build
    assert "-R '^(npy|cuda_smoke|cuda_ops)$'" in gpu01_build
    assert "-DCMAKE_CUDA_ARCHITECTURES=80" in gpu01_build
    assert "-DEVO_WARNINGS_AS_ERRORS=ON" in gpu01_build
    assert "-DEVO_NPY=ON" in gpu01_build
    assert "-DBUILD_TESTING=ON" in gpu01_build
    assert "-DEVO_LIBNPY_SOURCE_DIR=" in gpu01_build
    for excluded in (
        "'/CMakeUserPresets.json'",
        "'/.pytest_cache/'",
        "'/.ruff_cache/'",
    ):
        assert f"--exclude {excluded}" in gpu01_build

    gpu01_image = (
        args.source_dir / "scripts" / "gpu01_build_image.sh"
    ).read_text(encoding="utf-8")
    assert 'image_host="${EVO_GPU01_IMAGE_HOST:-gpu02}"' in gpu01_image
    assert "--delete" not in gpu01_image
    assert "EVO_REMOTE_ROOT" in gpu01_image
    assert "EVO_REMOTE_SOURCE_DIR" in gpu01_image
    assert "EVO_REMOTE_CONTAINER_PATH" in gpu01_image
    assert "EVO_REMOTE_CACHE_DIR" in gpu01_image
    assert "evo_quote_remote_command bash" in gpu01_image
    assert "evo.cpp-cuda12.4-rocky8.def" in gpu01_image
    assert "docker-archive://$archive" in gpu01_image
    assert "cuda12.4.1-devel-rockylinux8-amd64.docker.tar" in gpu01_image
    assert '"$apptainer" build "$partial" "$build_source"' in gpu01_image
    assert 'mv -- "$partial" "$image"' in gpu01_image

    build = (
        args.source_dir / "scripts" / "gpu02_build.sh"
    ).read_text(encoding="utf-8")
    assert 'rsync_retries="${EVO_RSYNC_RETRIES:-12}"' in build
    assert 'build_jobs="${EVO_BUILD_JOBS:-4}"' in build
    assert '  --build "$build_dir" -j"$build_jobs"' in build
    assert "evo_select_python gpu02_build" in build
    assert '-DPython3_EXECUTABLE="$python_bin"' in build
    assert "until rsync -az --delay-updates" in build
    assert "evo_quote_remote_command bash" in build
    assert "EVO_CTEST_REQUIRED_GPUS" not in build
    assert "CUDA_VISIBLE_DEVICES" not in build
    assert "gpu_count" not in build
    for variable in REMOTE_PATH_VARIABLES:
        assert variable in build
    assert "-DEVO_FLASH_ATTENTION_SOURCE_DIR=" in build
    assert "-DEVO_CUTLASS_SOURCE_DIR=" in build
    assert "-DEVO_LIBNPY_SOURCE_DIR=" in build
    assert "-DEVO_NPY=ON" in build
    assert "-DBUILD_TESTING=ON" in build
    assert "--delete" not in build, (
        "the source sync must not remove remote models, environments, or "
        "research artifacts that are outside the local checkout"
    )
    for excluded in (
        "'/.cache/'",
        "'/.venv/'",
        "'/CMakeUserPresets.json'",
        "'/.pytest_cache/'",
        "'/.ruff_cache/'",
        "'*.pt'",
        "'*.safetensors'",
        "'*.safetensors.index.json'",
    ):
        assert f"--exclude {excluded}" in build
    assert 'if (( rsync_attempt >= rsync_retries ))' in build
    assert 'sleep "$rsync_retry_delay"' in build
    assert "evo_select_cmake gpu02_build" in build
    assert 'fingerprint_file="$build_dir/.evo-source-fingerprint"' in build
    assert 'source_fingerprint_before="$(evo_source_fingerprint "$source_dir")"' in build
    assert 'source_fingerprint_after="$(evo_source_fingerprint "$source_dir")"' in build
    assert "source changed during build" in build
    assert 'mv -- "$fingerprint_partial" "$fingerprint_file"' in build
    assert 'evo_require_file gpu02_build "pinned libnpy header"' in build
    assert 'cmake_bin="$EVO_CMAKE_BIN_SELECTED"' in build
    assert "store/*-cmake-[3-9]*/bin/cmake" not in build
    assert "dnsh5jd817k0zddr0k6x3zmyl146bbs6" not in build
    configure_index = build.index('  -S "$source_dir"')
    compile_index = build.index('  --build "$build_dir"')
    assert configure_index < compile_index, (
        "V16: the canonical build entrypoint must configure its build "
        "directory before attempting to compile it"
    )

    smoke = (
        args.source_dir / "scripts" / "gpu02_smoke.sh"
    ).read_text(encoding="utf-8")
    assert 'cuda_visible_devices="${EVO_CUDA_VISIBLE_DEVICES:-0}"' in smoke
    assert 'required_gpus="${EVO_CTEST_REQUIRED_GPUS:-1}"' in smoke
    assert "evo_require_idle_multi_gpu_list gpu02_smoke" in smoke
    assert "evo_select_cmake gpu02_smoke" in smoke
    for variable in REMOTE_PATH_VARIABLES:
        assert variable in smoke

    for protected_script in (gpu01_build, gpu01_image, build):
        assert "--delete" not in protected_script, (
            "remote synchronization must never delete pre-existing artifacts"
        )

    local_build = (
        args.source_dir / "scripts" / "local_test.sh"
    ).read_text(encoding="utf-8")
    local_configure_index = local_build.index('  -S "$source_dir"')
    local_compile_index = local_build.index(
        '"$cmake_bin" --build "$build_dir"'
    )
    local_test_index = local_build.index(
        'ctest --test-dir "$build_dir"'
    )
    assert local_configure_index < local_compile_index < local_test_index, (
        "V16: local verification must configure before build and test"
    )
    assert '-DEVO_SANITIZE="$sanitize"' in local_build
    assert "EVO_SANITIZERS" not in local_build

    arc_wrapper = (
        args.source_dir / "scripts" / "convert_arc_checkpoint.sh"
    ).read_text(encoding="utf-8")
    assert "configs/model-registry.json" in arc_wrapper
    assert "convert_checkpoint.py" in arc_wrapper
    assert "EVO_SOURCE_SHA256" in arc_wrapper
    assert "EVO_DRY_RUN" in arc_wrapper
    assert 'env -u LD_LIBRARY_PATH "$inspector"' in arc_wrapper

    model_validation = (
        args.source_dir / "scripts" / "validate_model.sh"
    ).read_text(encoding="utf-8")
    assert '--gpu "$gpu_list"' in model_validation
    assert "EVO_EXPECTED_LOGITS" in model_validation
    assert "compare_logits.py" in model_validation

    prepare = (
        args.source_dir / "scripts" / "gpu02_prepare_40b.sh"
    ).read_text(encoding="utf-8")
    assert "/build/grp_icg/users/tang/.cache/huggingface}" in prepare
    assert "https://hf-mirror.com" in prepare
    assert 'export HF_HOME="$hf_home"' in prepare
    assert 'export HF_ENDPOINT="$mirror"' in prepare
    assert 'output_base="$model_dir/evo2-40b-e4m3sw.safetensors"' in prepare
    assert 'output="$output_base.index.json"' in prepare
    assert 'output_receipt="$output_base.sha256"' in prepare

    bionemo = (
        args.source_dir / "scripts" / "gpu02_prepare_bionemo_40b.sh"
    ).read_text(encoding="utf-8")
    assert "/build/grp_icg/users/tang/.cache/bionemo" in bionemo
    assert (
        "https://api.ngc.nvidia.com/v2/models/nvidia/clara/"
        "evo2-40b-1m-fp8-bf16-nemo2/versions/1.0/files/"
        in bionemo
    )
    assert 'archive_size="63680606710"' in bionemo
    assert (
        'archive_sha256="'
        "544b47e033d1fb0261b686a53f7c4fe240cd290253187d31e8c99dea9e35a680"
        '"' in bionemo
    )
    assert 'output="$output_base.index.json"' in bionemo
    assert 'output_receipt="$output_base.sha256"' in bionemo
    assert "convert_bionemo_checkpoint.py" in bionemo
    assert "evo2-40b-1m-bionemo-bf16.yml" in bionemo
    assert "--dry-run" in bionemo
    assert "nohup setsid" in bionemo
    assert 'status_partial="${status_file}.$$.partial"' in bionemo
    assert 'until launch_output="$(' in bionemo
    assert 'job_manifest="$(' in bionemo
    assert '"$source_dir/tools/evo/bionemo_checkpoint.py"' in bionemo
    assert 'test "$(cat "$status_file")" != "0"' in bionemo
    assert 'checkpoint_dir="$resource_dir/checkpoint"' not in bionemo, (
        "V32: locate the actual DCP .metadata instead of assuming archive layout"
    )
    assert "HF_ENDPOINT" not in bionemo, (
        "V34: NGC retrieval is independent of the Hugging Face endpoint"
    )

    bionemo_validation = (
        args.source_dir / "scripts" / "gpu02_validate_bionemo_40b.sh"
    ).read_text(encoding="utf-8")
    assert "evo2-40b-bionemo-bf16.safetensors" in bionemo_validation
    assert (
        'expected_greedy_sha256="'
        "b28b7e7e6b70661dfee15d5290c4bca097ca145f721c4fbc4de73ad1d1660b8b"
        '"' in bionemo_validation
    )
    assert "--score" in bionemo_validation
    assert "-n 8 --ctx 8192" in bionemo_validation
    assert "--top-k 1 --seed 1" in bionemo_validation
    assert "--dump-logits" in bionemo_validation
    assert "tools/compare_logits.py" in bionemo_validation
    assert "--minimum-cosine 0.999" in bionemo_validation
    assert bionemo_validation.count("--profile fast-q8-kv") == 2, (
        "V23: exact-unsupported BioNeMo validation must select its explicit "
        "experimental profile for score and generation"
    )
    assert "--reference-bytes" in bionemo_validation
    assert bionemo_validation.count("apptainer exec --nv") == 2
    assert "nohup setsid" in bionemo_validation
    assert 'binary="$HOME/evo.cpp/build-gpu/evo"' in bionemo_validation
    assert 'binary_sha256="$(sha256sum "$binary"' in bionemo_validation
    assert 'status_partial="${status_file}.$$.partial"' in bionemo_validation
    assert "nvidia-smi --query-gpu=" in bionemo_validation
    assert "wait_for_exclusive_devices" in bionemo_validation
    assert "EVO_BIONEMO_GPU_LIST" in bionemo_validation
    assert 'gpu_tag="${gpu_list//,/}"' in bionemo_validation
    assert 'test "$mode" = "Exclusive_Process"' in bionemo_validation
    assert 'test "$(cat "$status_file")" != "0"' in bionemo_validation

    esmc_validation = (
        args.source_dir / "scripts" / "gpu02_validate_esmc.sh"
    ).read_text(encoding="utf-8")
    for model_id in ("esmc_300m", "esmc_600m", "esmc_6b"):
        assert model_id in esmc_validation
    assert "tools/evo_fetch.py" in esmc_validation
    assert "--local-files-only" in esmc_validation
    assert "convert_esmc_checkpoint.py" in esmc_validation
    assert "generate_esmc_official_oracle.py" in esmc_validation
    assert "compare_esmc_oracle.py" in esmc_validation
    assert 'r["repo"].replace("/", "--")' in esmc_validation
    assert '["files"][0]["path"]' not in esmc_validation, (
        "HF receipt paths may resolve a snapshot symlink into the blobs directory"
    )
    assert esmc_validation.count("--profile exact") == 2
    assert "--pooling none" in esmc_validation
    assert 'models="${EVO_ESMC_MODELS:-esmc_300m esmc_600m esmc_6b}"' in esmc_validation
    assert 'source_fingerprint="$(dirname "$binary")/.evo-source-fingerprint"' in esmc_validation
    assert "git -C" not in esmc_validation, (
        "gpu02 builds intentionally exclude .git and publish a source fingerprint"
    )
    esmc_syntax = run_bash(
        "-n", str(args.source_dir / "scripts" / "gpu02_validate_esmc.sh")
    )
    assert esmc_syntax.returncode == 0, esmc_syntax.stderr

    benchmark_7b = (
        args.source_dir / "scripts" / "gpu02_benchmark_7b.sh"
    ).read_text(encoding="utf-8")
    assert "EVO_REMOTE_7B_BENCHMARK_WORKER" in benchmark_7b
    assert 'remote_host="${EVO_GPU02_HOST:-gpu02}"' in benchmark_7b
    assert "'bash -s' -- \"$requested_gpu\"" in benchmark_7b
    assert 'gpu="${EVO_7B_GPU:-3}"' in benchmark_7b
    assert "nvidia-smi --query-compute-apps=gpu_uuid" in benchmark_7b
    assert "CUDA device $gpu is not idle" in benchmark_7b
    assert "-B /data:/data" in benchmark_7b
    assert (
        "c66645929dc1b9c631f5be656da8726f38946315dc9167000a615dd626fcecf4"
        in benchmark_7b
    )
    assert benchmark_7b.count("--minimum-top1-agreement") == 2
    assert "--minimum-cosine 0.99999" in benchmark_7b
    assert "prefill_benchmark_gate.py" in benchmark_7b
    assert '--minimum-rate "1024=$minimum_rate_1024"' in benchmark_7b
    assert 'minimum_rate_1024="${EVO_7B_MIN_RATE_1024:-9300}"' in benchmark_7b
    assert "artifact-sha256.txt" in benchmark_7b

    quality = (
        args.source_dir / "scripts" / "gpu02_quality.sh"
    ).read_text(encoding="utf-8")
    assert "nohup setsid" in quality
    assert 'status_partial="${status_file}.$$.partial"' in quality
    assert 'until launch_output="$(' in quality
    assert 'until ssh "${ssh_options[@]}" "${remote_host}"' in quality
    assert "--start-index" in quality
    assert "/build/grp_icg/users/tang/.cache" in quality
    assert "https://hf-mirror.com" in quality
    assert "evo2-40b-e4m3sw.safetensors" in quality
    assert 'model_sha256="$(sha256sum "$model"' in quality
    assert 'test -x "$python_bin"' not in quality, (
        "V18: the pinned /nix Python exists only after the Apptainer mount"
    )
    assert quality.count("apptainer exec --nv") == 2
    assert quality.count(
        '"$python_bin" "$source_dir/tools/evo2_quality_gate.py"'
    ) == 2, (
        "V18: every pinned /nix Python invocation must be container-scoped"
    )

    environment = (
        args.source_dir / "docs" / "gpu02-environment.md"
    ).read_text(encoding="utf-8")
    assert "EVO_REMOTE_ROOT=/build/grp_icg/users/tang" in environment
    assert "EVO_REMOTE_CONTAINER_PATH" in environment
    assert "CMake >=3.25" in environment
    assert "does not use `rsync --delete`" in environment
    assert "EVO_CUDA_VISIBLE_DEVICES=0,1,2,3" in environment
    assert "EVO_CTEST_REQUIRED_GPUS=4" in environment
    assert "report is 3,777 bytes" in environment
    assert (
        "314f50939dade079ef69494a57f954488082c361172547930c49ca4a02ef3a40"
        in environment
    )
    gpu01_environment = (
        args.source_dir / "docs" / "gpu01-environment.md"
    ).read_text(encoding="utf-8")
    assert "EVO_REMOTE_ROOT=/build/grp_icg/users/tang" in gpu01_environment
    assert "EVO_CUDA_VISIBLE_DEVICES=2" in gpu01_environment
    assert "CMake >=3.25" in gpu01_environment

    readme = (args.source_dir / "README.md").read_text(encoding="utf-8")
    readme_zh = (args.source_dir / "README.zh_CN.md").read_text(encoding="utf-8")
    for readme_text in (readme, readme_zh):
        assert "EVO_REMOTE_ROOT" in readme_text
        assert "EVO_CUDA_VISIBLE_DEVICES" in readme_text

    libnpy_commit = "890ea4fcda302a580e633c624c6a63e2a5d422f6"
    libnpy_archive_sha256 = (
        "c7b275c6cb8e46df43a20271e65010bdf63945831f2c0931ea6f2eda6a842acd"
    )
    assert libnpy_commit in gpu01_build
    assert libnpy_commit in build
    assert libnpy_commit in gpu01_environment
    assert libnpy_commit in environment
    assert libnpy_archive_sha256 in gpu01_environment
    assert libnpy_archive_sha256 in environment
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
