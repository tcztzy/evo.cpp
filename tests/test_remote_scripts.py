#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import argparse
import re
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    args = parser.parse_args()

    test_script = args.source_dir / "scripts" / "gpu02_test.sh"
    text = test_script.read_text(encoding="utf-8")
    assert "apptainer exec --nv" in text, (
        "V18: the gpu02 CUDA test runner must expose the host NVIDIA driver "
        "with 'apptainer exec --nv'"
    )
    assert 'ctest_bin="' in text
    assert 'test -x "$ctest_bin"' not in text, (
        "V18: /nix is mounted only inside Apptainer, so container-only "
        "executables must not be preflighted on the gpu02 host"
    )
    assert "printf -v quoted_regex '%q'" in text, (
        "V18: the optional CTest regex must be shell-quoted before SSH "
        "transports it through the remote login shell"
    )
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

    build = (
        args.source_dir / "scripts" / "gpu02_build.sh"
    ).read_text(encoding="utf-8")
    assert 'rsync_retries="${EVO_RSYNC_RETRIES:-12}"' in build
    assert "until rsync -az --delay-updates" in build
    assert "--delete" not in build, (
        "the source sync must not remove remote models, environments, or "
        "research artifacts that are outside the local checkout"
    )
    for excluded in (
        "'/.cache/'",
        "'/.venv/'",
        "'*.pt'",
        "'*.safetensors'",
        "'*.safetensors.index.json'",
    ):
        assert f"--exclude {excluded}" in build
    assert 'if (( rsync_attempt >= rsync_retries ))' in build
    assert 'sleep "$rsync_retry_delay"' in build
    configure_index = build.index('  -S "$source_dir"')
    compile_index = build.index('  --build "$build_dir"')
    assert configure_index < compile_index, (
        "V16: the canonical build entrypoint must configure its build "
        "directory before attempting to compile it"
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

    model_validation = (
        args.source_dir / "scripts" / "validate_model.sh"
    ).read_text(encoding="utf-8")
    assert '--gpu "$gpu_list"' in model_validation
    assert "EVO_EXPECTED_LOGITS" in model_validation
    assert "compare_logits.py" in model_validation

    prepare = (
        args.source_dir / "scripts" / "gpu02_prepare_40b.sh"
    ).read_text(encoding="utf-8")
    assert "/build/grp_icg/users/tang/.cache}" in prepare
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
    assert "report is 3,777 bytes" in environment
    assert (
        "314f50939dade079ef69494a57f954488082c361172547930c49ca4a02ef3a40"
        in environment
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
