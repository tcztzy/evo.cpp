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
    assert '-DEVO2C_SANITIZE="$sanitize"' in local_build
    assert "EVO2C_SANITIZERS" not in local_build

    prepare = (
        args.source_dir / "scripts" / "gpu02_prepare_40b.sh"
    ).read_text(encoding="utf-8")
    assert "/build/grp_icg/users/tang/.cache}" in prepare
    assert "https://hf-mirror.com" in prepare
    assert 'export HF_HOME="$hf_home"' in prepare
    assert 'export HF_ENDPOINT="$mirror"' in prepare
    assert 'output="$model_dir/evo2-40b-e4m3sw.evo2"' in prepare
    assert 'output_size="82252717056"' in prepare
    assert (
        'output_sha256="'
        "d1619e3b2eef0fba7c5838bb61982e891cf63d55385ced865af06693222d6687"
        '"' in prepare
    )

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
    assert (
        "d1619e3b2eef0fba7c5838bb61982e891cf63d55385ced865af06693222d6687"
        in quality
    )
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
