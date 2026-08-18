#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import hashlib
import json
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EVO_GENERATOR = ROOT / "tools" / "generate_geneb_evo1_131k_upstream_oracle.py"
META_GENERATOR = ROOT / "tools" / "generate_geneb_metagene_1_upstream_oracle.py"
EVO_PREFLIGHT = (
    ROOT / "configs" / "evidence" / "audits" / "geneb-evo-1-131k-preflight.json"
)
META_PREFLIGHT = (
    ROOT / "configs" / "evidence" / "audits" / "geneb-metagene-1-preflight.json"
)
EXPECTED_HOSTS = {
    ("Darwin", "arm64"): "darwin-arm64",
    ("Linux", "x86_64"): "linux-x86_64",
}


def parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def assignment(tree: ast.Module, name: str) -> ast.expr:
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ):
            return node.value
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
            and node.value is not None
        ):
            return node.value
    raise AssertionError(f"missing assignment {name}")


def function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function {name}")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"{path} root is not an object")
    return value


class HighMemoryOracleContractTest(unittest.TestCase):
    def test_two_platform_host_contract_and_pending_linux_vectors(self) -> None:
        evo = parse(EVO_GENERATOR)
        meta = parse(META_GENERATOR)
        for tree in (evo, meta):
            self.assertEqual(
                ast.literal_eval(assignment(tree, "SUPPORTED_ORACLE_HOSTS")),
                EXPECTED_HOSTS,
            )
            self.assertEqual(
                ast.unparse(assignment(tree, "MINIMUM_ORACLE_HOST_MEMORY_BYTES")),
                "48 * 1024 ** 3",
            )
            host_gate = ast.unparse(function(tree, "validate_oracle_host"))
            self.assertIn("if full and (not memory_eligible)", host_gate)
            self.assertIn("Darwin/arm64 or Linux/x86_64", host_gate)
            self.assertIn("cross_isa_bit_exact_claimed", host_gate)

        self.assertEqual(
            ast.literal_eval(
                assignment(evo, "LINUX_X86_64_OPERATOR_OUTPUT_SHA256")
            ),
            "7793f22e0775624de4616355f92055ccc2a8f62e0b9662bfeb2ac718ba23f3f1",
        )
        self.assertEqual(
            ast.literal_eval(assignment(meta, "LINUX_X86_64_OPERATOR_SHA256")),
            {
                "rmsnorm_raw_f32_sha256": "5508a7b5b5b7bfb6d1a7d9d651ef94951bb6c6be0fdc75d6aed75098d2930165",
                "split_half_rope_raw_f32_sha256": "d9ae2aa864cdb7901dba7cfbcd72a8366bc50e7e05b615ee5a330fb6e58f7f12",
                "sdpa_raw_f32_sha256": "974c359338753008a899d2d1ad32de410ebcf361303e4dbabfaa016e4a113065",
                "swiglu_raw_f32_sha256": "6f361bf92696f003c143f8fb6004f89af3e5a127831e0c21f96fbe91b8f7f6a9",
            },
        )
        self.assertEqual(
            ast.literal_eval(assignment(evo, "OPERATOR_OUTPUT_SHA256")),
            "dd087a9938611dd258597594301a0d16518579d1b2db8fdfcb9edaf9a18467f8",
        )
        evo_packages = ast.unparse(function(evo, "package_versions"))
        self.assertIn("LINUX_X86_64_EXPECTED_PACKAGES", evo_packages)
        meta_environment = ast.unparse(function(meta, "validate_environment"))
        self.assertIn("required['torch'] = '2.7.1+cpu'", meta_environment)
        self.assertIn("official CPU-only Torch build", meta_environment)
        for tree, operator_name in (
            (evo, "operator_self_test"),
            (meta, "validate_operator_contract"),
        ):
            operator = ast.unparse(function(tree, operator_name))
            self.assertIn(
                "candidate-pending-remote-audit-only-freeze", operator
            )
            main = ast.unparse(function(tree, "main"))
            self.assertLess(
                main.index("platform_vector_status"),
                main.index("validate_receipt")
                if tree is evo
                else main.index("validate_full_sources"),
            )

    def test_metagene_loader_is_exact_cpu_f32_closure(self) -> None:
        tree = parse(META_GENERATOR)
        execute = function(tree, "execute_oracle")
        calls = [
            node
            for node in ast.walk(execute)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "from_pretrained"
        ]
        model_calls = [
            node
            for node in calls
            if any(keyword.arg == "output_loading_info" for keyword in node.keywords)
        ]
        self.assertEqual(len(model_calls), 1)
        keywords = {keyword.arg: keyword.value for keyword in model_calls[0].keywords}
        self.assertEqual(ast.literal_eval(keywords["device_map"]), {"": "cpu"})
        execute_source = ast.unparse(execute)
        self.assertNotIn("device_map='auto'", execute_source)
        for fragment in (
            "len(state) != SOURCE_TENSOR_COUNT",
            "state_logical_bytes != CHECKPOINT_LOGICAL_BYTES",
            "tensor.dtype != torch.float32",
            "tensor.device.type != 'cpu'",
            "tensor.is_meta",
            "parameter.dtype != torch.float32",
            "parameter.device.type != 'cpu'",
            "parameter.is_meta",
            "device_map != {'': 'cpu'}",
        ):
            self.assertIn(fragment, execute_source)

    def test_preflights_leave_remote_audit_and_full_output_pending(self) -> None:
        evo = load_json(EVO_PREFLIGHT)
        meta = load_json(META_PREFLIGHT)
        for evidence in (evo, meta):
            candidate = evidence["resource_plan"]["configured_larger_host_candidate"]
            self.assertNotIn("name", candidate)
            self.assertEqual(
                (candidate["system"], candidate["machine"]),
                ("Linux", "x86_64"),
            )
            self.assertEqual(
                candidate["required_promotion_status"],
                "verified-linux-x86_64-high-memory-host",
            )
            self.assertEqual(candidate["remote_audit_only_status"], "not-run")
            self.assertFalse(candidate["verified_high_memory_host"])
            self.assertFalse(candidate["full_oracle_allowed"])
            self.assertFalse(evidence["official_model_executed_before_freeze"])
            order = evidence["resource_plan"]["execution_order"]
            audit = next(index for index, value in enumerate(order) if "audit-only" in value)
            freeze = next(index for index, value in enumerate(order) if "operator-hashes" in value)
            run = next(index for index, value in enumerate(order) if value.startswith("run-") and "input-0" in value)
            self.assertLess(audit, freeze)
            self.assertLess(freeze, run)
            self.assertEqual(
                evidence["oracle"]["supported_host_profiles"],
                ["darwin-arm64", "linux-x86_64"],
            )
            self.assertFalse(evidence["oracle"]["cross_isa_bit_exact_claimed"])
            packages = evidence["oracle"]["planned_packages_by_host"]
            self.assertEqual(packages["darwin-arm64"]["torch"].endswith("+cpu"), False)
            self.assertTrue(packages["linux-x86_64"]["torch"].endswith("+cpu"))

        self.assertIn("B134", evo["spec"])
        self.assertIn("B134", meta["spec"])
        self.assertIn("B135", meta["spec"])
        self.assertEqual(meta["oracle"]["planned_device_map"], {"": "cpu"})
        self.assertEqual(meta["oracle"]["planned_state_tensor_count"], 291)
        self.assertEqual(
            meta["oracle"]["planned_state_logical_bytes"], 25_938_640_896
        )
        self.assertEqual(
            set(meta["oracle"]["planned_rejected_placements"]),
            {"cuda", "mps", "disk", "meta"},
        )

    def test_preflight_generator_descriptors_match(self) -> None:
        # The preflights froze the pre-Linux-freeze generator drafts.  After
        # the remote Linux-x86_64 audits the operator constants were frozen
        # and the real-run output stages were adapted to the runner vector
        # schema, so the live generators necessarily differ from the
        # recorded draft descriptors.  Pin the recorded draft descriptors
        # and require the live generators to carry the frozen constants.
        historical = {
            EVO_GENERATOR: (46_225, "3a1c8b3646f616b4007e19942f689a575705b4df56f84173aaba3bbe851372a4"),
            META_GENERATOR: (66_533, "e02fd1c9d40a1c572a08fb1a20f384ba414e4aad4a59f1817627ac6395bb0c8a"),
        }
        for generator, preflight in (
            (EVO_GENERATOR, EVO_PREFLIGHT),
            (META_GENERATOR, META_PREFLIGHT),
        ):
            descriptor = load_json(preflight)["oracle"]
            size, digest = historical[generator]
            self.assertEqual(descriptor["generator_size"], size)
            self.assertEqual(descriptor["generator_sha256"], digest)
            payload = generator.read_bytes()
            self.assertNotEqual(
                (len(payload), hashlib.sha256(payload).hexdigest()),
                (size, digest),
            )


if __name__ == "__main__":
    unittest.main()
