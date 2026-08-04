#!/usr/bin/env python3
"""Tests for the real-model vector comparator."""

import json
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError:
    print("SKIP: NumPy is not installed")
    raise SystemExit(77)

from compare_model_vectors import compare_directories


class ModelVectorComparatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="evo-vector-test-")
        self.root = Path(self.temporary.name)
        self.reference = self.root / "reference"
        self.native = self.root / "native"
        self.reference.mkdir()
        self.native.mkdir()
        (self.reference / "manifest.json").write_text(
            json.dumps({"layers": 2, "tokens": [65, 67]}),
            encoding="utf-8",
        )
        for layer in range(2):
            values = np.array([[1.0 + layer, -2.0], [3.0, 4.0]], dtype=np.float32)
            np.save(self.reference / f"layer_{layer:02d}.npy", values, allow_pickle=False)
            np.save(self.native / f"layer_{layer:02d}.npy", values, allow_pickle=False)
        logits = np.array([[0.0, 1.0, 2.0], [4.0, 3.0, 2.0]], dtype=np.float32)
        np.save(self.reference / "logits.npy", logits, allow_pickle=False)
        np.save(self.native / "logits.npy", logits, allow_pickle=False)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_identical_vectors_pass(self) -> None:
        report = compare_directories(self.reference, self.native, 0.999)
        self.assertTrue(report["passed"])
        self.assertIsNone(report["first_failing_layer"])
        self.assertEqual(report["logits"]["reference_last_top1"], 0)

    def test_first_bad_layer_and_top1_are_reported(self) -> None:
        bad_layer = np.load(self.native / "layer_01.npy", allow_pickle=False)
        bad_layer *= -1
        np.save(self.native / "layer_01.npy", bad_layer, allow_pickle=False)
        bad_logits = np.load(self.native / "logits.npy", allow_pickle=False)
        bad_logits[-1] = [1.0, 5.0, 2.0]
        np.save(self.native / "logits.npy", bad_logits, allow_pickle=False)

        report = compare_directories(self.reference, self.native, 0.999)
        self.assertFalse(report["passed"])
        self.assertEqual(report["first_failing_layer"], 1)
        self.assertEqual(report["logits"]["native_last_top1"], 1)

    def test_exact_mode_rejects_cosine_indistinguishable_drift(self) -> None:
        drifted = np.load(self.native / "layer_01.npy", allow_pickle=False)
        drifted.flat[0] = np.nextafter(drifted.flat[0], np.float32(np.inf))
        np.save(self.native / "layer_01.npy", drifted, allow_pickle=False)

        similarity_report = compare_directories(
            self.reference,
            self.native,
            0.999,
        )
        self.assertTrue(similarity_report["passed"])

        exact_report = compare_directories(
            self.reference,
            self.native,
            0.999,
            require_exact=True,
        )
        self.assertFalse(exact_report["passed"])
        self.assertEqual(exact_report["first_failing_layer"], 1)
        self.assertFalse(exact_report["layers"][1]["raw_exact"])
        self.assertEqual(exact_report["layers"][1]["unequal_elements"], 1)
        self.assertEqual(
            exact_report["layers"][1]["first_unequal_flat_index"],
            0,
        )

    def test_exact_mode_includes_debug_boundaries(self) -> None:
        (self.reference / "manifest.json").write_text(
            json.dumps({"layers": 2, "debug_layer": 0}),
            encoding="utf-8",
        )
        values = np.array([[1.0, 2.0]], dtype=np.float32)
        np.save(self.reference / "layer_0_probe.npy", values, allow_pickle=False)
        np.save(self.native / "layer_0_probe.npy", values, allow_pickle=False)
        report = compare_directories(
            self.reference,
            self.native,
            0.999,
            require_exact=True,
        )
        self.assertTrue(report["passed"])
        self.assertEqual(len(report["debug"]), 1)
        self.assertTrue(report["debug"][0]["raw_exact"])

        values[0, 1] = np.nextafter(values[0, 1], np.float32(np.inf))
        np.save(self.native / "layer_0_probe.npy", values, allow_pickle=False)
        report = compare_directories(
            self.reference,
            self.native,
            0.999,
            require_exact=True,
        )
        self.assertFalse(report["passed"])
        self.assertEqual(report["first_failing_debug"], "layer_0_probe.npy")

    def test_reference_only_diagnostics_are_not_native_requirements(self) -> None:
        (self.reference / "manifest.json").write_text(
            json.dumps({"layers": 2, "debug_layer": 0}),
            encoding="utf-8",
        )
        values = np.array([[1.0, 2.0]], dtype=np.float32)
        np.save(
            self.reference / "layer_0_reference_only.npy",
            values,
            allow_pickle=False,
        )
        report = compare_directories(
            self.reference,
            self.native,
            0.999,
            require_exact=True,
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["debug"], [])

    def test_checked_in_real_model_alignment_evidence_passes(self) -> None:
        evidence_path = Path(__file__).parent / "vectors" / "model_alignment.json"
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        native = evidence["evo_40b_bf16"]
        self.assertTrue(native["passed"])
        self.assertIsNone(native["first_failing_layer"])
        self.assertGreaterEqual(
            native["worst_layer"]["cosine"],
            native["minimum_cosine"],
        )
        self.assertTrue(native["logits"]["passed"])
        self.assertEqual(
            native["logits"]["reference_last_top1"],
            native["logits"]["native_last_top1"],
        )
        self.assertTrue(evidence["evo2_40b_bf16_reference"]["repeat_run_bit_exact"])
        for key in (
            "evo2_7b_bf16_reference",
            "evo2_40b_bf16_reference",
            "evo_40b_bf16",
        ):
            self.assertRegex(
                evidence[key]["vector_set_sha256"],
                r"^[0-9a-f]{64}$",
            )


if __name__ == "__main__":
    unittest.main()
