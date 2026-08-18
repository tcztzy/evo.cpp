#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate the pinned GENA-LM base GENEB oracle with official HF code."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any


RUNTIME_ID = "geneb-gena-lm"
SOURCE_REPO = "AIRI-Institute/gena-lm-bert-base"
REQUESTED_REVISION = "main"
SOURCE_REVISION = "416f055300346a5830ca49438daf5f4e136ed9a8"
CATALOG_CONTRACT_SHA256 = (
    "2b3a50188fe77daebbff2babda31ad2a28e7d6a9b7db37897b0821f24990719d"
)
EXTRACTOR_REPO = "ultimativity/GENEB"
EXTRACTOR_REVISION = "b465d2d6a11efbbc9a22c105e34832725ce50e05"
EXTRACTOR_SHA256 = (
    "6a9e5ba5720e93e7078881aabf575944f48959337444e37f19d9a6aa6962516d"
)
EXTRACTOR_PYPROJECT_SHA256 = (
    "616a15c0a751eb14687fefe691cd624fbf4cc104605f167a5782cb29d16060e0"
)
CANONICAL_SEQUENCE = "ACGTNACGTNACGTN"
CANONICAL_TOKEN_IDS = [1, 114, 9, 0, 114, 9, 0, 114, 9, 0, 2]
EXPECTED_SOURCE_FILES = {
    ".gitattributes": (
        1173,
        "983f034a5d3898b49e1c02cf4941beaf38849e8f01912ad6d8697e96270ee997",
    ),
    "README.md": (
        3830,
        "67174f94263b0844618c66f84601bb4c041ec511758ffec160efc553b877013b",
    ),
    "config.json": (
        754,
        "f4288ae1b2270570a803bb90b0c207f9927149876f333955df97f4e91eff2f67",
    ),
    "modeling_bert.py": (
        97101,
        "d641b0b99707af6badd0f3d11e0ae68505eaf4ef24ddc918b9c3f3995cf93ca0",
    ),
    "pytorch_model.bin": (
        543494664,
        "4310578f475a716f6343c7da90fba6230f2c917e80444726a481a611c1faf054",
    ),
    "special_tokens_map.json": (
        112,
        "303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3",
    ),
    "tokenizer.json": (
        1538643,
        "2fdbfdf74f08e8b98acc63e5a36c36d2ef44cf688f9485d95dde76acb6bd3034",
    ),
    "tokenizer_config.json": (
        46,
        "98fece87a533726e5d34601535095b6caae9515770e720d06bb2a62e8017641f",
    ),
}
EXPECTED_CONFIG = {
    "architectures": ["BertForPretraining"],
    "auto_map": {"AutoModel": "modeling_bert.BertForPreTraining"},
    "attention_probs_dropout_prob": 0.1,
    "gradient_checkpointing": False,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 768,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "layer_norm_eps": 1e-12,
    "last_layer_norm": False,
    "max_position_embeddings": 512,
    "model_type": "bert",
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "pad_token_id": 3,
    "position_embedding_type": "absolute",
    "pre_layer_norm": True,
    "transformers_version": "4.6.0.dev0",
    "type_vocab_size": 2,
    "use_cache": True,
    "vocab_size": 32000,
}


class OracleError(RuntimeError):
    """Raised when the pinned upstream oracle contract differs."""


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise OracleError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise OracleError(f"cannot load {label}: {error}") from error
    if not isinstance(value, dict):
        raise OracleError(f"{label} root must be an object")
    return value


def normalized_relative(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise OracleError(f"{label} must be a nonempty string")
    path = PurePosixPath(value)
    if path.is_absolute() or str(path) != value or any(
        part in ("", ".", "..") for part in path.parts
    ):
        raise OracleError(f"{label} must be a normalized relative path")
    return value


def validate_sources(
    receipt_path: Path, snapshot: Path
) -> dict[str, dict[str, Any]]:
    receipt = load_json(receipt_path, "source receipt")
    expected_fields = {
        "schema_version",
        "kind",
        "model_id",
        "repo",
        "requested_revision",
        "resolved_revision",
        "source_kind",
        "catalog_path",
        "catalog_contract_sha256",
        "load_path",
        "files",
    }
    if set(receipt) != expected_fields:
        raise OracleError("source receipt fields differ")
    identity = (
        receipt["schema_version"],
        receipt["kind"],
        receipt["model_id"],
        receipt["repo"],
        receipt["requested_revision"],
        receipt["resolved_revision"],
        receipt["source_kind"],
        receipt["catalog_contract_sha256"],
        receipt["load_path"],
    )
    expected_identity = (
        1,
        "source-checkpoint",
        RUNTIME_ID,
        SOURCE_REPO,
        REQUESTED_REVISION,
        SOURCE_REVISION,
        "huggingface",
        CATALOG_CONTRACT_SHA256,
        None,
    )
    if identity != expected_identity:
        raise OracleError(f"source receipt pinned provenance differs: {identity!r}")
    if snapshot.name != SOURCE_REVISION or not snapshot.is_dir():
        raise OracleError("snapshot path is not the pinned resolved revision")
    raw_files = receipt["files"]
    if not isinstance(raw_files, list) or len(raw_files) != len(
        EXPECTED_SOURCE_FILES
    ):
        raise OracleError("source receipt file count differs")

    verified: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(raw_files):
        label = f"source receipt files[{index}]"
        if not isinstance(item, dict) or set(item) != {
            "name",
            "path",
            "size",
            "sha256",
        }:
            raise OracleError(f"{label} fields differ")
        name = normalized_relative(item["name"], f"{label}.name")
        if name in verified or name not in EXPECTED_SOURCE_FILES:
            raise OracleError("source receipt file set is duplicated or unexpected")
        expected_size, expected_sha256 = EXPECTED_SOURCE_FILES[name]
        if item["size"] != expected_size or item["sha256"] != expected_sha256:
            raise OracleError(f"source receipt metadata differs for {name}")
        locator = Path(item["path"])
        logical = snapshot / name
        for candidate, candidate_label in (
            (locator, "receipt locator"),
            (logical, "snapshot file"),
        ):
            if not candidate.is_file():
                raise OracleError(f"{candidate_label} is missing for {name}")
            if (
                candidate.stat().st_size != expected_size
                or sha256_file(candidate) != expected_sha256
            ):
                raise OracleError(f"{candidate_label} differs for {name}")
        if logical.resolve(strict=True) != locator.resolve(strict=True):
            raise OracleError(f"snapshot file does not resolve to receipt blob: {name}")
        verified[name] = {"size": expected_size, "sha256": expected_sha256}
    if set(verified) != set(EXPECTED_SOURCE_FILES):
        raise OracleError("source receipt file set differs")
    snapshot_names = {
        path.relative_to(snapshot).as_posix()
        for path in snapshot.rglob("*")
        if path.is_file()
    }
    if snapshot_names != set(EXPECTED_SOURCE_FILES):
        raise OracleError("snapshot contains missing or extra regular files")
    return dict(sorted(verified.items()))


def git_output(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise OracleError(
            "git %s failed: %s" % (" ".join(arguments), result.stderr.strip())
        )
    return result.stdout.strip()


def validate_extractor(repository: Path, extractor: Path) -> dict[str, Any]:
    if git_output(repository, "rev-parse", "HEAD") != EXTRACTOR_REVISION:
        raise OracleError("GENEB extractor checkout revision differs")
    relative = extractor.resolve(strict=True).relative_to(repository.resolve(strict=True))
    committed = subprocess.run(
        ["git", "-C", str(repository), "show", f"HEAD:{relative.as_posix()}"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    payload = extractor.read_bytes()
    if (
        committed.returncode != 0
        or payload != committed.stdout
        or sha256_bytes(payload) != EXTRACTOR_SHA256
    ):
        raise OracleError("GENEB extractor differs from its pinned commit")
    pyproject = repository / "embedding_pipeline" / "pyproject.toml"
    if sha256_file(pyproject) != EXTRACTOR_PYPROJECT_SHA256:
        raise OracleError("GENEB extractor pyproject differs")
    required_fragments = (
        b"self.max_length = 512",
        b"outputs = self.model.bert(input_ids=input_ids,",
        b"last_hidden = outputs.last_hidden_state",
        b"sum_emb = (last_hidden * mask).sum(1)",
        b"emb = (sum_emb / lengths).cpu().numpy()",
    )
    if any(payload.count(fragment) != 1 for fragment in required_fragments):
        raise OracleError("GENEB extractor call/pooling contract differs")
    return {
        "revision": EXTRACTOR_REVISION,
        "source_sha256": EXTRACTOR_SHA256,
        "pyproject_sha256": EXTRACTOR_PYPROJECT_SHA256,
    }


def read_canonical_sequence(path: Path) -> tuple[str, str]:
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as error:
        raise OracleError(f"cannot read canonical FASTA: {error}") from error
    if len(lines) < 2 or not lines[0].startswith(">") or not lines[0][1:]:
        raise OracleError("canonical input must be one named FASTA record")
    if any(line.startswith(">") for line in lines[1:]):
        raise OracleError("canonical input must contain exactly one FASTA record")
    sequence = "".join(lines[1:])
    if sequence != CANONICAL_SEQUENCE:
        raise OracleError("canonical GENA-LM oracle sequence differs")
    return lines[0][1:], sequence


def package_lock() -> list[str]:
    packages = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name and distribution.version:
            packages.append(
                f"{name.lower().replace('_', '-')}=={distribution.version}"
            )
    return sorted(set(packages))


def write_atomic(path: Path, payload: bytes, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise OracleError(f"oracle output already exists: {path}")
    with tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{path.name}.", dir=path.parent, delete=False
    ) as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)


def generate(args: argparse.Namespace) -> dict[str, Any]:
    for name, value in {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "PYTHONHASHSEED": "0",
        "TZ": "UTC",
        "LANG": "C",
        "LC_ALL": "C",
    }.items():
        os.environ[name] = value

    snapshot = args.snapshot.resolve(strict=True)
    receipt = args.source_receipt.resolve(strict=True)
    extractor_repo = args.extractor_repo.resolve(strict=True)
    extractor_path = args.extractor.resolve(strict=True)
    input_path = args.input.resolve(strict=True)
    source_files = validate_sources(receipt, snapshot)
    extractor = validate_extractor(extractor_repo, extractor_path)
    _, sequence = read_canonical_sequence(input_path)
    config = load_json(snapshot / "config.json", "model config")
    if config != EXPECTED_CONFIG:
        raise OracleError("pinned GENA-LM model config differs")

    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer

    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")

    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    if (
        tokenizer.vocab_size != 32000
        or tokenizer.padding_side != "right"
        or tokenizer.cls_token_id != 1
        or tokenizer.sep_token_id != 2
        or tokenizer.pad_token_id != 3
        or tokenizer.mask_token_id != 4
    ):
        raise OracleError("official tokenizer vocabulary/special contract differs")
    encoded = tokenizer(
        [sequence],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    token_type_ids = encoded.get("token_type_ids")
    if (
        input_ids.tolist() != [CANONICAL_TOKEN_IDS]
        or attention_mask.tolist() != [[1] * len(CANONICAL_TOKEN_IDS)]
        or token_type_ids is None
        or token_type_ids.tolist() != [[0] * len(CANONICAL_TOKEN_IDS)]
    ):
        raise OracleError("official tokenizer IDs/mask/type IDs differ")

    model, loading = AutoModel.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        output_loading_info=True,
    )
    if any(
        loading.get(key)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise OracleError(f"official checkpoint loading info differs: {loading!r}")
    if model.__class__.__name__ != "BertForPreTraining" or not hasattr(model, "bert"):
        raise OracleError("AutoModel did not load the pinned pretraining wrapper/.bert")
    if (
        not bool(model.config.pre_layer_norm)
        or bool(model.config.last_layer_norm)
        or hasattr(model.bert.encoder, "last_layer_ln")
        or len(model.bert.encoder.layer) != 12
    ):
        raise OracleError("official pre-LN/no-final-norm topology differs")
    for index, layer in enumerate(model.bert.encoder.layer):
        if (
            not hasattr(layer, "pre_attention_ln")
            or not hasattr(layer, "post_attention_ln")
            or hasattr(layer.attention.output, "LayerNorm")
            or hasattr(layer.output, "LayerNorm")
        ):
            raise OracleError(f"official pre-LN layer topology differs at {index}")

    model.to(device="cpu", dtype=torch.float32).eval()
    with torch.inference_mode():
        # Exact pinned GENEB extractor path: call the `.bert` backbone only.
        outputs = model.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden.dtype)
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)
        pooled = pooled.to(device="cpu", dtype=torch.float32).contiguous()
    if tuple(last_hidden.shape) != (1, len(CANONICAL_TOKEN_IDS), 768):
        raise OracleError(f"official last hidden shape differs: {last_hidden.shape}")
    if tuple(pooled.shape) != (1, 768):
        raise OracleError(f"official pooled shape differs: {pooled.shape}")
    values = [float(value) for value in pooled[0].tolist()]
    if any(not math.isfinite(value) for value in values):
        raise OracleError("official oracle contains non-finite values")

    source_manifest = [
        {"name": name, "size": item["size"], "sha256": item["sha256"]}
        for name, item in source_files.items()
    ]
    environment = {
        "schema_version": 1,
        "oracle_contract": "geneb-independent-oracle-v1",
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "platform": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "packages": package_lock(),
        "torch": {
            "version": torch.__version__,
            "device": "cpu",
            "dtype": "float32",
            "num_threads": torch.get_num_threads(),
            "num_interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": (
                torch.are_deterministic_algorithms_enabled()
            ),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "mps_available_but_unused": bool(torch.backends.mps.is_available()),
        },
        "transformers": {
            "version": transformers.__version__,
            "tokenizers_version": importlib.metadata.version("tokenizers"),
            "local_files_only": True,
            "trust_remote_code": True,
            "model_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
            "backbone_class": (
                f"{model.bert.__class__.__module__}.{model.bert.__class__.__name__}"
            ),
            "tokenizer_class": (
                f"{tokenizer.__class__.__module__}.{tokenizer.__class__.__name__}"
            ),
        },
        "source_files": source_files,
        "model_semantics": {
            "hidden_size": 768,
            "layers": 12,
            "attention_heads": 12,
            "intermediate_size": 3072,
            "max_tokens": 512,
            "norm_placement": "pre",
            "final_layer_norm": False,
            "hidden_tap": "model.bert(...).last_hidden_state",
        },
    }
    provenance = {
        "oracle_contract": "geneb-independent-oracle-v1",
        "kind": "pinned-upstream-reference",
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "source_file_manifest_sha256": sha256_bytes(
            canonical_json(source_manifest)
        ),
        "checkpoint_sha256": EXPECTED_SOURCE_FILES["pytorch_model.bin"][1],
        "official_remote_modeling_code": True,
        "independent_of_evo_native_runtime": True,
        "generator_sha256": sha256_file(Path(__file__).resolve(strict=True)),
        "extractor_repo": EXTRACTOR_REPO,
        "extractor_revision": extractor["revision"],
        "extractor_module": "embedding_pipeline/extractors/genalm.py",
        "extractor_class": "GENALMExtractor",
        "extractor_sha256": extractor["source_sha256"],
        "extractor_pyproject_sha256": extractor["pyproject_sha256"],
        "model_entrypoint": "AutoModel(...).bert",
        "hidden_tap": "outputs.last_hidden_state",
        "pooling": "attention-mask-mean",
        "special_tokens": "include",
        "cls_token_id": 1,
        "sep_token_id": 2,
        "extractor_max_tokens": 512,
        "input_tokens": len(CANONICAL_TOKEN_IDS),
        "padding_tokens": 0,
        "norm_placement": "pre",
        "final_layer_norm": False,
    }
    return {
        "schema_version": 1,
        "kind": "geneb-independent-oracle-vector",
        "runtime_id": RUNTIME_ID,
        "input_sha256": sha256_file(input_path),
        "backend": "cpu",
        "profile": "cpu-f32",
        "values": values,
        "environment_lock": environment,
        "provenance": provenance,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--source-receipt", required=True, type=Path)
    parser.add_argument("--extractor-repo", required=True, type=Path)
    parser.add_argument("--extractor", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        oracle = generate(args)
        payload = canonical_json(oracle)
        write_atomic(args.output, payload, args.force)
        values = oracle["values"]
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "sha256": sha256_bytes(payload),
                    "values": len(values),
                    "minimum": min(values),
                    "maximum": max(values),
                    "l2_norm": math.sqrt(sum(value * value for value in values)),
                    "input_tokens": oracle["provenance"]["input_tokens"],
                },
                sort_keys=True,
            )
        )
    except (OracleError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
