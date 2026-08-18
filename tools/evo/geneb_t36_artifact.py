"""Closed checkpoint contracts shared by the T36 GENEB converters."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
import math
import os
import re
import struct
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

from evo.geneb_artifact import (
    GenebArtifactError,
    build_geneb_artifact_metadata,
    catalog_contract_sha256,
    converter_profile_contract_sha256,
    validate_hf_fetch_receipt_provenance,
)


MAX_HEADER_SIZE = 16 * 1024 * 1024
MAX_TENSOR_COUNT = 1000000
CHUNK_SIZE = 16 * 1024 * 1024
UINT64_MAX = (1 << 64) - 1
SHA256_RE = re.compile(r"[0-9a-f]{64}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
KEY_RE = re.compile(r"[A-Za-z0-9._-]+")
RUNTIME_NAME_RE = re.compile(r"[A-Za-z0-9._-]+")

PINNED_PROFILE_IDENTITIES = {
    "geneb-enformer": {
        "repo": "EleutherAI/enformer-official-rough",
        "requested_revision": "main",
        "revision": "affe5713ae9017460706a44108289b13c5fee16c",
        "config": (439, "ad7a0e07b2fe40fd4c93a8528416406b4429f303946f5a93209ef26f160a034f"),
        "weights": (1005149571, "99b09d602e195d89c7d4debe144bb2f43907ba0d74006e97098e99d9171c439c"),
        "canonical": (269, 919500800, "f4239e4472bf2d1d4b9e76b91cc46d59208acb421636f8407725d4ae4faad4b7"),
    },
    "geneb-space": {
        "repo": "yangyz1230/space",
        "requested_revision": "main",
        "revision": "63c9f5011877cc20b3e9d2d63dcdb1d547e62c18",
        "config": (700, "aa0c989abae4593aa22faa7b9f544237015d431dd33ebea8a6a80a4f60fc739c"),
        "weights": (2355245554, "f22ce262adc3347a8182bcd3bb5cd630ded4f1586b6c2a9841de6d14ea61cc81"),
        "canonical": (315, 2166370656, "27b5e9d233d07b4a11d9f36e8201cbcadcfff7a0a6ef319623dbe63b4dcd510b"),
    },
    "geneb-deepgene": {
        "source_url": "https://drive.google.com/drive/folders/1gb2IqO3NdSMbydKMLGZBFAbsbKhgm0i8",
        "config": (718, "09a9d102fa421711e4da2cf1018e042d98cce189ba3ee27c49cf0851f1f626f2"),
        "weights": (355268434, "4717e43cfe1036c5abc9aedd9ed158264aa1e8183e8cbe1375d4503c373e99d6"),
        "tokenizer": (167908, "5d178e8ce2ba55df97fff197f4b30f40133b95d7096be398c2df6b526c5d8cd3"),
        "canonical": (196, 352813056, "1808edcfd8f0bf05a8cbfaf73c4d9855e30f739a9518238085423cc6436e8368"),
    },
}

PINNED_IMPLEMENTATIONS = {
    "geneb-enformer": {
        "source": "enformer-pytorch v0.8.8@9ffeb8b62927d752b4983ef308a28bf70b34b160",
        "modeling_sha256": "7d3d8860560e353983e85c325ead0ba7aab4281957251828392c49dc7426c008",
        "data_sha256": "ec84cabc8476c7409726f5a0d0fc3ae4b88716f9f1d724cc8b87c7a2c3ca8a9a",
        "configuration_sha256": "715f799e2a55f0b7f991b434197a17718694a4e85e2bc7c47c5486114db2193a",
    },
    "geneb-space": {
        "source": "ZhuJiwei111/SPACE@4cdba18b80f948410623acee4b27a988ae7ddace",
        "modeling_sha256": "b2f8111f34fd85c800446d5148762bfe9fb32fde40d13098bdb89dd333a2408a",
        "modules_sha256": "56e5c00dcf4b17671cbe3184c797571f42c93e311c7e630b0028c1d2f017b0b7",
        "configuration_sha256": "493d517e70e172345a97643482d505b13194dd8068c0ecfb48f515d02cb538c9",
    },
    "geneb-deepgene": {
        "source": "wds-seu/DeepGene@486343e5212361d6cd7ed03c624f430ed3d5f02e",
        "modeling_sha256": "1849ee8eb6931d5479083eaf01ec7c2bff124555423c30c337e0bfe889fab3e4",
        "configuration_sha256": "28672584950f4bf08975024cdf529ab9cc3aa4497ef4f9a1b3863b2e971cc68e",
        "tokenizer_sha256": "5d178e8ce2ba55df97fff197f4b30f40133b95d7096be398c2df6b526c5d8cd3",
    },
}


class ConversionError(ValueError):
    """Raised when source data differs from a frozen T36 contract."""


def duplicate_checked_object(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result = {}  # type: Dict[str, Any]
    for key, value in pairs:
        if key in result:
            raise ConversionError("JSON object contains duplicate key %r" % key)
        result[key] = value
    return result


def load_json(path: Path, label: str) -> Tuple[Dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=duplicate_checked_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ConversionError("cannot read %s: %s" % (label, error))
    if not isinstance(value, dict):
        raise ConversionError("%s root must be an object" % label)
    return value, payload


def exact_keys(value: Mapping[str, Any], required: Iterable[str], optional: Iterable[str], label: str) -> None:
    required_set = set(required)
    allowed = required_set | set(optional)
    actual = set(value)
    if not required_set <= actual or not actual <= allowed:
        raise ConversionError(
            "%s fields differ: missing=%s extra=%s"
            % (label, sorted(required_set - actual), sorted(actual - allowed))
        )


def object_value(value: Any, label: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ConversionError("%s must be an object" % label)
    return value


def string_value(value: Any, label: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value) or "\x00" in value:
        raise ConversionError("%s must be a %sstring" % (label, "nonempty " if not allow_empty else ""))
    return value


def uint_value(value: Any, label: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < (1 if positive else 0) or value > UINT64_MAX:
        raise ConversionError("%s must be a%s uint64" % (label, " positive" if positive else ""))
    return value


def bool_value(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ConversionError("%s must be a boolean" % label)
    return value


def float_value(value: Any, label: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConversionError("%s must be a finite number" % label)
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ConversionError("%s must be a%s finite number" % (label, " positive" if positive else ""))
    return result


def normalized_relative_path(value: Any, label: str) -> str:
    text = string_value(value, label)
    path = PurePosixPath(text)
    if path.is_absolute() or str(path) != text or any(part in ("", ".", "..") for part in path.parts):
        raise ConversionError("%s must be a normalized relative path" % label)
    return text


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while True:
                chunk = source.read(CHUNK_SIZE)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as error:
        raise ConversionError("cannot hash %s: %s" % (path, error))
    return digest.hexdigest()


@dataclasses.dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: str
    shape: Tuple[int, ...]

    @property
    def nbytes(self) -> int:
        elements = 1
        for dimension in self.shape:
            if dimension <= 0 or elements > UINT64_MAX // dimension:
                raise ConversionError("tensor shape is invalid: %s" % self.name)
            elements *= dimension
        size = {"F32": 4, "I64": 8}.get(self.dtype)
        if size is None or elements > UINT64_MAX // size:
            raise ConversionError("tensor dtype/extent is invalid: %s" % self.name)
        return elements * size


@dataclasses.dataclass(frozen=True)
class TorchTensorSource:
    name: str
    dtype: str
    shape: Tuple[int, ...]
    nbytes: int
    tensor: Any

    def iter_chunks(self, chunk_size: int) -> Iterator[memoryview]:
        torch = importlib.import_module("torch")
        raw = self.tensor.detach().view(torch.uint8).reshape(-1).numpy()
        view = memoryview(raw).cast("B")
        if len(view) != self.nbytes:
            raise ConversionError("tensor %s exposed a wrong byte extent" % self.name)
        for offset in range(0, self.nbytes, chunk_size):
            yield view[offset : offset + min(chunk_size, self.nbytes - offset)]


@dataclasses.dataclass(frozen=True)
class RenamedTensorSource:
    name: str
    dtype: str
    shape: Tuple[int, ...]
    nbytes: int
    source: TorchTensorSource

    def iter_chunks(self, chunk_size: int) -> Iterator[memoryview]:
        return self.source.iter_chunks(chunk_size)


def spec(name: str, shape: Sequence[int], dtype: str = "F32") -> TensorSpec:
    return TensorSpec(name, dtype, tuple(shape))


def _common_cnn_specs(prefix: str = "") -> List[TensorSpec]:
    specs = [
        spec(prefix + "stem.0.weight", (768, 4, 15)),
        spec(prefix + "stem.0.bias", (768,)),
        spec(prefix + "stem.1.fn.0.weight", (768,)),
        spec(prefix + "stem.1.fn.0.bias", (768,)),
        spec(prefix + "stem.1.fn.0.running_mean", (768,)),
        spec(prefix + "stem.1.fn.0.running_var", (768,)),
        spec(prefix + "stem.1.fn.0.num_batches_tracked", (), "I64"),
        spec(prefix + "stem.1.fn.2.weight", (768, 768, 1)),
        spec(prefix + "stem.1.fn.2.bias", (768,)),
        spec(prefix + "stem.2.to_attn_logits.weight", (768, 768, 1, 1)),
    ]
    widths = (768, 896, 1024, 1152, 1280, 1536)
    previous = 768
    for index, width in enumerate(widths):
        base = prefix + "conv_tower.%d." % index
        specs.extend(
            [
                spec(base + "0.0.weight", (previous,)),
                spec(base + "0.0.bias", (previous,)),
                spec(base + "0.0.running_mean", (previous,)),
                spec(base + "0.0.running_var", (previous,)),
                spec(base + "0.0.num_batches_tracked", (), "I64"),
                spec(base + "0.2.weight", (width, previous, 5)),
                spec(base + "0.2.bias", (width,)),
                spec(base + "1.fn.0.weight", (width,)),
                spec(base + "1.fn.0.bias", (width,)),
                spec(base + "1.fn.0.running_mean", (width,)),
                spec(base + "1.fn.0.running_var", (width,)),
                spec(base + "1.fn.0.num_batches_tracked", (), "I64"),
                spec(base + "1.fn.2.weight", (width, width, 1)),
                spec(base + "1.fn.2.bias", (width,)),
                spec(base + "2.to_attn_logits.weight", (width, width, 1, 1)),
            ]
        )
        previous = width
    return specs


def _final_cnn_specs(prefix: str = "") -> List[TensorSpec]:
    return [
        spec(prefix + "final_pointwise.1.0.weight", (1536,)),
        spec(prefix + "final_pointwise.1.0.bias", (1536,)),
        spec(prefix + "final_pointwise.1.0.running_mean", (1536,)),
        spec(prefix + "final_pointwise.1.0.running_var", (1536,)),
        spec(prefix + "final_pointwise.1.0.num_batches_tracked", (), "I64"),
        spec(prefix + "final_pointwise.1.2.weight", (3072, 1536, 1)),
        spec(prefix + "final_pointwise.1.2.bias", (3072,)),
    ]


def enformer_primary_specs() -> List[TensorSpec]:
    result = _common_cnn_specs()
    for layer in range(11):
        prefix = "transformer.%d." % layer
        result.extend(
            [
                spec(prefix + "0.fn.0.weight", (1536,)),
                spec(prefix + "0.fn.0.bias", (1536,)),
                spec(prefix + "0.fn.1.rel_content_bias", (1, 8, 1, 64)),
                spec(prefix + "0.fn.1.rel_pos_bias", (1, 8, 1, 64)),
                spec(prefix + "0.fn.1.to_q.weight", (512, 1536)),
                spec(prefix + "0.fn.1.to_k.weight", (512, 1536)),
                spec(prefix + "0.fn.1.to_v.weight", (1536, 1536)),
                spec(prefix + "0.fn.1.to_out.weight", (1536, 1536)),
                spec(prefix + "0.fn.1.to_out.bias", (1536,)),
                spec(prefix + "0.fn.1.to_rel_k.weight", (512, 192)),
                spec(prefix + "1.fn.0.weight", (1536,)),
                spec(prefix + "1.fn.0.bias", (1536,)),
                spec(prefix + "1.fn.1.weight", (3072, 1536)),
                spec(prefix + "1.fn.1.bias", (3072,)),
                spec(prefix + "1.fn.4.weight", (1536, 3072)),
                spec(prefix + "1.fn.4.bias", (1536,)),
            ]
        )
    result.extend(_final_cnn_specs())
    return result


def _enformer_alias_name(name: str) -> str:
    mappings = (("stem.", "_trunk.1."), ("conv_tower.", "_trunk.2."), ("transformer.", "_trunk.4."), ("final_pointwise.", "_trunk.6."))
    for source, target in mappings:
        if name.startswith(source):
            return target + name[len(source) :]
    raise ConversionError("internal Enformer alias mapping is incomplete")


def enformer_source_specs() -> Tuple[List[TensorSpec], Dict[str, str]]:
    primary = enformer_primary_specs()
    aliases = {_enformer_alias_name(item.name): item.name for item in primary}
    result = list(primary)
    result.extend(spec(alias, next(item.shape for item in primary if item.name == canonical), next(item.dtype for item in primary if item.name == canonical)) for alias, canonical in aliases.items())
    result.extend(
        [
            spec("_heads.human.0.weight", (5313, 3072)),
            spec("_heads.human.0.bias", (5313,)),
            spec("_heads.mouse.0.weight", (1643, 3072)),
            spec("_heads.mouse.0.bias", (1643,)),
        ]
    )
    return result, aliases


def space_primary_specs() -> List[TensorSpec]:
    result = _common_cnn_specs("model.")
    result.extend(
        [
            spec("model.transformer.species_embedding.human", (1, 1, 1536)),
            spec("model.transformer.species_embedding.mouse", (1, 1, 1536)),
        ]
    )
    for layer in range(11):
        prefix = "model.transformer.transformer.%d." % layer
        result.extend(
            [
                spec(prefix + "attention.fn.0.weight", (1536,)),
                spec(prefix + "attention.fn.0.bias", (1536,)),
                spec(prefix + "attention.fn.1.rel_content_bias", (1, 8, 1, 64)),
                spec(prefix + "attention.fn.1.rel_pos_bias", (1, 8, 1, 64)),
                spec(prefix + "attention.fn.1.to_q.weight", (512, 1536)),
                spec(prefix + "attention.fn.1.to_k.weight", (512, 1536)),
                spec(prefix + "attention.fn.1.to_v.weight", (1536, 1536)),
                spec(prefix + "attention.fn.1.to_out.weight", (1536, 1536)),
                spec(prefix + "attention.fn.1.to_out.bias", (1536,)),
                spec(prefix + "attention.fn.1.to_rel_k.weight", (512, 192)),
                spec(prefix + "feed_forward.gates.human.0.weight", (4, 1536)),
                spec(prefix + "feed_forward.gates.human.0.bias", (4,)),
                spec(prefix + "feed_forward.gates.mouse.0.weight", (4, 1536)),
                spec(prefix + "feed_forward.gates.mouse.0.bias", (4,)),
                spec(prefix + "feed_forward.layer_norm.weight", (1536,)),
                spec(prefix + "feed_forward.layer_norm.bias", (1536,)),
                spec(prefix + "feed_forward.input.weight", (4, 1536, 3072)),
                spec(prefix + "feed_forward.input.bias", (4, 3072)),
                spec(prefix + "feed_forward.output.weight", (4, 3072, 1536)),
                spec(prefix + "feed_forward.output.bias", (4, 1536)),
            ]
        )
    result.extend(_final_cnn_specs("model."))
    return result


def space_drop_specs() -> List[TensorSpec]:
    result = [
        spec("model.heads.human.0.weight", (5313, 3072)),
        spec("model.heads.human.0.bias", (5313,)),
        spec("model.heads.mouse.0.weight", (1643, 3072)),
        spec("model.heads.mouse.0.bias", (1643,)),
        spec("model.tracks.gate_selector.weight", (8, 3072)),
        spec("model.tracks.gate_selector.bias", (8,)),
        spec("model.tracks.embedding_gate_selector.weight", (8, 1536)),
        spec("model.tracks.embedding_gate_selector.bias", (8,)),
    ]
    for index in range(8):
        result.extend(
            [
                spec("model.tracks.gates.%d.0.weight" % index, (8, 896)),
                spec("model.tracks.gates.%d.0.bias" % index, (8,)),
            ]
        )
    result.extend(
        [
            spec("model.tracks.layer_norm.weight", (896,)),
            spec("model.tracks.layer_norm.bias", (896,)),
            spec("model.tracks.input_proj.weight", (8, 896, 1792)),
            spec("model.tracks.input_proj.bias", (8, 1792)),
            spec("model.tracks.output_proj.weight", (8, 1792, 896)),
            spec("model.tracks.output_proj.bias", (8, 896)),
            spec("model.tracks.tracks_embedding.CAGE", (1, 1, 896)),
            spec("model.tracks.tracks_embedding.DNASE/ATAC", (1, 1, 896)),
            spec("model.tracks.tracks_embedding.Histone ChIP-seq", (1, 1, 896)),
            spec("model.tracks.tracks_embedding.TF ChIP-seq", (1, 1, 896)),
        ]
    )
    return result


def roformer_runtime_specs() -> List[TensorSpec]:
    result = [
        spec("roformer.embeddings.word_embeddings.weight", (4096, 768)),
        spec("roformer.embeddings.token_type_embeddings.weight", (2, 768)),
        spec("roformer.embeddings.LayerNorm.weight", (768,)),
        spec("roformer.embeddings.LayerNorm.bias", (768,)),
    ]
    for layer in range(12):
        prefix = "roformer.encoder.layer.%d." % layer
        for projection in ("query", "key", "value"):
            result.append(spec(prefix + "attention.self.%s.weight" % projection, (768, 768)))
            result.append(spec(prefix + "attention.self.%s.bias" % projection, (768,)))
        result.extend(
            [
                spec(prefix + "attention.output.dense.weight", (768, 768)),
                spec(prefix + "attention.output.dense.bias", (768,)),
                spec(prefix + "attention.output.LayerNorm.weight", (768,)),
                spec(prefix + "attention.output.LayerNorm.bias", (768,)),
                spec(prefix + "intermediate.dense.weight", (3072, 768)),
                spec(prefix + "intermediate.dense.bias", (3072,)),
                spec(prefix + "output.dense.weight", (768, 3072)),
                spec(prefix + "output.dense.bias", (768,)),
                spec(prefix + "output.LayerNorm.weight", (768,)),
                spec(prefix + "output.LayerNorm.bias", (768,)),
            ]
        )
    return result


def roformer_source_specs() -> Tuple[List[TensorSpec], Dict[str, str]]:
    runtime = roformer_runtime_specs()
    heads = [
        spec("cls.predictions.bias", (4096,)),
        spec("cls.predictions.transform.dense.weight", (768, 768)),
        spec("cls.predictions.transform.dense.bias", (768,)),
        spec("cls.predictions.transform.LayerNorm.weight", (768,)),
        spec("cls.predictions.transform.LayerNorm.bias", (768,)),
        spec("cls.predictions.decoder.weight", (4096, 768)),
        spec("cls.predictions.decoder.bias", (4096,)),
    ]
    aliases = {
        "cls.predictions.decoder.weight": "roformer.embeddings.word_embeddings.weight",
        "cls.predictions.decoder.bias": "cls.predictions.bias",
    }
    return runtime + heads, aliases


def load_torch_state(path: Path) -> List[TorchTensorSource]:
    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError as error:
        raise ConversionError("conversion requires offline PyTorch with weights_only+mmap") from error
    try:
        # PyTorch 2.1 requires a concrete string filename when mmap=True.
        state = torch.load(
            str(path), map_location="cpu", mmap=True, weights_only=True
        )
    except TypeError as error:
        raise ConversionError("installed PyTorch lacks weights_only=True,mmap=True") from error
    except Exception as error:
        raise ConversionError("cannot safely load checkpoint: %s" % error) from error
    if not isinstance(state, dict) or not state or len(state) > MAX_TENSOR_COUNT:
        raise ConversionError("checkpoint root/tensor count is invalid")
    result = []  # type: List[TorchTensorSource]
    for raw_name, tensor in state.items():
        if not isinstance(raw_name, str) or not raw_name or "\x00" in raw_name or len(raw_name.encode("utf-8")) >= 256:
            raise ConversionError("checkpoint contains an invalid tensor name")
        if not isinstance(tensor, torch.Tensor):
            raise ConversionError("checkpoint entry %s is not a tensor" % raw_name)
        dtype = "F32" if tensor.dtype == torch.float32 else "I64" if tensor.dtype == torch.int64 else ""
        if not dtype:
            raise ConversionError("tensor %s has unsupported dtype %s" % (raw_name, tensor.dtype))
        if tensor.device.type != "cpu" or tensor.layout != torch.strided or not tensor.is_contiguous():
            raise ConversionError("tensor %s is not dense contiguous CPU storage" % raw_name)
        shape = tuple(int(value) for value in tensor.shape)
        if len(shape) > 8 or any(value <= 0 for value in shape):
            raise ConversionError("tensor %s shape is invalid" % raw_name)
        expected_bytes = TensorSpec(raw_name, dtype, shape).nbytes
        if tensor.numel() * tensor.element_size() != expected_bytes:
            raise ConversionError("tensor %s byte extent differs" % raw_name)
        result.append(TorchTensorSource(raw_name, dtype, shape, expected_bytes, tensor))
    return result


def validate_manifest(tensors: Sequence[TorchTensorSource], expected: Sequence[TensorSpec]) -> Dict[str, TorchTensorSource]:
    provided = {}  # type: Dict[str, TorchTensorSource]
    for tensor in tensors:
        if tensor.name in provided:
            raise ConversionError("checkpoint tensor is duplicated: %s" % tensor.name)
        provided[tensor.name] = tensor
    wanted = {item.name: item for item in expected}
    if len(wanted) != len(expected):
        raise ConversionError("internal expected tensor manifest is duplicated")
    if set(provided) != set(wanted):
        raise ConversionError(
            "checkpoint tensor names differ: missing=%s extra=%s"
            % (sorted(set(wanted) - set(provided)), sorted(set(provided) - set(wanted)))
        )
    for name, requirement in wanted.items():
        actual = provided[name]
        if actual.dtype != requirement.dtype or actual.shape != requirement.shape or actual.nbytes != requirement.nbytes:
            raise ConversionError("checkpoint tensor dtype/shape differs: %s" % name)
    return provided


def validate_storage_aliases(tensors: Mapping[str, TorchTensorSource], aliases: Mapping[str, str]) -> None:
    for alias, canonical in aliases.items():
        left = tensors[alias].tensor
        right = tensors[canonical].tensor
        same = (
            left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
            and left.storage_offset() == right.storage_offset()
            and left.stride() == right.stride()
            and tuple(left.shape) == tuple(right.shape)
        )
        if not same:
            raise ConversionError("checkpoint storage alias differs: %s -> %s" % (alias, canonical))


def validated_sequence_tensors(path: Path, variant: str, contract: Mapping[str, Any]) -> List[RenamedTensorSource]:
    tensors = load_torch_state(path)
    if variant == "enformer":
        expected, aliases = enformer_source_specs()
        by_name = validate_manifest(tensors, expected)
        validate_storage_aliases(by_name, aliases)
        canonical_specs = [item for item in enformer_primary_specs() if item.dtype == "F32"]
        runtime = [RenamedTensorSource(item.name, "F32", item.shape, item.nbytes, by_name[item.name]) for item in canonical_specs]
    elif variant == "space":
        primary = space_primary_specs()
        expected = primary + space_drop_specs()
        by_name = validate_manifest(tensors, expected)
        canonical_specs = [item for item in primary if item.dtype == "F32"]
        runtime = [
            RenamedTensorSource(item.name[len("model.") :], "F32", item.shape, item.nbytes, by_name[item.name])
            for item in canonical_specs
        ]
    else:
        raise ConversionError("unsupported sequence-CNN variant")
    _validate_canonical_contract(runtime, contract)
    return runtime


def validated_roformer_tensors(path: Path, contract: Mapping[str, Any]) -> List[RenamedTensorSource]:
    expected, aliases = roformer_source_specs()
    by_name = validate_manifest(load_torch_state(path), expected)
    validate_storage_aliases(by_name, aliases)
    runtime = [RenamedTensorSource(item.name, item.dtype, item.shape, item.nbytes, by_name[item.name]) for item in roformer_runtime_specs()]
    _validate_canonical_contract(runtime, contract)
    return runtime


def _validate_canonical_contract(tensors: Sequence[RenamedTensorSource], contract: Mapping[str, Any]) -> None:
    if (
        len(tensors) != contract.get("canonical_f32_count")
        or any(item.dtype != "F32" for item in tensors)
        or sum(item.nbytes for item in tensors) != contract.get("canonical_bytes")
    ):
        raise ConversionError("canonical tensor count/dtype/byte contract differs")
    manifest = [
        {"name": item.name, "dtype": item.dtype, "shape": list(item.shape), "nbytes": item.nbytes}
        for item in tensors
    ]
    if len({item.name for item in tensors}) != len(tensors) or any(not RUNTIME_NAME_RE.fullmatch(item.name) for item in tensors):
        raise ConversionError("runtime tensor namespace is not closed")
    # This digest is converter-generated in addition to the independently
    # audited checkpoint digest pinned in the profile.
    json.dumps(manifest, sort_keys=True, separators=(",", ":"))


def _validate_profile_contract(profile: Mapping[str, Any], expected_format: str) -> None:
    runtime_id = string_value(profile.get("runtime_id"), "profile.runtime_id")
    pin = PINNED_PROFILE_IDENTITIES.get(runtime_id)
    if pin is None:
        raise ConversionError("profile runtime_id is not a pinned T36 model")
    if expected_format == "geneb-sequence-cnn-converter-v1":
        exact_keys(
            profile,
            [
                "runtime_id", "geneb_model_id", "paper_name",
                "catalog_architecture", "repo", "requested_revision",
                "revision", "source_files", "config_required",
                "source_contract", "topology",
            ],
            [],
            "sequence-CNN profile",
        )
        if runtime_id not in ("geneb-enformer", "geneb-space"):
            raise ConversionError("sequence-CNN profile identity differs")
        if any(profile.get(key) != pin[key] for key in ("repo", "requested_revision", "revision")):
            raise ConversionError("sequence-CNN profile source pin differs")
        variant = "enformer" if runtime_id == "geneb-enformer" else "space"
        expected_identity = (
            ("enformer-official-rough", "Enformer", "enformer")
            if variant == "enformer"
            else ("space", "SPACE", "space-multispecies-enformer")
        )
        if (
            profile.get("geneb_model_id"), profile.get("paper_name"),
            profile.get("catalog_architecture"),
        ) != expected_identity:
            raise ConversionError("sequence-CNN catalog identity differs")
        expected_topology = {
            "variant": variant,
            "input_length": 196608 if variant == "enformer" else 131072,
            "trunk_width": 1536,
            "output_width": 3072,
            "depth": 11,
            "heads": 8,
            "key_width": 64,
            "value_width": 192,
            "relative_feature_width": 192,
            "target_length": 896,
            "num_downsamples": 7,
            "batch_norm_epsilon": 0.00001,
            "gelu_sigmoid_scale": 1.702,
            "use_tf_gamma": False,
            "species_num_experts": 0 if variant == "enformer" else 4,
            "top_k": 0 if variant == "enformer" else 3,
            "gate_negative_slope": 0.0 if variant == "enformer" else 0.01,
            "species": "" if variant == "enformer" else "human",
            "weight_dtype": "F32",
        }
        if profile.get("topology") != expected_topology:
            raise ConversionError("sequence-CNN profile topology differs")
        expected_specs, aliases = (
            enformer_source_specs()
            if variant == "enformer"
            else (space_primary_specs() + space_drop_specs(), {})
        )
    elif expected_format == "geneb-roformer-converter-v1":
        exact_keys(
            profile,
            [
                "runtime_id", "geneb_model_id", "paper_name",
                "catalog_architecture", "source_url", "source_files",
                "tokenizer", "config_required", "source_contract",
                "topology",
            ],
            [],
            "RoFormer profile",
        )
        if runtime_id != "geneb-deepgene" or profile.get("source_url") != pin["source_url"]:
            raise ConversionError("RoFormer profile source identity differs")
        if (
            profile.get("geneb_model_id"), profile.get("paper_name"),
            profile.get("catalog_architecture"),
        ) != ("deepgene", "DeepGene", "deepgene-roformer-only"):
            raise ConversionError("RoFormer catalog identity differs")
        expected_topology = {
            "vocab_size": 4096,
            "tokenizer_vocab_size": 4096,
            "hidden_size": 768,
            "num_layers": 12,
            "num_attention_heads": 12,
            "head_dim": 64,
            "inner_size": 3072,
            "max_seqlen": 5120,
            "type_vocab_size": 2,
            "pad_token_id": 3,
            "cls_token_id": 1,
            "sep_token_id": 2,
            "layer_norm_epsilon": 1e-12,
            "rope_base": 10000.0,
            "rotary_value": False,
            "pooling": "attention-mask-mean",
            "manual_special_order": "cls-sep-payload",
            "pad_position_policy": "final-batch-position",
            "weight_dtype": "F32",
        }
        if profile.get("topology") != expected_topology:
            raise ConversionError("RoFormer profile topology differs")
        expected_tokenizer = {
            "manifest": "configs/tokenizers/geneb-deepgene-bpe-v1.json",
            "compiler_manifest_sha256": "5cc5a66863990a6bf7d50e69f6f066c698f6f7f7128ebb723ec22a533bd0be4f",
            "compiled_asset_sha256": "37cba9152a3a75dc7c655e033dce029376a64d519e6140a585caee06aae06fc6",
            "compiled_asset_size": 182157,
            "kind": "bpe",
            "emitted_vocab_size": 4096,
            "add_special_tokens": False,
            "manual_special_tokens": True,
            "padding_side": "right",
        }
        if profile.get("tokenizer") != expected_tokenizer:
            raise ConversionError("RoFormer tokenizer profile differs")
        expected_specs, aliases = roformer_source_specs()
    else:
        raise ConversionError("unsupported T36 profile format")

    files = {item["name"]: item for item in profile["source_files"]}
    for name, pin_key in (("config.json", "config"), ("pytorch_model.bin", "weights")):
        item = files.get(name)
        if item is None or (item.get("size"), item.get("sha256")) != pin[pin_key]:
            raise ConversionError("profile critical source pin differs: %s" % name)
    if runtime_id == "geneb-deepgene":
        item = files.get("tokenizer.json")
        if item is None or (item.get("size"), item.get("sha256")) != pin["tokenizer"]:
            raise ConversionError("profile tokenizer source pin differs")
    contract = object_value(profile.get("source_contract"), "profile.source_contract")
    required_contract = [
        "tensor_count", "f32_count", "canonical_f32_count",
        "canonical_bytes", "canonical_digest", "alias_pairs", "drop_policy",
    ]
    if expected_format == "geneb-sequence-cnn-converter-v1":
        required_contract.append("i64_count")
    exact_keys(contract, required_contract, [], "profile.source_contract")
    f32_count = sum(item.dtype == "F32" for item in expected_specs)
    i64_count = sum(item.dtype == "I64" for item in expected_specs)
    canonical_count, canonical_bytes, canonical_digest = pin["canonical"]
    if (
        contract.get("tensor_count") != len(expected_specs)
        or contract.get("f32_count") != f32_count
        or contract.get("i64_count", 0) != i64_count
        or contract.get("canonical_f32_count") != canonical_count
        or contract.get("canonical_bytes") != canonical_bytes
        or contract.get("canonical_digest") != canonical_digest
        or contract.get("alias_pairs") != len(aliases)
        or not isinstance(contract.get("drop_policy"), str)
        or not contract["drop_policy"]
    ):
        raise ConversionError("profile source tensor contract differs")


def load_profiles(path: Path, expected_format: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any], bytes]:
    root, payload = load_json(path, "T36 converter profiles")
    implementation_field = (
        "implementation_contracts"
        if expected_format == "geneb-sequence-cnn-converter-v1"
        else "implementation_contract"
        if expected_format == "geneb-roformer-converter-v1"
        else ""
    )
    if not implementation_field:
        raise ConversionError("unsupported T36 profile format")
    exact_keys(
        root,
        ["schema_version", "format", "models", implementation_field],
        [],
        "profile manifest",
    )
    if root["schema_version"] != 1 or root["format"] != expected_format:
        raise ConversionError("profile schema/format differs")
    if not isinstance(root["models"], list) or not root["models"]:
        raise ConversionError("profile models must be nonempty")
    result = {}  # type: Dict[str, Dict[str, Any]]
    for index, raw in enumerate(root["models"]):
        profile = object_value(raw, "models[%d]" % index)
        runtime_id = string_value(profile.get("runtime_id"), "profile.runtime_id")
        if runtime_id in result:
            raise ConversionError("profile runtime_id is duplicated")
        source_files = profile.get("source_files")
        if not isinstance(source_files, list) or not source_files:
            raise ConversionError("profile source_files must be nonempty")
        names = set()  # type: Set[str]
        for file_index, raw_file in enumerate(source_files):
            item = object_value(raw_file, "source_files[%d]" % file_index)
            exact_keys(item, ["name", "size", "sha256"], [], "source file")
            name = normalized_relative_path(item["name"], "source file name")
            if name in names:
                raise ConversionError("source file name is duplicated")
            names.add(name)
            uint_value(item["size"], "source file size", True)
            if not isinstance(item["sha256"], str) or not SHA256_RE.fullmatch(item["sha256"]):
                raise ConversionError("source file sha256 is invalid")
        _validate_profile_contract(profile, expected_format)
        result[runtime_id] = profile
    if expected_format == "geneb-sequence-cnn-converter-v1":
        implementations = object_value(
            root["implementation_contracts"], "implementation_contracts"
        )
        if set(implementations) != {"enformer", "space"}:
            raise ConversionError("sequence-CNN implementation identities differ")
        if (
            implementations["enformer"] != PINNED_IMPLEMENTATIONS["geneb-enformer"]
            or implementations["space"] != PINNED_IMPLEMENTATIONS["geneb-space"]
        ):
            raise ConversionError("sequence-CNN implementation pin differs")
    elif root["implementation_contract"] != PINNED_IMPLEMENTATIONS["geneb-deepgene"]:
        raise ConversionError("RoFormer implementation pin differs")
    return result, root, payload


def select_profile(receipt_path: Path, profiles: Mapping[str, Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any], bytes]:
    receipt, payload = load_json(receipt_path, "source receipt identity")
    model_id = receipt.get("model_id")
    if not isinstance(model_id, str) or model_id not in profiles:
        raise ConversionError("source receipt does not identify a converter profile")
    return profiles[model_id], receipt, payload


def validate_receipt(
    receipt_path: Path,
    profile: Mapping[str, Any],
    family: str,
    catalog_path: Optional[Path] = None,
    catalog_payload: Optional[bytes] = None,
    catalog_entry: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, Path], bytes]:
    receipt, payload = load_json(receipt_path, "source receipt")
    exact_keys(
        receipt,
        ["schema_version", "kind", "model_id", "files"],
        ["repo", "requested_revision", "resolved_revision", "source_kind", "source_url", "load_path", "catalog_path", "catalog_sha256", "catalog_contract_sha256"],
        "source receipt",
    )
    if receipt["schema_version"] != 1 or receipt["kind"] != "source-checkpoint" or receipt["model_id"] != profile["runtime_id"]:
        raise ConversionError("source receipt identity differs")
    if family == "sequence-cnn":
        if (
            receipt.get("repo") != profile["repo"]
            or receipt.get("requested_revision") != profile["requested_revision"]
            or receipt.get("resolved_revision") != profile["revision"]
            or receipt.get("source_kind", "huggingface") != "huggingface"
            or ("load_path" in receipt and receipt["load_path"] is not None)
        ):
            raise ConversionError("source receipt immutable HF identity differs")
        has_catalog_provenance = any(
            key in receipt
            for key in ("catalog_path", "catalog_sha256", "catalog_contract_sha256")
        )
        if has_catalog_provenance:
            if catalog_path is None or catalog_payload is None or catalog_entry is None:
                raise ConversionError(
                    "source receipt catalog provenance cannot be verified"
                )
            try:
                validate_hf_fetch_receipt_provenance(
                    receipt, catalog_path, catalog_payload, catalog_entry
                )
            except GenebArtifactError as error:
                raise ConversionError(str(error))
    else:
        if receipt.get("source_kind", "google-drive") != "google-drive" or receipt.get("source_url", profile["source_url"]) != profile["source_url"]:
            raise ConversionError("source receipt DeepGene identity differs")
    files = receipt["files"]
    if not isinstance(files, list) or not files:
        raise ConversionError("source receipt files must be nonempty")
    verified = {}  # type: Dict[str, Tuple[Path, int, str]]
    resolved_paths = set()  # type: Set[Path]
    for index, raw in enumerate(files):
        item = object_value(raw, "receipt.files[%d]" % index)
        exact_keys(item, ["name", "size", "sha256", "path"], [], "receipt file")
        name = normalized_relative_path(item["name"], "receipt file name")
        size = uint_value(item["size"], "receipt file size")
        digest = string_value(item["sha256"], "receipt file sha256")
        if not SHA256_RE.fullmatch(digest):
            raise ConversionError("receipt file sha256 is invalid")
        source_path = Path(string_value(item["path"], "receipt file path"))
        if source_path.is_symlink():
            raise ConversionError("receipt file must not be a symlink: %s" % name)
        path = source_path.resolve()
        if name in verified or path in resolved_paths:
            raise ConversionError("receipt file name/path is duplicated")
        try:
            actual_size = path.stat().st_size
        except OSError as error:
            raise ConversionError("cannot stat receipt file %s: %s" % (name, error))
        actual_digest = sha256_file(path)
        if (actual_size, actual_digest) != (size, digest):
            raise ConversionError("receipt integrity mismatch for %s" % name)
        verified[name] = (path, size, digest)
        resolved_paths.add(path)
    # V65/B64: every receipt file above is integrity-checked first.  Only now
    # select the exact family-owned critical subset; unrelated same-revision
    # snapshot files do not influence tensor selection.
    critical = {"config.json", "pytorch_model.bin"}
    if not critical <= set(verified):
        raise ConversionError("source receipt is missing critical files: %s" % sorted(critical - set(verified)))
    pinned = {item["name"]: item for item in profile["source_files"]}
    for name in critical:
        expected = pinned.get(name)
        if expected is None or (verified[name][1], verified[name][2]) != (expected["size"], expected["sha256"]):
            raise ConversionError("critical source file differs from pinned profile: %s" % name)
    # If a receipt includes a named file from the profile's audited snapshot,
    # require that snapshot digest too. Other same-revision files remain valid
    # extras after their receipt integrity gate.
    for name in set(verified) & set(pinned):
        if (verified[name][1], verified[name][2]) != (pinned[name]["size"], pinned[name]["sha256"]):
            raise ConversionError("audited snapshot file differs: %s" % name)
    return {name: verified[name][0] for name in critical}, payload


def validate_config(path: Path, profile: Mapping[str, Any]) -> Dict[str, Any]:
    config, payload = load_json(path, "source config")
    pinned = next(item for item in profile["source_files"] if item["name"] == "config.json")
    if len(payload) != pinned["size"] or sha256_bytes(payload) != pinned["sha256"]:
        raise ConversionError("source config receipt differs")
    required = object_value(profile.get("config_required"), "profile.config_required")
    wrong = {key: (config.get(key), value) for key, value in required.items() if key not in config or config[key] != value}
    if wrong:
        raise ConversionError("source config semantic fields differ: %s" % wrong)
    return config


def load_catalog_entry(path: Path, profile: Mapping[str, Any], family: str) -> Tuple[Dict[str, Any], Dict[str, Any], bytes]:
    root, payload = load_json(path, "GENEB catalog")
    if root.get("schema_version") != 1 or not isinstance(root.get("models"), list):
        raise ConversionError("GENEB catalog schema differs")
    matches = [item for item in root["models"] if isinstance(item, dict) and item.get("runtime_id") == profile["runtime_id"]]
    if len(matches) != 1:
        raise ConversionError("GENEB catalog identity is not unique")
    entry = matches[0]
    expected = {
        "runtime_id": profile["runtime_id"],
        "geneb_model_id": profile["geneb_model_id"],
        "paper_name": profile["paper_name"],
        "architecture": profile["catalog_architecture"],
        "family": "cnn-transformer" if profile["runtime_id"] == "geneb-enformer" else "cnn-transformer-moe" if profile["runtime_id"] == "geneb-space" else "transformer-encoder",
    }
    if any(entry.get(key) != value for key, value in expected.items()):
        raise ConversionError("GENEB catalog family identity differs")
    topology = profile["topology"]
    presets = entry.get("embedding_presets")
    context = entry.get("context")
    if (
        not isinstance(presets, dict)
        or not isinstance(context, dict)
        or context.get("declared_max_tokens") != (topology["input_length"] if family == "sequence-cnn" else topology["max_seqlen"])
        or any(
            not isinstance(presets.get(name), dict)
            or presets[name].get("output_width") != topology["output_width" if family == "sequence-cnn" else "hidden_size"]
            for name in ("reference", "normalized")
        )
    ):
        raise ConversionError("GENEB catalog context/output contract differs")
    if family == "sequence-cnn":
        source = entry.get("source")
        if not isinstance(source, dict) or source.get("kind") != "huggingface" or source.get("repo") != profile["repo"] or source.get("revision") != profile["revision"] or source.get("immutable") is not True:
            raise ConversionError("GENEB catalog sequence-CNN source differs")
        if any(presets[name].get("pooling") != "spatial-mean" for name in ("reference", "normalized")):
            raise ConversionError("GENEB catalog sequence-CNN pooling differs")
    else:
        if any(
            presets[name].get("pooling") != "attention-mask-mean"
            or presets[name].get("special_tokens") != "include-manual-cls-sep"
            or presets[name].get("mask_domain") != "attention-mask"
            for name in ("reference", "normalized")
        ):
            raise ConversionError("GENEB catalog RoFormer embedding semantics differ")
    return entry, root, payload


TOKENIZER_DESCRIPTOR_KEYS = {
    "converter.schema", "converter.version", "compiler_manifest_sha256", "source_receipt_contract_sha256",
    "tokenizer.profile", "tokenizer.path", "tokenizer.sha256", "tokenizer.size",
}


def validate_tokenizer_descriptor(
    descriptor_path: Path,
    tokenizer_root: Optional[Path],
    artifact_root: Path,
    profile: Mapping[str, Any],
) -> Tuple[Dict[str, Any], str]:
    descriptor, payload = load_json(descriptor_path, "tokenizer descriptor")
    if set(descriptor) != TOKENIZER_DESCRIPTOR_KEYS:
        raise ConversionError("tokenizer descriptor fields differ")
    expected = profile["tokenizer"]
    if (
        descriptor["converter.schema"] != "evo-tokenizer-conversion-receipt"
        or descriptor["converter.version"] != 1
        or descriptor["tokenizer.profile"] != "evo-tokenizer-v1"
        or not isinstance(descriptor["source_receipt_contract_sha256"], str)
        or not SHA256_RE.fullmatch(descriptor["source_receipt_contract_sha256"])
        or descriptor["compiler_manifest_sha256"] != expected["compiler_manifest_sha256"]
        or descriptor["tokenizer.sha256"] != expected["compiled_asset_sha256"]
        or descriptor["tokenizer.size"] != expected["compiled_asset_size"]
    ):
        raise ConversionError("tokenizer descriptor differs from pinned compiler output")
    relative = normalized_relative_path(descriptor["tokenizer.path"], "tokenizer.path")
    root = (tokenizer_root if tokenizer_root is not None else descriptor_path.parent).resolve()
    asset = (root / relative).resolve()
    try:
        asset.relative_to(root)
    except ValueError:
        raise ConversionError("tokenizer path escapes tokenizer root")
    if (artifact_root.resolve() / relative).resolve() != asset:
        raise ConversionError("tokenizer asset must be staged relative to runtime artifact")
    try:
        size = asset.stat().st_size
    except OSError as error:
        raise ConversionError("cannot stat tokenizer asset: %s" % error)
    if size != descriptor["tokenizer.size"] or sha256_file(asset) != descriptor["tokenizer.sha256"]:
        raise ConversionError("tokenizer asset integrity differs")
    return ({key: descriptor[key] for key in ("tokenizer.profile", "tokenizer.path", "tokenizer.sha256", "tokenizer.size")}, sha256_bytes(payload))


def encode_metadata_value(value: Any) -> str:
    if isinstance(value, bool):
        return "b:%d" % int(value)
    if isinstance(value, int) and not isinstance(value, bool):
        if value < 0 or value > UINT64_MAX:
            raise ConversionError("metadata integer is outside uint64")
        return "u:%d" % value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConversionError("metadata float is non-finite")
        return "f:%016x" % struct.unpack("<Q", struct.pack("<d", value))[0]
    if isinstance(value, str):
        return "s:" + value
    raise ConversionError("metadata type is unsupported")


def encoded_header(profile_name: str, metadata: Mapping[str, Any], tensors: Sequence[RenamedTensorSource]) -> bytes:
    encoded = {}  # type: Dict[str, str]
    for key in sorted(metadata):
        if not KEY_RE.fullmatch(key) or len(key.encode("ascii")) > 255 or key == "runtime.profile":
            raise ConversionError("invalid/reserved metadata key %r" % key)
        encoded[key] = encode_metadata_value(metadata[key])
    encoded["runtime.profile"] = "s:" + profile_name
    root = {"__metadata__": dict(sorted(encoded.items()))}  # type: Dict[str, Any]
    offset = 0
    for tensor in tensors:
        end = offset + tensor.nbytes
        root[tensor.name] = {"dtype": tensor.dtype, "shape": list(tensor.shape), "data_offsets": [offset, end]}
        offset = end
    raw = json.dumps(root, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    header = raw + b" " * ((-len(raw)) % 8)
    if not header or len(header) > MAX_HEADER_SIZE:
        raise ConversionError("runtime Safetensors header exceeds 16 MiB")
    return header


def write_artifact(
    output_path: Path,
    profile_name: str,
    metadata: Mapping[str, Any],
    tensors: Sequence[RenamedTensorSource],
    force: bool,
) -> None:
    if output_path.suffix != ".safetensors" or not tensors:
        raise ConversionError("output must be a nonempty .safetensors artifact")
    header = encoded_header(profile_name, metadata, tensors)
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        raise FileExistsError("output already exists: %s" % output_path)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".%s." % output_path.name, suffix=".tmp", dir=str(output_path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w+b") as output:
            total_size = 8 + len(header) + sum(tensor.nbytes for tensor in tensors)
            output.truncate(total_size)
            output.seek(0)
            output.write(struct.pack("<Q", len(header)))
            output.write(header)
            for tensor in tensors:
                written = 0
                for raw in tensor.iter_chunks(CHUNK_SIZE):
                    chunk = memoryview(raw).cast("B")
                    if written + len(chunk) > tensor.nbytes:
                        raise ConversionError("tensor %s yielded too many bytes" % tensor.name)
                    output.write(chunk)
                    written += len(chunk)
                if written != tensor.nbytes:
                    raise ConversionError("tensor %s yielded a wrong byte count" % tensor.name)
            output.flush()
            os.fsync(output.fileno())
        if force:
            os.replace(str(temporary), str(output_path))
        else:
            os.link(str(temporary), str(output_path))
            temporary.unlink()
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def base_metadata(
    profile: Mapping[str, Any],
    receipt_payload: bytes,
    profile_contract_sha256: str,
    catalog_payload: bytes,
    catalog_root: Mapping[str, Any],
    catalog_entry: Mapping[str, Any],
) -> Dict[str, Any]:
    result = build_geneb_artifact_metadata(catalog_root, catalog_entry, catalog_payload)
    result.update(
        {
            "model.id": profile["runtime_id"],
            "source.receipt_sha256": sha256_bytes(receipt_payload),
            "source.catalog_contract_sha256": catalog_contract_sha256(
                catalog_root, catalog_entry
            ),
            "source.converter_profile_contract_sha256": profile_contract_sha256,
            "source.config_sha256": next(item["sha256"] for item in profile["source_files"] if item["name"] == "config.json"),
            "source.weights.sha256": next(item["sha256"] for item in profile["source_files"] if item["name"] == "pytorch_model.bin"),
            "source.weights.size": next(item["size"] for item in profile["source_files"] if item["name"] == "pytorch_model.bin"),
            "source.canonical_digest": profile["source_contract"]["canonical_digest"],
            "source.canonical_tensor_count": profile["source_contract"]["canonical_f32_count"],
            "source.canonical_bytes": profile["source_contract"]["canonical_bytes"],
        }
    )
    return result


def sequence_metadata(
    profile: Mapping[str, Any],
    implementation_contracts: Mapping[str, Any],
    receipt_payload: bytes,
    catalog_payload: bytes,
    catalog_root: Mapping[str, Any],
    catalog_entry: Mapping[str, Any],
) -> Dict[str, Any]:
    topology = profile["topology"]
    implementation = implementation_contracts[topology["variant"]]
    profile_contract = converter_profile_contract_sha256(
        1,
        "geneb-sequence-cnn-converter-v1",
        profile,
        {"implementation_contracts": implementation_contracts},
    )
    result = base_metadata(
        profile,
        receipt_payload,
        profile_contract,
        catalog_payload,
        catalog_root,
        catalog_entry,
    )
    result.update(
        {
            "runtime.abi": "geneb-sequence-cnn-pytorch-v1",
            # The native sequence-CNN ABI exposes only the returned final
            # sequence embedding. Public layer 0 is that tensor; transformer
            # depth remains independently recorded below.
            "runtime.embedding_layer_count": 1,
            "runtime.tokenizer_vocabulary_size": 4,
            "model.architecture": "GenebSequenceCnnEncoder",
            "config.hidden_size": topology["output_width"],
            "config.num_layers": topology["depth"],
            "config.max_seqlen": topology["input_length"],
            "sequence_cnn.variant": topology["variant"],
            "sequence_cnn.input_length": topology["input_length"],
            "sequence_cnn.trunk_width": topology["trunk_width"],
            "sequence_cnn.output_width": topology["output_width"],
            "sequence_cnn.depth": topology["depth"],
            "sequence_cnn.heads": topology["heads"],
            "sequence_cnn.key_width": topology["key_width"],
            "sequence_cnn.value_width": topology["value_width"],
            "sequence_cnn.relative_feature_width": topology["relative_feature_width"],
            "sequence_cnn.target_length": topology["target_length"],
            "sequence_cnn.num_downsamples": topology["num_downsamples"],
            "sequence_cnn.batch_norm_epsilon": topology["batch_norm_epsilon"],
            "sequence_cnn.gelu_sigmoid_scale": topology["gelu_sigmoid_scale"],
            "sequence_cnn.use_tf_gamma": topology["use_tf_gamma"],
            "sequence_cnn.species_num_experts": topology["species_num_experts"],
            "sequence_cnn.top_k": topology["top_k"],
            "sequence_cnn.gate_negative_slope": topology["gate_negative_slope"],
            "sequence_cnn.species": topology["species"],
            "sequence_cnn.hidden_tap": "returned-sequence-embedding" if topology["variant"] == "space" else "returned-embeddings",
            "sequence_cnn.pooling": "spatial-mean",
            "sequence_cnn.mask_domain": "all-896-spatial-rows",
            "sequence_cnn.special_tokens": "none",
            "source.repo": profile["repo"],
            "source.revision": profile["revision"],
            "source.implementation_contract": implementation["source"],
        }
    )
    for key, value in implementation.items():
        if key != "source":
            result["source.implementation_%s" % key] = value
    return result


def roformer_metadata(
    profile: Mapping[str, Any],
    implementation: Mapping[str, Any],
    tokenizer: Mapping[str, Any],
    tokenizer_descriptor_sha256: str,
    receipt_payload: bytes,
    catalog_payload: bytes,
    catalog_root: Mapping[str, Any],
    catalog_entry: Mapping[str, Any],
) -> Dict[str, Any]:
    topology = profile["topology"]
    profile_contract = converter_profile_contract_sha256(
        1,
        "geneb-roformer-converter-v1",
        profile,
        {"implementation_contract": implementation},
    )
    result = base_metadata(
        profile,
        receipt_payload,
        profile_contract,
        catalog_payload,
        catalog_root,
        catalog_entry,
    )
    result.update(
        {
            "runtime.abi": "geneb-roformer-pytorch-v1",
            "runtime.embedding_layer_count": topology["num_layers"] + 1,
            "runtime.tokenizer_vocabulary_size": topology["tokenizer_vocab_size"],
            "model.architecture": "GenebRoformerEncoder",
            "config.vocab_size": topology["vocab_size"],
            "config.hidden_size": topology["hidden_size"],
            "config.num_layers": topology["num_layers"],
            "config.max_seqlen": topology["max_seqlen"],
            "roformer.vocab_size": topology["vocab_size"],
            "roformer.width": topology["hidden_size"],
            "roformer.layers": topology["num_layers"],
            "roformer.heads": topology["num_attention_heads"],
            "roformer.head_width": topology["head_dim"],
            "roformer.inner_width": topology["inner_size"],
            "roformer.max_sequence_length": topology["max_seqlen"],
            "roformer.type_vocabulary_size": topology["type_vocab_size"],
            "roformer.pad_token_id": topology["pad_token_id"],
            "roformer.cls_token_id": topology["cls_token_id"],
            "roformer.sep_token_id": topology["sep_token_id"],
            "roformer.layer_norm_epsilon": topology["layer_norm_epsilon"],
            "roformer.rope_base": topology["rope_base"],
            "roformer.rotary_value": topology["rotary_value"],
            "roformer.pooling": topology["pooling"],
            "roformer.manual_special_order": topology["manual_special_order"],
            "roformer.pad_position_policy": topology["pad_position_policy"],
            "roformer.hidden_tap": "roformer-last-hidden-state",
            "roformer.mask_domain": "attention-mask",
            "roformer.special_tokens": "include-manual-cls-sep",
            "source.url": profile["source_url"],
            "source.implementation_contract": implementation["source"],
            "source.tokenizer_descriptor_sha256": tokenizer_descriptor_sha256,
        }
    )
    for key, value in implementation.items():
        if key != "source":
            result["source.implementation_%s" % key] = value
    result.update(tokenizer)
    return result


__all__ = [
    "ConversionError", "GenebArtifactError", "load_profiles", "select_profile",
    "validate_receipt", "validate_config", "load_catalog_entry",
    "validate_tokenizer_descriptor", "validated_sequence_tensors",
    "validated_roformer_tensors", "sequence_metadata", "roformer_metadata",
    "write_artifact", "sha256_bytes",
]
