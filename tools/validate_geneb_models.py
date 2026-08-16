#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate the immutable GENEB v4 model and provenance catalog."""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple


BENCHMARK_COMMIT = "b54d018903e7f6b874ee45b74e275936deff4cd3"
DATASET_REVISION = "4edd705be573e48c585c2cf79dc320f9f43c7b04"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
EXTRACTOR_COMMIT = "b465d2d6a11efbbc9a22c105e34832725ce50e05"
MODEL_META_SHA256 = "24c89d45a576c266ab24a710aabd192905cc4fee0585bd4ed7a8aecebecfad09"
ENFORMER_PATCH_PATH = "configs/geneb-reference-patches/enformer-seq-length.patch"
ENFORMER_PATCH_SHA256 = "77dfce0361962f4a5bbe5c2ad0c80bd9eaf5899fffb64bbc988fba01b477afda"
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
RUNTIME_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

# model_meta key -> (Table 4 display name, pinned parameter count)
EXPECTED_MODELS = {
    "METAGENE-1": ("METAGENE-1", 7_000_000_000),
    "evo-1-131k-base": ("Evo-1-131k", 7_000_000_000),
    "GenomeOcean-4B": ("GenomeOcean-4B", 4_000_000_000),
    "GENERator-eukaryote-3b-base": ("GENERator-Eukaryote-3B", 3_000_000_000),
    "dna_gpt3b_m": ("DNA-GPT-3B-M", 3_000_000_000),
    "nucleotide-transformer-2.5b-multi-species": ("NT-2.5B-MS", 2_500_000_000),
    "LucaOne-default-step36M": ("LucaOne", 2_000_000_000),
    "GENERator-eukaryote-1.2b-base": (
        "GENERator-Eukaryote-1.2B",
        1_200_000_000,
    ),
    "Omni-DNA-1B": ("Omni-DNA-1B", 1_000_000_000),
    "agro-nucleotide-transformer-1b": ("Agro-NT-1B", 1_000_000_000),
    "space": ("SPACE", 588_750_000),
    "eccdnamamba_weights": ("eccDNAMamba", 537_000_000),
    "GenomeOcean-500M": ("GenomeOcean-500M", 500_000_000),
    "gena-lm-bert-large-t2t": ("GENA-LM-Large-T2T", 336_000_000),
    "Omni-DNA-300M": ("Omni-DNA-300M", 300_000_000),
    "BioFM-265M": ("BioFM-265M", 265_000_000),
    "enformer-official-rough": ("Enformer", 252_000_000),
    "nucleotide-transformer-v2-250m-multi-species": (
        "NT-v2-250M-MS",
        250_000_000,
    ),
    "PlantCaduceus_l32": ("PlantCaduceus", 225_000_000),
    "OmniNA-220m": ("OmniNA-220M", 220_000_000),
    "gpt2_gene_multi_v2_ft": ("GPT2-Gene-Multi-v2", 200_000_000),
    "gpt2_gene_v1": ("GPT2-Gene-v1", 200_000_000),
    "genomics-fm": ("Genomics-FM", 120_000_000),
    "DNABERT-S": ("DNABERT-S", 117_070_082),
    "dnabert2-no-flashattention": ("DNABERT-2", 117_000_000),
    "gena-lm-bert-base-t2t-multi": ("GENA-LM-T2T-Multi", 110_000_000),
    "gena-lm-bert-base": ("GENA-LM", 110_000_000),
    "dna_gpt0.1b_h": ("DNA-GPT-0.1B-H", 100_000_000),
    "nucleotide-transformer-v2-100m-multi-species": (
        "NT-v2-100M-MS",
        100_000_000,
    ),
    "GROVER": ("GROVER", 87_101_793),
    "MutBERT": ("MutBERT", 86_000_000),
    "deepgene": ("DeepGene", 85_000_000),
    "hyenadna-large-1m-seqlen-hf": ("HyenaDNA-Large-1M", 54_600_000),
    "nucleotide-transformer-v2-50m-3mer-multi-species": (
        "NT-v2-50M-3mer-MS",
        50_000_000,
    ),
    "nucleotide-transformer-v2-50m-multi-species": (
        "NT-v2-50M-MS",
        50_000_000,
    ),
    "hyenadna-medium-160k-seqlen-hf": (
        "HyenaDNA-Medium-160k",
        14_200_000,
    ),
    "caduceus-ps_seqlen-131k_d_model-256_n_layer-16": (
        "Caduceus-PS-131k",
        8_000_000,
    ),
    "JanusDNA_72_w": ("JanusDNA-72-w", 1_980_000),
    "JanusDNA_72_wo": ("JanusDNA-72-wo", 1_980_000),
    "caduceus-ph_seqlen-1k_d_model-256_n_layer-4_lr-8e-3": (
        "Caduceus-PH-1k",
        1_930_000,
    ),
}

EXPECTED_SUBMISSION_SHA256 = {
    "BioFM-265M": "0ce161016474dba79f76fa4ef66a81d6b274f337d018453a37107f5a3a06bc33",
    "DNABERT-S": "f967da6c1ea405a1b12c7dd526ef56bfbb0e87451881404232d106d6c20ddbae",
    "GENERator-eukaryote-1.2b-base": "08ba30bc835ae81cfbc26b2e427c36efd47563a57876591c690ea86722a8eda3",
    "GENERator-eukaryote-3b-base": "d7d24829d40250fd414a7e8cfedadbd04307b07259d5b803e614cc9c924b69aa",
    "GROVER": "582799257e560dd751f141f1883b9aa63f9d9700826c1f7f48f911a154c3e8e8",
    "GenomeOcean-4B": "08df45d9d6485ab6bbf6f4fbc4ea131d7dea343e31be0e0aaccafbcf88794240",
    "GenomeOcean-500M": "a78484b1cf946e0dd4f8826f3d1ab4d25a7d3e291e6e74d3db678894adc9567e",
    "JanusDNA_72_w": "2ab5e421de438c9af917d9c4454e9e323dff517118a52ba18cc5bd1b71149876",
    "JanusDNA_72_wo": "7be3af8b68bb1bdbfbcb799a24c4cf72214e8ad0af5b76a8104de8052bbbcef4",
    "LucaOne-default-step36M": "0ff9ad1e327655ff36de40cd1faed631c40e7f81d0ce7049d2daf1fbbd84260b",
    "METAGENE-1": "14dcc77e40bb00300753d71d805c94a65b91a76b01e21ce0fa28c5999c384759",
    "MutBERT": "4aa05fbebe358e5f1fbb4172fe5cc495abe4515a19aeedac020e8901e5d25f13",
    "Omni-DNA-1B": "5b14482f9b18d378b023a221bf8a75aa9490235398b128ad6728e1866b2271bd",
    "Omni-DNA-300M": "55b689a843c92f0464edd1f4946b6e1e4d2ff78df3ff0f3293551bdec735a272",
    "OmniNA-220m": "1391f041ef8feb4166b1ad6cc46135754570eab2d133683c7a6cb3a3229ddd9c",
    "PlantCaduceus_l32": "9d67d22743c9065970b656d39e2c30d4c1326de6fda9c9b46b2eba8d0f840692",
    "agro-nucleotide-transformer-1b": "a0be21a0a80e3d53fef7a3bb1a8c689d734986080ae0cbdb1e997e1a97f671c4",
    "caduceus-ph_seqlen-1k_d_model-256_n_layer-4_lr-8e-3": "0c34a0fb70eb0251f0cdd34d8d456b688db0e4fe7d4e6fc9b085dcd0029e39fb",
    "caduceus-ps_seqlen-131k_d_model-256_n_layer-16": "1154401bc2d800f186b1389cea18907032575186385bc6bde2cc943da062d946",
    "deepgene": "e2ac0f19d70e82fc3f5d352e7848c1c32636c285ea594bf764a04faabdda7e2a",
    "dna_gpt0.1b_h": "a841776def89642707be869bd6f60edbec5c8ab7e41d75a6aee2159746eb6dbc",
    "dna_gpt3b_m": "225386f210fc65ba1d05f8623f664e5375b5ec185a024cb0832091cc88d9a801",
    "dnabert2-no-flashattention": "88322872ecb271df51bd2e2584ea9a442243ac4f9a105409d8b6ab5111a5898a",
    "eccdnamamba_weights": "51d2d2823c5e2864c1155266f8d4d5d28b95017a07a574366fc062d725d9f51c",
    "enformer-official-rough": "49f0f2a62cf9228fea7438eed82ff176678a13d71c8f594287c690152938e8d7",
    "evo-1-131k-base": "b4e73142b172f94aa890ced5bf4ee98c254a6c2dcc986faa175c3293cbd46231",
    "gena-lm-bert-base-t2t-multi": "6e6616aea3631e2f2bcecb1ebb17644a45c643038a69881a7fe4894a36e13db5",
    "gena-lm-bert-base": "0dcb878ed1fba21fe7920ecb20002777ad7c7152b177cf74db6b31e749a2b7c1",
    "gena-lm-bert-large-t2t": "187e6c9c491890e383a24905bf1dc634e6dafcd05d4b3d4825622c8859171a6c",
    "genomics-fm": "b5e424a246d9817c58a26cb39b22223b2f1d9540f604fae4cd9c8b94b7055c95",
    "gpt2_gene_multi_v2_ft": "a1ae18fc358eac06c416adf8a92e5d25201b494be2df2a1371e765d5094623a0",
    "gpt2_gene_v1": "1dfeb7eb370d603cca45441a0503725124d1863ae01423462a2c089bed17592c",
    "hyenadna-large-1m-seqlen-hf": "847e6ccb9e6cf1c9ac172ed60bba95ffb0ae2f0c1bb5f46452e391c99b52ffbb",
    "hyenadna-medium-160k-seqlen-hf": "d74a9e4b6270710c7e951906bf5596a95764ecac5c8ea31c03a665e99360a070",
    "nucleotide-transformer-2.5b-multi-species": "c98fbdb7df1ef9176d85388054f46ffc78ce87d224d2a3e2ffdd14fb24f9b1dc",
    "nucleotide-transformer-v2-100m-multi-species": "6ea77a76216b67b0e08dbd043cda3c3a3293c9116d68440420611573b383c949",
    "nucleotide-transformer-v2-250m-multi-species": "ac481d593f7e2c7564b049de14af02491f1df7b082c5317ef5f28388fd087a3b",
    "nucleotide-transformer-v2-50m-3mer-multi-species": "d1740d2e9572ffbaa69eb7dc3b64eafb090e2e91c87fdb880b51e20ff4252a69",
    "nucleotide-transformer-v2-50m-multi-species": "933b4f7e462191098826d192d6fd64d9791e6a3f550351041527b1691d0cdb3e",
    "space": "eae92567fe6e9940dcb2a5ad4f795087be199c381ca225c3b185037b2768d2f4",
}

EXPECTED_HF_SOURCES = {
    "METAGENE-1": ("metagene-ai/METAGENE-1", "ad8a1e0ee62b85058bfc05d823d8e8d4759edc48"),
    "evo-1-131k-base": ("togethercomputer/evo-1-131k-base", "c206aab77ae5967a069c4200ecb1858588528c9d"),
    "GenomeOcean-4B": ("DOEJGI/GenomeOcean-4B", "2bed2fc3ed47c5f6955ba3e64563512c9b338dfb"),
    "GENERator-eukaryote-3b-base": ("GenerTeam/GENERator-eukaryote-3b-base", "7515b17659f092997335226c3fb6aafd06c0add9"),
    "nucleotide-transformer-2.5b-multi-species": ("InstaDeepAI/nucleotide-transformer-2.5b-multi-species", "b746b125aacd1b0970c05a32fd71ba726754542e"),
    "LucaOne-default-step36M": ("LucaGroup/LucaOne-default-step36M", "f0d2807eb097a1507b911022d979b502cd62b74c"),
    "GENERator-eukaryote-1.2b-base": ("GenerTeam/GENERator-eukaryote-1.2b-base", "5e872c94264891f9adf59d8ea64e426bb68badb5"),
    "Omni-DNA-1B": ("zehui127/Omni-DNA-1B", "0ea9d54e356b4e7354dc40e2d980e9ebcb2ccfd5"),
    "agro-nucleotide-transformer-1b": ("InstaDeepAI/agro-nucleotide-transformer-1b", "b0e1ea1f53a2bf5bb29f8eab7a7e553bf06c1ab1"),
    "space": ("yangyz1230/space", "63c9f5011877cc20b3e9d2d63dcdb1d547e62c18"),
    "eccdnamamba_weights": ("eccDNAMamba/eccDNAMamba-1M", "36d95a3f783fa93640ce9e070ebde3f0ebed175d"),
    "GenomeOcean-500M": ("DOEJGI/GenomeOcean-500M", "0d9f453925aca9c278505cf0103b6b4311092052"),
    "gena-lm-bert-large-t2t": ("AIRI-Institute/gena-lm-bert-large-t2t", "f997a8a7c1ba5feb6d11de46354a41c51ffe9660"),
    "Omni-DNA-300M": ("zehui127/Omni-DNA-300M", "23587a0177a1f00d18b7079a2e81150b80e63f1c"),
    "BioFM-265M": ("m42-health/BioFM-265M", "52218bbdd383c123f6ced585c9cf2f62ae1fbb17"),
    "enformer-official-rough": ("EleutherAI/enformer-official-rough", "affe5713ae9017460706a44108289b13c5fee16c"),
    "nucleotide-transformer-v2-250m-multi-species": ("InstaDeepAI/nucleotide-transformer-v2-250m-multi-species", "c0f0359229f36ff6bc3a021247eefe0a9c344bd1"),
    "PlantCaduceus_l32": ("kuleshov-group/PlantCaduceus_l32", "e624c13c3d35415348b854c87a218893b23564f7"),
    "OmniNA-220m": ("XLS/OmniNA-220m", "64ea6ce7b250fc611773215ddcdd1ecca232de67"),
    "gpt2_gene_multi_v2_ft": ("dnagpt/gpt2_gene_multi_v2_ft", "b74ac7de9a489e936f612ded2036e01a33eec743"),
    "gpt2_gene_v1": ("dnagpt/gpt2_gene_v1", "ffcc7a4135ec0439800ee95fdc413706685136e8"),
    "DNABERT-S": ("zhihan1996/DNABERT-S", "00e47f96cdea35e4b6f5df89e5419cbe47d490c6"),
    "dnabert2-no-flashattention": ("zhihan1996/DNABERT-2-117M", "7bce263b15377fc15361f52cfab88f8b586abda0"),
    "gena-lm-bert-base-t2t-multi": ("AIRI-Institute/gena-lm-bert-base-t2t-multi", "4633e5a1ada905bb7afee6877d71cc12578a95a5"),
    "gena-lm-bert-base": ("AIRI-Institute/gena-lm-bert-base", "416f055300346a5830ca49438daf5f4e136ed9a8"),
    "nucleotide-transformer-v2-100m-multi-species": ("InstaDeepAI/nucleotide-transformer-v2-100m-multi-species", "f34324c6fde36a4f635f0f1f06cac5d25acd6798"),
    "GROVER": ("PoetschLab/GROVER", "6b223110f0d6963e849f55bc2a2f3cff0e38c7a4"),
    "MutBERT": ("JadenLong/MutBERT", "b68d8d6c9ccd8167639b25fb979cbd39a5c5c60c"),
    "hyenadna-large-1m-seqlen-hf": ("LongSafari/hyenadna-large-1m-seqlen-hf", "0a629abf9c7f85b4ec9aa6a1aefa3adcf1907446"),
    "nucleotide-transformer-v2-50m-3mer-multi-species": ("InstaDeepAI/nucleotide-transformer-v2-50m-3mer-multi-species", "ff82eaf931e483feeb6bf7ecf03f1febe6b2fe76"),
    "nucleotide-transformer-v2-50m-multi-species": ("InstaDeepAI/nucleotide-transformer-v2-50m-multi-species", "81b29e5786726d891dbf929404ef20adca5b36f1"),
    "hyenadna-medium-160k-seqlen-hf": ("LongSafari/hyenadna-medium-160k-seqlen-hf", "7ebf71773d22c0ede2cc55cb2be15ee8c289e1ce"),
    "caduceus-ps_seqlen-131k_d_model-256_n_layer-16": ("kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16", "d89eeb853136ea64da7feb3d0c8e909771b17ae6"),
    "caduceus-ph_seqlen-1k_d_model-256_n_layer-4_lr-8e-3": ("kuleshov-group/caduceus-ph_seqlen-1k_d_model-256_n_layer-4_lr-8e-3", "9865108985311772704c2ece0bac082153a6167b"),
}

EXPECTED_MANUAL_SOURCES = {
    "dna_gpt3b_m": ("google-drive", "https://drive.google.com/drive/folders/10UPPx6V13oQW6knuLV7d8SRIA3D6hYor"),
    "genomics-fm": ("google-drive", "https://drive.google.com/drive/folders/17-MQdeD9G-F7uiuHWWMdP50MmgJvpJb9"),
    "dna_gpt0.1b_h": ("google-drive", "https://drive.google.com/drive/folders/10UPPx6V13oQW6knuLV7d8SRIA3D6hYor"),
    "deepgene": ("google-drive", "https://drive.google.com/drive/folders/1gb2IqO3NdSMbydKMLGZBFAbsbKhgm0i8"),
    "JanusDNA_72_w": ("harvard-dataverse", "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi%3A10.7910%2FDVN%2FHDT0RN"),
    "JanusDNA_72_wo": ("harvard-dataverse", "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi%3A10.7910%2FDVN%2FHDT0RN"),
}


class CatalogError(RuntimeError):
    """Raised when catalog data violates the frozen GENEB contract."""


def _object_pairs(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result = {}  # type: Dict[str, Any]
    for key, value in pairs:
        if key in result:
            raise CatalogError("duplicate JSON key: " + key)
        result[key] = value
    return result


def _read_catalog(path: Path) -> Dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise CatalogError("cannot read catalog {}: {}".format(path, error))
    try:
        value = json.loads(text, object_pairs_hook=_object_pairs)
    except (CatalogError, json.JSONDecodeError) as error:
        raise CatalogError("invalid catalog JSON: {}".format(error))
    return _mapping(value, "$catalog")


def _mapping(value: Any, path: str) -> Dict[str, Any]:
    if type(value) is not dict:
        raise CatalogError("{} must be an object".format(path))
    return value


def _array(value: Any, path: str) -> List[Any]:
    if type(value) is not list:
        raise CatalogError("{} must be an array".format(path))
    return value


def _keys(
    value: Mapping[str, Any],
    path: str,
    required: Set[str],
    optional: Optional[Set[str]] = None,
) -> None:
    optional = set() if optional is None else optional
    missing = required - set(value)
    unknown = set(value) - required - optional
    if missing:
        raise CatalogError(
            "{} missing fields: {}".format(path, ", ".join(sorted(missing)))
        )
    if unknown:
        raise CatalogError(
            "{} has unknown fields: {}".format(path, ", ".join(sorted(unknown)))
        )


def _string(value: Any, path: str, nullable: bool = False) -> Optional[str]:
    if nullable and value is None:
        return None
    if type(value) is not str or not value:
        raise CatalogError("{} must be a nonempty string".format(path))
    return value


def _boolean(value: Any, path: str) -> bool:
    if type(value) is not bool:
        raise CatalogError("{} must be a boolean".format(path))
    return value


def _integer(value: Any, path: str, nullable: bool = False) -> Optional[int]:
    if nullable and value is None:
        return None
    if type(value) is not int or value <= 0:
        raise CatalogError("{} must be a positive integer".format(path))
    return value


def _choice(value: Any, path: str, choices: Set[str]) -> str:
    result = _string(value, path)
    if result not in choices:
        raise CatalogError(
            "{} must be one of: {}".format(path, ", ".join(sorted(choices)))
        )
    return result


def _digest(value: Any, path: str, nullable: bool = False) -> Optional[str]:
    result = _string(value, path, nullable=nullable)
    if result is not None and DIGEST_RE.fullmatch(result) is None:
        raise CatalogError("{} must be a lowercase SHA256".format(path))
    return result


def _commit(value: Any, path: str) -> str:
    result = _string(value, path)
    if COMMIT_RE.fullmatch(result) is None:
        raise CatalogError("{} must be a lowercase 40-hex commit".format(path))
    return result


def _exact(value: Any, expected: Any, path: str) -> None:
    if value != expected or type(value) is not type(expected):
        raise CatalogError("{} must equal {!r}".format(path, expected))


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _safe_relative(value: Any, path: str) -> str:
    result = _string(value, path)
    parsed = PurePosixPath(result)
    if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
        raise CatalogError("{} must be a normalized relative path".format(path))
    return result


def _validate_suite(value: Any) -> None:
    suite = _mapping(value, "suite")
    _keys(
        suite,
        "suite",
        {
            "id",
            "paper",
            "benchmark_commit",
            "extractor_commit",
            "model_meta_path",
            "model_meta_sha256",
            "dataset",
            "reference_batching",
        },
    )
    _exact(suite["id"], "geneb-v4", "suite.id")
    _exact(suite["benchmark_commit"], BENCHMARK_COMMIT, "suite.benchmark_commit")
    _exact(suite["extractor_commit"], EXTRACTOR_COMMIT, "suite.extractor_commit")
    _exact(suite["model_meta_path"], "benchmark/model_meta.json", "suite.model_meta_path")
    _exact(suite["model_meta_sha256"], MODEL_META_SHA256, "suite.model_meta_sha256")

    paper = _mapping(suite["paper"], "suite.paper")
    _keys(
        paper,
        "suite.paper",
        {"arxiv_id", "version", "table", "evaluated_models"},
    )
    _exact(paper["arxiv_id"], "2606.04525", "suite.paper.arxiv_id")
    _exact(paper["version"], "v4", "suite.paper.version")
    _exact(paper["table"], "Table 4", "suite.paper.table")
    _exact(paper["evaluated_models"], 40, "suite.paper.evaluated_models")

    dataset = _mapping(suite["dataset"], "suite.dataset")
    _keys(
        dataset,
        "suite.dataset",
        {"repo", "revision", "tasks", "categories", "license"},
    )
    _exact(dataset["repo"], "darlednik/geneb-tasks", "suite.dataset.repo")
    _exact(dataset["revision"], DATASET_REVISION, "suite.dataset.revision")
    _exact(dataset["tasks"], 100, "suite.dataset.tasks")
    _exact(dataset["categories"], 13, "suite.dataset.categories")
    _exact(dataset["license"], "apache-2.0", "suite.dataset.license")

    batching = _mapping(suite["reference_batching"], "suite.reference_batching")
    _keys(
        batching,
        "suite.reference_batching",
        {
            "batch_size",
            "order",
            "split_boundary",
            "final_batch",
            "pad_to_batch_max",
            "padding_side",
        },
    )
    _exact(batching["batch_size"], 8, "suite.reference_batching.batch_size")
    _exact(batching["order"], "input", "suite.reference_batching.order")
    _exact(batching["split_boundary"], "flush", "suite.reference_batching.split_boundary")
    _exact(
        batching["final_batch"],
        "actual",
        "suite.reference_batching.final_batch",
    )
    _exact(
        batching["pad_to_batch_max"],
        True,
        "suite.reference_batching.pad_to_batch_max",
    )
    _exact(
        batching["padding_side"],
        "model-preset",
        "suite.reference_batching.padding_side",
    )


def _validate_tokenizer(value: Any, path: str) -> None:
    tokenizer = _mapping(value, path)
    required_fields = {
        "kind",
        "asset_source",
        "assets",
        "add_special_tokens",
        "padding_side",
        "pad_to",
        "max_tokens",
    }
    optional_fields = {"k", "dynamic", "unknown_fields"}
    fields = required_fields | optional_fields
    _keys(tokenizer, path, required_fields, optional_fields)
    _string(tokenizer["kind"], path + ".kind")
    _string(tokenizer["asset_source"], path + ".asset_source")
    if tokenizer["assets"] is not None:
        _array(tokenizer["assets"], path + ".assets")
    _boolean(tokenizer["add_special_tokens"], path + ".add_special_tokens")
    _string(tokenizer["padding_side"], path + ".padding_side")
    _string(tokenizer["pad_to"], path + ".pad_to")
    _integer(tokenizer["max_tokens"], path + ".max_tokens", nullable=True)
    if "k" in tokenizer:
        _integer(tokenizer["k"], path + ".k")
    if "dynamic" in tokenizer:
        _boolean(tokenizer["dynamic"], path + ".dynamic")
    unknown = _array(tokenizer.get("unknown_fields", []), path + ".unknown_fields")
    if len(unknown) != len(set(unknown)):
        raise CatalogError(path + ".unknown_fields contains duplicates")
    for index, field in enumerate(unknown):
        field_path = "{}.unknown_fields[{}]".format(path, index)
        field_name = _string(field, field_path)
        if field_name not in fields:
            raise CatalogError(field_path + " names an unknown tokenizer field")
        if tokenizer[field_name] is not None:
            raise CatalogError(field_path + " must name a null field")


def _validate_context(value: Any, path: str) -> None:
    context = _mapping(value, path)
    required_fields = {
        "declared_max_tokens",
        "reference_max_tokens",
        "unit",
        "length_policy",
    }
    optional_fields = {"unknown_fields"}
    fields = required_fields | optional_fields
    _keys(context, path, required_fields, optional_fields)
    _integer(context["declared_max_tokens"], path + ".declared_max_tokens", True)
    _integer(context["reference_max_tokens"], path + ".reference_max_tokens", True)
    _choice(context["unit"], path + ".unit", {"tokens", "bases"})
    _choice(
        context["length_policy"],
        path + ".length_policy",
        {"tokenizer-truncate", "reject", "raw-crop", "fixed-pad", "model-preset"},
    )
    unknown = _array(context.get("unknown_fields", []), path + ".unknown_fields")
    if len(unknown) != len(set(unknown)):
        raise CatalogError(path + ".unknown_fields contains duplicates")
    for index, field in enumerate(unknown):
        field_path = "{}.unknown_fields[{}]".format(path, index)
        field_name = _string(field, field_path)
        if field_name not in fields:
            raise CatalogError(field_path + " names an unknown context field")
        if context[field_name] is not None:
            raise CatalogError(field_path + " must name a null field")


def _validate_input_transform(value: Any, path: str) -> None:
    transform = _mapping(value, path)
    _keys(
        transform,
        path,
        {
            "case",
            "u_to_t",
            "invalid",
            "frame_trim",
            "raw_crop",
            "fixed_pad",
            "prefix",
            "special_tokens",
            "token_truncation",
        },
    )
    _choice(transform["case"], path + ".case", {"preserve", "upper", "lower"})
    _boolean(transform["u_to_t"], path + ".u_to_t")
    _string(transform["invalid"], path + ".invalid")
    _string(transform["prefix"], path + ".prefix", nullable=True)
    _string(transform["special_tokens"], path + ".special_tokens")
    _choice(
        transform["token_truncation"],
        path + ".token_truncation",
        {"none", "left", "right", "model-preset"},
    )

    frame_trim = transform["frame_trim"]
    if frame_trim is not None:
        trim = _mapping(frame_trim, path + ".frame_trim")
        _keys(trim, path + ".frame_trim", {"multiple", "remove_from"})
        _integer(trim["multiple"], path + ".frame_trim.multiple")
        _choice(
            trim["remove_from"],
            path + ".frame_trim.remove_from",
            {"left", "right"},
        )

    raw_crop = transform["raw_crop"]
    if raw_crop is not None:
        crop = _mapping(raw_crop, path + ".raw_crop")
        _keys(crop, path + ".raw_crop", {"length", "keep"})
        _integer(crop["length"], path + ".raw_crop.length")
        _choice(
            crop["keep"],
            path + ".raw_crop.keep",
            {"prefix", "suffix", "center"},
        )

    fixed_pad = transform["fixed_pad"]
    if fixed_pad is not None:
        pad = _mapping(fixed_pad, path + ".fixed_pad")
        _keys(
            pad,
            path + ".fixed_pad",
            {"length", "side", "value", "balance"},
        )
        _integer(pad["length"], path + ".fixed_pad.length")
        side = _choice(
            pad["side"],
            path + ".fixed_pad.side",
            {"left", "right", "both"},
        )
        _string(pad["value"], path + ".fixed_pad.value")
        balance = _string(pad["balance"], path + ".fixed_pad.balance", True)
        if balance is not None and balance not in {"extra-left", "extra-right"}:
            raise CatalogError(
                path + ".fixed_pad.balance must be null, extra-left, or extra-right"
            )
        if side == "both" and balance is None:
            raise CatalogError(path + ".fixed_pad both requires an odd-difference balance")
        if side != "both" and balance is not None:
            raise CatalogError(path + ".fixed_pad one-sided padding requires null balance")


def _validate_embedding_presets(value: Any, path: str) -> None:
    presets = _mapping(value, path)
    _keys(presets, path, {"reference", "normalized"})
    for preset_name in ("reference", "normalized"):
        preset_path = path + "." + preset_name
        preset = _mapping(presets[preset_name], preset_path)
        _keys(
            preset,
            preset_path,
            {
                "hidden_tap",
                "pooling",
                "special_tokens",
                "mask_domain",
                "output_width",
            },
        )
        for field in ("hidden_tap", "pooling", "special_tokens", "mask_domain"):
            _string(preset[field], preset_path + "." + field)
        _integer(preset["output_width"], preset_path + ".output_width", True)


def _validate_source(value: Any, path: str) -> str:
    source = _mapping(value, path)
    _keys(
        source,
        path,
        {
            "kind",
            "repo",
            "requested_revision",
            "revision",
            "immutable",
            "url",
            "required_files",
            "receipt",
            "manual_instructions",
        },
    )
    kind = _choice(
        source["kind"],
        path + ".kind",
        {"huggingface", "google-drive", "harvard-dataverse"},
    )
    repo = _string(source["repo"], path + ".repo", nullable=True)
    _string(source["requested_revision"], path + ".requested_revision", nullable=True)
    revision = _string(source["revision"], path + ".revision", nullable=True)
    _boolean(source["immutable"], path + ".immutable")
    url = _string(source["url"], path + ".url")
    if not url.startswith("https://"):
        raise CatalogError(path + ".url must use https")
    required_files = source["required_files"]
    if required_files is not None:
        required_files = _array(required_files, path + ".required_files")
        if not required_files:
            raise CatalogError(path + ".required_files must be null or nonempty")
        required_paths = []  # type: List[str]
        for index, file_value in enumerate(required_files):
            file_path = "{}.required_files[{}]".format(path, index)
            file_entry = _mapping(file_value, file_path)
            _keys(file_entry, file_path, {"path", "size", "sha256"})
            relative = _safe_relative(file_entry["path"], file_path + ".path")
            if relative in required_paths:
                raise CatalogError(path + ".required_files contains duplicate paths")
            required_paths.append(relative)
            _integer(file_entry["size"], file_path + ".size", True)
            _digest(file_entry["sha256"], file_path + ".sha256", True)
    manual = _string(
        source["manual_instructions"],
        path + ".manual_instructions",
        nullable=True,
    )

    receipt = _mapping(source["receipt"], path + ".receipt")
    _keys(
        receipt,
        path + ".receipt",
        {"required", "per_file_size", "per_file_sha256", "manifest_status"},
    )
    for field in ("required", "per_file_size", "per_file_sha256"):
        _exact(receipt[field], True, path + ".receipt." + field)
    manifest_status = _choice(
        receipt["manifest_status"],
        path + ".receipt.manifest_status",
        {"resolved-at-fetch", "required-after-manual-download", "pinned"},
    )

    if kind == "huggingface":
        if repo is None or "/" not in repo:
            raise CatalogError(path + ".repo must be OWNER/NAME for Hugging Face")
        if revision is None or COMMIT_RE.fullmatch(revision) is None:
            raise CatalogError(path + ".revision must pin a Hugging Face commit")
        if source["immutable"] is not True:
            raise CatalogError(path + ".immutable must be true for Hugging Face")
        if manual is not None:
            raise CatalogError(path + ".manual_instructions must be null for Hugging Face")
        if manifest_status == "required-after-manual-download":
            raise CatalogError(path + ".receipt uses a manual status for Hugging Face")
    else:
        if manual is None:
            raise CatalogError(path + ".manual_instructions is required for manual sources")
        if required_files is None:
            raise CatalogError(path + ".required_files is required for manual sources")
        files_complete = all(
            item["size"] is not None and item["sha256"] is not None
            for item in required_files
        )
        expected_status = "pinned" if files_complete else "required-after-manual-download"
        if manifest_status != expected_status:
            raise CatalogError(
                "{}.receipt.manifest_status must be {} for its file evidence".format(
                    path, expected_status
                )
            )
    return kind


def _validate_provenance(value: Any, path: str, source_dir: Path) -> Dict[str, Any]:
    provenance = _mapping(value, path)
    _keys(
        provenance,
        path,
        {
            "extractor",
            "reference_patch",
            "decisions",
            "known_defects",
            "normalization_decisions",
            "normalization_patch_sha256",
        },
    )
    extractor = _mapping(provenance["extractor"], path + ".extractor")
    _keys(extractor, path + ".extractor", {"module", "class", "commit"})
    module = _safe_relative(extractor["module"], path + ".extractor.module")
    if not module.endswith(".py"):
        raise CatalogError(path + ".extractor.module must name a Python source file")
    _string(extractor["class"], path + ".extractor.class")
    _exact(extractor["commit"], EXTRACTOR_COMMIT, path + ".extractor.commit")

    patch = _mapping(provenance["reference_patch"], path + ".reference_patch")
    _keys(
        patch,
        path + ".reference_patch",
        {"status", "path", "sha256", "operations"},
    )
    patch_status = _choice(
        patch["status"], path + ".reference_patch.status", {"none", "applied"}
    )
    patch_sha = _digest(patch["sha256"], path + ".reference_patch.sha256")
    operations = _array(patch["operations"], path + ".reference_patch.operations")
    patch_path = _string(patch["path"], path + ".reference_patch.path", True)
    if patch_status == "none":
        if patch_path is not None or patch_sha != EMPTY_SHA256 or operations:
            raise CatalogError(
                path + ".reference_patch none must use null path, empty SHA256, and []"
            )
    else:
        if patch_path is None or not operations:
            raise CatalogError(path + ".reference_patch applied requires path and operations")
        relative = _safe_relative(patch_path, path + ".reference_patch.path")
        local_path = source_dir / relative
        try:
            payload = local_path.read_bytes()
        except OSError as error:
            raise CatalogError(
                "{} cannot read {}: {}".format(path + ".reference_patch.path", relative, error)
            )
        actual_sha = hashlib.sha256(payload).hexdigest()
        if actual_sha != patch_sha:
            raise CatalogError(
                "{} hash mismatch: expected {}, got {}".format(
                    path + ".reference_patch.path", patch_sha, actual_sha
                )
            )

    _array(provenance["decisions"], path + ".decisions")
    _array(provenance["known_defects"], path + ".known_defects")
    normalization = _array(
        provenance["normalization_decisions"], path + ".normalization_decisions"
    )
    expected_normalization_sha = _canonical_digest(normalization)
    normalization_sha = _digest(
        provenance["normalization_patch_sha256"],
        path + ".normalization_patch_sha256",
    )
    if normalization_sha != expected_normalization_sha:
        raise CatalogError(
            "{}.normalization_patch_sha256 must hash canonical normalization_decisions"
            .format(path)
        )
    return {
        "extractor_commit": extractor["commit"],
        "reference_patch_sha256": patch_sha,
        "normalization_patch_sha256": normalization_sha,
        "reference_patch_status": patch_status,
        "reference_patch_path": patch_path,
    }


def _validate_license_component(value: Any, path: str) -> None:
    component = _mapping(value, path)
    _keys(component, path, {"id", "status", "source"})
    status = _choice(
        component["status"], path + ".status", {"declared", "unknown", "not-applicable"}
    )
    license_id = _string(component["id"], path + ".id", True)
    source = _string(component["source"], path + ".source", True)
    if status == "declared" and (license_id is None or source is None):
        raise CatalogError(path + " declared license requires id and source")
    if status != "declared" and license_id is not None:
        raise CatalogError(path + " non-declared license must use null id")


def _validate_licenses(value: Any, path: str) -> None:
    licenses = _mapping(value, path)
    _keys(
        licenses,
        path,
        {"weights", "code", "tokenizer", "dataset", "aup", "redistribution"},
    )
    for component in ("weights", "code", "tokenizer", "dataset"):
        _validate_license_component(licenses[component], path + "." + component)
    aup = _mapping(licenses["aup"], path + ".aup")
    _keys(aup, path + ".aup", {"required", "status", "url"})
    if aup["required"] is not None:
        _boolean(aup["required"], path + ".aup.required")
    aup_status = _choice(
        aup["status"], path + ".aup.status", {"declared", "none", "unknown"}
    )
    aup_url = _string(aup["url"], path + ".aup.url", True)
    if aup_status == "declared" and aup_url is None:
        raise CatalogError(path + ".aup declared status requires url")
    if aup["required"] is True and aup_status != "declared":
        raise CatalogError(path + ".aup required=true requires declared status")

    redistribution = _mapping(licenses["redistribution"], path + ".redistribution")
    _keys(redistribution, path + ".redistribution", {"weights", "metadata"})
    _exact(redistribution["weights"], False, path + ".redistribution.weights")
    _exact(redistribution["metadata"], True, path + ".redistribution.metadata")


def _validate_oracle(value: Any, path: str) -> Dict[str, Any]:
    oracle = _mapping(value, path)
    _keys(
        oracle,
        path,
        {"status", "environment_lock", "input_digest", "tolerances", "evidence"},
    )
    status = _choice(oracle["status"], path + ".status", {"missing", "passed", "failed"})
    environment = oracle["environment_lock"]
    input_digest = oracle["input_digest"]
    tolerances = oracle["tolerances"]
    evidence = oracle["evidence"]
    if status == "missing":
        if any(item is not None for item in (environment, input_digest, tolerances, evidence)):
            raise CatalogError(path + " missing oracle must use null evidence fields")
    else:
        _digest(input_digest, path + ".input_digest")
        _mapping(environment, path + ".environment_lock")
        tolerance_map = _mapping(tolerances, path + ".tolerances")
        _keys(tolerance_map, path + ".tolerances", {"max_abs", "mean_abs", "cosine"})
        for field in ("max_abs", "mean_abs", "cosine"):
            number = tolerance_map[field]
            if type(number) not in (int, float):
                raise CatalogError(path + ".tolerances." + field + " must be numeric")
        if tolerance_map["max_abs"] < 0 or tolerance_map["mean_abs"] < 0:
            raise CatalogError(path + ".tolerances absolute errors must be nonnegative")
        if not -1 <= tolerance_map["cosine"] <= 1:
            raise CatalogError(path + ".tolerances.cosine must be in [-1, 1]")
        _mapping(evidence, path + ".evidence")
    return {
        "oracle_status": status,
        "oracle_env": environment,
        "oracle_input_digest": input_digest,
    }


def _validate_runtime_support(value: Any, path: str) -> Dict[str, Any]:
    support = _mapping(value, path)
    _keys(support, path, {"status", "artifact_profile", "reason"})
    status = _choice(
        support["status"],
        path + ".status",
        {"cataloged", "manual-source", "experimental", "unsupported", "supported"},
    )
    artifact_profile = _string(support["artifact_profile"], path + ".artifact_profile", True)
    reason = _string(support["reason"], path + ".reason", True)
    if status == "supported" and artifact_profile is None:
        raise CatalogError(path + " supported status requires artifact_profile")
    if status != "supported" and reason is None:
        raise CatalogError(path + " non-supported status requires reason")
    return {"status": status, "artifact_profile": artifact_profile, "reason": reason}


def _validate_benchmark_provenance(value: Any, path: str) -> Dict[str, Any]:
    provenance = _mapping(value, path)
    _keys(
        provenance,
        path,
        {
            "reference_status",
            "normalized_status",
            "upstream_status",
            "reason",
            "official_submission_path",
            "official_submission_sha256",
        },
    )
    reference = _choice(
        provenance["reference_status"],
        path + ".reference_status",
        {"eligible", "blocked"},
    )
    _exact(
        provenance["normalized_status"],
        "protocol-compatible",
        path + ".normalized_status",
    )
    upstream = _choice(
        provenance["upstream_status"],
        path + ".upstream_status",
        {"executable", "broken"},
    )
    reason = _string(provenance["reason"], path + ".reason", True)
    if reference == "blocked" and reason is None:
        raise CatalogError(path + " blocked reference status requires reason")
    submission_path = _safe_relative(
        provenance["official_submission_path"], path + ".official_submission_path"
    )
    submission_sha = _digest(
        provenance["official_submission_sha256"],
        path + ".official_submission_sha256",
    )
    return {
        "reference_status": reference,
        "normalized_status": provenance["normalized_status"],
        "upstream_status": upstream,
        "reason": reason,
        "official_submission_path": submission_path,
        "official_submission_sha256": submission_sha,
    }


def _validate_backends(value: Any, path: str) -> Dict[str, str]:
    backends = _mapping(value, path)
    _keys(backends, path, {"cpu", "cuda", "mps"})
    result = {}  # type: Dict[str, str]
    for backend_name in ("cpu", "cuda", "mps"):
        backend_path = path + "." + backend_name
        backend = _mapping(backends[backend_name], backend_path)
        _keys(backend, backend_path, {"status", "evidence"})
        status = _choice(
            backend["status"],
            backend_path + ".status",
            {"not-promoted", "promoted", "unsupported"},
        )
        evidence = backend["evidence"]
        if status == "promoted" and evidence is None:
            raise CatalogError(backend_path + " promoted status requires evidence")
        if status == "not-promoted" and evidence is not None:
            raise CatalogError(backend_path + " not-promoted status requires null evidence")
        result[backend_name] = status
    return result


def _validate_model(value: Any, index: int, source_dir: Path) -> Dict[str, Any]:
    path = "models[{}]".format(index)
    model = _mapping(value, path)
    _keys(
        model,
        path,
        {
            "runtime_id",
            "geneb_model_id",
            "paper_name",
            "params",
            "family",
            "architecture",
            "tokenizer",
            "context",
            "input_transform",
            "embedding_presets",
            "source",
            "provenance",
            "licenses",
            "oracle",
            "runtime_support",
            "benchmark_provenance",
            "backends",
            "promotion_state",
        },
    )
    runtime_id = _string(model["runtime_id"], path + ".runtime_id")
    if RUNTIME_ID_RE.fullmatch(runtime_id) is None:
        raise CatalogError(path + ".runtime_id must be lowercase kebab-case")
    model_id = _string(model["geneb_model_id"], path + ".geneb_model_id")
    paper_name = _string(model["paper_name"], path + ".paper_name")
    params = _integer(model["params"], path + ".params")
    _string(model["family"], path + ".family")
    _string(model["architecture"], path + ".architecture")
    _validate_tokenizer(model["tokenizer"], path + ".tokenizer")
    _validate_context(model["context"], path + ".context")
    _validate_input_transform(model["input_transform"], path + ".input_transform")
    _validate_embedding_presets(model["embedding_presets"], path + ".embedding_presets")
    source_kind = _validate_source(model["source"], path + ".source")
    provenance = _validate_provenance(model["provenance"], path + ".provenance", source_dir)
    _validate_licenses(model["licenses"], path + ".licenses")
    licenses = _mapping(model["licenses"], path + ".licenses")
    _exact(licenses["dataset"]["id"], "apache-2.0", path + ".licenses.dataset.id")
    _exact(
        licenses["dataset"]["status"],
        "declared",
        path + ".licenses.dataset.status",
    )
    _exact(
        licenses["dataset"]["source"],
        "dataset-card",
        path + ".licenses.dataset.source",
    )
    oracle = _validate_oracle(model["oracle"], path + ".oracle")
    support = _validate_runtime_support(model["runtime_support"], path + ".runtime_support")
    benchmark = _validate_benchmark_provenance(
        model["benchmark_provenance"], path + ".benchmark_provenance"
    )
    backends = _validate_backends(model["backends"], path + ".backends")
    promotion_state = _choice(
        model["promotion_state"],
        path + ".promotion_state",
        {"cataloged", "runtime-supported"},
    )

    if model_id not in EXPECTED_MODELS:
        raise CatalogError(path + ".geneb_model_id is not in pinned model_meta")
    expected_paper, expected_params = EXPECTED_MODELS[model_id]
    _exact(paper_name, expected_paper, path + ".paper_name")
    _exact(params, expected_params, path + ".params")
    source = _mapping(model["source"], path + ".source")
    if model_id in EXPECTED_HF_SOURCES:
        expected_repo, expected_revision = EXPECTED_HF_SOURCES[model_id]
        _exact(source_kind, "huggingface", path + ".source.kind")
        _exact(source["repo"], expected_repo, path + ".source.repo")
        _exact(source["revision"], expected_revision, path + ".source.revision")
        _exact(
            source["url"],
            "https://huggingface.co/" + expected_repo,
            path + ".source.url",
        )
    else:
        expected_kind, expected_url = EXPECTED_MANUAL_SOURCES[model_id]
        _exact(source_kind, expected_kind, path + ".source.kind")
        _exact(source["repo"], None, path + ".source.repo")
        _exact(source["url"], expected_url, path + ".source.url")
    if model_id == "enformer-official-rough":
        _exact(
            provenance["reference_patch_status"],
            "applied",
            path + ".provenance.reference_patch.status",
        )
        _exact(
            provenance["reference_patch_path"],
            ENFORMER_PATCH_PATH,
            path + ".provenance.reference_patch.path",
        )
        _exact(
            provenance["reference_patch_sha256"],
            ENFORMER_PATCH_SHA256,
            path + ".provenance.reference_patch.sha256",
        )
    else:
        _exact(
            provenance["reference_patch_status"],
            "none",
            path + ".provenance.reference_patch.status",
        )
    if source_kind == "huggingface" and support["status"] == "manual-source":
        raise CatalogError(path + " Hugging Face source must not be manual-source")
    if (
        source_kind != "huggingface"
        and support["status"] != "manual-source"
        and source["receipt"]["manifest_status"] != "pinned"
    ):
        raise CatalogError(
            path + " Drive/Dataverse promotion requires a complete pinned file receipt"
        )
    expected_submission = "submissions/{}.json".format(model_id)
    _exact(
        benchmark["official_submission_path"],
        expected_submission,
        path + ".benchmark_provenance.official_submission_path",
    )
    _exact(
        benchmark["official_submission_sha256"],
        EXPECTED_SUBMISSION_SHA256[model_id],
        path + ".benchmark_provenance.official_submission_sha256",
    )
    if benchmark["reference_status"] == "eligible" and oracle["oracle_status"] != "passed":
        raise CatalogError(path + " reference-eligible provenance requires a passed oracle")
    if support["status"] == "supported":
        if oracle["oracle_status"] != "passed":
            raise CatalogError(path + " supported runtime requires a passed oracle")
        if backends["cpu"] != "promoted":
            raise CatalogError(path + " supported runtime requires promoted CPU evidence")
        if promotion_state != "runtime-supported":
            raise CatalogError(path + " supported runtime requires runtime-supported promotion")
    elif promotion_state != "cataloged":
        raise CatalogError(path + " non-supported runtime must remain cataloged")

    return {
        "runtime_id": runtime_id,
        "geneb_model_id": model_id,
        "paper_name": paper_name,
        "runtime_support": support["status"],
        "benchmark_provenance": benchmark,
        "backends": backends,
        "extractor_commit": provenance["extractor_commit"],
        "reference_patch_sha256": provenance["reference_patch_sha256"],
        "normalization_patch_sha256": provenance["normalization_patch_sha256"],
        "oracle_status": oracle["oracle_status"],
        "oracle_env": oracle["oracle_env"],
        "oracle_input_digest": oracle["oracle_input_digest"],
        "official_submission_path": benchmark["official_submission_path"],
        "official_submission_sha256": benchmark["official_submission_sha256"],
    }


def validate_catalog(catalog_path: Path, source_dir: Path) -> Dict[str, Any]:
    """Validate one GENEB catalog and return its deterministic gate summary."""
    catalog = _read_catalog(catalog_path)
    _keys(catalog, "$catalog", {"schema_version", "suite", "aliases", "models"})
    _exact(catalog["schema_version"], 1, "schema_version")
    if set(EXPECTED_SUBMISSION_SHA256) != set(EXPECTED_MODELS):
        raise CatalogError("internal submission manifest does not cover the model set")
    if set(EXPECTED_HF_SOURCES) | set(EXPECTED_MANUAL_SOURCES) != set(EXPECTED_MODELS):
        raise CatalogError("internal source manifest does not cover the model set")
    if set(EXPECTED_HF_SOURCES) & set(EXPECTED_MANUAL_SOURCES):
        raise CatalogError("internal source manifests overlap")
    _validate_suite(catalog["suite"])

    models = _array(catalog["models"], "models")
    if len(models) != 40:
        raise CatalogError("models must contain exactly 40 records")
    summaries = [
        _validate_model(model, index, source_dir) for index, model in enumerate(models)
    ]
    runtime_ids = [model["runtime_id"] for model in summaries]
    model_ids = [model["geneb_model_id"] for model in summaries]
    paper_names = [model["paper_name"] for model in summaries]
    for label, values in (
        ("runtime_id", runtime_ids),
        ("geneb_model_id", model_ids),
        ("paper_name", paper_names),
    ):
        if len(set(values)) != 40:
            raise CatalogError(label + " values must be unique")
    if set(model_ids) != set(EXPECTED_MODELS):
        missing = set(EXPECTED_MODELS) - set(model_ids)
        extra = set(model_ids) - set(EXPECTED_MODELS)
        raise CatalogError(
            "geneb_model_id set mismatch; missing={} extra={}".format(
                sorted(missing), sorted(extra)
            )
        )
    expected_paper_names = {entry[0] for entry in EXPECTED_MODELS.values()}
    if set(paper_names) != expected_paper_names:
        raise CatalogError("paper_name set does not equal GENEB v4 Table 4")

    aliases = _mapping(catalog["aliases"], "aliases")
    _keys(aliases, "aliases", {"presets", "models"})
    preset_aliases = _mapping(aliases["presets"], "aliases.presets")
    _keys(preset_aliases, "aliases.presets", {"geneb"})
    _exact(preset_aliases["geneb"], "geneb-v4-normalized", "aliases.presets.geneb")
    model_aliases = _mapping(aliases["models"], "aliases.models")
    identity_values = set(runtime_ids) | set(model_ids) | set(paper_names)
    for alias, target in model_aliases.items():
        _string(alias, "aliases.models key")
        target_value = _string(target, "aliases.models." + alias)
        if alias in identity_values:
            raise CatalogError("model alias collides with a canonical identity: " + alias)
        if target_value not in set(runtime_ids):
            raise CatalogError("model alias target is not a runtime_id: " + target_value)

    status_counts = {}  # type: Dict[str, int]
    provenance_counts = {}  # type: Dict[str, int]
    for model in summaries:
        support = model["runtime_support"]
        status_counts[support] = status_counts.get(support, 0) + 1
        reference = model["benchmark_provenance"]["reference_status"]
        provenance_counts[reference] = provenance_counts.get(reference, 0) + 1
    raw_sha = hashlib.sha256(catalog_path.read_bytes()).hexdigest()
    return {
        "schema_version": 1,
        "suite": "geneb-v4",
        "valid": True,
        "catalog_sha256": raw_sha,
        "model_count": 40,
        "runtime_support_counts": dict(sorted(status_counts.items())),
        "reference_provenance_counts": dict(sorted(provenance_counts.items())),
        "models": summaries,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate the pinned 40-model GENEB v4 catalog"
    )
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="project/share root used to verify cataloged patch files",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    catalog_path = args.catalog.resolve()
    source_dir = (
        args.source_dir.resolve()
        if args.source_dir is not None
        else catalog_path.parent.parent.resolve()
    )
    try:
        summary = validate_catalog(catalog_path, source_dir)
    except (CatalogError, OSError) as error:
        print("GENEB catalog validation failed: {}".format(error), file=sys.stderr)
        return 2
    payload = json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n"
    if args.output is None:
        sys.stdout.write(payload)
    else:
        try:
            args.output.write_text(payload, encoding="utf-8")
        except OSError as error:
            print("cannot write GENEB summary: {}".format(error), file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
