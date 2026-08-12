# Artifact acquisition and releases

`evo.cpp` keeps model weights outside Git. It distinguishes upstream source
checkpoints from converted `evo2-runtime-v1` artifacts so a downloaded `.pt`
file is never mistaken for an inference-ready model.

## Fetch an official source checkpoint

The fetch helper is an offline/conversion tool, not a dependency of the C++
inference process. Install its pinned client in a separate environment:

```sh
python3 -m venv .venv-fetch
. .venv-fetch/bin/activate
python3 -m pip install -r requirements-fetch.txt
```

Fetch by registered model ID:

```sh
python3 tools/evo_fetch.py --print-path source evo2_7b
```

An installed package exposes the same command as `evo-fetch`. The registry
supplies the repository, immutable 40-hex commit, filename, byte size, and
SHA256. A successful invocation verifies the complete file and writes an
atomic JSON receipt. `evo2_40b` and `evo2_40b_base` print two ordered part
paths; they still require the documented verified merge before conversion.

Cache selection is, in order:

1. `--cache-dir PATH`;
2. `$EVO_CACHE_HOME/huggingface/hub`;
3. `$HF_HOME/hub`;
4. `$XDG_CACHE_HOME/huggingface/hub` or `~/.cache/huggingface/hub`.

The layout and downloader are Hugging Face's commit-aware `refs`, `blobs`, and
`snapshots` cache. `--local-files-only` prohibits network access. Every return
path is size- and SHA256-verified; a corrupt online cache entry is force
refetched, while a corrupt offline entry fails explicitly.

## Fetch a converted runtime artifact

A repository publishing converted weights must contain `evo-artifact.json`:

```json
{
  "schema_version": 1,
  "artifact_profile": "evo2-runtime-v1",
  "model_id": "evo2_7b",
  "load_path": "model.safetensors.index.json",
  "files": [
    {
      "path": "model.safetensors.index.json",
      "size": 1234,
      "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    }
  ]
}
```

Every index and shard must be listed with its exact size and SHA256. Paths must
be normalized relative paths; duplicates, traversal, unknown profile, or a
`load_path` outside the list fail closed.

```sh
MODEL="$(evo-fetch --print-path runtime OWNER/REPO@COMMIT)"
evo -m "$MODEL" --score sequences.fa --ctx 8192 --gpu 0
```

A branch or tag is resolved once to an immutable commit and recorded in the
receipt. Online resolution force-refreshes the small manifest from that commit;
offline resolution requires its SHA256 to match the last verified receipt. For
reproducible automation, pass the commit explicitly.

## Binary releases

Tags matching the CMake project version build a Linux x86-64 CUDA 12.8 exact
archive from pinned CMake, Conan, lockfile, container, and architecture inputs.
The archive contains binaries, libraries, headers, the model registry, and the
fetch helper, but never model weights.

Each release carries:

- `SHA256SUMS` covering the archive and canonical metadata sidecar;
- archive size, SHA256, tag, commit, platform, backend, CUDA version, pinned
  build-image digest, runtime profile, and the explicit
  `contains_model_weights: false` field;
- a GitHub/Sigstore build-provenance attestation generated from the checksum
  file.

After downloading a release:

```sh
sha256sum -c SHA256SUMS
gh attestation verify evo-0.1.0-linux-x86_64-cuda12.8.tar.gz \
  --repo OWNER/evo.cpp
```

Repository owners should enable GitHub immutable releases so published tags
and assets cannot later be replaced.
