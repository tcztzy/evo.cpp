# gpu02 reproducible environment

Validated on 2026-07-23:

- Rocky Linux 8.10, x86-64
- four NVIDIA A800 80GB PCIe GPUs
- NVIDIA driver 580.126.20
- bidirectional CUDA P2P read/write available between every GPU pair
- Apptainer 1.4.5
- CUDA 12.8.93 inside `$HOME/evo2c-cuda12.8-rocky8.sif`
- image: 4,843,704,320 bytes, SHA256
  `d5f2682c742a4bd1af0f2d70c7c8d5e63bdbc6ca76c41df0c941b31849fe5667`
- native CUDA target: `sm_80`
- aria2 1.37.0 from the user's read-only Nix profile
- conversion Python 3.12 and CPU PyTorch 2.10.0 from the user's read-only
  Nix profile; the script bootstraps `$HOME/.venv-evo2c-convert`

The checkpoint cache is remote-only:
`/build/grp_icg/users/tang/.cache/huggingface`. The official Hugging Face
revision is `d529aa57c30771814217ad89baaeaf6e2315c7d7`; gpu02 uses
`https://hf-mirror.com` (equivalent to `HF_ENDPOINT=https://hf-mirror.com`)
because the canonical endpoint is not reachable from the server.

Official checkpoint parts:

| File | Bytes | SHA256 |
|---|---:|---|
| `evo2_40b.pt.part0` | 41,126,745,847 | `3b74fa4e6158d49265e3e270ba8869390d064358f8bf3d2af0b3e1772728f485` |
| `evo2_40b.pt.part1` | 41,126,745,847 | `bdc4a76e0f23f8295e7061c2f0deff24f723bd916dc4cdc4d9216cac9c2d49d5` |

Validated generated artifacts:

| File | Bytes | SHA256 |
|---|---:|---|
| `$HOME/evo2c-models/evo2_40b.pt` | 82,253,491,694 | `dd299612b1c1cdded0dfdcaf4d16f98fc97458261d80f4d662429f0ccb316bc3` |
| `$HOME/evo2c-models/evo2-40b-bf16.evo2` | 82,252,673,728 | `7452b1412919d6724172dfe4a76b01605ccdb20522873dea6f06df4ea76c8ac7` |

From the local repository, build and test with:

```sh
scripts/gpu02_build.sh
scripts/gpu02_smoke.sh
scripts/gpu02_prepare_40b.sh
```

`gpu02_build.sh` constructs `$HOME/evo2c-cuda12.8-rocky8.sif` from
`containers/evo2c-cuda12.8-rocky8.def` when the image is absent, then reuses it
on later runs.

The preparation script resumes both official checkpoint parts into the shared
Hugging Face cache with ordinary HTTP Range requests against the mirror. This
avoids depending on FP8 packages or the Xet reconstruction client. It verifies
exact sizes and SHA256 values before publishing cache blobs, merges them in
numeric order, records the merged SHA in EVO2C metadata, performs the streaming
BF16 conversion, and validates the resulting file with `evo2c-inspect`. Model
files remain under `$HOME/evo2c-models` on gpu02.

The long-running preparation phase executes as a detached remote worker. Its
PID, log, and atomically published exit status are kept under
`$HOME/evo2c-models/.prepare-40b-jobs`; the local wrapper reconnects and resumes
polling after SSH or jump-host interruptions.
