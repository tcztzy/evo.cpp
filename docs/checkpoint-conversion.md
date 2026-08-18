# Checkpoint conversion

生产 runtime 读取严格注册的 `evo2-runtime-v1`、`hyenadna-runtime-v1` 与
`esmc-runtime-v1` Safetensors，以及 CPU embedding 用的
`geneb-decoder-runtime-v1`、`geneb-olmo-runtime-v1`、`geneb-esm-runtime-v1`、
`geneb-bert-runtime-v1`、`geneb-gpt2-runtime-v1`、
`geneb-dna-gpt-runtime-v1`、`geneb-custom-encoder-runtime-v1` 与
`geneb-mamba-runtime-v1`、`geneb-hyenadna-runtime-v1` 与
`geneb-evo1-runtime-v1`、`geneb-janusdna-runtime-v1`、
`geneb-sequence-cnn-runtime-v1`，以及 `geneb-roformer-runtime-v1`
Safetensors。Arc `.pt`、BioNeMo DCP 与 DNA-GPT 源 `.pth` 使用隔离的 PyTorch
转换环境；HyenaDNA/ESMC converter 不依赖 PyTorch。任何产品推理路径都不加载
Python、PyTorch 或 libtorch。

GENEB decoder converter 只在 BioFM profile 写入可选
`decoder.attention_kernel=torch-cpu-flash-bf16-portable`；其他 profile 省略该键
并使用 eager 语义。Runtime 对未知值、非 BF16 activation 或 sliding-window
组合 fail closed，因此该选择不能由模型名或后端自动推断。

Evo 2/ESMC 的 model ID、revision、checksum、source/runtime manifest 以
[`configs/model-registry.json`](../configs/model-registry.json) 为机器可读来源。
HyenaDNA v1 的显式 model ID/revision/hash 由其 converter 写入并校验。

可先用注册表锁定的 fetch helper 获取并验证源 checkpoint：

```sh
python3 -m pip install -r requirements-fetch.txt
CHECKPOINT="$(python3 tools/evo_fetch.py --print-path source evo2_7b)"
```

cache、离线模式、runtime artifact manifest 与 release provenance 见
[`artifact-distribution.md`](artifact-distribution.md)。

## Arc `.pt`

创建独立转换环境：

```sh
python3 -m venv .venv-convert
. .venv-convert/bin/activate
python3 -m pip install -r requirements-convert.txt
```

通过注册表 wrapper 转换：

```sh
scripts/convert_arc_checkpoint.sh \
  evo2_7b evo2_7b.pt evo2-7b.safetensors
```

或直接指定 config：

```sh
PYTHONPATH=tools python3 tools/convert_checkpoint.py \
  --input CHECKPOINT.pt \
  --config configs/REGISTERED_CONFIG.yml \
  --output MODEL.safetensors \
  --source-sha256 CHECKPOINT_SHA256
```

设置 `EVO_SOURCE_SHA256` 可通过 wrapper 记录已验证的源 hash；
`EVO_DRY_RUN=1` 只验证而不写出。split checkpoint 必须先按 `.part0`,
`.part1`, … 数字顺序合并，直接传入 part 会被拒绝。

转换器固定使用：

```python
torch.load(path, map_location="cpu", mmap=True, weights_only=True)
```

它严格校验完整 tensor manifest、连续 storage、dtype/shape 和 TE extra state。
BF16 norm/rope 小向量可精确 widen 到 F32；Arc software-FP8 projection
按 chunk 直接编码为最终 `F8_E4M3`，不会在 host 内存中同时物化完整源权重和
完整结果。20B 的 projection 源权重虽为 F32，但官方 Vortex 会先把它加载进
BF16 parameter，再由 Transformer Engine 转为 E4M3；converter 明确保留这次
F32 → BF16 舍入。1B/40B 的 projection 源权重本身就是 BF16。

默认按 4096 MiB tensor payload 贪心分片。`--output MODEL.safetensors` 是
命名基准；大模型实际生成
`MODEL-00001-of-000NN.safetensors` 和
`MODEL.safetensors.index.json`。可用 `--max-shard-mib` 调整目标大小；它不会
拆开单个 tensor，也不会按 layer 或 GPU 建立特殊分片。

## BioNeMo BF16 40B

```sh
PYTHONPATH=tools python3 tools/convert_bionemo_checkpoint.py \
  --input /path/to/checkpoint/weights \
  --config configs/evo2-40b-1m-bionemo-bf16.yml \
  --output evo2-40b-bionemo-bf16.safetensors \
  --source-sha256 544b47e033d1fb0261b686a53f7c4fe240cd290253187d31e8c99dea9e35a680
```

该 checkpoint 是 NVIDIA 独立微调的 BF16 权重，不是 Arc 40B 删除 FP8 state
后的等价文件。converter 验证 DCP mapping、拆分 FC1、tie embedding/unembedding
并生成 runtime 所需的 HCM/HCL F32 tensor。

## 检查与运行

```sh
build/evo-inspect MODEL.safetensors.index.json
build/evo-inspect MODEL.safetensors.index.json --tensor norm.scale

scripts/validate_model.sh MODEL.safetensors.index.json 0,1 8192
```

发布大型 artifact 时，在仓库外计算并保存 SHA256：

```sh
sha256sum MODEL-*.safetensors MODEL.safetensors.index.json \
  > MODEL.safetensors.sha256
sha256sum -c MODEL.safetensors.sha256
```

仓库只保存 source lock、转换代码和格式/profile 测试，不保存 checkpoint、
转换结果或易失的本地绝对路径。

## Tokenizer asset

非内建 tokenizer 使用独立、可审计的离线编译步骤。输入 manifest 描述 tokenizer
来源和可表示语义；source receipt 为每个 `tokenizer.json`、`vocab.txt` 或自定义
spec 固定 size/SHA-256：

```sh
python3 tools/convert_tokenizer_asset.py \
  --manifest tokenizer-compiler-manifest.json \
  --receipt tokenizer-source-receipt.json \
  --output tokenizer.evo.json \
  --descriptor tokenizer-descriptor.json \
  --asset-path assets/tokenizer.evo.json
```

输出 descriptor 包含 converter/manifest/receipt contract provenance，以及可直接写入模型
artifact metadata 的 `tokenizer.profile/path/sha256/size` 四项。模型 converter 只
复制这四项，不把 tokenizer converter 的通用 `converter.*` 键混入模型自身的
provenance namespace。`source_receipt_contract_sha256` 在验证真实 locator 后只
hash schema/kind 与每个文件的 role/name/size/content SHA，因此 cache 绝对根不
影响 descriptor 或模型 artifact。writer 使用 canonical JSON + trailing newline 与同目录
atomic publish；已有输出除非显式 `--force` 不会覆盖。

当前 compiler 支持严格子集的 Hugging Face BPE、WordPiece、`use_regex=false` 的
byte-BPE、vocab-text k-mer，以及显式 character/single-nucleotide、mixed/BioToken
longest-match 和 k-mer+BPE spec。不能精确表示的 normalizer、pre-tokenizer、
AddedToken 或 regex 行为会 typed fail，必须先提供显式自定义 spec 和 oracle，
不能静默近似。

## HyenaDNA Hugging Face Safetensors

HyenaDNA 转换不需要 PyTorch 或 `safetensors` Python package；converter 严格读取
上游 Safetensors header、F32 tensor manifest 与 `config.json`，再写独立的
`hyenadna-runtime-v1` typed artifact：

```sh
PYTHONPATH=tools python3 tools/convert_hyenadna_checkpoint.py \
  --input model.safetensors --config config.json \
  --output hyenadna.safetensors \
  --model-id LongSafari/hyenadna-tiny-1k-seqlen-hf \
  --revision e8c1effa8673814e257e627d2e1eda9ea5a373f6 \
  --source-sha256 5ce2146c21e9c4baa6bddc4998fd3d029903ae84a563bf80218644082194a12d

build/evo-inspect hyenadna.safetensors
build/evo -m hyenadna.safetensors --backend cpu --ctx 1026 \
  --score sequences.fa
```

首版接受 official F32 causal-LM tensor layout、order 2、3-tap short filter、
12-token source vocabulary padded 到 16，以及 `max_seq_len ≤ 4096`。其他
topology、dtype、缺失/额外 tensor 或 source hash 不一致均 typed fail。完整
runtime/tokenizer 边界见 [architecture registry](architectures.md)。

## Biohub ESMC Hugging Face Safetensors

ESMC 300M、600M 与 6B 使用独立 `esmc-runtime-v1` profile。转换器不依赖
PyTorch 或 Python `safetensors` package；输入必须是 `evo-fetch source` 生成的
hash-verified receipt：

```sh
python3 tools/evo_fetch.py source esmc_300m > fetch.json
RECEIPT="$(python3 -c 'import json; print(json.load(open("fetch.json"))["receipt"])')"
PYTHONPATH=tools python3 tools/convert_esmc_checkpoint.py \
  --receipt "$RECEIPT" --output esmc-300m.safetensors
```

converter 在写出前校验 pinned repo/revision、全部文件 hash、config、tokenizer、
完整 F32 tensor manifest 及预期的 zero-byte U8 `_extra_state`。6B 自动产生标准
Safetensors shards/index。架构、模型 ID、权重名与运行命令见
[ESMC native inference](esmc.md)。
