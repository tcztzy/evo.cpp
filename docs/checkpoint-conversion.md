# Evo 2 checkpoint conversion

生产 runtime 只读取 `evo2-runtime-v1` Safetensors。Arc `.pt` 与 BioNeMo DCP
都在离线环境中转换；推理进程不加载 Python、PyTorch 或 libtorch。

模型、revision、checksum、source/runtime manifest 的机器可读来源是
[`configs/model-registry.json`](../configs/model-registry.json)。

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
完整结果。

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
