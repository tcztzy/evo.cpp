# SPEC

## 目标

让 Evo 2 在没有原生 FP8 的 CUDA 设备上完成 batch-1 推理。生产 runtime
只接受 `evo2-runtime-v1` Safetensors，不依赖 Python、PyTorch、libtorch、
Vortex、Transformer Engine 或硬件 FP8 指令。

上游 checkpoint 更新时，只更新
[`configs/model-registry.json`](configs/model-registry.json)、必要的 tensor
映射并重新转换；仓库不维护另一套私有容器协议。

## 边界

- 支持注册表内的官方 Evo 2 1B、7B、20B、40B 及现有变体。
- 支持 sequence scoring 和 generation，首要场景是 batch=1、1–4 张 CUDA GPU。
- `.pt` 和 BioNeMo DCP 只是离线转换输入。
- Python/PyTorch 只存在于 `tools/` 转换环境，不进入生产进程。
- native runtime 为 C++17/CUDA，使用 cuBLASLt 和 cuFFT。
- 7B 与 BioNeMo BF16 使用 BF16；Arc 1B/20B/40B 的指定 Hyena projection
  使用软件 E4M3。
- 首要非 Hopper 目标为 Ampere A100/A800。

非目标：训练、微调、通用模型归档、HTTP 服务、跨节点并行、压缩与 batch>1。

## 唯一生产输入

```sh
evo -m model.safetensors.index.json -p ACGT -n 100 --ctx 8192 --gpu 0,1
evo -m model.safetensors.index.json --score sequences.fasta --gpu 0,1
evo-inspect model.safetensors.index.json
```

runtime 不直接读取 `.pt`，也没有 Python/libtorch fallback。输入不是严格 profile、
模型签名不匹配、dtype 不支持或显存不足时必须明确失败。

## 离线转换

Arc 转换器以以下安全边界读取可信 `.pt`：

```python
torch.load(path, map_location="cpu", mmap=True, weights_only=True)
```

转换器必须：

- 按注册表检查模型 ID、来源 revision、文件大小/SHA256 和完整 tensor manifest；
- 检查 tensor 名、dtype、shape、layout、storage offset 和 payload 范围；
- 接受 storage offset 非零但连续的 tensor，拒绝非连续 tensor；
- 只忽略 manifest 明确列出的可重建数据；
- 验证 Transformer Engine 的 `scale_fwd`、`scale_inv_fwd` 和
  `amax_history_fwd`，不能静默退回 BF16；
- 以有界 chunk 流式读取和写入，临时文件完成后 `fsync` 并原子发布。

大 checkpoint 和转换结果不提交到 Git。注册表锁定源 artifact；发布后的
Safetensors 用外部 SHA256/sidecar 锁定。

## Safetensors runtime profile

磁盘格式是标准 Safetensors：

```text
u64 little-endian JSON header length
8-byte-aligned UTF-8 JSON header
dense tensor data buffer
```

严格子集与 typed metadata 编码见
[`docs/model-format.md`](docs/model-format.md)。

核心约束：

- 默认按 4 GiB payload 贪心分片，使用标准 `model-00001-of-000NN.safetensors`
  和 `model.safetensors.index.json` 命名；小模型可为单文件；
- 分片只看大小和 tensor 顺序，不按 Evo 2 layer、GPU 或模型大小命名；
- tensor data_offsets 相对各 shard 的 data buffer；
- payload 无洞、无重叠，完整覆盖文件尾；
- 只接受 `F32`、`BF16`、`F8_E4M3`；
- metadata 必须包含 `evo2.profile=s:evo2-runtime-v1` 和
  `runtime.abi=s:evo2-safetensors-v1`；
- global tensor 先写，随后按 layer/native load order 写；
- 软件 FP8 projection 在转换时写成最终 `F8_E4M3` code；
- 每个软件 FP8 projection 只额外保留相邻的 F32 `[input_scale,
  output_scale]`；
- 不存 pickle、训练状态、源 FP8 history、压缩数据或 GPU 专属分片。

标准格式负责互操作和 mmap；真实性由外部 artifact SHA256 负责。启动时不为
内嵌 checksum 再扫描一次全部权重。

## Native 加载

1. 只读 mmap 文件，解析并严格校验 header、metadata、shape、size 与所有范围。
2. 用 tensor payload bytes 规划连续 layer 到 1–4 张 GPU。
3. 直接把最终 BF16/F32/F8_E4M3 payload 上传到目标设备。
4. 所有层加载成功后发布 model ready 状态。

加载过程中禁止启动 Python/libtorch或重新量化完整 projection。runtime 可使用
软件 E4M3 kernel 解码/计算，但不能要求设备支持原生 FP8。

## 正确性与验收

- byte tokenizer 保持 byte→同值 token id，EOS=0、PAD=1、vocab=512。
- E4M3 编码必须与 `float8_e4m3fn` bit pattern 一致。
- 官方 BF16 7B 的精确模式必须与固定 Vortex/PyTorch reference 的 BF16/F32
  位型逐元素一致；不得用 cosine 或 top-1 代替。paged-Q8、software-FP8 与
  BioNeMo 路径保留各自明确声明的误差门禁，不能冒充该 bit-exact 模式。
- 同一 build、checkpoint 和 seed 的 greedy 输出 byte-identical。
- cuBLASLt 与 cuFFT 复用测试必须直接断言 plan build/generation 计数，不能用
  wall-clock 猜测 cache 是否命中。
- 7B 性能 gate 只在空闲 GPU 上运行，固定 checkpoint/input，分别记录首次与
  repeated prefill，并把阈值、官方对比和 artifact hash 写入 JSON。
- parser 测试覆盖 malformed JSON、错误 profile/dtype/shape、overflow、洞、
  重叠、截断和尾随 payload。
- converter 测试覆盖 mmap、restricted unpickling、非零 storage offset、
  非连续 storage、FP8 extra state、最终 E4M3 code 和原子失败。
- 真实 7B BF16 与至少一个 Arc software-E4M3 模型必须在非 Hopper GPU
  上通过 score、logit 和 greedy gate。
- 生产进程的动态依赖和调用链中不得出现 Python、PyTorch、libtorch、Vortex
  或 Transformer Engine。
