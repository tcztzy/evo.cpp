# evo.cpp

**不用 Python 运行 Evo 2；对 7B 而言，连一个 bit 的数值漂移都不接受。**

[English](README.md) | **简体中文**

`evo.cpp` 是专注于 batch-1 Evo 2 推理的 C++17/CUDA runtime。它从标准
Safetensors checkpoint 运行官方 1B、7B、20B 和 40B 模型家族，支持 1–4 张
NVIDIA GPU；推理过程不依赖 PyTorch、Vortex、Transformer Engine、Python 或
硬件 FP8。

项目只解决一个问题：让 Evo 2 更容易部署，同时**绝不把数值相似冒充正确**。

## 为什么选择 evo.cpp？

- **7B 位级精确。**审计范围内的每个 BF16/F32 bit 都与固定版本的
  Vortex/PyTorch reference 相同，包括全部 logits、全部 32 层 block 输出、
  cached generation、2,048-key softmax 边界，以及官方 3,000-token prompt
  forcing 切换点。这里没有 cosine、容差或 top-1 兜底。
- **足够小的原生 runtime。**推理进程是为 Evo 2 定制的 C++/CUDA，而不是一套
  Python framework。它严格加载有界 Safetensors，并提供 scoring、generation
  和白盒 tensor dump。
- **精确不等于放弃性能。**单张 A800 上，最终 exact 7B 路径在
  16 / 128 / 1,024-token prefill 下达到
  1,071.5 / 7,619.2 / 11,369.3 tok/s；在同一组对比中分别是预热后官方
  runtime 的 1.80× / 1.60× / 1.21×。
- **在 Ampere 上保留原版 Evo 2 语义。**Arc 1B/20B/40B checkpoint 依赖
  Transformer Engine FP8 行为。`evo.cpp` 用软件复现其固定 E4M3 和
  H100-QGMMA 累加语义，因此部署不受 Hopper FP8 指令限制。

设计刻意保持克制：一种架构、明确的数值契约、原生 kernel，以及用显式失败代替
静默近似。它更接近 `llama.cpp` 和 `ds4.c`，而不是通用训练框架。

## 用证据说话

| 主张 | 实测结果 |
|---|---|
| BF16 7B exactness | 全部审计 prefill block 与 logits、16-step cached generation、长 softmax dispatch 边界及 CUDA 12.8/13.3 对比均为零 raw-bit 差异 |
| Exact 7B prefill，1×A800 | 16 / 128 / 1,024 tokens 下为 1,071.5 / 7,619.2 / 11,369.3 tok/s；预热后的官方 Vortex 为 595 / 4,776 / 9,412 tok/s |
| Exact 7B cached decode，1×A800 | 中位数 72.62 tok/s |
| 原版 Arc 40B 质量门禁，4×A800 | 500-base identity 为 99.2%、97.2%、78.2%、99.2%，均值 93.45%；prompt 0 与独立 oracle byte-identical |
| BioNeMo BF16 40B，2×A800 | 历史同 checkpoint 测试中，8-token generation 为 1.288 s 对 8.710 s，每卡记录峰值约 45.4 GB 对 55.546 GB |

以上是文档所述 A800 环境中的 batch-1 结果，不是可外推到所有环境的厂商基准。
40B 数字产生于当前 Safetensors-only runtime 之前，因此明确标为历史验证。完整
命令、hash、first-divergence 分析和测试边界见
[7B bit-exact 审计](docs/vortex-7b-bit-exactness.md)和
[GPU 验证记录](docs/gpu02-environment.md)。

## 快速开始

PyTorch 只用于独立的离线转换环境；原生 runtime 不会加载它。

```sh
python3 -m venv .venv-convert
. .venv-convert/bin/activate
python3 -m pip install -r requirements-convert.txt

scripts/convert_arc_checkpoint.sh \
  evo2_7b /models/evo2_7b.pt /models/evo2-7b.safetensors
```

构建 CUDA runtime：

```sh
cmake -S . -B build \
  -DEVO_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build -j
```

检查、生成或评分：

```sh
MODEL=/models/evo2-7b.safetensors.index.json

build/evo-inspect "$MODEL"
build/evo -m "$MODEL" -p ACGT -n 32 --ctx 8192 --gpu 0
build/evo -m "$MODEL" --score sequences.fa --ctx 8192 --gpu 0
```

converter 使用安全的 CPU mapping，逐一验证注册 tensor 的名称、shape 和 dtype，
再写入标准的按大小分片 Safetensors。缺失、重复或未知 tensor 都会直接报错。详见
[转换指南](docs/checkpoint-conversion.md)和[格式契约](docs/model-format.md)。

## 支持路径

| 模型路径 | 数值契约 | ctx 8K 建议起点 | 验证状态 |
|---|---|---:|---|
| 官方 7B | BF16，与固定 Vortex/PyTorch 语义 bit-exact | 1×24 GB | 真实 checkpoint 已转换、benchmark，并在 A800 上完成位级审计 |
| Arc 1B / 20B | 训练时 FP8 projection 使用软件 E4M3 | 1×16 GB / 2×40–48 GB | registry、转换、拓扑和 kernel 契约已测试；真实 checkpoint GPU 运行待完成 |
| 原版 Arc 40B | 软件 E4M3 与 H100-QGMMA 累加模拟 | 2×80 GB | legacy runtime 已在 4×A800 验证；Safetensors 重跑待完成 |
| BioNeMo BF16 40B | BF16 | 2×80 GB | legacy runtime 已在 2×A800 对齐；Safetensors 重跑待完成 |

registry 包含官方 base 和长上下文变体。硬件数字是保守起点而不是硬下限；实际显存
还受上下文长度、KV 格式、driver overhead 和其他 GPU 进程影响。

## 保证的边界

- **“Exact”是刻意收窄的承诺。**它覆盖官方 BF16 7B checkpoint 和审计固定的
  reference。软件 FP8、BioNeMo 40B、Q8 KV、sampling 和其他 GPU 架构有各自的
  验证结论。
- **8K 是生产质量门禁。**chunked prefill 和 paged Q8 KV 支持更大上下文；
  131K 与 1M 结果是容量和短 decode smoke，不代表完整上下文 prefill 已经高效。
- **这是 inference runtime。**训练、微调、LoRA、分布式服务、speculative
  decoding 和稳定 library ABI 不在范围内。
- **不支持的情况会明确失败。**runtime 不会偷偷把 exact 路径切换到数学上近似的
  实现。

## 深入实现

- [7B 如何精确到第一个发生差异的 primitive](docs/vortex-7b-bit-exactness.md)
- [软件 E4M3 如何在 Ampere 上保留 Arc checkpoint 语义](docs/software-fp8.md)
- [模型数值契约](docs/math-semantics.md)
- [Safetensors 格式与转换](docs/model-format.md)
- [可复现 GPU 环境与验证 artifact](docs/gpu02-environment.md)
- [架构和验收标准](SPEC.md)

`evo.cpp` 是独立 runtime，不属于 Arc Institute 或 NVIDIA。模型架构和 checkpoint
来自 [Arc Institute 官方 Evo 2 项目](https://github.com/arcinstitute/evo2)。仓库
代码使用 Apache-2.0；模型权重仍遵循其上游许可证。
