# evo2c

[English](README.md) | **简体中文**

`evo2c` 是覆盖全部官方 Evo 2 尺寸（1B、7B、20B、40B，以及受支持的 base/
长上下文变体）的独立 C++17/CUDA 推理实现。batch-1 推理可使用 1–4 张 CUDA
GPU；推理进程不依赖 PyTorch、Vortex、Transformer Engine，也不需要硬件 FP8
指令。

项目思路接近 `llama.cpp` 和 `ds4.c`：只实现一个经过检查的模型容器、一种模型
架构、必要的原生 kernel、离线权重转换和可复现的数值测试。它不是新的生物学模型，
也没有重新训练或修改 checkpoint；它解决的是 Evo 2 在非 FP8 硬件上的推理问题。

## 支持与验证矩阵

每一行都已通过本地 topology、source manifest、converter、native registry 与
corruption 测试。“GPU 已验证”只表示真实转换 checkpoint 已在 CUDA 上实际运行
或对齐；单独的 Python/Vortex reference vector 不算 native runtime 验证。

| Model ID | Config | Projection 语义 | ctx 8K 建议起点 | 真实 checkpoint native 状态 |
|---|---|---|---|---|
| `evo2_1b_base` | `evo2-1b-8k.yml` | 固定 TE 2.3 software E4M3；source weight BF16 | 1×16 GB | 尚未运行（本地无 checkpoint/GPU） |
| `evo2_7b` | `evo2-7b-1m.yml` | BF16 | 1×24 GB | 尚未运行；现有 7B vector 仅是 reference evidence |
| `evo2_7b_base` | `evo2-7b-8k.yml` | BF16 | 1×24 GB | 尚未运行 |
| `evo2_7b_262k` | `evo2-7b-262k.yml` | BF16 | 1×24 GB | 尚未运行 |
| `evo2_20b` | `evo2-20b-1m.yml` | 固定 TE 2.3 software E4M3；source weight F32 | 2×40–48 GB | 尚未运行 |
| `evo2_40b` | `evo2-40b-1m.yml` | 固定 TE 2.3 software E4M3 | 2×80 GB | 已在 4×A800 80 GB 验证 |
| `evo2_40b_base` | `evo2-40b-8k.yml` | 固定 TE 2.3 software E4M3 | 2×80 GB | 尚未运行 |
| `evo2_40b_bionemo_bf16` | `evo2-40b-1m-bionemo-bf16.yml` | BF16 | 2×80 GB | 已在 2×A800 80 GB 验证并对齐 |

以上是保守起点，不是 runtime 硬编码下限。pipeline 按 layer payload bytes 选择
连续边界；实际显存还受 context、KV 格式、driver overhead 和其他进程影响。

## 给科研使用者的结论

- **与 BioNeMo 对齐：**使用同一 NVIDIA
  `evo2/40b-1m-fp8-bf16:1.0` checkpoint 时，16×512 个 logits 全部有限，
  16/16 行 top-1 完全一致，8-token greedy continuation 逐字节一致。
- **数值差异很小且可解释：**最低 row cosine 为 `0.999998748`；
  8192 个 logits 中 2723 个完全相同，平均绝对差为 `0.112738`，而 logit
  的最大绝对值为 `23.625`。15 个实际 target 的平均绝对 log-probability
  差仅为 `0.008972 nat/token`。
- **gpu02 上观测到更低的延迟和显存：**同为 2×A800、同一 BF16 checkpoint
  的 8-token greedy 测试中，BioNeMo 2.4 用时 `8.71 s`，evo2c 用时
  `1.288 s`，即 BioNeMo 用时为 evo2c 的 `6.76×`；每卡峰值记录由
  `55.546 GB` 降至约 `45.4 GB`。
- **原始 Arc checkpoint 也经过独立验证：**4 个官方长 prompt 各生成
  500 bases，identity 为 `99.2%`、`97.2%`、`78.2%`、`99.2%`，
  平均 `93.45%`；其中 prompt 0 与独立 Vortex software-H100-QGMMA
  oracle 逐字节相同。
- **边界明确：**`--ctx 8192` 是质量门禁覆盖的生产路径；131K 和 1M 是
  KV 容量与短 decode 数值 smoke，不代表当前实现能高效 prefill 完整 1M prompt。

这些结果说明当前实现适合在已验证配置上做 sequence scoring 和 generation。
它们不等于“所有下游生物学任务都已经等价验证”；正式研究仍应在自己的任务、
序列分布和统计指标上做一次小规模交叉验证。

## evo2c、Arc/Vortex 与 BioNeMo 的关系

| 项目 | 提供什么 | 40B 权重 | 典型软件栈 | 非 Hopper 支持 |
|---|---|---|---|---|
| [Arc Institute Evo 2](https://github.com/arcinstitute/evo2) | 原始模型、checkpoint、论文与研究接口 | 原始 Arc 40B/40B-base | Python + Vortex | 官方说明 40B 的数值准确性依赖 Hopper FP8 |
| [Vortex](https://github.com/Zymrael/vortex) | Arc 使用的 StripedHyena 2 推理与数值实现 | 读取原始 Arc 权重 | PyTorch + Transformer Engine，可选 FlashAttention | 7B 可纯 BF16；40B 需要 Transformer Engine/FP8 |
| [NVIDIA BioNeMo](https://docs.nvidia.com/bionemo-framework/latest/models/evo2/index.html) | Evo 2 训练、微调、预测和推理框架 | 原始权重及 NVIDIA 微调变体 | PyTorch + NeMo/Megatron + Transformer Engine | NVIDIA 微调的 40B checkpoint 原生支持 Ampere+ BF16 |
| **evo2c** | 面向官方 1B/7B/20B/40B profile 的轻量原生 runtime | 支持 Arc 变体和 BioNeMo BF16 40B | C++17 + CUDA + cuBLASLt/cuFFT | Arc 1B/20B/40B 用 software E4M3；7B 与 BioNeMo 用 BF16 |

[Arc 官方说明](https://github.com/arcinstitute/evo2#requirements)指出原始 1B、
20B、40B 对训练时的 Vortex-style FP8 数值语义敏感，直接改成 BF16 可能损害
生物学准确性；7B 是例外。[NVIDIA 的说明](https://docs.nvidia.com/bionemo-recipes/latest/main/examples/bionemo-evo2/examples/fine-tuning-tutorial/index.html#fp8-and-hardware-compatibility)
也给出相同结论，并提供了为 Ampere+ FP8/BF16 微调的
`evo2/40b-1m-fp8-bf16:1.0`。NVIDIA 同时注明，该 40B 变体相对原始模型在
Hopper FP8 上有轻微准确率回退。

因此两条路径不能混为一谈：

1. **希望在 gpu02/A800 上直接使用稳定的 BF16 40B：**选择 BioNeMo
   `evo2/40b-1m-fp8-bf16:1.0`。这是推荐路径，也是下面 BioNeMo 性能与
   logits 对比使用的 checkpoint。
2. **希望复现 Arc 原始 40B 权重和 Vortex FP8 语义：**选择
   `arcinstitute/evo2_40b`。evo2c 在 Ampere 上以 software E4M3 模拟
   checkpoint 固定的 Transformer Engine 2.3 scale 和 H100 QGMMA 累加语义。
3. **不要直接比较两种 checkpoint 的 score：**BioNeMo 40B 是 NVIDIA
   微调过的独立权重，不是原始 Arc 权重的另一种文件格式。跨 checkpoint
   的 score 差异包含模型权重差异，不能归因于 runtime。

## 与 BioNeMo 的同 checkpoint 对比

### 测试条件

| 项目 | BioNeMo oracle | evo2c |
|---|---|---|
| Checkpoint | `evo2/40b-1m-fp8-bf16:1.0` | 同一 checkpoint 转换为只读 `.evo2` |
| GPU | 2×A800 80GB | 同一台 gpu02 的 2×A800 80GB |
| 数值路径 | BF16 | BF16 |
| 并行方式 | Megatron tensor parallel，TP=2 | 25+25 层 pipeline，PP=2 |
| Batch/context | batch 1，`ctx=8192` | batch 1，`ctx=8192` |
| Greedy prompt | `ACGTACGTACGTACGT` | 相同 |
| 输出 tokens | 8 | 8 |

BioNeMo oracle 使用 BioNeMo Recipes 2.4、Megatron Bridge 0.4.1、
Megatron Core 0.17.0rc0、PyTorch `2.13.0a0+8145d630e8.nv26.06`、
Transformer Engine `2.16.0+4220403e` 和 CUDA 13.3。evo2c 使用 CUDA
12.8.93。完整环境和 artifact hash 见
[`docs/gpu02-environment.md`](docs/gpu02-environment.md)。

### 性能列表

| 指标 | BioNeMo 2.4 | evo2c | 观测差异 |
|---|---:|---:|---:|
| 8-token generation time，不含模型加载 | 8.710 s | 1.288 s | evo2c `6.76×` 更短 |
| 端到端 output rate | 报告 0.9 tok/s | 6.21 tok/s | 约 `6.9×` |
| 稳态 cached decode | 未单独报告 | 8.248 tok/s | — |
| 16-token prefill | 未单独报告 | 0.440 s / 36.394 tok/s | — |
| 每卡峰值记录 | 55.546 GB | 45.393 / 45.376 GB | 约少 `18.3%` |
| 模型加载 | 未单独报告 | 54.878 s | — |

evo2c 的 generation time 为 `0.439633 s` prefill 加 `0.848684 s`
的 7 个 measured decode steps；8 个输出 token 除以总时间得到
`6.21 tok/s`。BioNeMo 的 `0.9 tok/s` 是其日志报告值，`6.76×` 则直接由
`8.710 / 1.288` 计算。

这是一组**同服务器、同 GPU 型号、同 checkpoint 的观测对比**，但不是严格的
vendor benchmark：两次运行时服务器上的其他任务不完全相同，并行策略也不同；
BioNeMo 的显存数字是框架报告的 per-rank peak，evo2c 数字是进程的 CUDA
allocation delta。它适合说明本次部署的实际量级，不应外推成所有长度、GPU 和
batch size 下都固定快 `6.76×`。

## logits 差异列表

### 总体差异

score 输入为 `ACGTGATTACAACGTT`，两端都输出 `[16, 512]` F32 NPY。
下表中的 signed delta 定义为 `evo2c - BioNeMo`。

| 指标 | 数值 |
|---|---:|
| 元素总数 | 8192 |
| 完全相同 | 2723（33.24%） |
| 全局平均绝对差 | 0.112738370 |
| 全局 RMS 差 | 0.164874028 |
| 最大绝对差 | 0.5 |
| 两端最大绝对 logit | 23.625 |
| 最低 row cosine | 0.999998748028 |
| top-1 一致 | 16/16 |
| 所有值 finite | 是 |

最常见的逐元素 signed delta 为：

| `evo2c - BioNeMo` | 元素数 | 占 8192 比例 |
|---:|---:|---:|
| `-0.125` | 2946 | 35.96% |
| `0` | 2723 | 33.24% |
| `+0.125` | 1459 | 17.81% |
| `-0.25` | 479 | 5.85% |
| `-0.5` | 476 | 5.81% |
| `-0.375` | 28 | 0.34% |
| `+0.25` | 10 | 0.12% |
| 其他 BF16 网格小差值 | 71 | 0.87% |

这些离散的 `0.125` 倍数是 BF16 在该数值尺度上的典型量化间隔。它们说明差异
是有限精度计算路径逐层累积的结果，而不是随机内存错误或 NaN。

### 逐位置差异

`target` 是 score 时下一个真实 token；最后一行没有下一个 target，但保留该
位置 logits 以便完整比较。`exact` 表示该行 512 个 logits 中数值完全相等
的个数。

| row | input | target | BioNeMo top-1 | evo2c top-1 | cosine | max abs | mean abs | exact / 512 |
|---:|:---:|:---:|:---:|:---:|---:|---:|---:|---:|
| 0 | A | C | T | T | 0.999999052520 | 0.25 | 0.240627 | 1 |
| 1 | C | G | T | T | 0.999998748028 | 0.125 | 0.115118 | 38 |
| 2 | G | T | A | A | 0.999999388966 | 0.5 | 0.487663 | 0 |
| 3 | T | G | T | T | 0.999999602599 | 0.25 | 0.125381 | 6 |
| 4 | G | A | T | T | 0.999999762142 | 0.125 | 0.001856 | 501 |
| 5 | A | T | A | A | 0.999999212200 | 0.125 | 0.116225 | 33 |
| 6 | T | T | A | A | 0.999999460517 | 0.125 | 0.004314 | 490 |
| 7 | T | A | T | T | 0.999999872079 | 0.125 | 0.001130 | 504 |
| 8 | A | C | T | T | 0.999999650671 | 0.125 | 0.120406 | 14 |
| 9 | C | A | A | A | 0.999999423607 | 0.125 | 0.117972 | 26 |
| 10 | A | A | T | T | 0.999999138835 | 0.125 | 0.115395 | 35 |
| 11 | A | C | A | A | 0.999999991730 | 0.0625 | 0.000132 | 508 |
| 12 | C | G | A | A | 0.999999855896 | 0.125 | 0.001003 | 504 |
| 13 | G | T | A | A | 0.999999210265 | 0.125 | 0.117914 | 25 |
| 14 | T | T | G | G | 0.999999491825 | 0.125 | 0.119436 | 19 |
| 15 | T | — | G | G | 0.999999469196 | 0.125 | 0.119240 | 19 |

row 2 看起来最大，但该行的 mean signed delta 是 `-0.487434`，接近所有
logits 一起平移约 `-0.5`。softmax 对完全相同的整行常数平移严格不变，因此
`max abs = 0.5` 不能直接解释为概率误差很大。实际概率影响应看
log-softmax，而不是只看 raw logit 的最大差。

### 对 sequence score 的实际影响

对同一 15 个 target token 做 stable log-softmax：

| 指标 | BioNeMo | evo2c | 差异 |
|---|---:|---:|---:|
| 总 log-likelihood | -20.7778942673 | -20.7006628617 | +0.0772314056 |
| 平均 log-likelihood | -1.3851929512 | -1.3800441908 | +0.0051487604 |
| Perplexity | 3.99560 | 3.97508 | -0.02052 |

15 个 token 的平均绝对 log-probability 差为 `0.0089722188 nat/token`，
最大为 `0.0607365483 nat`。这比 raw logits 的 `0.5` 最大差更能反映
score 的真实影响。该短序列只是数值回归向量，不是生物学 benchmark；
研究结论仍应使用足够大的任务数据集和置信区间。

另一个独立的 generation 检查使用 prompt `ACGTACGTACGTACGT`：
BioNeMo 与 evo2c 都 greedy 生成 `ACGTACGT`，8 bytes 完全相同。

## 为什么 cosine 不是 1

cosine 只有在两个向量方向完全一致时才等于 1。当前最低值
`0.999998748028` 对应的 `1 - cosine` 只有约 `1.25×10⁻⁶`，说明两个
512 维向量方向几乎相同，但并非逐元素完全相同。

已经确认的事实：

1. **不是换了 checkpoint。**对比使用同一个 NGC archive；转换器对普通 BF16
   weight payload 做 bit-exact 搬运，并严格检查 506 个源 tensor 到 537 个
   输出 tensor 的名称、形状和 dtype 映射。
2. **不是最后一次简单 cast。**BioNeMo 导出的 final logits 文件是 F32，
   但其中每个值都已能由 BF16 精确表示；evo2c logits 也在 BF16 网格上。
   将 BioNeMo 最终结果再 round 一次 BF16 并不能得到 evo2c 结果。
3. **差异不改变该测试的离散决策。**16/16 top-1 相同，greedy continuation
   逐字节相同，所有 logits 都是 finite。

最可能的差异来源按执行顺序列出：

1. **并行分解不同。**BioNeMo oracle 使用 TP=2，把 GEMM 分片后做 collective
   reduction；evo2c 使用 PP=2，每一层的完整 GEMM 只在所属 GPU 上计算。
   浮点加法不满足结合律，分片和 reduction 顺序会改变末位。
2. **GEMM tiling 与 epilogue 不同。**BioNeMo 通过 PyTorch、
   Transformer Engine/Megatron 的 fused kernel；evo2c 通过 cuBLASLt，
   BF16 输入、FP32 accumulation、BF16 layer output。两者 reduction tree、
   tile 大小和 bias/epilogue 融合位置不同。
3. **归一化 reduction 不同。**evo2c 的 BF16 RMSNorm 是自定义 warp reduction；
   Transformer Engine 使用自己的 fused 实现。平方和顺序的微小变化会影响
   normalization scale。
4. **Hyena 与 attention kernel 不同。**FIR/IIR filter、FFT、RoPE、online
   softmax 和 cache 读写均为独立实现。每层产生的少数 ULP 差异会在 50 个
   StripedHyena 2 block 中累积。
5. **逐层 BF16 写回。**即使 GEMM 内部都以 FP32 累加，各层边界写回 BF16
   时仍会舍入；上一步末位差异可能让下一层落到相邻 BF16 值。

当前能严格确定差异已出现在 final logits，但**还不能声称第一个分歧精确发生在
哪一层或哪一个算子**：官方 oracle 只导出了 final logits，没有导出 50 层内部
activation。若要定位首个分歧，需要给 BioNeMo 增加只读 layer hooks，并与
evo2c 的 `--dump-layer` 依次比较 `pre_norm`、mixer output、residual、
post-norm、MLP 和 block output。README 把这一点明确保留为未完成的更细粒度
诊断，而不是把“最可能来源”写成已经证明的唯一原因。

## 原始 Arc/Vortex checkpoint 的质量门禁

以下结果使用 `arcinstitute/evo2_40b` 原始权重和 evo2c software E4M3 路径，
与上面的 BioNeMo BF16 checkpoint **不是同一组权重**。每个 prompt 先并行
prefill 3000 bytes，再 teacher-force 剩余 prompt，最后 greedy 生成 500 bases。

| 官方 prompt | Prompt bytes | Generated bases | 对官方 target 的 identity |
|---:|---:|---:|---:|
| 0 | 3268 | 500 | 99.2% |
| 1 | 3528 | 500 | 97.2% |
| 2 | 3080 | 500 | 78.2% |
| 3 | 3808 | 500 | 99.2% |
| **平均** | — | 2000 | **93.45%** |

固定 gate 为 `91.15 ± 3` percentage points，当前结果通过。prompt 0 的
500-base continuation 与独立 Vortex software-H100-QGMMA oracle
逐字节相同；同一 binary 的重复短 generation 也逐字节相同。

原始 Arc 路径在空闲的 4×A800 上，prompt 0 的观测性能为：

| 阶段 | 工作量 | 时间 | 吞吐 |
|---|---:|---:|---:|
| Model load | 40B | 60.290 s | — |
| Parallel prefill | 3000 tokens | 135.261 s | 22.179 tok/s |
| Teacher force | 268 tokens | 99.037 s | 2.706 tok/s |
| Cached decode | 9 tokens | 3.484 s | 2.583 tok/s |

这条路径用了 4 张 GPU 和压缩的 software-E4M3 projection weights，不能把它的
显存与速度直接拿来和上面的 2-GPU BioNeMo BF16 表比较。

## gpu02 快速开始

gpu02 的已验证环境是 Rocky Linux 8.10、4×A800 80GB PCIe、driver
580.126.20、Apptainer 内 CUDA 12.8.93，任意 GPU 对之间均有双向 P2P。

### 构建与测试

在本机仓库运行：

```sh
scripts/gpu02_build.sh
scripts/gpu02_test.sh
```

### 准备 BioNeMo BF16 40B（推荐）

```sh
scripts/gpu02_prepare_bionemo_40b.sh
```

NGC archive 使用独立 cache：

```text
/build/grp_icg/users/tang/.cache/bionemo
```

生成并校验：

```text
$HOME/evo2c-models/evo2-40b-bionemo-bf16.evo2
```

文件大小为 `82,254,509,184` bytes，SHA256 为
`3fb2ec7ed2c89c4f88dcb9c4c6f675e46c2b37722ee82778ce0ff84794dfa5c8`。

在任选两张空闲 GPU 上复现 BioNeMo 对齐 gate：

```sh
EVO2C_BIONEMO_GPU_LIST=1,2 scripts/gpu02_validate_bionemo_40b.sh
```

### 准备原始 Arc 40B

脚本默认使用用户指定的 Hugging Face cache 和镜像：

```sh
export HF_HOME=/build/grp_icg/users/tang/.cache
export HF_ENDPOINT=https://hf-mirror.com
scripts/gpu02_prepare_40b.sh
```

它固定官方 revision，支持断点续传，校验两个分片及合并文件，再生成：

```text
$HOME/evo2c-models/evo2-40b-e4m3sw.evo2
```

文件大小为 `82,252,717,056` bytes，SHA256 为
`d1619e3b2eef0fba7c5838bb61982e891cf63d55385ced865af06693222d6687`。
`HF_ENDPOINT` 只用于 Arc/Hugging Face 路径，不参与 NGC 下载。

### 手动离线转换

gpu02 preparation script 已封装下载、恢复、hash 和转换。若 checkpoint 已在
本地，也可直接调用转换器；PyTorch 只在此离线步骤中需要：

```sh
python3 -m venv .venv-convert
. .venv-convert/bin/activate
python3 -m pip install -r requirements-convert.txt

scripts/convert_arc_checkpoint.sh \
  evo2_7b evo2_7b.pt evo2-7b.evo2

python3 tools/convert_checkpoint.py \
  --input evo2_40b.pt \
  --config configs/evo2-40b-1m.yml \
  --output evo2-40b-e4m3sw.evo2 \
  --dtype bf16

python3 tools/convert_bionemo_checkpoint.py \
  --input /path/to/nemo2/weights \
  --config configs/evo2-40b-1m-bionemo-bf16.yml \
  --output evo2-40b-bionemo-bf16.evo2 \
  --source-sha256 544b47e033d1fb0261b686a53f7c4fe240cd290253187d31e8c99dea9e35a680
```

wrapper 接受支持矩阵中的任意 Arc model ID。精确 upstream revision、hash、
manifest、precision 规则和每个尺寸的 smoke 命令见
[`docs/checkpoint-conversion.md`](docs/checkpoint-conversion.md)。

### 运行 generation

在 gpu02 上：

```sh
image="$HOME/evo2c-cuda12.8-rocky8.sif"
nix_root="$HOME/.local/share/nix-root"
binary="$HOME/evo2c/build-gpu/evo2c"
model="$HOME/evo2c-models/evo2-40b-bionemo-bf16.evo2"

apptainer exec --nv -B "$nix_root:/nix:ro" "$image" \
  "$binary" -m "$model" -p ACGTACGTACGTACGT -n 8 --ctx 8192 \
  --gpu 1,2 --top-k 1 --seed 1
```

生成序列写入 stdout；诊断信息和最终 `evo2c_metrics` JSON 写入 stderr。

### 运行 FASTA scoring

```sh
apptainer exec --nv -B "$nix_root:/nix:ro" "$image" \
  "$binary" -m "$model" --score sequence.fa --ctx 8192 \
  --gpu 1,2 >scores.jsonl 2>metrics.log
```

每条 FASTA record 输出一个 JSON object，包含 token 数、总/平均
log-likelihood、perplexity 和逐 token log-likelihood。长于 8192 的输入会
state-preserving chunk prefill，score 仍基于所有位置 logits。

### 原始 Arc 长 prompt gate

```sh
scripts/gpu02_quality.sh
```

任务可跨 SSH 重连继续运行，汇总报告位于
`$HOME/evo2c-artifacts/t13-native-quality-4x500.json`。

## 长上下文支持与限制

以下 32K/131K/1M 结果来自原始 Arc `.evo2` 和 4×A800 路径；尚未把同样的
长上下文容量 gate 扩展到 2-GPU BioNeMo BF16 路径。

- 质量覆盖的生产目标是 batch 1、`--ctx 8192`。
- 一个真实 8193-token prompt 已在 `--ctx 32768` 下跨过 8192-token
  activation arena，验证 state-preserving chunking。
- `--ctx 131072` 及以上自动选择 16384-token page 的 Q8 KV；物理 page
  只在 token 首次写入时分配。
- 131K smoke 的 2×512 logits 相对 BF16-cache baseline 最低 cosine 为
  `0.9999995896`，top-1 与输出 bytes 完全一致。
- 同一 binary 已通过 `--ctx 1048576` 的真实模型容量 smoke。
- 完整填充的 1M Q8 KV 预计每卡约 33 GiB；加上权重、state 和 activation，
  4 个 stage 预计各需 52.37–53.74 GiB，可容纳于空闲 A800 80GB。
- **没有完成完整 1M prompt 的 prefill 性能验证。**当前 attention prefill
  仍是 quadratic 且没有跨 query 复用 K/V tile；1M capacity 适合增量增长的
  sequence，不代表完整 1M prefix 已经实用。

示例：

```sh
arc_model="$HOME/evo2c-models/evo2-40b-e4m3sw.evo2"

apptainer exec --nv -B "$nix_root:/nix:ro" "$image" \
  "$binary" -m "$arc_model" -p ACGTACGTACGTACGT -n 2 --ctx 131072 \
  --gpu 0,1,2,3 --top-k 1 --seed 1 --dump-logits q8-logits.npy
```

## 实现范围

- registry 选择 24/25/32/50-layer StripedHyena 2，支持 HCS、HCM、HCL、
  MHA/RoPE、MLP。
- KV、FIR、IIR cache 与 cached autoregressive decode。
- byte tokenizer，文本/FASTA scoring，greedy、top-k/top-p sampling。
- 1–4 GPU、按 payload 平衡的连续 layer pipeline，batch 1。
- BioNeMo BF16 Hyena projection 原生路径。
- Arc 原始 checkpoint 的 E4M3FN one-byte weight cache 和
  software-H100-QGMMA accumulation。
- 8192-token 固定 activation arena 与 state-preserving chunked prefill。
- 131K+ fixed-page Q8 KV 和 online-softmax 内 F32 dequantization。
- `EVO2C` v1 mmap container 在任何 CUDA allocation 前完成 header、
  tensor、shape、dtype、offset、checksum 和 model metadata 检查。
- `--dump-tokens`、`--dump-logits`、`--dump-layer` 可用于外部数值审计。

当前不支持通用模型加载、训练、LoRA、batch > 1、服务框架或非 CUDA 后端。
已验证的生产硬件是 A800 `sm_80`；不对更老架构、其他 CUDA architecture
或小显存设备作未经测试的承诺。

multi-chunk scoring 和 generation 已支持；一个 prefill 跨多个 activation
chunk 时，`--dump-layer` 会明确拒绝，因为单个 NPY 不能表示多次独立的
stage-local invocation。

## 可复现性与 artifact

| Artifact | 大小（bytes） | SHA256 |
|---|---:|---|
| Arc 合并 checkpoint | 82,253,491,694 | `dd299612b1c1cdded0dfdcaf4d16f98fc97458261d80f4d662429f0ccb316bc3` |
| Arc `.evo2` | 82,252,717,056 | `d1619e3b2eef0fba7c5838bb61982e891cf63d55385ced865af06693222d6687` |
| BioNeMo NGC archive | 63,680,606,710 | `544b47e033d1fb0261b686a53f7c4fe240cd290253187d31e8c99dea9e35a680` |
| BioNeMo BF16 `.evo2` | 82,254,509,184 | `3fb2ec7ed2c89c4f88dcb9c4c6f675e46c2b37722ee82778ce0ff84794dfa5c8` |
| BioNeMo official logits NPY | 32,896 | `9e16b0de532e57350b0b0ffdb9c48728b339c584925070ede75ea38d308d51d6` |
| evo2c logits NPY | 32,896 | `99c2c6de5291a7b9e525921f1c4fa9a089b94b96eab39320a4d87a738cda2244` |
| logits comparison report | 3,110 | `864f8f64ffaf18de005c770412c9ab31a1775c98556dd1da67ff49e0b984e44c` |

Arc checkpoint 固定 Hugging Face revision
`d529aa57c30771814217ad89baaeaf6e2315c7d7`。每个准备脚本都在发布最终文件
前验证固定大小和 SHA256，已存在但不匹配的 artifact 不会被静默复用。

BioNeMo DCP manifest 含 506 个 BF16 data tensors 和 210 个 metadata
entries；转换后为 537 个 runtime tensors。普通 BF16 payload bit-exact，
缺失、重复、未知 tensor 或不一致 shape/dtype 都会使转换失败，而不是静默跳过。

## 本地构建与测试

CPU-only validation 需要 CMake、C++17 compiler 和 Python 3：

```sh
cmake -S . -B build-cpu -DEVO2C_CUDA=OFF \
  -DEVO2C_WARNINGS_AS_ERRORS=ON
cmake --build build-cpu -j
ctest --test-dir build-cpu --output-on-failure
```

CUDA build 需要 CUDA 12.x、cuBLASLt、cuFFT 和 `sm_80` target：

```sh
cmake -S . -B build-gpu -DEVO2C_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DEVO2C_WARNINGS_AS_ERRORS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-gpu -j
ctest --test-dir build-gpu --output-on-failure
```

canonical local entrypoint：

```sh
scripts/local_test.sh build-release
EVO2C_SANITIZE=ON scripts/local_test.sh build-sanitize
```

当前状态：

- Local Release：21/21 passed。
- ASan/UBSan：21/21 passed。
- 最近一次 multi-size 修改前的 gpu02 baseline：28/28 passed，包括 9 个 CUDA
  tests 和 2 个 multi-GPU tests。本次修改尚未在 gpu02 重建/运行，因为当前
  host 没有 CUDA compiler/GPU。
- 真实 PyTorch DCP converter integration：5/5 passed。

PyTorch 只在离线 checkpoint conversion 和可选 oracle test 中使用，
production binary 不链接 PyTorch、Vortex 或 Transformer Engine。

## 设计与审计文档

- [`SPEC.md`](SPEC.md)：可执行 invariant、validation gate 和任务记录。
- [`docs/model-format.md`](docs/model-format.md)：`.evo2` checked container。
- [`docs/checkpoint-conversion.md`](docs/checkpoint-conversion.md)：Arc 与
  BioNeMo checkpoint 的严格转换规则。
- [`docs/math-semantics.md`](docs/math-semantics.md)：Vortex-compatible
  layer 数学语义。
- [`docs/software-fp8.md`](docs/software-fp8.md)：为什么原始 1B/20B/40B
  不能简单换成 BF16，以及 Ampere 如何模拟所需 FP8 语义。
- [`docs/gpu02-environment.md`](docs/gpu02-environment.md)：gpu02 环境、
  benchmark、artifact 路径和 SHA256。

## 上游来源与许可

- [Evo 2 paper](https://www.nature.com/articles/s41586-026-10176-5)
- [Arc Institute Evo 2 repository](https://github.com/arcinstitute/evo2)
- [Arc `evo2_40b` checkpoint](https://huggingface.co/arcinstitute/evo2_40b)
- [Vortex inference repository](https://github.com/Zymrael/vortex)
- [BioNeMo Evo 2 documentation](https://docs.nvidia.com/bionemo-framework/latest/models/evo2/index.html)
- [BioNeMo FP8/BF16 hardware compatibility](https://docs.nvidia.com/bionemo-recipes/latest/main/examples/bionemo-evo2/examples/fine-tuning-tutorial/index.html#fp8-and-hardware-compatibility)

本项目是独立的 Apache-2.0 实现。上游研究、代码与 checkpoint attribution
见 [`NOTICE`](NOTICE)；使用模型权重时也应遵守对应发布源的条款。
