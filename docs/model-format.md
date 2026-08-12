# Safetensors runtime profiles

`evo.cpp` 使用标准 Safetensors 作为生产模型容器。Evo 2 保持严格、确定性的
`evo2-runtime-v1` profile；第二个 model family 使用独立的
`hyenadna-runtime-v1` profile；ESMC 使用 `esmc-runtime-v1`。单文件使用
`.safetensors` 后缀；大模型使用
标准 Safetensors 分片名与 `model.safetensors.index.json`。

## 为什么直接建立在 Safetensors 上

- 每个 shard 的 header 小、payload 连续，适合只读 mmap 和顺序上传；
- dtype、shape 和 byte range 已是标准字段，不需要维护私有 descriptor；
- 上游更新只影响转换映射和 manifest；
- Python、Rust 与其他生态工具可以直接查看 tensor；
- runtime 只实现一个很小的严格 parser，不需要 JSON 或 Safetensors 第三方库。

Safetensors 是存储协议，不是完整的 Evo 2 模型契约。因此 runtime 仍会校验
profile metadata、官方模型签名和精确 tensor 集。

## 物理布局

```text
8-byte little-endian header length
UTF-8 JSON header padded with ASCII spaces to an 8-byte boundary
dense tensor data buffer
```

`data_offsets` 相对 data buffer 起点。所有 tensor 必须按 offset 连续排列，
不允许洞、重叠、别名、尾随数据或截断。Evo 2 profile 的物理顺序为：

1. `embedding_layer.weight`
2. `unembed.weight`
3. `norm.scale`
4. 按 block 编号和 native load order 排列的 layer tensor

其他已注册 profile 使用各自 converter manifest 中的确定性顺序；不依赖源文件
JSON key 顺序。shard 不绑定 GPU 数量；runtime 根据各 layer 的 payload bytes
选择连续 pipeline 分段。

## 按大小分片

转换器默认以 4 GiB tensor payload 为目标大小，按对应 profile 的确定性物理顺序
贪心装填 shard：

```text
model-00001-of-00004.safetensors
model-00002-of-00004.safetensors
model-00003-of-00004.safetensors
model-00004-of-00004.safetensors
model.safetensors.index.json
```

不按 Evo 2 layer、GPU 或模型大小添加特殊后缀，也不拆开单个 tensor。若一个
tensor 本身超过目标大小，它单独占一个 shard。小于目标大小的模型仍只写
`model.safetensors`，不生成 index。

index 使用通用 Safetensors/Hugging Face 结构：

```json
{
  "metadata": {"total_size": 13170533120},
  "weight_map": {
    "embedding_layer.weight": "model-00001-of-00004.safetensors"
  }
}
```

runtime 对 index 和所有 shard 做只读 mmap，并校验 `weight_map`、总 payload
大小、tensor 唯一性以及各 shard metadata 完全一致。`--max-shard-mib` 只改变
物理文件数量，不改变 tensor、数值或 GPU 划分。

## 支持 dtype

| Runtime dtype | Safetensors dtype | Payload |
|---|---|---:|
| F32 | `F32` | `elements × 4` |
| BF16 | `BF16` | `elements × 2` |
| 软件 E4M3 | `F8_E4M3` | `elements` |

其他 Safetensors dtype 一律拒绝。维度必须为正，rank 最大为 8，shape product
和 payload size 必须精确匹配。

`F8_E4M3` 存的是转换阶段生成的最终 `float8_e4m3fn` code，不表示 runtime
需要硬件 FP8。对应 kernel 可以在 Ampere 上用普通整数/BF16 指令消费这些字节。

每个软件 FP8 Hyena projection 附带：

```text
blocks.N.projections.fp8_runtime_scales
dtype = F32
shape = [2]
value = [input_scale, output_scale]
```

上游 weight scale 已在转换时应用；`scale_inv_fwd` 和
`amax_history_fwd` 不写入 runtime 文件。

## Typed metadata

Safetensors 的 `__metadata__` 值只能是字符串。profile 用稳定前缀保留原始类型：

| 前缀 | 类型 | 示例 |
|---|---|---|
| `s:` | UTF-8 string | `s:StripedHyena2` |
| `u:` | canonical uint64 decimal | `u:8192` |
| `f:` | IEEE-754 F64 bits，16 位小写 hex | `f:3eb0c6f7a0b5ed8d` |
| `b:` | bool | `b:0` / `b:1` |
| `l:` | comma-separated uint64 list | `l:3,10,17` |
| `x:` | bytes，lowercase hex | `x:00ff` |

Evo 2 必需键保持不变：

```text
evo2.profile = s:evo2-runtime-v1
runtime.abi  = s:evo2-safetensors-v1
```

HyenaDNA 使用不冒充 Evo 2 的独立键值：

```text
runtime.profile = s:hyenadna-runtime-v1
runtime.abi     = s:hyenadna-safetensors-v1
```

ESMC 同样使用独立键值：

```text
runtime.profile = s:esmc-runtime-v1
runtime.abi     = s:esmc-safetensors-v1
```

ESMC runtime tensors 保持官方 F32 names/shapes，按 embedding、每层 attention/
FFN、final norm、LM head 的确定顺序写出；唯一不写入 runtime 的 source entries
是 converter 精确枚举的 zero-byte U8 `_extra_state`。
`runtime.embedding_layer_count` 记录 `num_layers + 1`，对应官方 hidden-state
indexing。

每个 shard 必须恰好包含一种已注册 profile key，且所有 shard metadata 完全
相同。profile、ABI、`model.architecture` 和 backend support 由 architecture
registry 联合校验，文件名不参与识别。

转换器还记录 model ID、拓扑、source repo/revision、checkpoint 文件名/大小/
SHA、precision policy 和 converter producer。tensor descriptor 与官方 registry
signature 是推理 payload 的最终权威。

## 安全与完整性

reader 在 CUDA allocation 前验证：

- index 只引用同目录的 `.safetensors` 文件，不能路径穿越；
- `weight_map` 与 shard tensor 集、`total_size` 精确一致；
- header length 非零、8-byte 对齐且不超过 16 MiB；
- JSON root、字段和键唯一；
- metadata 均使用已知 typed prefix；
- tensor 名、dtype、rank、shape、size、offset 和总文件范围；
- artifact profile、runtime ABI、architecture registry 与对应 tensor signature。

Safetensors header/range 校验防止越界和结构损坏。artifact 的真实性及完整文件
校验由可信渠道中的 SHA256 负责；转换结果不提交 Git，发布时在外部 manifest
或 `.sha256` sidecar 中锁定。

## 写入事务

writer 先按 payload bytes 规划 shard，再计算确定性的紧凑 JSON header 和
最终文件大小。每个文件都在同目录临时创建、预分配，以 bounded chunk 顺序
写入并 `fsync`；index 最后发布。默认不写 timestamp，不压缩，也不二次扫描
整个输出生成内嵌 checksum。
