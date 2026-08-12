# SPEC

## §G GOAL

evo.cpp !成为本地、可嵌入、可移植的生物序列基础模型推理平台；Evo 2 exact inference 为首个且不可回退的参考实现，并原生覆盖 Biohub ESMC 300M/600M/6B 蛋白质 encoder inference。

## §C CONSTRAINTS

- C1: 注册表内 Evo 2 1B/7B/20B/40B exact profile !保持既有 raw-bit gate、确定性与显式失败语义；见 `docs/math-semantics.md`、`docs/model-size-validation.md`。
- C2: 生产 runtime !为 C++17；⊥ Python、PyTorch、libtorch、Vortex、Transformer Engine 运行时依赖。Python/PyTorch 仅限离线转换与 oracle。
- C3: Evo 2 exact artifact !继续使用严格 `evo2-runtime-v1` Safetensors；⊥为平台化重写已验证格式。其他 architecture/量化/容器只能使用显式独立 profile 或 adapter。
- C4: checkpoint 转换、mmap、manifest、FP8 extra-state 与原子发布契约 !保持 `docs/model-format.md`、`docs/checkpoint-conversion.md` 约束。
- C5: 训练/微调由 Savanna 或 NVIDIA BioNeMo 负责；本仓库只加载其推理产物与未来 adapter。
- C6: CUDA/Ampere 仍为 exact 主后端；新后端与 fast profile ⊥冒充 exact。
- C7: 平台首先覆盖 DNA 序列；多模型架构必须抽象扩展点，⊥在 Evo 2 CUDA 类上继续堆叠公共接口。
- C8: 大 checkpoint、转换结果、数据集 ⊥提交 Git；下载缓存与 revision/hash !可审计。
- C9: ESMC canonical source !为 `biohub/ESMC-300M`、`biohub/ESMC-600M`、`biohub/ESMC-6B`；2024-12 HF repos 仅作 deprecated alias/source 说明，⊥隐式混载旧权重布局。
- C10: ESMC production runtime !为 C++17 CPU/CUDA；Python/PyTorch/官方 Transformers fork 仅限隔离的 converter/oracle，不进入产品依赖图。
- C11: ESMC v1 artifact !使用独立 `esmc-runtime-v1` strict Safetensors profile、F32 tensors、pinned source receipt；⊥伪装 `evo2-runtime-v1` 或 silent dtype/layout conversion。
- C12: ESMC 是最大 2048 tokens（含 `<cls>`/`<eos>`）的双向 masked encoder；v1 只承诺 logits/hidden-state embedding，⊥autoregressive generation、causal score、variant likelihood 或 recurrent chunking。
- C13: ESMC v1 CUDA !先支持单 device exact path；multi-GPU/CPU+GPU offload 必须 typed unsupported，⊥静默退化。
- C14: ESMC 接入 !保持 Evo 2/HyenaDNA artifact、tokenization、CLI、C ABI 与数值 gate 行为不变。
- C15: gpu02 Hugging Face 工作 !统一 `HF_HOME=/build/grp_icg/users/tang/.cache/huggingface`；repo cache !位于 `$HF_HOME/hub`，⊥在低配额 `$HOME` 重建 checkpoint cache。

## §I INTERFACES

- I1 existing-cli: `evo -m MODEL -p DNA -n N --ctx N --gpu IDS` 与 `evo -m MODEL --score INPUT --ctx N --gpu IDS` !保持兼容。
- I2 cli: `evo run|score|embed|variant-score|serve|bench ...`; `-hf/--hf-repo REPO[@REV]` !解析已验证本地 HF cache artifact；⊥推理进程调用 Python downloader。
- I3 c-api: `include/evo/evo.h` 暴露 opaque `evo_model`、`evo_context`、`evo_batch`、`evo_sampler`; C ABI + version query。
- I4 artifact: strict Safetensors + index JSON + typed metadata；exact/fast/profile/revision/hash !可查询。
- I5 bio-input: FASTA、raw、stdin；后续 FASTQ/gzip/VCF/reference FASTA；record name、坐标、strand !保留。
- I6 output: scoring/bench JSONL；embedding NPY/Safetensors；generation stdout `raw|fasta`；错误→nonzero + typed status。
- I7 server: `/v1/generate`、`/v1/score`、`/v1/embeddings`、`/v1/variants`、`/health`、`/metrics`；请求可取消、限长、隔离 context。
- I8 install: `cmake --install` 提供 library、headers、CLI、CMake package；release 提供校验和与平台元数据。
- I9 esmc-source: `evo_fetch.py source esmc_{300m,600m,6b}` 获取 pinned config/tokenizer/checkpoint；`convert_esmc_checkpoint.py --receipt ...` 离线产出严格 runtime artifact。
- I10 esmc-cli: `evo logits -m MODEL --input INPUT --output DIR` 输出逐 token 64-way F32 NPY；既有 `evo embed ...` 输出选定 hidden layer，蛋白质输入默认添加 `<cls>`/`<eos>`。
- I11 esmc-artifact: metadata 至少含 architecture/profile/model-id/revision、layers/width/heads/vocab/context、tokenizer identity、tensor manifest 与 source receipt hash。
- I12 esmc-c-api: `evo_context_prefill` callback 返回全部 encoded positions logits；`evo_context_embed` layer `0` 为 token embedding、`1..n-1` 为前一 block 输出、`n` 为官方 post-final-LayerNorm embedding；`decode/generate/causal score` → typed unsupported。

## §R RESEARCH

id|topic|finding|src
R1|llama.cpp platform|核心覆盖 C/C++ library、HF 直载、1.5–8 bit 量化、CPU/CUDA/Metal/HIP/Vulkan/SYCL 与 CPU+GPU hybrid|https://github.com/ggml-org/llama.cpp
R2|llama.cpp serving|server 提供 parallel decoding、continuous batching、embeddings、monitoring 与多用户支持|https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
R3|Evo 2 inference|官方推理面包含 forward scoring、intermediate embeddings、generation；最长 checkpoint context 1M bp|https://github.com/ArcInstitute/evo2
R4|Evo 2 training|官方把训练/微调交给 Savanna 或 NVIDIA BioNeMo|https://github.com/ArcInstitute/evo2#training-and-finetuning
R5|OpenGenome2 IO|raw 数据为大规模异构 FASTA；流式 record 处理优先于全量 materialization|https://huggingface.co/datasets/arcinstitute/opengenome2
R6|ESMC canonical models|Biohub 当前推荐 `ESMC-300M`/`ESMC-600M`/`ESMC-6B`；旧 `*-2024-12` repos 仅保留 backward compatibility|https://huggingface.co/biohub/ESMC-6B
R7|ESMC topology|三者分别为 `(d_model,heads,layers)=(960,15,30),(1152,18,36),(2560,40,80)`，vocab=64、max positions=2048、F32 source weights|https://huggingface.co/biohub/ESMC-300M/blob/a59b831785f907e96e6a246b1d142bfb76df31ee/config.json
R8|ESMC forward|官方实现为 pre-LN Transformer：scaled residual、Q/K LayerNorm、non-interleaved RoPE、bidirectional SDPA、SwiGLU、final LayerNorm 与 GELU/LayerNorm LM head|https://github.com/Biohub/transformers/blob/3a8956fb4d4ea16b0ec8e71deef2c2909b6a5cbf/src/transformers/models/esmc/modeling_esmc.py
R9|ESMC tokenizer|固定 64-slot vocab；33 个已分配 token，`<cls>=0,<pad>=1,<eos>=2,<unk>=3,<mask>=32`，character BPE 自动包围 cls/eos|https://github.com/Biohub/transformers/blob/3a8956fb4d4ea16b0ec8e71deef2c2909b6a5cbf/src/transformers/models/esmc/tokenization_esmc.py
R10|ESMC outputs|官方 masked-LM 返回 per-token logits、last hidden state 与可选全部 hidden states；⊥causal generation contract|https://huggingface.co/biohub/ESMC-6B
R11|ESMC license|Biohub ESM code/models 使用 MIT license，并要求遵循 Biohub Acceptable Use Policy|https://github.com/Biohub/esm/blob/26b0bc2b771e3e419ea74f445a5f35cc094a1509b6a5cbf/README.md#license
R12|ESMC weights|当前 300M/600M revision 分别为 `a59b831…`/`a7e8201…` 单 F32 Safetensors；6B `45b0fa5…` 为 6-shard F32 Safetensors + index|https://huggingface.co/biohub/ESMC-6B/tree/45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a

## §V INVARIANTS

- V1: ∀ registered exact artifact → checkpoint signature、dtype、shape、数值 gate 与既有 generated bytes 不变。
- V2: ∀ production binary dependency/call chain → Python/PyTorch/libtorch/Vortex/Transformer Engine ∉ graph。
- V3: ∀ multi-record sequence input → reader 每次最多 materialize 1 record；⊥读取整个 FASTA 后再推理。
- V4: 默认 scoring host logits memory ∈ O(chunk_tokens×vocab)；仅显式 `--dump-logits` 可 materialize O(sequence_tokens×vocab)。
- V5: ∀ FASTA record → name、record order、sequence bytes 与错误行号保持；CRLF/wrapped lines 与既有严格 whitespace 语义不回退。
- V6: ∀ record → length 在模型 load/执行前或 record 进入执行时校验 `2 ≤ length ≤ ctx`; 错误包含 record name。
- V7: exact/fast/quantized/backend mode !显式输出；fast/quantized !拥有独立数值 + 生物任务 acceptance gate，⊥标记 exact。
- V8: model weights 与 mutable context/session state !分离；多个 context 可安全共享只读 model。
- V9: 同 build+artifact+mode+seed 的 greedy output !byte-identical。
- V10: unsupported model/profile/shape/dtype/backend/OOM !typed fail；⊥silent fallback。
- V11: embedding !支持指定 intermediate layer 与明确 pooling；shape、layer、dtype 写入输出 metadata。
- V12: variant scoring !记录 reference、alternate、window coordinates、strand 与归一化策略；reference mismatch → fail。
- V13: server ∀ request → 独立 context、输入上限、取消路径；dynamic batching ⊥混淆 record/session state。
- V14: public C ABI 仅 opaque handles；ABI version query + ownership/free pair !有 contract tests。
- V15: current I1 CLI scripts/tests !继续通过；新 command hierarchy 只能兼容扩展。
- V16: ∀ completed §T → named test 覆盖对应 invariant；full CPU suite !green，相关 CUDA/exact gate 在目标环境 !green。
- V17: sequence byte limit !在读取/append 前检查；超限输入 ⊥先 materialize 再拒绝。
- V18: streaming input 后续 record 失败 → 已输出 JSONL 仅含完整 record 行 + process nonzero；⊥半行或成功退出。
- V19: remote GPU build 缺依赖 → actionable path error；成功 build !写 source fingerprint；remote test !拒绝缺失/不匹配 fingerprint，⊥运行 stale binary。
- V20: public sequence IO ∀ multi-record path → only streaming callback API；⊥暴露全量 `vector<SequenceRecord>` 聚合 helper。
- V21: bench ∀ report → architecture、artifact/profile、backend、input identity、warmup、repetitions、token count、timing statistic !显式；失败/unsupported → typed nonzero。
- V22: ∀ production exact model ID → pinned real-checkpoint raw-bit gate evidence !存在；缺证据 ID !显式 experimental/unsupported，⊥继承同 family 证据。
- V23: ∀ maintained runner 使用 exact-unsupported model ID → !显式 non-exact profile 或断言 typed unsupported；⊥依赖默认 `exact`。
- V24: ESMC registry ∀ size → canonical HF ID/revision、Biohub hosted alias、topology 与 exact capability !显式；未知/legacy-incompatible source → typed fail。
- V25: ∀ ESMC protein sequence → tokenizer token IDs 与 R9 bit-exact；默认首尾分别为 0/2，`<mask>`/`|` 与已分配字符按官方 special-token/BPE 规则解析，非法/非词表 byte → 3，`encoded_length ≤ 2048` 在分配/执行前检查。
- V26: ESMC forward ∀ backend → R8 operation order、epsilon、RoPE positions、residual scale、exact GELU 与 F32 source semantics !固定；同输入 logits/final embedding 对官方 oracle 满足 `max_abs≤5e-3,mean_abs≤5e-4,cosine≥0.99999`，否则 fail。
- V27: ESMC capability !仅 `logits|embed`；generation/decode/causal score/variant → typed unsupported，⊥借用 Evo 2 sampling/scoring path。
- V28: ESMC conversion ∀ canonical size → pinned receipt/config/tokenizer、完整 tensor names/shapes/dtypes/sizes/hashes !先验证；缺失/多余 real tensor、非 F32、zero-byte extra-state 之外异常或 hash mismatch → 原子失败。
- V29: ESMC ∀ forward/embed → 单次完整 encoded sequence 执行双向 attention；⊥跨 chunk 复用 prefix state；embedding metadata !记录 point。
- V30: ESMC tests !含 tiny deterministic CPU/CUDA fixture、tokenizer vectors、converter corruption gates；gpu02 ∀三 canonical artifacts → conversion/load + short-sequence official logits/last-hidden oracle gate。
- V31: ESMC change 后 full CPU suite 与既有相关 gpu02 Evo 2/HyenaDNA gates !green；⊥修改已有模型生成/token/logit semantics。
- V32: ESMC CUDA v1 ∀ device selection → exactly one CUDA device；0/2+ devices 或 unsupported compute/runtime → typed fail，且失败前⊥部分输出。
- V33: ESMC hidden index !bit-exact 对齐 pinned Transformers `layers_to_collect`：`0=token_embedding`、`i∈[1,n-1]=block(i-1)_output`、`n=final_layer_norm`；⊥沿用旧 ESM SDK 的逐 block-output convention。
- V34: gpu02 ESMC fetch/conversion/oracle gate → !export C15 `HF_HOME` 且默认 `cache_dir=$HF_HOME/hub`；显式 override !可审计。

## §T TASKS

id|status|task|cites
T1|x|stream FASTA/raw records；移除 CLI 双重读取；保持 parser contract|V3,V5,V6,V15,V16,V17,V18,V19,I1,I5
T2|x|stream scoring chunk logits；默认⊥累计完整 logits matrix|V1,V4,V6,V15,V16,I1,I6
T3|x|抽离 backend-neutral model/context/batch；发布稳定 C ABI + install package|V1,V2,V8,V10,V14,V15,V16,I3,I8
T4|x|实现 first-class intermediate embedding API/CLI + metadata|V1,V7,V11,V15,V16,I2,I6
T5|x|实现 strand/coordinate-aware variant scoring CLI|V1,V6,V7,V12,V16,I2,I5,I6
T6|x|实现 HF revision/hash/cache 获取、CI 与预编译 release pipeline|V2,V10,V16,I2,I4,I8
T7|x|实现共享 model 的 scheduler、dynamic batching 与 bio-native server|V8,V10,V13,V16,I7
T8|x|设计并实现 fast/quantized profiles + 数值/科学 benchmark gates|V1,V7,V9,V10,V16,I4
T9|x|实现向量化 CPU backend 与显式 CPU+GPU offload policy|V2,V7,V9,V10,V16,I2,I3
T10|x|抽象 architecture registry；接入第二个生物序列模型 family|V7,V8,V10,V11,V16,I2,I3,I4
T11|x|补 FASTQ/gzip/stdin/VCF/reference IO 与坐标输出格式|V3,V5,V6,V12,V16,I5,I6
T12|x|完善贡献指南、兼容策略、benchmark matrix 与 release 文档|V7,V14,V15,V16,I8
T13|x|实现 `run|score|bench` CLI hierarchy、内建 reproducible bench 与 generation `raw|fasta` 输出|V7,V9,V10,V15,V16,V21,I1,I2,I6
T14|x|移除公共全量 sequence reader；测试与 consumer 统一 streaming API|V3,V16,V17,V20,I5
T15|x|实现 `-hf/--hf-repo REPO[@REV]` 已验证本地 cache artifact 解析|V2,V10,V16,I2,I4
T16|x|审计 production exact model ID；补真实 checkpoint raw-bit 证据或显式降级为 experimental/unsupported|V1,V10,V16,V22,I4
T17|x|核实官方 ESMC IDs/架构/tokenizer/权重/license/reference；审计扩展点并固化兼容边界|C9,C10,C12,C14,R6,R7,R8,R9,R10,R11,R12,V24,V27
T18|x|注册三尺寸；实现 pinned HF source fetch/receipt、`esmc-runtime-v1` converter/loader 与 corruption tests|C8,C9,C11,V10,V24,V28,I4,I9,I11
T19|x|实现 bit-exact protein tokenizer 与 CPU F32 forward/logits/embedding；tiny oracle tests|C10,C12,V11,V25,V26,V27,V29,V30,I10,I12
T20|x|实现单卡 CUDA F32 ESMC forward/logits/embedding 与 typed unsupported 边界|C10,C12,C13,V26,V27,V29,V30,V32,I10,I12
T21|x|接入 C API/CLI/HF offline artifact validation；补 logits/embedding metadata、能力 gate 与用户文档|C9,C12,C14,V24,V25,V27,V29,I2,I6,I10,I11,I12
T22| |生成 pinned 官方 oracle；gpu02 验证三尺寸；跑 full regression、记录证据并清理非制品文件|C8,C10,C14,C15,V16,V26,V28,V30,V31,V34

## §B BUGS

id|date|cause|fix
B1|2026-08-12|gpu02 缺 pinned libnpy 时 build 静默退出；test 未校验 build/source 新鲜度→stale binary 被执行|V19
B2|2026-08-12|CUDA static archive 的默认可见符号绕过 shared-target visibility→C++/CUDA internals 泄漏进 C ABI DSO|V14
B3|2026-08-12|初版 C API context 仅共享 mmap artifact、重复上传 GPU weights→大模型 multi-context 显存不可扩展|V8
B4|2026-08-12|纯 CLI contract 写死 GPU 0,1；单卡 CUDA_VISIBLE_DEVICES 映射下逻辑 1 不存在→环境相关失败|V15,V16
B5|2026-08-12|GCC 8 的 std::filesystem 仍在独立 stdc++fs；streaming NPY cleanup 首次引入该依赖→link 失败|V16
B6|2026-08-12|HF fetch contract fixture 只支持空 work dir；CTest 重跑遇到既有 fake package 目录→非幂等失败|V16
B7|2026-08-12|CUDA Python test 在 source 下生成 `__pycache__`；source fingerprint 将执行产物当源码→首次 test 后误报 stale build|V16,V19
B8|2026-08-12|profile flag 在 CPU-only C ABI 中完成校验但仅由 CUDA 分支消费→`-Werror` unused-variable 破坏 CPU build|V16
B9|2026-08-12|NEON dot-product 分支后 scalar fallback 仍参与同一作用域编译，且 FIR helper 残留无效参数→Apple ARM `-Werror` build 失败|V16
B10|2026-08-12|CPU CLI 把 `std::filesystem` 引入 `evo_core`，但 GCC 8 的 `stdc++fs` 只链接到两个旧 test target→新 CPU consumer 链接失败|V16
B11|2026-08-12|新增 CPU backend 时改写 CPU-only binary 的既有 CUDA unsupported 文案→旧 CLI contract/script 失配|V15,V16
B12|2026-08-12|HyenaDNA internal header 与实现同在 `src/cpu`，include 却重复添加 `cpu/`；target 未暴露 `src` include root→T10 首次 build 失败|V16
B13|2026-08-12|sampler 用全局 Evo 2 `512` 常量校验 logits/top-k→第二个 architecture 的 16-token vocabulary 无法 generation|V7,V10,V16
B14|2026-08-12|server load gate 独立重复 `vocab == 512` 假设→HyenaDNA 已通过 registry/C ABI 仍在启动末端被拒绝|V7,V10,V16
B15|2026-08-12|C ABI sampler 测试把 Evo 2 的 512 logits 当作全局契约→动态词表实现正确后仍被旧断言判失败|V7,V10,V16
B16|2026-08-12|metadata-only C ABI fixture 未声明新增的 architecture/runtime ABI→全量 contract 在注册表校验前失败|V10,V14,V16
B17|2026-08-12|model-format fixture 增加 registry metadata 后仍写死旧 metadata count→格式解析成功却被脆弱断言判失败|V10,V16
B18|2026-08-12|HyenaDNA acceptance fixture 依赖 NumPy，但 gpu02 的可复现 Nix Python 不含该包→CUDA 门无法运行独立 oracle|V2,V16,V19
B19|2026-08-12|FASTQ quality 范围循环隐式把 signed char 转 unsigned char→AppleClang `-Werror` 拒绝构建|V16
B20|2026-08-12|VCF callback 混用显式 `Status{}` 与推导 initializer-list return，且 variant helper 参数名被反向链局部变量遮蔽→严格编译失败|V16
B21|2026-08-12|BioNeMo behavioral gate 省略 profile→model ID 降级后仍隐式请求 default exact|V23
B21|2026-08-12|gpu02 CUDA image 只有 versioned `libz.so.1` runtime、没有 zlib-devel headers/unversioned linker name→标准 FindZLIB 配置失败|V2,V16,V19
B22|2026-08-12|预先把 `EVO_ZLIB_LIBRARY` 缓存变量定义为空会让 `find_library` 视为用户已赋值而跳过搜索→本机 link 未携带 zlib|V16
B23|2026-08-12|AppleClang 未报告 lambda 参数遮蔽外层 helper 参数，GCC 8 `-Wshadow -Werror` 在 gpu02 拒绝构建|V16,V19
B24|2026-08-12|CUDA variant token-dump lambda 同样复用外层 `sequence` 名，CPU-only 本机门未编译该 TU→gpu02 才触发 GCC shadow gate|V16,V19
B25|2026-08-12|初版 ESMC 规范沿用旧 ESM SDK hidden-state convention，未逐行核对 pinned Transformers `layers_to_collect`→中间层索引整体错位|V33
B26|2026-08-12|ESMC softmax kernel 使用未由当前 include graph 暴露的 CUDA 私有 infinity macro→gpu02 CUDA 12.8 编译失败|V30
B27|2026-08-12|全量回归发现 legacy converter/documentation contracts 假设 registry.models 全是 Evo 2：新增 ESMC profile 后索引不存在的 `config` 并漏报三模型 exact 支持|V24,V31
B28|2026-08-12|ESMC 验收脚本假设 gpu02 同步目录含 `.git`，但标准 build 明确排除 VCS metadata→真实远端在运行前失败|V16,V30
B29|2026-08-12|ESMC oracle 从 receipt 首文件父目录推断 HF snapshot，但 cached symlink 返回 resolved `blobs/` 路径→官方 loader 找不到同目录制品|V28,V30
B30|2026-08-12|ESMC oracle 假设 config 暴露 `expansion_ratio`，但 pinned 实现固定 `8/3` 并向上取整至 256→真实 config 拓扑自检异常|R8,V26,V30
B31|2026-08-12|通用 Transformers 5.1 loader 把 TE 的 Safetensors `_extra_state` 当普通 attribute，无法装载官方权重→oracle 改用 PyTorch extra-state-aware `load_state_dict` 且逐 shard 核对完整 key set|R8,R12,V28,V30
B32|2026-08-12|BioNeMo 容器的可选 FlashAttention Triton RoPE 找不到无版本 `libcuda.so`→oracle 固定使用 pinned 官方 pure-PyTorch RoPE fallback 与 manual F32 attention，避免环境相关 kernel dispatch|R8,V26,V30
B33|2026-08-12|ESMC 验收脚本在 gpu02 宿主直接启动容器构建的 CUDA binary→动态链接器找不到 `libcudart.so.12`|V16,V30
B34|2026-08-12|初次 oracle 隐式采用可选 Transformer Engine fused reduction，与原生/官方 portable fallback 的顺序不同且将 hidden 微差放大为 logits 超阈值→oracle manifest 固定并校验官方 PyTorch F32 fallback|R8,V26,V30
B35|2026-08-12|ESMC gpu02 gate 默认在低配额 `$HOME` 建 HF cache，违背集群共享高容量缓存约定并触发重复下载→统一 export C15 `HF_HOME`，repo cache 固定其 `hub/` child|C15,V34
B36|2026-08-12|全量本机回归偶发让 test thread 在两次顺序 submit 间停顿超过 40ms，合法形成 2 batches 却被脆弱时序断言判失败→batch contract 使用宽窗口，取消/异常 case 隔离到 zero-window scheduler|V16
B37|2026-08-12|官方 tokenizer 对用户字面 `<pad>` 仍输出 attention-mask 1；原生 attention 仅凭 token ID 1 当自动 padding→batch-one runtime 取消隐式 ID mask，tiny oracle 纳入显式 `<pad>`|V25,V26,V30
B38|2026-08-12|共享 `HF_HOME` 已有精确 revision snapshot 且 21/21 注册 SHA256 正确，但 `huggingface_hub` 离线 API 因客户端元数据布局差异拒绝命中→`evo-fetch source --local-files-only` 对 immutable commit snapshot 直接定位 registry 文件并先验 size/SHA256；缺失或不符仍 fail closed|C15,V28,V34
