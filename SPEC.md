# SPEC

## §G GOAL

evo.cpp !成为本地、可嵌入、可移植的生物序列基础模型推理平台；Evo 2 exact inference 为首个且不可回退的参考实现，原生覆盖 Biohub ESMC 300M/600M/6B，并完整支持 GENEB v4 Table 4 全部 40 checkpoint 的 frozen embedding 推理与可复现评测。

## §C CONSTRAINTS

- C1: 注册表内 Evo 2 1B/7B/20B/40B exact profile !保持既有 raw-bit gate、确定性与显式失败语义；见 `docs/math-semantics.md`、`docs/model-size-validation.md`。
- C2: 生产 runtime !为 C++17；⊥ Python、PyTorch、libtorch、Vortex、Transformer Engine 运行时依赖。Python/PyTorch 仅限离线转换与 oracle。
- C3: Evo 2 exact artifact !继续使用严格 `evo2-runtime-v1` Safetensors；⊥为平台化重写已验证格式。其他 architecture/量化/容器只能使用显式独立 profile 或 adapter。
- C4: checkpoint 转换、mmap、manifest、FP8 extra-state 与原子发布契约 !保持 `docs/model-format.md`、`docs/checkpoint-conversion.md` 约束。
- C5: 训练/微调由 Savanna 或 NVIDIA BioNeMo 负责；本仓库只加载其推理产物与未来 adapter。
- C6: CUDA/Ampere 仍为 exact 主后端；新后端与 fast profile ⊥冒充 exact。
- C7: 平台覆盖 DNA 与蛋白质序列；多模型架构必须抽象扩展点，⊥在任一 architecture 私有类上继续堆叠公共接口。
- C8: 大 checkpoint、转换结果、数据集 ⊥提交 Git；下载缓存与 revision/hash !可审计。
- C9: ESMC canonical source !为 `biohub/ESMC-300M`、`biohub/ESMC-600M`、`biohub/ESMC-6B`；2024-12 HF repos 仅作 deprecated alias/source 说明，⊥隐式混载旧权重布局。
- C10: ESMC production runtime !为 C++17 CPU/CUDA/MPS；Python/PyTorch/官方 Transformers fork 仅限隔离的 converter/oracle，不进入产品依赖图。
- C11: ESMC v1 artifact !使用独立 `esmc-runtime-v1` strict Safetensors profile、F32 tensors、pinned source receipt；⊥伪装 `evo2-runtime-v1` 或 silent dtype/layout conversion。
- C12: ESMC 是最大 2048 tokens（含 `<cls>`/`<eos>`）的双向 masked encoder；v1 只承诺 logits/hidden-state embedding，⊥autoregressive generation、causal score、variant likelihood 或 recurrent chunking。
- C13: ESMC v1 CUDA !先支持单 device exact path；multi-GPU/CPU+GPU offload 必须 typed unsupported，⊥静默退化。
- C14: ESMC 接入 !保持 Evo 2/HyenaDNA artifact、tokenization、CLI、C ABI 与数值 gate 行为不变。
- C15: gpu02 Hugging Face 工作 !统一 `HF_HOME=/build/grp_icg/users/tang/.cache/huggingface`；repo cache !位于 `$HF_HOME/hub`，⊥在低配额 `$HOME` 重建 checkpoint cache。
- C16: MPS v1 !macOS arm64；C++17 core + 最小 Objective-C++ bridge；MPS/Metal 执行 GEMM，host 执行 state/nonlinear ops；⊥MLX/PyTorch runtime，⊥exact claim，⊥silent CPU fallback。
- C17: GENEB scope != 领域调研全集；! 精确覆盖 arXiv:2606.04525v4 Table 4 的 40 个 evaluated checkpoint；Table 5 的 13 个 excluded model 不得冒充已支持。
- C18: “supported” !表示 canonical checkpoint 可完成 pinned fetch/receipt、strict conversion/load、模型一致 tokenization/context/hidden-state/pooling 与真实权重 oracle gate；仅 registry/catalog 行 ≠ support。
- C19: GENEB production inference !遵守 C2；Python/PyTorch/JAX/官方自定义代码仅限隔离 converter/oracle/benchmark probe，⊥进入 `libevo` 依赖或推理调用链。
- C20: 权重/license 不同构；项目只分发 Apache-2.0 自有代码/元数据/转换器；第三方 checkpoint 按原 license/AUP 由用户获取，⊥重新打包非商业或授权不明权重。
- C21: GENEB 参考语义 !固定 `darlednik/GENEB@b465d2d6a11efbbc9a22c105e34832725ce50e05` + 已审计 patch/decision manifest；`geneb-v4-reference` 仅修复不可执行缺陷并保留 upstream 有效行为，`geneb-v4-normalized` 按记录独立消除 padding/batch 污染，二者结果不得混报。
- C22: 每个 GENEB checkpoint !至少有 portable C++ CPU F32/BF16 correctness path；CUDA/MPS capability 按真实 kernel/设备证据声明，缺少则 typed unsupported，⊥以隐式截断或 CPU fallback 冒充加速支持。
- C23: GENEB `runtime_support` 与 `benchmark_provenance` !独立；评测的“论文可复现”与“协议兼容” !分开声明；缺少可执行 reference extractor/官方 submission/digest 的模型可达 runtime supported 但只能产出 normalized protocol-compatible 结果，⊥声称重现论文榜单。

## §I INTERFACES

- I1 existing-cli: `evo -m MODEL -p DNA -n N --ctx N --gpu IDS` 与 `evo -m MODEL --score INPUT --ctx N --gpu IDS` !保持兼容。
- I2 cli: `evo run|score|embed|variant-score|serve|bench ...`; `-hf/--hf-repo REPO[@REV]` !解析已验证本地 HF cache artifact；⊥推理进程调用 Python downloader。
- I3 c-api: `include/evo/evo.h` 暴露 opaque `evo_model`、`evo_context`、`evo_batch`、`evo_sampler`; C ABI + version query。
- I4 artifact: strict Safetensors + index JSON + typed metadata；exact/fast/profile/revision/hash !可查询。
- I5 bio-input: FASTA、raw、stdin；后续 FASTQ/gzip/VCF/reference FASTA；record name、坐标、strand !保留。
- I6 output: scoring/bench JSONL；embedding NPY/Safetensors；generation stdout `raw|fasta`；错误→nonzero + typed status。
- I7 server: `/v1/generate`、`/v1/score`、`/v1/embeddings`、`/v1/variants`、`/health`、`/metrics`；请求可取消、限长、隔离 context。
- I8 install: `cmake --install` 提供 library、headers、CLI、CMake package；release 提供校验和与平台元数据，含 registry 对齐的 production architectures/artifact profiles。
- I9 esmc-source: `evo_fetch.py source esmc_{300m,600m,6b}` 获取 pinned config/tokenizer/checkpoint；`convert_esmc_checkpoint.py --receipt ...` 离线产出严格 runtime artifact。
- I10 esmc-cli: `evo logits -m MODEL --input INPUT --output DIR` 输出逐 token 64-way F32 NPY；既有 `evo embed ...` 输出选定 hidden layer，蛋白质输入默认添加 `<cls>`/`<eos>`。
- I11 esmc-artifact: metadata 至少含 architecture/profile/model-id/revision、layers/width/heads/vocab/context、tokenizer identity、tensor manifest 与 source receipt hash。
- I12 esmc-c-api: `evo_context_prefill` callback 返回全部 encoded positions logits；`evo_context_embed` layer `0` 为 token embedding、`1..n-1` 为前一 block 输出、`n` 为官方 post-final-LayerNorm embedding；`decode/generate/causal score` → typed unsupported。
- I13 mps-cli-c-api: `--backend mps [--profile mps-f32]`；`--gpu|--gpu-layers` → invalid_argument；C ABI append `EVO_BACKEND_MPS=3`,`EVO_STATUS_MPS=7`；ABI minor `1.4`。
- I14 geneb-catalog: `configs/geneb-models.json` → 精确 40 条；`geneb_model_id`(pinned model_meta key)、`paper_name`(Table 4)、`runtime_id` 三者双射；alias 另表且不计数；family/architecture/params/tokenizer/context/embedding/input_transform/source files、extractor commit+patch hash、oracle env/input digest、weight/code/tokenizer/dataset license+AUP+redistribution、`runtime_support`、`benchmark_provenance` 与 backend/promotion state !分项可查询；reference batching 固定 `{batch_size,order,split_boundary,final_batch,pad_to_batch_max,padding_side}`。
- I15 geneb-cli: `evo models --suite geneb [--json]`；`evo embed -m MODEL --input INPUT --output DIR --preset geneb-v4-{reference,normalized}` → 按 pinned preset 有界缓冲、保持输入顺序输出 per-record NPY + metadata，禁止同时传 `--layer|--pooling`；`geneb` alias 固定指向 normalized 且在 metadata 展开。
- I16 geneb-fetch: `evo_fetch.py source MODEL --catalog configs/geneb-models.json` → pinned HF/HTTP/Dataverse/Drive receipt；无法自动获取的授权 checkpoint → typed manual-source 指引。
- I17 geneb-artifact: strict family-specific Safetensors profile + typed tokenizer assets/forward config/embedding preset/source receipt；`input_transform` !声明 case/U→T/invalid/frame trim/crop side+offset/fixed pad/special-token/token-truncation policy；reference batch !按 I14 以输入顺序每 8 条分组、train/test 边界处 flush、末批保持实际大小、仅 pad 到当批最长（方向按模型 preset）；不兼容族 !独立 architecture/ABI，⊥伪装 `evo2-runtime-v1`。
- I18 geneb-oracle: `tools/validate_geneb_models.py` → 40-model catalog/converter/load/token/hidden/pool gate summary JSON，每条证据含 `extractor_commit`、`normalization_patch_sha256`、`oracle_env`、`oracle_input_digest`；`tools/run_geneb.py` 使用 dataset revision `4edd705be573e48c585c2cf79dc320f9f43c7b04` 与 exact Python/NumPy/scikit-learn lock、完整 LogisticRegression kwargs/solver、thread env、submission schema 执行 100-task×`full|10shot|1shot` frozen probe，reference/normalized 输出分 namespace。
- I19 geneb-embed-abi: C ABI minor `1.5` append `evo_context_embed_ex`、size/versioned options `{preset,layer,pooling}` 与 callback-scoped `evo_embedding_result_info` metadata（resolved preset/hidden tap/pooling/original+effective length/crop+pad/token count/rows/columns），旧 `evo_context_embed` 保持；server `/v1/embeddings` 接受 `preset:"geneb-v4-reference|geneb-v4-normalized"` 且与 `layer|pooling` 互斥。

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
R13|MPS GEMM|`MPSMatrixMultiplication` 执行 `C=alpha*op(A)*op(B)+beta*C`；matrix transpose + command-buffer encode 为公开 contract|https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixmultiplication
R14|MPS matrix storage|`MPSMatrix` row-major，底层使用 `MTLBuffer`；CPU/GPU coherency 遵守 Metal buffer 规则|https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrix
R15|Metal CLI device|`MTLCreateSystemDefaultDevice()` 返回系统默认 GPU；macOS 非图形 CLI !显式链接 CoreGraphics|https://developer.apple.com/documentation/metal/mtlcreatesystemdefaultdevice%28%29?language=objc
R16|GitHub macOS GPU|标准 arm64 macOS runner ⊥GPU acceleration contract；8-core GPU hardware acceleration 仅由 macOS xlarge runner 明示|https://docs.github.com/en/actions/reference/runners/larger-runners#available-macos-larger-runners-and-labels
R17|GENEB model scope|v4 Table 4 定义 40 evaluated models：13 decoder、15 encoder + DeepGene、3 Hyena/StripedHyena、4 Mamba/Caduceus、2 JanusDNA、2 CNN-Transformer|https://arxiv.org/abs/2606.04525v4
R18|GENEB protocol|GENEB 对 frozen sequence representations 使用统一 probing protocol；官方 main 说明 40-model reference extractors 位于 dev branch|https://github.com/darlednik/GENEB/tree/b465d2d6a11efbbc9a22c105e34832725ce50e05/embedding_pipeline/extractors
R19|GENEB dataset|100 tasks/13 categories；task dataset immutable revision `4edd705be573e48c585c2cf79dc320f9f43c7b04`；probe=logistic regression，seeds=`13,17,42,123,997`|https://github.com/darlednik/GENEB/blob/b54d018903e7f6b874ee45b74e275936deff4cd3/benchmark/benchmark_spec.json
R20|GENEB exclusions|13 surveyed models 因 private/broken/missing code/计算或 wrapper 原因排除；Evo2 不在 evaluated 40 中|https://arxiv.org/abs/2606.04525v4
R21|GENEB source heterogeneity|official extractors 混用 HF AutoModel、自定义 PyTorch/JAX、Google Drive、Harvard Dataverse 与本地 checkpoint；转换层 !分离 source acquisition 与 runtime artifact|https://github.com/darlednik/GENEB/tree/b465d2d6a11efbbc9a22c105e34832725ce50e05/embedding_pipeline/extractors
R22|GENEB harness reproducibility|official requirements 仅给 NumPy/scikit-learn 开放下界，harness 继承 LogisticRegression 多个版本默认值，validator 不重算 metrics；可复现 runner !额外锁定环境/参数/输出|https://github.com/darlednik/GENEB/blob/b54d018903e7f6b874ee45b74e275936deff4cd3/harness/run_GENEB.py

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
- V35: ∀ platform-level docs/help/release metadata → !表述 biological-sequence platform 且与 registry production architectures/artifact profiles 对齐；architecture-specific 页面 !显式限定 scope；⊥把 evo.cpp 全局等同 Evo 2 runtime。
- V36: MPS !显式选择、显式输出 backend/profile/kernel；device/framework/command-buffer/OOM failure → typed nonzero，⊥partial output/fallback。
- V37: MPS `mps-f32` ∀ registered architecture → operation order/token/cache semantics 保持；StripedHyena2 tiny `max_abs<0.08`；HyenaDNA `max_abs≤1e-5`；ESMC !满足 V26；否则 fail gate。
- V38: MPS model weights !跨 context 共享；mutable state !隔离；同 build+artifact+profile+input → deterministic。
- V39: `EVO_MPS=ON` 仅 Apple arm64；其他 platform → actionable configure error；`EVO_MPS=OFF` !保持 Linux/CPU/CUDA build graph。
- V40: backend dispatch !exhaustive；新 enum ⊥被既有 non-CUDA branch 当 CPU；MPS 与 `--gpu|--gpu-layers|device_count>0` 互斥。
- V41: `libevo` 链入的 ∀ CXX/CUDA/OBJCXX backend object !编译期 hidden visibility；`v41_c_api_symbols` !严格匹配 C ABI allowlist，⊥导出内部 C++/backend symbol。
- V42: MPS CI !始终编译 backend + 跑 portable contracts；`MTLCreateSystemDefaultDevice()==nil` 时仅 `v42_mps_*` hardware gates → CTest skip 77；skip ⊥作为 runtime 数值通过证据，真实 Metal host !全绿。
- V43: GENEB catalog ∀ release → `paper_name` set bit-exact = v4 Table 4 40 显示名，`geneb_model_id` set bit-exact = pinned model_meta 40 key，两集各无重复/遗漏/多余且与 40 个 `runtime_id` 双射；alias ⊥计数。
- V44: ∀ GENEB `runtime_support=supported` → canonical real checkpoint conversion+load+embed gate 存在且对 pinned input 通过独立 model oracle；只有 `benchmark_provenance=reference-eligible` 才额外要求 commit+patch official reference 在 clean locked env 产出已校验向量；catalog-only/tiny-only/source-unavailable → experimental|manual-source|unsupported，upstream-broken 可 runtime supported 但⊥ reference-eligible。
- V45: ∀ GENEB embedding → token IDs、有效 token mask、hidden tap、pooling 与所选 C21 preset 一致；oracle dtype/backend/env/`max_abs|mean_abs|cosine` 阈值 !按 model+profile 在 catalog 校准并固定，⊥将 F32 通用阈值套给 BioFM BF16 等路径。
- V46: architecture dispatch !由 artifact architecture + factory 决定；tokenizer/pooling/backend 是独立属性；⊥`tokenizer == architecture`、`bool esmc`或 unknown 默认 StripedHyena 分支。
- V47: tokenizer ∀ runtime → vocab/merges/k-mer/special IDs/normalization/padding/RC 资产在 artifact 中 strict hash+schema 校验；⊥网络、locale、Python tokenizer 或未登记 fallback。
- V48: ∀ third-party model → source revision/file hash、weight/code/tokenizer/dataset license、AUP、redistribution 状态在 receipt/catalog 分项可查；非商业/授权不明权重 ⊥进入 Git/release artifact/cache mirror。
- V49: ∀ GENEB record → 先在分配前校验 raw safety cap，再执行 I17 显式 transform；仅 `length_policy=reject` 超限才拒绝，crop/trim/pad !可重现；原序列只 materialize 1 record；输出含 model/revision/profile/preset/layer/pooling/original+effective length/crop+pad/token_count/input identity。
- V50: ∀ HyenaDNA/Mamba/StripedHyena 长上下文 checkpoint → 按模型声明上限执行且使用渐近算法；⊥静默 crop/硬限 4096/为规避实现而降低 context metadata。
- V51: ∀ model/context → immutable weights 可共享；attention/cache/SSM/conv/MoE state 隔离；同 artifact/profile/input/preset → deterministic embedding bytes。
- V52: ∀ backend capability → registry、CLI、C ABI、server/help/release metadata 一致；缺 kernel/device/memory → typed unsupported/OOM，⊥ partial output/silent CPU fallback。
- V53: ∀ GENEB runtime family → tiny deterministic converter/tokenizer/CPU/backend fixture + corruption gates；∀ 40 canonical checkpoint → pinned short-sequence independent model oracle evidence；∀ `reference-eligible` checkpoint → commit+patch official reference 在 fresh locked env 产出并通过追加 evidence。
- V54: GENEB probe → R19 dataset revision/tasks/splits/seeds/logreg 固定；`reference` 仅使用 hashed unblock patch，按 I14/I17 固定 input order、batch=8、split flush、末批不补齐与 batch-max padding，保留可执行 upstream 语义；`normalized` 逐记录消除 padding contamination/只取 batch[0] 等缺陷且只标 protocol-compatible；undefined `seq_length` 等 decision !在 T25 固定后才开工。
- V55: GENEB change 后 ∀ existing Evo2/HyenaDNA/ESMC CPU/CUDA/MPS/C ABI/CLI/server/format/fetch/release gate !green；⊥改写已验证 artifact/token/logit/embedding semantics。
- V56: ∀ promoted GENEB preset → CLI/C ABI `embed_ex`/server 的输出向量、transform/preset/shape/result-info metadata bitwise 或按 catalog tolerance parity；旧 C ABI 行为不变，互斥参数 typed fail。
- V57: GENEB full run → 每个已支持模型完整 100 tasks×`full|10shot|1shot`×`MCC|Acc|F1`，lock/submission/env digest 齐全；仅 `benchmark_provenance=reference-eligible` + reference preset + pinned official submission 可比较 per-task metric `abs≤1e-6`，normalized-only 结果只标 protocol-compatible，⊥补数/混 namespace/声称榜单复现。
- V58: ∀ embedding preset → hidden tap、CLS/mean/last/spatial pooling、special-token inclusion、mask domain 与 output width !显式固定；⊥用现有 `none|mean|last` 近似不同语义。

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
T22|x|生成 pinned 官方 oracle；gpu02 验证三尺寸；跑 full regression、记录证据并清理非制品文件|C8,C10,C14,C15,V16,V26,V28,V30,V31,V34
T23|x|审计全仓 scope；修复 platform-level Evo 2-only 表述与 release metadata；增加 registry-driven contract|C7,V10,V16,V35,I8
T24|x|实现 macOS arm64 MPS backend、CLI/C ABI/server、数值与构建契约|C2,C6,C10,C14,C16,R13,R14,R15,R16,V1,V2,V7,V8,V9,V10,V14,V15,V16,V21,V26,V31,V35,V36,V37,V38,V39,V40,V41,V42,I2,I3,I6,I7,I8,I10,I12,I13
T25|x|冻结 GENEB 40-model 三 ID 双射、canonical source/file receipt、分项 license/AUP、reference/normalized extractor patch+input-transform+preset decision、oracle/env lock 与完整性 contract|C17,C18,C20,C21,C23,R17,R18,R19,R20,R21,R22,V43,V44,V45,V48,V54,V57,V58,I14,I16,I18
T26|.|重构通用 model/architecture/backend factory、registry-driven artifact profile 与 capability dispatch|C2,C3,C7,C14,C19,V1,V2,V8,V10,V14,V15,V46,V51,V52,V55,I2,I3,I4,I7,I8,I17
T27|.|实现 artifact-driven BPE/k-mer/SN/mixed/BioToken tokenizer、strict asset converter 与 catalog input-transform engine|C2,C18,C19,C21,V5,V10,V17,V25,V43,V45,V47,V49,V53,V55,V58,I4,I5,I17
T28|.|实现 METAGENE、GenomeOcean×2、GENERator×2、BioFM、OmniNA 的 Llama/Mistral-derived CPU embedding/converter/oracle|C18,C19,C20,C21,C22,C23,R17,R18,V8,V10,V11,V44,V45,V48,V49,V51,V53,V55,V58,I15,I16,I17,I18
T29|.|实现 OmniDNA×2、GPT2-Gene×2、DNA-GPT×2 的 OLMo/GPT2/custom-decoder CPU embedding/converter/oracle|C18,C19,C20,C21,C22,C23,R17,R18,V8,V10,V11,V44,V45,V48,V49,V51,V53,V55,V58,I15,I16,I17,I18
T30|.|实现 NT×5 + Agro-NT 的 ESM-derived encoder CPU embedding/converter/oracle|C18,C19,C20,C21,C22,C23,R17,R18,V8,V10,V11,V44,V45,V48,V49,V51,V53,V55,V58,I15,I16,I17,I18
T31|.|实现 GENA×3、DNABERT-S/2、GROVER、MutBERT 的 BERT/Mosaic/soft-input encoder CPU embedding/converter/oracle|C18,C19,C20,C21,C22,C23,R17,R18,V8,V10,V11,V44,V45,V48,V49,V51,V53,V55,V58,I15,I16,I17,I18
T32|.|实现 LucaOne 与 Genomics-FM custom encoder CPU embedding/converter/oracle|C18,C19,C20,C21,C22,C23,R17,R18,V8,V10,V11,V44,V45,V48,V49,V51,V53,V55,V58,I15,I16,I17,I18
T33|.|实现 Evo-1-131k + HyenaDNA Medium-160k/Large-1M 长上下文原生 CPU embedding/converter/oracle|C18,C19,C20,C21,C22,C23,V1,V8,V10,V11,V44,V45,V48,V49,V50,V51,V53,V55,V58,I15,I16,I17,I18
T34|.|实现 eccDNAMamba、PlantCaduceus、Caduceus PS/PH 双向/RC Mamba 原生 CPU embedding/converter/oracle|C18,C19,C20,C21,C22,C23,V8,V10,V11,V44,V45,V48,V49,V50,V51,V53,V55,V58,I15,I16,I17,I18
T35|.|实现 JanusDNA w/wo attention 的 Mamba+attention+MoE 原生 CPU embedding/converter/oracle|C18,C19,C20,C21,C22,C23,V8,V10,V11,V44,V45,V48,V49,V50,V51,V53,V55,V58,I15,I16,I17,I18
T36|.|实现 Enformer、SPACE 与 DeepGene pinned RoFormer-only GENEB embedding path 的原生 CPU converter/oracle；DeepGene graph stage 非 GENEB profile 不在本任务内|C18,C19,C20,C21,C22,C23,V8,V10,V11,V44,V45,V48,V49,V51,V53,V54,V55,V58,I15,I16,I17,I18
T37|.|为有实作 kernel 与证据的 GENEB family 接入 CUDA/MPS promotion matrix、backend factory、C ABI `embed_ex`/CLI/server；其余固定 typed unsupported|C2,C6,C14,C16,C19,C22,V2,V7,V8,V9,V10,V14,V16,V36,V37,V38,V39,V40,V41,V42,V45,V51,V52,V53,V55,V56,I2,I3,I7,I8,I13,I15,I19
T38|.|实现 `models --suite geneb`、两个 embedding preset、pinned multi-source fetch/receipt 与 locked GENEB 100-task×3-regime probe runner|C8,C17,C18,C19,C20,C21,C23,V3,V5,V10,V11,V16,V17,V18,V20,V43,V44,V47,V48,V49,V52,V54,V55,V57,V58,I2,I4,I5,I6,I14,I15,I16,I18
T39|.|在真实 checkpoint 上验证 CPU 40/40 conversion/load/token/embedding；验证 promoted CUDA/MPS matrix 与其余 typed unsupported；跑 full regression 并完成 docs/release evidence|C8,C15,C17,C18,C20,C22,C23,V16,V19,V22,V34,V35,V41,V42,V43,V44,V45,V48,V49,V52,V53,V54,V55,V56,V57,V58,I8,I14,I18,I19

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
B39|2026-08-12|gpu02 宿主可读共享 `HF_HOME`，但 BioNeMo Apptainer image 默认不 bind `/build/grp_icg`→ESMC fetch/conversion/oracle 容器统一显式只读 bind C15 `HF_HOME` 到同路径，避免误报 snapshot 缺失或复制权重|C15,V28,V34
B40|2026-08-12|共享 snapshot 按 B39 只读挂载后，`evo-fetch` 默认尝试在 cache child 写 receipt→新增独立 `--receipt-dir`；gpu02 gate 将 receipt 写入本次验收制品，模型 cache 全程只读|C15,V28,V34
B41|2026-08-13|ESMC benchmark contract 用 `==` 比较由二进制浮点样本计算的 median，正确聚合因表示误差失败→数值断言改用严格容差比较|V21
B42|2026-08-13|ESMC CLI 的 `evo_metrics` 按整个多记录输入聚合，benchmark 错把它当逐记录样本→CUDA logits 另发逐记录计时，summarizer 同时校验逐记录样本与单条 aggregate|V21
B43|2026-08-13|ESMC official benchmark 只同步 device logits，而原生 `prefill` 计时包含 logits 回传主机→两侧统一 `forward_with_host_logits` timing scope 并由 summarizer 拒绝混用|V21
B44|2026-08-16|`CXX_VISIBILITY_PRESET` 不作用于 OBJCXX 编译→MPS bridge 内部 C++ symbol 泄漏进 C ABI DSO|V41
B45|2026-08-16|CI 把 Apple-silicon architecture 误当 Metal GPU 可用性保证→标准 macOS runner 可能让合法 MPS hardware gate 误失败|V42
