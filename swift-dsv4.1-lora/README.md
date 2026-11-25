# DeepSeek-V4.1-Flash 4 层代理模型：ms-swift Megatron 训练镜像（ROCm）

镜像内只有环境和脚本，不含模型、数据集和运行记录，首次使用需先下载。

## 使用

```bash
docker run -it --device /dev/kfd --device /dev/dri --group-add video --privileged --network=host \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size=2048g \
    --ulimit memlock=-1 --ulimit stack=67108864 amdagi/training-example:swift_rocm10.2_torch2.16_1123_py312
bash /workspace/dsv41_proxy/download_model_and_data.sh   # 约 127GB，下载到 /workspace/dsv41_proxy
bash /workspace/dsv41_proxy/run_proxy_lora.sh            # LoRA；全参 SFT 用 run_proxy_sft.sh
```

在 8×MI300X 上，`run_proxy_lora.sh` 默认 100 步约 3 分钟，单卡显存约 64GiB。

- 默认从 HuggingFace 下载。`HUB=modelscope` 改用 ModelScope（文件完全相同），`HF_ENDPOINT=https://hf-mirror.com` 使用 HF 镜像站。
- 重复执行会跳过已下载的文件。
- 训练输出写到 `/workspace/dsv41_proxy/output/`。
- 下载的数据在容器的可写层里。不要对这样的容器做 `docker commit`，否则新镜像会带上这约 127GB 数据。

## `run_proxy_lora.sh` 相对 `megatron_sft.sh` 的改动

以 `/workspace/ms-swift/examples/models/deepseek_v41/megatron_sft.sh`（原版全参 SFT 脚本）为基准，`run_proxy_lora.sh` 只改了下面几处。其余参数完全相同，包括 TP=1、CP=1、EP=8、`--max_length 8192`、100 步、full recompute、`--attention_backend flash`、MoE 融合、precision-aware optimizer 和 safetensors 保存。

- **运行环境：** 去掉了 `cd "$(git rev-parse --show-toplevel)"` 和 `LD_LIBRARY_PATH=${CONDA_PREFIX}/lib`，因为镜像用的是 `/opt/venv` 而不是 conda。新增 `unset NVTE_FLASH_ATTN NVTE_FUSED_ATTN NVTE_UNFUSED_ATTN`：基础镜像预设的 `NVTE_FLASH_ATTN=0` / `NVTE_FUSED_ATTN=1` 与 `--attention_backend flash` 冲突，不 unset 会直接报错。
- **模型和数据：** `MODEL` 由 `deepseek-ai/DeepSeek-V4.1-Flash` 改为本地 4 层代理 `/workspace/dsv41_proxy/DeepSeek-V4.1-Flash-4L`。`DATASET` 由 ModelScope ID 改为同一份数据的本地文件 `/workspace/dsv41_proxy/data/distill_r1_110k_sft.jsonl`。
- **并行度：** `PP_SIZE` 由 4 改为 1。原脚本的 PP=4×EP=8 需要 4 机 32 卡；4 层模型在单机 8 卡上用 PP=1，DP 和 global batch 仍为 8，每卡 48 个专家，与原配置相同。
- **训练方式：** `--tuner_type full` 改为 `lora`，rank 8、alpha 32，并加 `--merge_lora false`，只保存 adapter。`--target_modules` 覆盖注意力、CSA2 compressor / indexer 和 MoE 的线性层。没有用 all-linear，因为 mHC、Engram 投影等模块的 LoRA 无法导出到 HF adapter。
- **学习率：** 由 `1e-5 → 1e-6` 改为 LoRA 常用的 `1e-4 → 1e-5`。
- **优化器：** 去掉 `--optimizer_cpu_offload true`。LoRA 的可训练参数很少，优化器状态直接放在显存里即可。
- **输出：** `--output_dir` 改为 `/workspace/dsv41_proxy/output/dsv41-proxy-lora`，并在末尾加 `"$@"` 透传额外参数，例如 `bash run_proxy_lora.sh --max_length 4096`。

## 镜像如何构建（相对 `rocm/primus:v26.9` 的改动）

- **基础环境保持不变：** ROCm 10.2、PyTorch 2.16、TransformerEngine 2.20、flash_attn、apex、triton、transformers 5.10 均为基础镜像原版。安装时用 `constraints.txt` 锁定 torch、torchvision、triton、TE、flash_attn、apex 和 numpy，避免被替换。
- **以 editable 方式安装三个仓库：**

  | 包 | 仓库 / commit | 位置 |
  | --- | --- | --- |
  | megatron-core 0.19.0 | `tastelikefeet/Megatron-LM`，分支 `dsv41-pr7224-engram-local`，`827e940` | `/workspace/Megatron-LM-dsv41` |
  | ms-swift 4.7.0.dev0 | `modelscope/ms-swift` main，`e42be49` | `/workspace/ms-swift` |
  | mcore-bridge 1.8.0.dev0 | `modelscope/mcore-bridge` main，`9e11d36` | `/workspace/ms-swift/mcore-bridge` |

- **本地补丁**（可在对应目录用 `git diff` 查看）：
  - Megatron `csa.py`：原生（未融合）稀疏注意力改为直接索引取 KV，避免反向物化 O(S²·D) 的梯度（8k 序列单次申请约 100GB 显存会导致 OOM）。ROCm 上没有 cuDNN/TileLang 融合 kernel，只能走这条原生路径。
  - Megatron `finalize_model_grads.py`：路由专家偏置更新按专家数分组，兼容 DSpark（128 专家）与主干（384 专家）共存的情况。
  - mcore-bridge `deepseek_v41.py`：导出时跨 EP 收集 Engram 表（否则只会写出当前 rank 的分片，EP=8 时只有 1/8 的行），加载 LoRA adapter 时跳过 Engram 表。
## 模型处理方式（`download_model_and_data.sh` + `make_proxy_from_shards.py`）

1. **只下载需要的分片：** 从 `deepseek-ai/DeepSeek-V4.1-Flash` 只下载 48 个分片中的 8 个（1–6、43、47 号，约 126GB，完整模型为 510GB）。这些分片包含前 4 层、embed、norm/head、视觉塔，以及第 1 层的 Engram 表（单表约 101GB）。
2. **组装 4 层模型：** 用硬链接把这些分片组装成 `DeepSeek-V4.1-Flash-4L`，不复制数据。权重索引只保留前 4 层。`config.json` 中 `text_config` 的改动如下：
   - **层数：** `num_hidden_layers=4`。
   - **注意力布局：** `compress_ratios=[0,0,2,2]`。第 0、1 层为滑动窗口注意力，第 2 层为 CSA2 ratio-2 的 KV/indexer source 层，第 3 层复用第 2 层的结果。对应 `kv_source_layer_ids=index_source_layer_ids=[2]`。
   - **关闭 candidate：** `candidate_source_layer_id=null`，`candidate_topk_blocks=candidate_block_size=0`。
   - **Engram：** `engram_layer_ids=[1]`，`engram_num_embeddings=[384006168]`。Engram 表大小是按层依次分配的素数，截断后第 1 层的行数不变。
   - **去掉 DSpark/MTP：** `num_nextn_predict_layers=0`，并删除所有 `dspark_*` 字段。
   - **保持原样：** hidden size 5120、384 个专家、FP8/FP4 量化格式、tokenizer 和视觉塔。
3. **加载时的处理：** mcore-bridge 把 FP8/FP4 权重反量化为 BF16 用于训练。Engram 表以 FP8 常驻显存（EP 切分），不参与训练。
4. **生成 tokenizer map：** 用 `tools/engram/generate_tokenizer_map.py` 生成 `engram_tokenizer_map.json`，参数为 `--layer-ids 1 --max-ngram-order 4 --pad-token-id 2 --padded-vocab-size 129280`。生成的压缩词表大小为 99092，与配置一致。

另一种做法是镜像中原有的 `make_proxy_config.py` + `run_bootstrap.sh` + `proxy_init_sft.py`。它不下载权重，而是构造随机初始化的代理模型：层布局为 0/2/1/1，保留 DSpark，Engram 词表缩小到 16 万，去掉 FP8/FP4 量化。它的输出目录同样是 `DeepSeek-V4.1-Flash-4L`，与下载脚本二选一。`run_bootstrap.sh` 需要自备 `data/distill_subset.jsonl`。

## 数据集处理方式

- **来源：** HF `Congliu/Chinese-DeepSeek-R1-Distill-data-110k-SFT`（ModelScope 上为 `liucong/Chinese-DeepSeek-R1-Distill-data-110k-SFT`，即原版 `megatron_sft.sh` 的默认数据集）中的 `distill_r1_110k_sft.jsonl`。
- **内容：** 共 11 万条中文 R1 蒸馏样本，格式为 Alpaca 风格的 `instruction` / `input` / `output`，`output` 中包含 `<think>` 推理过程。
- **处理：** 不做离线预处理，原样使用。训练时 ms-swift 自动将其转为 messages，套用 `deepseek_v41` 对话模板，按 `--max_length 8192` 截断，并用 `--split_dataset_ratio 0.01` 留出 1% 作为验证集。tokenize 在训练时逐样本懒加载进行。
