# DiffSynth-Studio 训练镜像（ROCm 10.2 / PyTorch 2.16 / Python 3.12）

镜像：`amdagi/training-example:DiffSynth_rocm10.2_torch2.16_1124_py312`

- **基础镜像**：`rocm/primus:v26.9`（`sha256:921604bee3a7289e95f37db5a4d990dc530f64bdbcabd80efd952aef98647d27`）
- **DiffSynth-Studio**：[modelscope/DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) `main` 分支 commit `3093e5599430e5605054cd052631344ae4298891`（2026-11-24），已 `pip install -e` 安装
- **验证范围**：`examples/minimax_h3/model_training` 下全部 24 个训练脚本，在 8× AMD Instinct MI350X 上**全部能够正常训练**（结果见第 7 节）
- **构建方式**：镜像只通过本目录下的 `Dockerfile` 用 `docker build` 生成，**不包含任何模型权重和数据集**

## 1. 镜像内容

```
/workspace/DiffSynth-Studio/          DiffSynth-Studio 源码（含 .git，commit 3093e559），工作目录
/workspace/DiffSynth-ROCm/
├── Dockerfile                        生成本镜像的 Dockerfile
├── .dockerignore
├── README.md                         本文档
├── requirements-rocm.txt             在基础镜像之上新增的 Python 包（固定版本）
├── tools/
│   ├── check_env.py                  环境自检（docker build 时跑 CPU 部分；--gpu 跑 GPU 部分）
│   ├── download_models.py            下载并校验全部脚本用到的模型（HF 镜像优先，sha256 校验）
│   ├── download_datasets.sh          下载官方样例数据集 diffsynth_example_dataset
│   └── quick_check.py                24 个训练脚本的快速训练验证与汇总报表
└── results/                          测试结果（汇总表、每个脚本的训练日志 / 实际执行的脚本 / loss 记录）
```

## 2. 相对基础镜像的修改

ROCm、PyTorch 系列（torch / torchvision / torchaudio / triton）以及 transformers、accelerate、peft、diffusers 等均**保持基础镜像原版本不变**：Dockerfile 先把它们的版本写入 `/opt/base-constraints.txt`，之后所有 `pip install` 都带 `-c` 约束，避免被替换成 CUDA 版本。

| 组件 | 基础镜像 | 本镜像 | 原因 |
|---|---|---|---|
| torchcodec | 无 | 0.17.0（源码编译，CPU 解码，链接系统 FFmpeg 6.1） | 镜像里的 torchaudio 2.11 读音频依赖 torchcodec；缺失时 DiffSynth 只打印一条 warning，然后在没有音频的情况下继续训练。PyPI 的 torchcodec 是 CUDA 版（依赖 `libnvrtc.so.13`），无法在 ROCm 上加载 |
| torchao | 0.15.0（源码编译） | 0.16.0（PyPI） | 镜像自带的 peft 0.21.2 要求 torchao ≥ 0.16，否则注入 LoRA 时直接 `ImportError`。0.16.0 的 C++ 扩展与 torch 2.16 不匹配会被自动跳过，纯 Python 部分可用 |
| bitsandbytes | 无 | 0.50.2 | NF4 量化（`DiffSynth-Studio/MiniMax-H3-NF4` 权重和 `special/quant_training`）。没有 ROCm 7.17 的预编译库，会自动使用 ROCm 7.15 的版本，并打印 `No prebuilt binary for ROCm 7.17, loading ROCm 7.15 instead`，已验证可用 |
| comfy-kitchen | 无 | 0.2.42 | ComfyUI 的 int8_convrot / fp8_scaled 权重（`Comfy-Org/MiniMax-H3`）。在 ROCm 上 DiffSynth 会禁用它的 CUDA 后端；HIP 后端只支持 RDNA，不支持 gfx950；实际运行在 triton / eager 后端上 |
| deepspeed | 无 | 0.20.0 | `full/*` 中 ZeRO-3 全量训练 |
| modelscope、imageio(+ffmpeg)、ftfy、entropack、librosa 等 | 无 | 见 `requirements-rocm.txt` | DiffSynth-Studio 及其 audio / training 依赖 |
| diffsynth | 无 | 2.2.3（editable，`/workspace/DiffSynth-Studio`） | 本镜像的主体 |
| apt 包 | — | FFmpeg 开发头文件、libjpeg-turbo / libpng / libwebp 开发包 | 编译 torchcodec |

## 3. 使用方法

### 3.1 启动容器

模型和数据集放在宿主机上，挂载进容器。下面把它们挂到仓库内的 `models/`、`data/`，这样官方脚本里的相对路径可以直接使用，训练产出（`./models/train/...`）也会写回宿主机：

```bash
docker run -it --name diffsynth \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host --network=host --shm-size=256g \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v /path/to/models:/workspace/DiffSynth-Studio/models \
  -v /path/to/data:/workspace/DiffSynth-Studio/data \
  amdagi/training-example:DiffSynth_rocm10.2_torch2.16_1124_py312

# 容器内自检：版本、torchaudio/torchcodec、peft+torchao、注意力后端、NF4 / int8 / fp8 量化线性层
python /workspace/DiffSynth-ROCm/tools/check_env.py --gpu
```

`import diffsynth` 时会触发 aiter 的 JIT，它需要通过 `rocminfo` 检测 GPU，所以在没有挂载 GPU 的容器里导入会失败。

### 3.2 下载模型和数据集

```bash
cd /workspace/DiffSynth-ROCm/tools
python download_models.py      # 全部约 630 GB，下载到 $H3_MODELS（默认 /workspace/DiffSynth-Studio/models）并校验 sha256
bash download_datasets.sh      # 约 80 MB，下载到 $H3_DATA（默认 /workspace/DiffSynth-Studio/data）
```

| ModelScope ID（DiffSynth 的本地目录名） | HuggingFace 镜像 | 用到的文件 | 大小 |
|---|---|---|---|
| `MiniMax/MiniMax-H3` | `MiniMaxAI/MiniMax-H3` | `FL2VA/*`、`Ref2VA/*` | 288 GB |
| `Comfy-Org/MiniMax-H3` | `Comfy-Org/MiniMax-H3` | 8 个剪枝 / FP8 / Int8 DiT 和 Int8 文本编码器 | 260 GB |
| `DiffSynth-Studio/MiniMax-H3-NF4` | 同名 | 全部 | 73 GB |
| `PAI/MiniMax-H3-Fun-Controlnet-Union` | `alibaba-pai/MiniMax-H3-Fun-Controlnet-Union` | 全部 | 6.8 GB |
| `lightx2v/Minimax-h3-Turbo` | 同名 | `minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors`（仅 Text-Embeddings 的验证脚本使用） | 1.4 GB |
| `DiffSynth-Studio/MiniMax-H3-Text-Embeddings` | 同名 | 全部 | 50 MB |
| `DiffSynth-Studio/MiniMax-H3-TrainingAdapter` | 无（只能从 ModelScope 下载） | 全部（LoRA 脚本中注释掉的可选项） | 2.5 GB |

- **预下载之后**：设置 `DIFFSYNTH_SKIP_DOWNLOAD=True`，否则多卡训练时每个进程都会去请求 ModelScope。

### 3.3 运行官方训练脚本

```bash
cd /workspace/DiffSynth-Studio
export DIFFSYNTH_SKIP_DOWNLOAD=True
# 脚本第一行会用 modelscope 再拉一次样例数据集；modelscope.cn 很慢时可以改用国际站
export MODELSCOPE_ENDPOINT=https://modelscope.ai MODELSCOPE_DOMAIN=modelscope.ai
bash examples/minimax_h3/model_training/lora/MiniMax-H3-FL2VA.sh
```

- `accelerate launch` 没有指定配置时会使用全部可见 GPU，可以用 `HIP_VISIBLE_DEVICES` 限定。`full/` 下的 ZeRO-3 脚本使用 `accelerate_config_zero3.yaml`，固定 8 个进程。
- 两阶段脚本的第二阶段，每个进程的步数 = `dataset_repeat × num_epochs`，与 GPU 数无关（第一阶段每个进程各缓存一份样本）。
- 很多脚本共用同一个输出目录，例如 `./models/train/MiniMax-H3-T2VA-split`；验证脚本也按这些固定路径读取 checkpoint。连续跑多个脚本时请分目录运行，或者先把上一次的产出挪走。

### 3.4 复现本文的快速验证

需要 8 张 GPU，并已按 3.2 下载好模型和数据集：

```bash
cd /workspace/DiffSynth-ROCm/tools
H3_OUT=/workspace/h3_quick_check python quick_check.py              # 全部 24 个，约 45 分钟
H3_JOBS=lora_FL2VA,full_Ref2VA python quick_check.py                # 只跑部分任务
H3_OUT=/workspace/h3_quick_check python quick_check.py --report     # 汇总表
```

### 3.5 重新构建镜像

```bash
mkdir ctx && cp -r /workspace/DiffSynth-Studio /workspace/DiffSynth-ROCm ctx/
cp /workspace/DiffSynth-ROCm/Dockerfile /workspace/DiffSynth-ROCm/.dockerignore ctx/
cd ctx && docker build -t amdagi/training-example:DiffSynth_rocm10.2_torch2.16_1124_py312 .
```

## 4. 测试环境

- 8× AMD Instinct MI350X（gfx950，每卡 252 GiB HBM），宿主机 amdgpu 驱动 6.16.13，Linux 5.15.0-70，384 CPU 线程
- 软件环境即本镜像（ROCm 10.2 / HIP 7.17，PyTorch 2.16.0a0，Python 3.12.3）。DiffSynth 自动选择的注意力后端是 ROCm 版 flash_attention_2
- 测试日期：2026-11-24

## 5. 测试方法

快速验证（逻辑即 `tools/quick_check.py`）：

- **测试环境**：24 个脚本都在开发容器中测过。开发容器是 `rocm/primus:v26.9`，手工安装了与本镜像完全相同的包版本，本 Dockerfile 就是把这套环境固化下来。
- **执行范围**：每个官方脚本都完整执行，包括第一阶段数据缓存、第二阶段训练和保存 checkpoint。
- **唯一的改动**：第二阶段缩短为 1 个 epoch，每进程约 5 步。两阶段脚本设 `--dataset_repeat 5`；单阶段脚本（NF4、quant_training、Text-Embeddings）设 `--dataset_repeat 10`，2 个进程各 5 步。
- **辅助修改**（`results/run_scripts/` 中可以看到实际执行的脚本）：
  - 跳过已预下载的数据集下载行
  - 加 `--main_process_port`，让并发任务使用不同端口
  - 加 `--enable_csv_log` 记录 loss
  - ZeRO-3 任务存出整模型 checkpoint 后，记录大小再删除（单个 40~66 GB）
- **隔离**：每个脚本在独立工作目录运行，`examples/`、`data/` 和模型目录用软链共享。
- **GPU 分配**：非 ZeRO-3 脚本每个 2 卡、4 个并行（仍然走多卡 DDP）；ZeRO-3 脚本独占 8 卡。
- **通过标准**：退出码为 0、各阶段 loss 都是有限值、checkpoint 成功写出。
- **不覆盖**：训练后的推理验证脚本（`validate_`*）；性能调优和基准。表中的 s/it 来自 5 步的短跑，包含预热，只供参考。

此外，`lora/MiniMax-H3-FL2VA.sh` 按官方完整步数跑过一次，见 7.2。

## 6. 结论

`examples/minimax_h3/model_training` 下的 24 个训练脚本覆盖了以下全部组合，在本镜像中都能正常训练：

- FL2VA / Ref2VA 两个分区
- 官方 bf16、Comfy 剪枝 bf16、FP8、Int8 ConvRot、DiffSynth NF4、bitsandbytes 动态 NF4 等权重格式
- LoRA / 全量 / ControlNet / Text-Embeddings / split / quant 等训练方式
- DDP 与 DeepSpeed ZeRO-3

## 7. 测试结果

### 7.1 快速验证（24/24 通过）

| # | 脚本 | DiT 权重 / 训练方式 | GPU | 每个任务的训练步数 | 最后一步 loss | checkpoint | 耗时 | 结果 |
|---|---|---|---|---|---|---|---|---|
| 1 | `full/MiniMax-H3-FL2VA.sh` | 官方 FL2VA bf16，全量 DiT，ZeRO-3 | 8 | 5 + 5 | 0.5801 | 2 × 整模型 66 GB（记录后删除） | 16 min | 通过 |
| 2 | `lora/MiniMax-H3-FL2VA.sh` | 官方 FL2VA bf16 | 2 | 5 + 5 | 0.3569 | 2 × 0.31 GB | 8 min | 通过 |
| 3 | `special/split_training/MiniMax-H3-FL2VA.sh` | 官方 FL2VA bf16，`--offload_models` 分阶段卸载 | 2 | 5 | 0.4888 | 1 × 0.31 GB | 2 min | 通过 |
| 4 | `special/quant_training/MiniMax-H3-FL2VA-bitsandbytes_nf4.sh` | 官方 FL2VA bf16，加载时 bitsandbytes NF4 量化 | 2 | 5 + 5 | 0.2056 | 2 × 0.31 GB | 4 min | 通过 |
| 5 | `lora/MiniMax-H3-Pruned-FL2VA.sh` | Comfy FL2VA 剪枝 bf16 | 2 | 5 + 5 | 0.5785 | 2 × 0.31 GB | 4 min | 通过 |
| 6 | `lora/MiniMax-H3-FP8-Pruned-FL2VA.sh` | Comfy FL2VA 剪枝 FP8 scaled（comfy-kitchen） | 2 | 5 + 5 | 0.5285 | 2 × 0.31 GB | 4 min | 通过 |
| 7 | `lora/MiniMax-H3-Int8-ConvRot-FL2VA.sh` | Comfy FL2VA Int8 ConvRot + Int8 文本编码器（comfy-kitchen） | 2 | 5 + 5 | 0.5912 | 2 × 0.31 GB | 4 min | 通过 |
| 8 | `lora/MiniMax-H3-Int8-ConvRot-Pruned-FL2VA.sh` | Comfy FL2VA 剪枝 Int8 ConvRot（comfy-kitchen） | 2 | 5 + 5 | 0.2797 | 2 × 0.31 GB | 4 min | 通过 |
| 9 | `lora/MiniMax-H3-NF4-FL2VA.sh` | DiffSynth FL2VA NF4（单阶段，bitsandbytes） | 2 | 5 + 5 | 0.6160 | 2 × 0.31 GB | 3 min | 通过 |
| 10 | `lora/MiniMax-H3-NF4-Pruned-FL2VA.sh` | DiffSynth FL2VA 剪枝 NF4（单阶段，bitsandbytes） | 2 | 5 + 5 | 0.0000 | 2 × 0.31 GB | 3 min | 通过 |
| 11 | `lora/MiniMax-H3-Ref2VA.sh` | 官方 Ref2VA bf16（参考图 + 参考音频） | 2 | 5 | 0.3833 | 1 × 0.31 GB | 10 min | 通过 |
| 12 | `lora/MiniMax-H3-Pruned-Ref2VA.sh` | Comfy Ref2VA 剪枝 bf16 | 2 | 5 | 0.3136 | 1 × 0.31 GB | 6 min | 通过 |
| 13 | `lora/MiniMax-H3-FP8-Pruned-Ref2VA.sh` | Comfy Ref2VA 剪枝 FP8 scaled（comfy-kitchen） | 2 | 5 | 0.2085 | 1 × 0.31 GB | 5 min | 通过 |
| 14 | `lora/MiniMax-H3-Int8-ConvRot-Ref2VA.sh` | Comfy Ref2VA Int8 ConvRot（comfy-kitchen） | 2 | 5 | 0.4239 | 1 × 0.31 GB | 3 min | 通过 |
| 15 | `lora/MiniMax-H3-Int8-ConvRot-Pruned-Ref2VA.sh` | Comfy Ref2VA 剪枝 Int8 ConvRot（comfy-kitchen） | 2 | 5 | 0.5959 | 1 × 0.31 GB | 2 min | 通过 |
| 16 | `lora/MiniMax-H3-NF4-Ref2VA.sh` | DiffSynth Ref2VA NF4（单阶段，bitsandbytes） | 2 | 5 | 0.7321 | 1 × 0.31 GB | 2 min | 通过 |
| 17 | `lora/MiniMax-H3-NF4-Pruned-Ref2VA.sh` | DiffSynth Ref2VA 剪枝 NF4（单阶段，bitsandbytes） | 2 | 5 | 0.1090 | 1 × 0.31 GB | 2 min | 通过 |
| 18 | `lora/MiniMax-H3-Fun-Controlnet-Union.sh` | 官方 FL2VA bf16 + PAI ControlNet（LoRA 训 DiT） | 2 | 5 | 0.9507 | 1 × 0.31 GB | 2 min | 通过 |
| 19 | `special/quant_training/MiniMax-H3-Ref2VA-bitsandbytes_nf4.sh` | 官方 Ref2VA bf16，加载时 bitsandbytes NF4 量化 | 2 | 5 | 0.6864 | 1 × 0.31 GB | 3 min | 通过 |
| 20 | `full/MiniMax-H3-Fun-Controlnet-Union.sh` | 官方 FL2VA bf16（冻结）+ 全量训练 PAI ControlNet | 2 | 5 | 0.9161 | 1 × 6.80 GB | 2 min | 通过 |
| 21 | `full/MiniMax-H3-Text-Embeddings.sh` | DiffSynth FL2VA 剪枝 NF4 + 全量训练文本嵌入模板 | 2 | 5 | 0.1119 | 1 × 0.5 MB | 1 min | 通过 |
| 22 | `full/MiniMax-H3-Ref2VA.sh` | 官方 Ref2VA bf16，全量 DiT，ZeRO-3 | 8 | 5 | 0.1330 | 1 × 整模型 66 GB（记录后删除） | 4 min | 通过 |
| 23 | `full/MiniMax-H3-Pruned-FL2VA.sh` | Comfy FL2VA 剪枝 bf16，全量 DiT，ZeRO-3 | 8 | 5 + 5 | 0.2183 | 2 × 整模型 40 GB（记录后删除） | 5 min | 通过 |
| 24 | `full/MiniMax-H3-Pruned-Ref2VA.sh` | Comfy Ref2VA 剪枝 bf16，全量 DiT，ZeRO-3 | 8 | 5 | 0.6924 | 1 × 整模型 40 GB（记录后删除） | 3 min | 通过 |

24/24 通过。

- **原始记录**：每个脚本的训练日志、实际执行的脚本副本和 loss 记录分别在 `results/logs/`、`results/run_scripts/`、`results/loss/`；汇总表见 `results/quick_check_report.txt`（开发容器）和 `results/image_check_report.txt`（最终镜像）。

### 7.2 `lora/MiniMax-H3-FL2VA.sh` 官方完整步数

运行环境与本镜像的软件环境一致，8 卡 DDP，官方参数未修改，只加了 `--enable_csv_log`。

- **T2VA 第二阶段**：500 步（5 epoch × 100），稳定在 5.56 s/step，每卡显存约 84 GiB，保存了 5 个 LoRA checkpoint（每个 310 MB，覆盖 52 个 block 的 `qkv_proj / out_proj / fc1 / fc2`，共 1.55 亿参数）。各 epoch 平均 loss 依次为 0.301、0.146、0.117、0.103、0.096。
- **FL2VA**：第一阶段完成；第二阶段训练 79 步（约 6.4 s/step，平均 loss 0.255）后，因切换到快速验证而手动停止。

日志和 loss 记录分别在 `results/logs/lora_FL2VA_full_schedule.train.log` 和 `results/loss/lora_FL2VA_full_schedule/`。

## 8. 已知问题与注意事项

1. **上游脚本笔误**：`lora/MiniMax-H3-Pruned-FL2VA.sh` 的 T2VA 第二阶段写成了 `--num_epochs 1000 --dataset_repeat 500`，约 50 万步，并会保存 1000 个 checkpoint。对应的验证脚本读取 `epoch-4`，应当与其他脚本一样是 `5 × 100`。
2. **上游验证脚本笔误**：`validate_full/MiniMax-H3-Fun-Controlnet-Union.py` 第 31 行写的是 `torch.bflaot16`，运行到这里会报 `AttributeError`，应为 `torch.bfloat16`。
3. **音频被静默丢弃的风险**：如果 torchcodec 不可用，DiffSynth 读不到训练音频时只会打一条 warning，然后在没有音频的情况下训练。本镜像已修复，`check_env.py` 会做一次读写往返校验。
4. **可以忽略的日志**：
   - torchao 的 `Skipping import of cpp extensions due to incompatible torch version`
   - bitsandbytes 的 `loading ROCm 7.15 instead`
   - peft 探测 bitsandbytes 时打印的 `kernels-community` 提示
5. **comfy-kitchen 的 Int8 / FP8 算子**：在 gfx950 上走 triton / eager 后端，功能正确，但没有针对性能优化。
6. **ZeRO-3 全量训练的 checkpoint**：每个 epoch 保存一次整模型 bf16 checkpoint（官方 DiT 66 GB、剪枝版 40 GB），需要预留足够磁盘。
