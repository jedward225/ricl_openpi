# aha_ricl — RICL on RLBench (RoboRetry)

> RICL (Retrieval-based In-Context Learning) 适配 RLBench 数据。基于 Pi0-FAST 自回归模型，通过 DINOv2 KNN 检索成功 demo 作为 in-context examples。
> 本目录 fork 自 [openpi-ricl](https://github.com/Physical-Intelligence/openpi) 的 RICL 分支，适配 RLBench 数据格式。

---

## 1. 背景与动机

### 1.1 为什么从 aha_openpi 转向 aha_ricl

`aha_openpi` (Pi0, flow matching) 的实验发现 **conditioning collapse** — 模型忽略 failure context：

- per-k loss 完全 flat，context ablation 显示 context 反而有害
- same-scene pairing (data_v2) 也未能解决问题

**RICL** 提供了已验证的 in-context learning 范式：

- Pi0-FAST (自回归) 天然适合 ICL — 语言模型的 few-shot prompting
- **Block-diagonal + lower-triangular attention**: 每个 retrieved observation 是独立的 "in-context example"
- **Action interpolation**: 检索到的 demo actions 与模型预测加权融合，提供显式的 action prior

### 1.2 当前阶段：Success-Only RICL 复现

先验证 Pi0-FAST + RLBench 的 ICL pipeline 跑通，从同任务成功 demo 中检索 top-4 neighbors：

```
Query observation → DINOv2 embedding → KNN (per-task index)
    → top-4 retrieved success frames
    → [retrieved_0, retrieved_1, retrieved_2, retrieved_3, query] → Pi0-FAST-RICL → actions
```

后续扩展方向：failure chunk retrieval（从失败 demo 中检索相关片段作为 context）。

---

## 2. 当前状态 (2026-03-01)

### 已完成

- ✅ uv 环境搭建 (pyproject.toml 适配, lerobot/autofaiss/opencv-headless)
- ✅ 数据预处理脚本 (`preprocessing/process_rlbench_demos.py`)
- ✅ KNN 检索预处理脚本 (`preprocessing/retrieve_within_rlbench.py`)
- ✅ 数据加载器 (`RiclRLBenchDataset` in `data_loader.py`)
- ✅ Policy transform (`RiclRLBenchInputs` / `RiclRLBenchOutputs`)
- ✅ 训练配置 (`pi0_fast_rlbench_ricl` in `config.py`)
- ✅ 训练脚本 (`scripts/train_ricl_rlbench.py`)
- ✅ Norm stats (复用 aha_openpi 的 8D state + 7D actions 统计)

### 待完成

- ⬜ 运行数据预处理 (Step 1, 需 GPU for DINOv2)
- ⬜ 运行 KNN 检索 (Step 2)
- ⬜ 训练验证 (Step 3, 需集群 8×A100)
- ⬜ 推理/评测脚本适配
- ⬜ Failure chunk retrieval 扩展

---

## 3. 关键参数

| 参数 | 值 | 备注 |
|------|---|------|
| cameras | 3: front + overhead + wrist | 映射: front→top_image→base_0_rgb, overhead→right_image→base_1_rgb, wrist→wrist_image→left_wrist_0_rgb |
| action_dim | 7 | delta-EE: [dx, dy, dz, drx, dry, drz, gripper] |
| state_dim | 8 | EE: [x, y, z, rx, ry, rz, gripper, 0] (同 LIBERO 先例) |
| action_horizon | 10 | 同 LIBERO/DROID-base |
| num_retrieved_observations | 4 | 可配置 0-4, default 4 |
| retrieval scope | per-task | 8 个独立 KNN index (每 task 25 episodes) |
| image encoder | SigLIP frozen | 只 fine-tune Gemma LM |
| action_interpolation | ON, lamda=10.0 | success-only 阶段 |
| max_token_len | 250 | RICL 多 observation 需要更大 token budget |
| training data | 25 eps/task × 8 tasks = 200 eps | |
| base checkpoint | pi0_fast_base | 本地: `/home/ruoqu/jjliu/shared/models/pi0_fast_base/params/` |
| batch_size | 16 | RICL: 5 observations × 3 cameras = 15 images/sample |
| DINOv2 embedding | 仅 front_rgb (top_image) | 49152-dim (ViT-B14, 64-patch spatial reduction) |

---

## 4. 代码结构

```
aha_ricl/
├── preprocessing/
│   ├── process_rlbench_demos.py          # ★ RLBench pkl+PNG → processed_demo.npz
│   ├── retrieve_within_rlbench.py        # ★ Per-task KNN 检索 → indices_and_distances.npz
│   ├── process_collected_demos.py        # (原有) DROID h5py → processed_demo.npz
│   └── retrieve_within_collected_demo_groups.py  # (原有) DROID KNN 检索
├── scripts/
│   ├── train_ricl_rlbench.py             # ★ RLBench RICL 训练 (convenience wrapper)
│   ├── train_pi0_fast_ricl.py            # (原有) 通用 RICL 训练脚本 (CLI config 选择)
│   ├── compute_norm_stats_rlbench.py     # ★ 从 processed npz 计算 norm stats
│   ├── compute_norm_stats.py             # (原有) 从 LeRobot dataset 计算
│   ├── serve_policy_ricl.py              # RICL 推理服务
│   └── train.py                          # (原有) 通用训练脚本 (non-RICL)
├── src/openpi/
│   ├── models/
│   │   ├── pi0_fast_ricl.py              # ★ RICL 模型 (block-diagonal attention, action interpolation)
│   │   ├── pi0_fast.py                   # Pi0-FAST 基础模型
│   │   ├── pi0.py                        # Pi0 (flow matching)
│   │   ├── model.py                      # BaseModel, RiclObservation, Observation
│   │   ├── tokenizer.py                  # FASTTokenizerRicl
│   │   └── siglip.py                     # SigLIP image encoder (frozen)
│   ├── policies/
│   │   ├── rlbench_policy.py             # ★ RiclRLBenchInputs / RiclRLBenchOutputs
│   │   ├── droid_policy.py               # RiclDroidInputs / RiclDroidOutputs (参考)
│   │   ├── libero_policy.py              # LiberoInputs (state_dim=8 先例)
│   │   └── utils.py                      # DINOv2 加载 + embedding 计算
│   ├── training/
│   │   ├── config.py                     # ★ RiclRLBenchDataConfig + pi0_fast_rlbench_ricl
│   │   ├── data_loader.py               # ★ RiclRLBenchDataset + get_action_chunk_rlbench
│   │   ├── optimizer.py                  # CosineDecaySchedule, AdamW
│   │   ├── checkpoints.py               # orbax checkpoint management
│   │   ├── sharding.py                   # FSDP sharding
│   │   └── weight_loaders.py             # CheckpointWeightLoader
│   ├── transforms.py                     # ResizeImagesRicl, TokenizeFASTInputsRicl, Normalize
│   └── shared/
│       ├── normalize.py                  # NormStats, load/save norm_stats.json
│       └── array_typing.py              # JAX array type annotations
├── assets/
│   ├── pi0_fast_rlbench_ricl/rlbench/norm_stats.json  # ★ 归一化统计 (8D state, 7D actions)
│   ├── pi0_fast_droid_ricl/droid/norm_stats.json       # (原有) DROID norm stats
│   └── max_distance.json                               # (原有) DROID 距离归一化
├── pyproject.toml                        # ★ 修改: autofaiss, opencv-headless, uv constraints
└── packages/openpi-client/               # WebSocket 推理客户端
```

`★` 标记为 RLBench 适配新增/修改的文件。

---

## 5. 数据管线

### 5.1 数据格式: processed_demo.npz

每个 episode 预处理为一个 npz 文件：

```
processed_rlbench/{task}/episode_{N}/
├── processed_demo.npz
│   ├── state:                (T, 8)       float32  # EE-space [x,y,z,rx,ry,rz,gripper,0]
│   ├── actions:              (T, 7)       float32  # delta-EE [dx,dy,dz,drx,dry,drz,gripper]
│   ├── top_image:            (T, 224, 224, 3)  uint8  # front_rgb resized 256→224
│   ├── right_image:          (T, 224, 224, 3)  uint8  # overhead_rgb resized
│   ├── wrist_image:          (T, 224, 224, 3)  uint8  # wrist_rgb resized
│   ├── top_image_embeddings: (T, 49152)   float32  # DINOv2 ViT-B14 (仅 front_rgb)
│   └── prompt:               str                   # VLA_TASK_DESCRIPTIONS[task]
└── indices_and_distances.npz  (Step 2 生成)
    ├── retrieved_indices:     (T, k, 2)   int32   # [ep_idx, step_idx] pairs
    ├── query_indices:         (T, 2)      int32
    └── distances:             (T, k+1)    float64  # k个 retrieved + 1个 query-to-first
```

### 5.2 全局元数据 (Step 2 生成)

```
processed_rlbench/
├── ep_idxs_to_fol.json          # {ep_idx: folder_path}
├── fols_to_ep_idxs.json         # {folder_path: ep_idx}
├── groups_to_ep_fols.json       # {task: [folder_paths]}
├── groups_to_ep_idxs.json       # {task: [ep_idxs]}
└── max_distance.json            # {"max_distance": float} 距离归一化
```

Episode indices 从 100000 开始 (RICL convention)。

### 5.3 共享模块依赖

预处理脚本通过 `sys.path.insert(0, "../shared")` 引用 `shared/rlbench_io.py`：

```python
# shared/rlbench_io.py 关键函数
extract_state(obs) → np.ndarray (8,)       # [x,y,z,rx,ry,rz,gripper,0]
extract_delta_action(obs1, obs2) → np.ndarray (7,)  # [dx,dy,dz,drx,dry,drz,gripper]
load_episode_data(episode_path) → dict      # 读取 low_dim_obs.pkl (SafeUnpickler)
VLA_TASK_DESCRIPTIONS: dict                 # task_name → prompt string
```

### 5.4 Camera 映射

```
RLBench camera    →  RICL npz field    →  Model image key
───────────────────────────────────────────────────────────
front_rgb         →  top_image         →  base_0_rgb
overhead_rgb      →  right_image       →  base_1_rgb
wrist_rgb         →  wrist_image       →  left_wrist_0_rgb
```

---

## 6. RICL 模型架构

### 6.1 Block-Diagonal Attention

```
[retrieved_0] [retrieved_1] [retrieved_2] [retrieved_3] [query]
     ↓              ↓              ↓              ↓          ↓
  self-attn     self-attn      self-attn      self-attn   self-attn
  + attn to 0   + attn to     + attn to      + attn to   + attn to
                  0,1           0,1,2          0,1,2,3     0,1,2,3

Lower-triangular: 每个位置看到自己 + 之前所有 retrieved observations
Query 看到所有 4 个 retrieved + 自身
```

每个 observation 包含: 3 images (SigLIP tokens) + prompt + state → ~768 tokens
总 token 数: 5 observations × ~768 = ~3840 tokens

### 6.2 Action Interpolation

训练和推理时，模型预测的 actions 与第一个 retrieved demo 的 actions 加权融合：

```python
weight = exp(-lamda * distance)  # distance = L2(query_embedding, retrieved_0_embedding) / max_distance
final_actions = weight * retrieved_0_actions + (1 - weight) * model_predicted_actions
```

`lamda=10.0`, `max_distance` 从预处理 Step 2 的 `max_distance.json` 获取。

### 6.3 Pi0FASTRiclConfig (关键配置)

```python
Pi0FASTRiclConfig(
    action_dim=7,              # RLBench: 7 (delta-EE)
    action_horizon=10,         # action chunk length
    max_token_len=250,         # per-observation token budget
    num_retrieved_observations=4,  # top-4 KNN neighbors
    use_action_interpolation=True,
    lamda=10.0,
    paligemma_variant="gemma_2b",  # Gemma LM (fine-tuned)
    # SigLIP image encoder frozen via freeze_filter
)
```

---

## 7. 核心代码详解

### 7.1 RiclRLBenchDataset (data_loader.py)

```python
class RiclRLBenchDataset(Dataset):
    def __init__(self, model_config, processed_dir):
        # 1. 加载 ep_idxs_to_fol.json 等映射
        # 2. 加载所有 episode 的 indices_and_distances.npz
        # 3. 归一化 distances (÷ max_distance)
        # 4. 从 npz 读取 prompts

    def __getitem__(self, index) -> dict:
        # 返回 dict:
        #   retrieved_{0..3}_{top,right,wrist}_image: (224,224,3) uint8
        #   retrieved_{0..3}_state: (8,) float32
        #   retrieved_{0..3}_actions: (10, 7) float32
        #   retrieved_{0..3}_prompt: str
        #   query_{top,right,wrist}_image, query_state, query_actions, query_prompt: 同上
        #   exp_lamda_distances: (5, 1) float32  # if action interpolation ON
```

### 7.2 RiclRLBenchInputs (rlbench_policy.py)

```python
@dataclasses.dataclass(frozen=True)
class RiclRLBenchInputs(transforms.DataTransformFn):
    # 映射 dataset dict → model input format
    # top_image → base_0_rgb, right_image → base_1_rgb, wrist_image → left_wrist_0_rgb
    # image_mask: all True (3 cameras always present)
    # 传递: state, actions, prompt, exp_lamda_distances

class RiclRLBenchOutputs(transforms.DataTransformFn):
    # 裁剪 model output actions 到 7D (RLBench action dim)
```

### 7.3 Data Loader 路由 (create_data_loader)

```python
if "rlbench_ricl" in config.name:        # ← 新增
    dataset = RiclRLBenchDataset(config.model, config.processed_dir)
elif "ricl" in config.name:               # 原有 DROID RICL
    dataset = RiclDroidDataset(config.model, config.finetuning_collected_demos_dir)
elif "pi0_fast_droid___finetune_on_" in config.name:
    dataset = Pi0FastDroidFinetuneDataset(...)
else:
    dataset = create_dataset(data_config, config.model)  # LeRobot datasets
```

### 7.4 Action Chunk 提取

```python
def get_action_chunk_rlbench(actions, step_idx, action_horizon):
    """7D delta-EE actions. Padding: zeros + repeat last gripper."""
    # actions: (T, 7)
    # Returns: (action_horizon, 7)
    # 超出范围时: delta actions 补零, gripper (idx 6) 重复最后一帧值
```

---

## 8. 训练配置

### 8.1 预定义配置

```python
# config.py 中的 _CONFIGS
pi0_fast_rlbench_ricl       # ★ RLBench success-only retrieval
pi0_fast_droid_ricl         # (原有) DROID RICL
pi0_fast_droid_ricl___finetune_on_new_task  # (原有) DROID finetuning
pi0_fast_libero             # (原有) LIBERO fine-tuning
# ... 其他 aloha/droid/debug 配置
```

### 8.2 pi0_fast_rlbench_ricl 配置详情

```python
TrainConfig(
    name="pi0_fast_rlbench_ricl",
    processed_dir="./processed_rlbench",
    model=Pi0FASTRiclConfig(
        action_dim=7, action_horizon=10, max_token_len=250,
        num_retrieved_observations=4,
        use_action_interpolation=True, lamda=10.0,
    ),
    data=RiclRLBenchDataConfig(
        repo_id=None,
        assets=AssetsConfig(asset_id="rlbench"),
    ),
    weight_loader=CheckpointWeightLoader(".../pi0_fast_base/params"),
    num_train_steps=20_000,
    batch_size=16,
    freeze_filter=...get_freeze_filter_with_frozen_img_encoder(),
    ema_decay=None,
    lr_schedule=CosineDecaySchedule(
        warmup_steps=300, peak_lr=2.5e-5,
        decay_steps=15000, decay_lr=2.5e-6,
    ),
)
```

### 8.3 Norm Stats

路径: `assets/pi0_fast_rlbench_ricl/rlbench/norm_stats.json`

当前复用 aha_openpi 的统计 (基于 2000 success episodes)。
如需从 25 episodes 子集重新计算:

```bash
python scripts/compute_norm_stats_rlbench.py \
    --processed_dir ./processed_rlbench \
    --output_dir ./assets/pi0_fast_rlbench_ricl/rlbench
```

---

## 9. 路径映射

### 本地开发 (Ubuntu, RTX 5090 32GB)

| 用途 | 路径 |
|------|------|
| 代码 | `/home/ruoqu/jjliu/RoboRetry/aha_ricl/` |
| AHA 原始数据 | `/home/ruoqu/jjliu/AHA/data_v1/` |
| Pi0-FAST base checkpoint | `/home/ruoqu/jjliu/shared/models/pi0_fast_base/params/` |
| 预处理输出 | `./processed_rlbench/` (相对于 aha_ricl) |
| 训练 checkpoints | `./checkpoints/pi0_fast_rlbench_ricl/{exp_name}/` |
| Norm stats | `./assets/pi0_fast_rlbench_ricl/rlbench/norm_stats.json` |
| shared 模块 | `/home/ruoqu/jjliu/RoboRetry/shared/` (rlbench_io.py) |

### 集群 (8×A100)

| 用途 | 路径 |
|------|------|
| 代码 | `/root/RoboRetry/aha_ricl/` |
| 原始数据 | `/cephfs/shared/jjliu/data_v1/` |
| Pi0-FAST base | `/cephfs/shared/models/pi0_fast_base/params/` (需确认) |
| 预处理输出 | `./processed_rlbench/` |
| 训练 checkpoints | `./checkpoints/` |

集群无法 SSH 直连，代码通过 GitHub 同步。

---

## 10. 常用命令

```bash
cd /home/ruoqu/jjliu/RoboRetry/aha_ricl

# === Step 1: 数据预处理 (需 GPU for DINOv2, ~8GB 输出) ===
python preprocessing/process_rlbench_demos.py \
    --data_root /home/ruoqu/jjliu/AHA/data_v1 \
    --output_dir ./processed_rlbench \
    --tasks all --num_episodes 25

# === Step 2: KNN 检索预处理 ===
python preprocessing/retrieve_within_rlbench.py \
    --processed_dir ./processed_rlbench \
    --knn_k 100 --embedding_type top_image

# === Step 3 (可选): 重新计算 norm stats ===
python scripts/compute_norm_stats_rlbench.py \
    --processed_dir ./processed_rlbench

# === Step 4: 训练 ===
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
uv run scripts/train_ricl_rlbench.py --exp_name=rlbench_ricl_v1

# 或者用通用训练脚本:
uv run scripts/train_pi0_fast_ricl.py pi0_fast_rlbench_ricl --exp_name=rlbench_ricl_v1

# === 验证预处理结果 ===
python -c "
import numpy as np
d = np.load('processed_rlbench/pick_up_cup/episode_0/processed_demo.npz', allow_pickle=True)
for k in d.files:
    v = d[k]
    print(f'{k}: {v.shape if hasattr(v, \"shape\") else v}')
"

# === 验证 KNN 检索结果 ===
python -c "
import numpy as np, json
d = np.load('processed_rlbench/pick_up_cup/episode_0/indices_and_distances.npz')
print(f'retrieved_indices: {d[\"retrieved_indices\"].shape}')
print(f'distances: {d[\"distances\"].shape}')
m = json.load(open('processed_rlbench/max_distance.json'))
print(f'max_distance: {m[\"max_distance\"]:.4f}')
"

# === 验证 config 加载 ===
uv run python -c "
import openpi.training.config as config
c = config.get_config('pi0_fast_rlbench_ricl')
print(f'action_dim={c.model.action_dim}, horizon={c.model.action_horizon}, retrieved={c.model.num_retrieved_observations}')
"
```

---

## 11. 与 aha_openpi 的关键差异

| 方面 | aha_openpi (ICRL) | aha_ricl (RICL) |
|------|-------------------|-----------------|
| 模型 | Pi0 (flow matching) | Pi0-FAST (autoregressive) |
| Context 来源 | 失败 demo (failure context) | 成功 demo (KNN retrieval) |
| Context 注入方式 | context_mask + 额外 image slots | block-diagonal attention (独立 observations) |
| Action 生成 | flow matching (连续) | FAST tokenizer (离散 256-bin) |
| 数据格式 | LeRobot parquet | processed_demo.npz |
| Prompt 格式 | "Attempt 1: {desc}. Task: {task}" | 单纯 task description |
| 检索 | 无 (预分配 failure context) | DINOv2 KNN (per-task, top-4) |
| Action interpolation | 无 | exp(-λ·dist) 加权融合 |
| Images/sample | 2-5 (front + wrist + 0~3 context) | 15 (5 obs × 3 cameras) |

---

## 12. 与原有 RICL-DROID 的关键差异

| 方面 | RICL-DROID (原有) | RICL-RLBench (新增) |
|------|-------------------|---------------------|
| action_dim | 8 (joint_vel 7 + gripper 1) | 7 (delta-EE 6 + gripper) |
| state_dim | 8 (joint_pos 7 + gripper 1) | 8 (EE 6 + gripper + 0) |
| action_horizon | 15 | 10 |
| action chunk 函数 | `get_action_chunk()` (split vel/gripper) | `get_action_chunk_rlbench()` (单数组) |
| 数据来源 | DROID h5py + 采集 demo | RLBench pkl + PNG |
| DINOv2 embedding | wrist_image (检索用) | top_image/front_rgb (检索用) |
| Episode idx 分支 | `if ep_idx < 100000` (DROID) else (collected) | 全部 ≥100000 (无 DROID 分支) |
| Prompt | 从文件夹名推断 | 从 npz prompt 字段读取 |
| max_distance 格式 | `assets/max_distance.json: {distances: {max: float}}` | `processed_rlbench/max_distance.json: {max_distance: float}` |

---

## 13. DINOv2 Embedding 细节

```python
# src/openpi/policies/utils.py
EMBED_DIM = 49152  # 768 (ViT-B14 hidden) × 64 (spatial patches)

load_dinov2()  # → dinov2_vitb14 from torch.hub, eval mode, CUDA
embed_with_batches(images, model, batch_size=256)
# images: (N, 224, 224, 3) uint8
# output: (N, 49152) float32 — 64 patch tokens flattened
```

KNN 检索使用 L2 距离，`autofaiss.build_index()` 自动优化索引结构。

---

## 14. 踩坑记录

1. **pyav 依赖**: lerobot 依赖 pyav，但 pyav 在非 x86_64 Linux 上不可用。解决: `[tool.uv] environments = ["sys_platform == 'linux' and platform_machine == 'x86_64'"]`
2. **lerobot 版本**: 必须与 aha_openpi 使用相同的 lerobot commit (`0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`)，否则依赖冲突
3. **override-dependencies**: 需要 `ml-dtypes==0.4.1` 和 `tensorstore==0.1.74` 固定版本
4. **npz prompt 读取**: numpy 存储 string 为 `np.str_`，读取时需 `str(npz["prompt"])` 转换
5. **Action padding**: RLBench 7D actions padding 时 delta 补零，gripper 重复最后值 (idx 6)
6. **DROID action padding**: DROID 8D actions 分为 joint_vel (7) + gripper (1) 两部分处理，与 RLBench 不同
7. **Pi0-FAST action_dim < state_dim**: RLBench action_dim=7 < state_dim=8，不需要 state padding (LIBERO 先例)
8. **FAST tokenizer 精度**: 7D delta-EE (含 Euler 角) 通过 256-bin 离散化可能精度下降，需验证
9. **DINOv2 显存**: `embed_with_batches` 默认 batch_size=256，若 GPU 显存不足需调小
10. **集群 checkpoint 路径**: pi0_fast_base 在集群上的位置需要确认 (不同于 pi0_base)
11. **opencv-python-headless**: 避免 `qt.qpa.plugin: Could not find "xcb"` 错误

---

## 15. 存储估算

| 内容 | 大小 |
|------|------|
| Images (3 cameras × 200 eps × ~150 frames × 224×224×3) | ~6.3 GB |
| DINOv2 embeddings (仅 front_rgb, 200 eps × ~150 × 49152 × 4B) | ~2 GB |
| State + Actions (200 eps × ~150 × (8+7) × 4B) | ~18 MB |
| KNN indices (200 eps × ~150 × 100 × 2 × 4B) | ~24 MB |
| **总计 (processed_rlbench/)** | **~8.4 GB** |

---

## 16. 后续扩展方向

### 16.1 Failure Chunk Retrieval

将 failure demo 也构建 KNN index，查询时同时检索 success demo 和 failure demo。
retrieved observations 可以包含 "这里失败了" 的 context，类似 aha_openpi 的 failure context 但通过检索自动关联。

### 16.2 Cross-Task Retrieval

当前 per-task index，后续可尝试跨任务检索 (参考 ExpReS-VLA)。
需要更大的 embedding 空间和更精细的距离度量。

### 16.3 与 aha_openpi Failure Context 结合

RICL retrieval (success demo context) + ICRL failure context (failure description/image) 的混合方案。

---

*最后更新: 2026-03-01*
