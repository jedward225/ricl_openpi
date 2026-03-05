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

## 2. 当前状态 (2026-03-05)

### 已完成

- ✅ uv 环境搭建 (pyproject.toml 适配, lerobot/autofaiss/opencv-headless)
- ✅ 数据预处理脚本 (`preprocessing/process_rlbench_demos.py`)
- ✅ KNN 检索预处理脚本 (`preprocessing/retrieve_within_rlbench.py`)
- ✅ 数据加载器 (`RiclRLBenchDataset` in `data_loader.py`)
- ✅ Policy transform (`RiclRLBenchInputs` / `RiclRLBenchOutputs`)
- ✅ 训练配置 (`pi0_fast_rlbench_ricl` in `config.py`)
- ✅ 训练脚本 (`scripts/train_ricl_rlbench.py`)
- ✅ Euler angle wrapping fix (`shared/rlbench_io.py`, see §17.1)
- ✅ SafeUnpickler (no CoppeliaSim dependency for pkl loading)
- ✅ 数据预处理 (集群 3090, euler fix 后全量重新处理)
- ✅ Norm stats 重新计算 (euler fix 后, drz 从偏斜 [-0.024, 0.144] → 对称 [-0.059, 0.075])
- ✅ KNN 检索完成
- ✅ **Success-only RICL 训练完成** (8×A100, 20k steps, loss ~0.01, `rlbench_ricl_v2`)
- ✅ 评测脚本 (`scripts/eval_ricl_rlbench.py`)
- ✅ `policy.py` 修复 max_distance.json 加载 (支持 RLBench 格式 + demos_dir 路径)

### 待完成

- ⬜ **RICL baseline 评测** (下载 checkpoint → 本地预处理 → eval, 最高优先级)
- ⬜ DPO pair 诊断实验 (expert vs failure action chunk 的 L2 距离分布)
- ⬜ Failure chunk bank 构建 (DINOv2 embedding + per-waypoint index)
- ⬜ Stage-aware failure retrieval + DPO 训练 (see §18)
- ⬜ 全量评测 + ablation

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
| Pi0-FAST base | `/cephfs_ssd_xumengdi/models/pi0_fast_base` |
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

见 §18 的完整提案 (Failure-Aware In-Context Policy Learning via Stage-Aware Contrastive Retrieval)。

---

## 17. 技术修复与验证 (2026-03-02 ~ 03-05)

### 17.1 Euler Angle Wrapping Bug Fix

**问题**: `shared/rlbench_io.py` 的 `extract_delta_action()` 未处理 euler angle 在 ±π 边界的不连续。
例如 rz 从 3.14 → -3.14 时，delta 计算为 -6.28 而非 ≈0。

**修复** (line 164):
```python
delta_euler = (delta_euler + np.pi) % (2 * np.pi) - np.pi
```

**Norm stats 验证结果** (修复前 vs 修复后):

| 维度 | 修复前 q01/q99 | 修复后 q01/q99 | 变化 |
|------|---------------|---------------|------|
| drz  | [-0.024, 0.144] (偏斜) | [-0.059, 0.075] (对称) | Range 0.168 → 0.134 |
| drx  | [-0.031, 0.028] | [-0.031, 0.028] | 无变化 (不受影响) |
| dry  | [-0.023, 0.051] | [-0.023, 0.051] | 无变化 |

drz 从明显右偏 (正方向异常尾巴) 变为对称分布，确认 wrapping bug 已修复。

### 17.2 SafeUnpickler

在 `shared/rlbench_io.py` 中添加 `SafeUnpickler` 类，使得读取 RLBench `low_dim_obs.pkl` 不再需要 CoppeliaSim/PyRep 环境。
所有预处理脚本和 `build_failure_bank.py` 均使用此 unpickler。

### 17.3 FAST Tokenizer 精度验证

**测试脚本**: `scripts/test_fast_roundtrip.py`

对全部 8 tasks × 25 episodes 做 action → tokenize → detokenize roundtrip 精度验证。

**结论: PASS** — FAST tokenizer 精度对 RLBench 完全够用。

| 维度 | Mean Error | P95 Error | Mean/Range |
|------|-----------|-----------|------------|
| dx/dy/dz | 0.15-0.45mm | 0.4-1.2mm | 0.4-0.7% |
| drx/dry/drz | 0.001-0.0015 rad | 0.002-0.004 rad | 0.5-0.8% |
| gripper | 0.007 | 0.015 | 0.7% |

insert_usb_in_computer (最精细任务): position p95 < 0.87mm, rotation p95 < 0.003 rad。

### 17.4 Success-Only RICL 训练

- **集群**: 8×A100 80GB
- **Exp name**: `rlbench_ricl_v2`
- **Steps**: 20,000
- **Final loss**: ~0.01，收敛良好
- **Checkpoint**: `checkpoints/pi0_fast_rlbench_ricl/rlbench_ricl_v2/20000/`
- **Base model**: `pi0_fast_base` (from `/cephfs_ssd_xumengdi/models/pi0_fast_base`)

### 17.5 Eval 脚本 & policy.py 修复

- 新增 `scripts/eval_ricl_rlbench.py`: 完整 RLBench 评测脚本 (3 cameras, action chunk 执行, per-task max steps, video recording)
- 修复 `policy.py` 中 `max_distance.json` 加载: 支持 RLBench 格式 (`{max_distance: float}`) + DROID 格式 (`{distances: {max: float}}`), 优先查 `demos_dir/max_distance.json`

---

## 18. 提案: Failure-Aware In-Context Policy Learning via Stage-Aware Contrastive Retrieval

### 18.1 核心问题

前期实验证明 **BC (behavior cloning) 结构性无法利用 failure context**:

1. **Conditioning collapse** (Pi0, flow matching): 模型直接忽略 context conditioning (Cocos, NeurIPS 2025 理论证实)
2. **P(a\* | s, c_fail) = P(a\* | s)**: 在 BC 目标下，expert action 本就不依赖 failure context — 优化器自然学会忽略 context
3. **Evidence**: per-k loss flat, context ablation 显示 context 反而有害, same-scene pairing (data_v2) 也无效

**Key insight**: 需要打破 BC 的 conditional independence，让 context 对 action prediction 产生 **因果影响**。

### 18.2 为什么 Pi0-FAST (AR) + RICL 有可能成功

1. **AR 模型不存在 conditioning collapse**: 自回归生成中，prefix tokens (retrieved context) 直接影响后续 token 生成，结构上无法 bypass (vs flow matching 的 cross-attention 可被忽略)
2. **RICL 已验证 ICL 在 Pi0-FAST 上 work**: success-only retrieval 已证明模型能从 retrieved observations 中提取 useful information
3. **Chunk-level context > Episode-level context**: RICL 检索的是 (state, action_chunk) 对而非单张图像 + 文字描述 — 信息密度远高于 aha_openpi 的 failure context

### 18.3 方法: Stage-Aware Contrastive Retrieval

#### Stage 1: Success-Only RICL Baseline (已完成)

训练 Pi0-FAST-RICL on success-only demos。验证 ICL pipeline + action interpolation 在 RLBench 上的基础性能。

#### Stage 2: Failure Chunk Bank 构建

```
data_v1/{task}/failures/{fail_type}_wp{X}/episodes/episode{N}/
    → 按 waypoint 切分为 (state, action_chunk, image, DINOv2_embedding) chunks
    → 标注: task, waypoint_idx, fail_type, VLM_description
    → 存入 per-task failure_chunk_bank.npz
```

结构 (per-task):
```
failure_chunk_bank_{task}.npz:
├── images:        (N_fail, 224, 224, 3)  uint8
├── states:        (N_fail, 8)            float32
├── action_chunks: (N_fail, 10, 7)        float32
├── embeddings:    (N_fail, 49152)         float32  # DINOv2
├── waypoints:     (N_fail,)              int32
├── fail_types:    (N_fail,)              str
└── descriptions:  (N_fail,)              str       # VLM descriptions
```

索引: per-(task, waypoint) 的 KNN index。

#### Stage 3: Mixed Retrieval Training (BC phase)

推理时检索流程:
```
Query obs → DINOv2 embedding → 检索 top-k success chunks + top-m failure chunks
    → [fail_0, fail_1, ..., success_0, success_1, ..., query] → Pi0-FAST-RICL → actions
```

**Attention mask 设计**:
- 失败 chunks 放在前面 (prefix)
- 成功 chunks 放在中间
- Query 在最后
- 所有 chunk 用 lower-triangular block attention (RICL 原有设计)

训练: 在 success demos 上继续 BC 训练，但 retrieved context 混合了 success + failure chunks。

#### Stage 4: DPO (Direct Preference Optimization)

**Valid DPO pair 构造** (解决前期 pair 构造错误):

```
Same query obs + same retrieved context (success + failure chunks)
    ├── Preferred (y_w): expert action chunk (from success demo at same waypoint)
    └── Dispreferred (y_l): failure action chunk (from failure demo at same waypoint)

Key: 两个 response 共享完全相同的 prompt (obs + context)，只是 action chunk 不同。
```

Waypoint 对齐: 通过 DINOv2 embedding 距离匹配 success 和 failure demos 在相同 stage 的 frames。

### 18.4 Proactive vs Reactive Inference

| 模式 | 触发条件 | Context | 期望效果 |
|------|---------|---------|---------|
| **Proactive** | 每步 (attempt 0) | 检索相关 failure chunks | 预防性避免常见失败 (如: "看到类似场景下其他人夹爪抓太高") |
| **Reactive** | 实际失败后 retry | 当前 episode 的失败 chunk + 检索相似 failure | 针对性修正 (如: "刚才就是夹爪抓太高，这次调低") |

Proactive 模式是核心创新点 — 不需要先失败再学习，而是从 failure bank 预见潜在风险。

### 18.5 Diagnostic Experiment (DPO 前必做)

验证 DPO preference pair 的信号强度:

```python
# 对每个 (task, waypoint):
#   1. 找到 success demos 和 failure demos 在该 waypoint 的 frames
#   2. 用 DINOv2 embedding 匹配最近的 success-failure pairs
#   3. 计算 action chunk 的 L2 距离

for task in ALL_TASKS:
    for wp_idx in waypoints[task]:
        success_chunks = get_success_chunks_at_waypoint(task, wp_idx)
        failure_chunks = get_failure_chunks_at_waypoint(task, wp_idx)
        distances = compute_pairwise_l2(success_chunks, failure_chunks)
        print(f"{task}/wp{wp_idx}: mean_L2={distances.mean():.4f}, "
              f"separable={distances.mean() > threshold}")
```

如果 success/failure action chunks 在相同 stage 下有显著 L2 距离 → DPO 有足够的 preference signal。
如果距离太小 → failure 更多是 timing/precision 问题而非方向错误 → 需要更细粒度的 action 比较。

---

## 19. 技术讨论 & 开放问题 (2026-03-05)

### 19.1 π_ref 对 failure chunks 的暴露

**问题**: DPO 中的 reference policy (π_ref) 是 success-only RICL baseline。它从未见过 failure chunks 作为 context，因此:
- π_ref(y_w | x, fail_context) 和 π_ref(y_l | x, fail_context) 的输出可能是 noisy (因为 context 分布不同于训练)
- 标准 DPO loss 可能不稳定

**可能方案**:
1. Stage 3 (mixed retrieval BC) 先让模型适应 failure chunks 作为 context → 用训练后的 model 作为 π_ref
2. 使用 IPO 或 SimPO 等不依赖 π_ref 的变体
3. 如果效果不好，考虑 RLHF 替代 DPO

### 19.2 DPO on FAST Tokens

DPO 最初设计用于 LLM (natural language token)。FAST tokens 是 BPE + DCT 编码的连续动作离散化 — 它们不像 NL tokens 有丰富的语义。

**潜在问题**: DPO 的 log-probability ratio 可能在 FAST token 空间上 variance 很大，导致训练不稳定。

**缓解**: 参考 HAPO (Hierarchical Action Preference Optimization) 在机器人动作上做 preference learning 的实践。

### 19.3 Per-(task, waypoint) Index 稀疏度

| Task | wps | fail_types | fails/type | Total chunks |
|------|-----|-----------|-----------|-------------|
| push_button_hard | 2 | 11 | ~12 | ~137 |
| pick_up_cup | 3 | 10 | ~12 | ~122 |
| close_box_hard | 5 | 14 | ~14 | ~200 |
| insert_usb | 5 | 44 | ~15 | ~660 |
| stack_cups | 10 | 52 | ~14 | ~715 |

某些 (task, waypoint) 组合可能只有 1-5 个 failure chunks → KNN 检索可能返回不相关结果。

**方案**: 设置最小距离阈值，距离过远时 fallback 到 success-only context。

### 19.4 Attention Mask: 失败 chunks 的位置

RICL 原始设计: 所有 retrieved observations 地位平等 (block-diagonal + lower-triangular)。
加入 failure chunks 后需要考虑:

1. **分组分隔**: failure chunks 和 success chunks 用不同的 positional encoding / segment ID？
2. **顺序**: failure 放前面 (让模型先 "看到" 失败) 还是混合排列？
3. **label**: 是否需要告诉模型 "这个 chunk 是失败的"？ (prompt 中标注 "failed" / "success")

当前计划: failure chunks 放 prefix，success chunks 紧跟其后，prompt 中标注来源。

---

## 20. Immediate Next Steps (2026-03-05)

### Step 1: RICL Baseline 评测 (最高优先级)

```bash
# 1. 下载 checkpoint 到本地
scp -r cluster:checkpoints/pi0_fast_rlbench_ricl/rlbench_ricl_v2/20000 \
    ./checkpoints/pi0_fast_rlbench_ricl/rlbench_ricl_v2/

# 2. 本地预处理 (如果还没跑过)
python preprocessing/process_rlbench_demos.py \
    --data_root /home/ruoqu/jjliu/AHA/data_v1 \
    --output_dir ./processed_rlbench \
    --tasks all --num_episodes 25

python preprocessing/retrieve_within_rlbench.py \
    --processed_dir ./processed_rlbench \
    --knn_k 100 --embedding_type top_image

# 3. 评测 (需要 CoppeliaSim)
export COPPELIASIM_ROOT=/home/ruoqu/jjliu/CoppeliaSim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

python scripts/eval_ricl_rlbench.py \
    --checkpoint checkpoints/pi0_fast_rlbench_ricl/rlbench_ricl_v2/20000 \
    --demos_dir ./processed_rlbench \
    --task all --episodes 25 --save_video
```

### Step 2: DPO Diagnostic Experiment

验证 failure vs success action chunks 的可分离性 (见 §18.5)。

### Step 3: Failure Chunk Bank 构建

扩展现有 `build_failure_bank.py` (from aha_openpi) 到 RICL 格式:
- 添加 DINOv2 embedding
- 添加 action chunks (10-step)
- 添加 state vectors
- Per-(task, waypoint) 索引

### Step 4: Mixed Retrieval Training + DPO

根据 Step 1-3 的结果决定具体实现方案。

---

*最后更新: 2026-03-05*
