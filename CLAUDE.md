# aha_ricl — RICL on RLBench (RoboRetry)

> RICL (Retrieval-based In-Context Learning) 适配 RLBench 数据。基于 Pi0-FAST 自回归模型，通过 DINOv2 KNN 检索成功 demo 作为 in-context examples。
> 本目录 fork 自 [openpi-ricl](https://github.com/Physical-Intelligence/openpi) 的 RICL 分支，适配 RLBench 数据格式。

---

## 1. 背景

`aha_openpi` (Pi0, flow matching) 发现 **conditioning collapse** — 模型忽略 failure context。RICL 提供了已验证的 in-context learning 范式：

- Pi0-FAST (自回归) 天然适合 ICL
- **Block-diagonal + lower-triangular attention**: 每个 retrieved observation 是独立的 "in-context example"
- **Action interpolation**: 检索到的 demo actions 与模型预测加权融合

---

## 2. 实验结果与结论

### RICL v3 评测 (2026-03-08)

Checkpoint: `/home/ruoqu/jjliu/shared/trained_models/ricl_rlbench/19999` (20k steps, norm stats 修复后)
评测: 7 tasks × 25 episodes, headless, CoppeliaSim v4.1

| Task                   | Pi0 baseline (10k) | RICL v3 (20k) |
| ---------------------- | ------------------ | ------------- |
| close_box_hard         | **86.0%**          | 64.0%         |
| insert_usb_in_computer | **4.0%**           | 0.0%          |
| pick_and_lift          | **3.0%**           | 0.0%          |
| pick_up_cup            | **54.0%**          | 40.0%         |
| push_button_hard       | **49.0%**          | 40.0%         |
| slide_block_to_target  | **12.0%**          | 0.0%          |
| stack_blocks           | 0.0%               | 0.0%          |
| stack_cups             | 0.0%               | 0.0%          |
| **Average**            | **26.0%**          | ~18.0%        |

### 瓶颈分析

1. **Token 生成质量差**: 每个 action chunk 仅生成 ~17 个 FAST token (正常应 50-100+)
2. **Action interpolation 过度依赖 retrieved**: query 权重仅 0.0016，模型预测被压制
3. **成功 episodes 步数偏多**: 2-3× 于 Pi0 baseline (interpolation 压小动作幅度)
4. **数据量不公平**: RICL 每样本 15 张图 bs=16 vs Pi0 2 张图 bs=64

### 结论

RICL 的 block-diagonal attention 架构可行，但 action interpolation + success-only retrieval 在 RLBench 上不适用。Pi0-FAST ICRL (`aha_openpi/`) 保留了 attention 架构，去掉了 action interpolation，改用 failure context。

---

## 3. 关键参数

| 参数                       | 值                              | 备注                                                     |
| -------------------------- | ------------------------------- | -------------------------------------------------------- |
| cameras                    | 3: front + overhead + wrist     | front→top_image, overhead→right_image, wrist→wrist_image |
| action_dim                 | 7                               | delta-EE: [dx, dy, dz, drx, dry, drz, gripper]           |
| state_dim                  | 8                               | EE: [x, y, z, rx, ry, rz, gripper, 0]                    |
| action_horizon             | 10                              | action chunk length                                      |
| num_retrieved_observations | 4                               | top-4 KNN neighbors                                      |
| action_interpolation       | ON, lamda=10.0                  |                                                          |
| max_token_len              | 250                             | per-observation token budget                             |
| training data              | 25 eps/task × 8 tasks = 200 eps |                                                          |
| batch_size                 | 16                              | 5 obs × 3 cameras = 15 images/sample                     |

---

## 4. 代码结构

```
aha_ricl/
├── preprocessing/
│   ├── process_rlbench_demos.py          # RLBench pkl+PNG → processed_demo.npz
│   └── retrieve_within_rlbench.py        # Per-task KNN 检索 → indices_and_distances.npz
├── scripts/
│   ├── train_ricl_rlbench.py             # 训练 wrapper
│   ├── train_pi0_fast_ricl.py            # 通用 RICL 训练 (CLI config)
│   ├── eval_ricl_rlbench.py              # 评测 (CoppeliaSim + video recording)
│   ├── compute_norm_stats_rlbench.py     # 从 processed npz 计算 norm stats
│   └── test_ricl_inference.py            # 推理测试 (无需 CoppeliaSim)
├── src/openpi/
│   ├── models/
│   │   ├── pi0_fast_ricl.py              # RICL 模型 (block-diagonal attention, action interpolation)
│   │   ├── model.py                      # BaseModel, RiclObservation
│   │   └── tokenizer.py                  # FASTTokenizerRicl
│   ├── policies/
│   │   ├── rlbench_policy.py             # RiclRLBenchInputs / RiclRLBenchOutputs
│   │   ├── policy.py                     # DINOv2 检索 + 推理
│   │   └── utils.py                      # DINOv2 embedding 工具
│   └── training/
│       ├── config.py                     # Pi0FASTRiclConfig, RiclRLBenchDataConfig
│       └── data_loader.py               # RiclRLBenchDataset
├── assets/
│   └── pi0_fast_rlbench_ricl/rlbench/norm_stats.json  # 归一化统计 (prefixed keys)
└── eval_results/                         # 评测结果 JSON + 视频
```

---

## 5. 数据格式: processed_demo.npz

```
processed_rlbench/{task}/episode_{N}/
├── processed_demo.npz
│   ├── state:                (T, 8)       float32
│   ├── actions:              (T, 7)       float32
│   ├── top_image:            (T, 224, 224, 3)  uint8   # front_rgb
│   ├── right_image:          (T, 224, 224, 3)  uint8   # overhead_rgb
│   ├── wrist_image:          (T, 224, 224, 3)  uint8   # wrist_rgb
│   ├── top_image_embeddings: (T, 49152)   float32      # DINOv2 ViT-B14
│   └── prompt:               str
└── indices_and_distances.npz
    ├── retrieved_indices:     (T, k, 2)   int32         # [ep_idx, step_idx]
    └── distances:             (T, k+1)    float64
```

---

## 6. 模型架构

### Block-Diagonal Attention

```
[retrieved_0] [retrieved_1] [retrieved_2] [retrieved_3] [query]
     ↓              ↓              ↓              ↓          ↓
  self-attn      + see 0        + see 0,1      + see 0,1,2  + see all
```

每个 observation: 3 images (SigLIP) + prompt + state → ~768 tokens。总计 5 × ~768 = ~3840 tokens。

### Action Interpolation

```python
weight = exp(-lamda * distance)  # distance = L2(query_emb, retrieved_0_emb) / max_distance
final_actions = weight * retrieved_0_actions + (1 - weight) * model_predicted_actions
```

---

## 7. 路径映射

| 用途                     | 本地路径                                                     | 集群路径                                    |
| ------------------------ | ------------------------------------------------------------ | ------------------------------------------- |
| 代码                     | `/home/ruoqu/jjliu/RoboRetry/aha_ricl/`                      | `/root/RoboRetry/aha_ricl/`                 |
| AHA 数据                 | `/home/ruoqu/jjliu/AHA/data_v1/`                             | `/cephfs/shared/jjliu/data_v1/`             |
| Pi0-FAST base checkpoint | `/home/ruoqu/jjliu/shared/models/pi0_fast_base/params/`      | `/cephfs_ssd_xumengdi/models/pi0_fast_base` |
| 预处理输出               | `./processed_rlbench/`                                       | `./processed_rlbench/`                      |
| Norm stats               | `./assets/pi0_fast_rlbench_ricl/rlbench/norm_stats.json`     |                                             |
| RICL v3 checkpoint       | `/home/ruoqu/jjliu/shared/trained_models/ricl_rlbench/19999` |                                             |

---

## 8. 常用命令

```bash
cd /home/ruoqu/jjliu/RoboRetry/aha_ricl

# Step 1: 数据预处理 (需 GPU for DINOv2)
python preprocessing/process_rlbench_demos.py \
    --data_root /home/ruoqu/jjliu/AHA/data_v1 \
    --output_dir ./processed_rlbench \
    --tasks all --num_episodes 25

# Step 2: KNN 检索预处理
python preprocessing/retrieve_within_rlbench.py \
    --processed_dir ./processed_rlbench \
    --knn_k 100 --embedding_type top_image

# Step 3: 训练
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
uv run scripts/train_ricl_rlbench.py --exp_name=rlbench_ricl_v3

# Step 4: 评测
python scripts/eval_ricl_rlbench.py \
    --checkpoint /home/ruoqu/jjliu/shared/trained_models/ricl_rlbench/19999 \
    --demos_dir ./processed_rlbench --task all --episodes 25 --save_video
```

---

## 9. 踩坑记录

1. **Norm stats key 不匹配 (重大)**: `state`/`actions` vs `query_state`/`query_actions` → 归一化从未生效 → FAST tokenizer 只编码 4 token → 输出全零。已修复: `compute_norm_stats_rlbench.py` 生成 prefixed keys
2. **Euler angle wrapping**: delta euler 在 ±π 边界不连续。修复: `(delta + π) % (2π) - π`
3. **FAST tokenizer 精度**: roundtrip 验证通过 (position p95 < 1.2mm, rotation p95 < 0.004 rad)
4. **autofaiss 挂起**: 32k 向量上 `autofaiss.build_index` 卡住 1h+，改用 `faiss.IndexFlatL2`
5. **orbax API 变更**: 0.11.33 的 `ckptr.metadata()` 返回 `StepMetadata` 对象而非 dict，需 `hasattr` 兼容
6. **SafeUnpickler**: 读取 RLBench pkl 无需 CoppeliaSim 环境
7. **RTX 5090 (sm_120)**: 需要 JAX 0.9.1 + CUDA 12.9，pyproject.toml override 仅限本地
8. **KNN 检索重复**: top-4 可能全来自同一 episode 相邻帧 (RICL 原版行为)

---

## 10. 与 Pi0-FAST ICRL 的关系

RICL 验证了 block-diagonal attention 在 Pi0-FAST 上的可行性。Pi0-FAST ICRL (`aha_openpi/`) 继承了这个架构，做了以下关键改变：

|                      | RICL (本目录)              | Pi0-FAST ICRL (aha_openpi)                        |
| -------------------- | -------------------------- | ------------------------------------------------- |
| Context 来源         | 检索成功 demo (KNN)        | 失败上下文 (failure bank)                         |
| Action interpolation | 有 (exp(-λ·dist) blend)    | **无** (failure actions 是错误的)                 |
| Context 内容         | 3 images + state + actions | 2 images + state + actions + description + advice |
| 数据格式             | processed_demo.npz         | LeRobot parquet + OnlineContextDataset            |
| 效果                 | ~18% (低于 baseline)       | 待验证                                            |

*最后更新: 2026-03-14*