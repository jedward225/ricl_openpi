```
cd aha_ricl

# 1. 数据预处理 (RLBench → processed_demo.npz + DINOv2 embeddings)
python preprocessing/process_rlbench_demos.py \
    --data_root {data_root} --output_dir ./processed_rlbench \
    --tasks all --num_episodes {25,50}

# 2. KNN 检索预处理
# Origin (KNN retrieve)
python preprocessing/retrieve_within_rlbench.py \
    --processed_dir ./processed_rlbench --knn_k 100 --embedding_type top_image
# Random sample
python preprocessing/random_sample_within_rlbench.py \
    --processed_dir ./processed_rlbench_25 --knn_k 100 --embedding_type top_image

# 3. 计算 norm stats
# Origin (KNN retrieve)
python scripts/compute_norm_stats_rlbench.py --processed_dir ./processed_rlbench

# 4. 训练
# Adjust random_sample in src/openpi/training/config.py to enable random sampling.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
uv run scripts/train_ricl_rlbench.py --exp_name=rlbench_ricl_n{25,50}

# 5. 评测
# Origin (KNN retrieve)
python scripts/eval_ricl_rlbench.py \
    --checkpoint ./checkpoints/.../latest --demos_dir ./processed_rlbench \
    --task all --episodes 25 --save_video
# Random sample
python scripts/eval_ricl_rlbench.py \
    --checkpoint ./checkpoints/.../latest --demos_dir ./processed_rlbench \
    --task all --episodes 25 --save_video --random
```