"""
Generate random retrieval indices for RICL-Random ablation.

Reads an existing processed_rlbench_v4/ directory (DINOv2 KNN precomputed) and
outputs processed_rlbench_v4_random/ with the same layout, but each query
frame's retrieved_indices are sampled UNIFORMLY AT RANDOM from all OTHER
episodes in the same task (no DINOv2 re-embedding needed).

Symlinks processed_demo.npz from the source dir to avoid duplicating the
~17 GB of image data. Distances are filled with zeros and max_distance=1.0
since the RICL-Random config runs with use_action_interpolation=False.

Usage:
    cd aha_ricl
    python preprocessing/make_random_indices.py \
        --src_dir ./processed_rlbench_v4 \
        --dst_dir ./processed_rlbench_v4_random \
        --knn_k 100 --seed 0
"""

import argparse
import json
import logging
import os
from collections import defaultdict

import numpy as np

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, required=True,
                        help="Existing processed dir with KNN indices (e.g., ./processed_rlbench_v4)")
    parser.add_argument("--dst_dir", type=str, required=True,
                        help="Output dir for random indices (e.g., ./processed_rlbench_v4_random)")
    parser.add_argument("--knn_k", type=int, default=100,
                        help="Number of random retrievals per query frame (match RICL training config)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    src = os.path.abspath(args.src_dir)
    dst = os.path.abspath(args.dst_dir)
    os.makedirs(dst, exist_ok=True)

    # Load task/episode mappings from src
    groups_to_ep_fols_src = json.load(open(os.path.join(src, "groups_to_ep_fols.json")))
    ep_idxs_to_fol_src = json.load(open(os.path.join(src, "ep_idxs_to_fol.json")))
    groups_to_ep_idxs = json.load(open(os.path.join(src, "groups_to_ep_idxs.json")))

    # Build dst mappings with rewritten paths (src -> dst)
    groups_to_ep_fols_dst = {}
    ep_idxs_to_fol_dst = {}
    fols_to_ep_idxs_dst = {}
    for task, fols in groups_to_ep_fols_src.items():
        new_fols = []
        for fol in fols:
            new_fol = os.path.abspath(fol).replace(src, dst)
            new_fols.append(new_fol)
        groups_to_ep_fols_dst[task] = new_fols
    for ep_idx_str, fol in ep_idxs_to_fol_src.items():
        new_fol = os.path.abspath(fol).replace(src, dst)
        ep_idxs_to_fol_dst[ep_idx_str] = new_fol
        fols_to_ep_idxs_dst[new_fol] = int(ep_idx_str)

    # For each task, iterate episodes and emit random indices
    total_frames = 0
    for task_idx, (task, ep_idxs) in enumerate(groups_to_ep_idxs.items()):
        ep_idxs = [int(e) for e in ep_idxs]
        num_eps = len(ep_idxs)

        # Build (ep_idx, step_idx) pool for this task from processed_demo.npz
        all_pairs = []   # list of (ep_idx, step_idx) across ALL eps in task
        ep_to_T = {}
        for ep_idx in ep_idxs:
            ep_fol = ep_idxs_to_fol_src[str(ep_idx)]
            demo_path = os.path.join(ep_fol, "processed_demo.npz")
            demo = np.load(demo_path, mmap_mode="r")
            T = len(demo["state"])
            ep_to_T[ep_idx] = T
            all_pairs.extend([(ep_idx, t) for t in range(T)])
        all_pairs = np.asarray(all_pairs, dtype=np.int32)   # (N_task_frames, 2)
        logger.info(f"[{task_idx+1}/{len(groups_to_ep_idxs)}] {task}: {num_eps} eps, "
                    f"{len(all_pairs)} total frames")

        # For each episode, sample random retrievals from all OTHER episodes
        for ep_idx in ep_idxs:
            src_ep_fol = ep_idxs_to_fol_src[str(ep_idx)]
            dst_ep_fol = os.path.abspath(src_ep_fol).replace(src, dst)
            os.makedirs(dst_ep_fol, exist_ok=True)

            # Symlink processed_demo.npz from src (avoid duplicating)
            src_demo = os.path.join(src_ep_fol, "processed_demo.npz")
            dst_demo = os.path.join(dst_ep_fol, "processed_demo.npz")
            if not os.path.exists(dst_demo):
                os.symlink(os.path.abspath(src_demo), dst_demo)

            # Candidate pool: all (ep, step) pairs EXCEPT those from this episode
            pool_mask = all_pairs[:, 0] != ep_idx
            pool = all_pairs[pool_mask]                           # (N_pool, 2)
            assert len(pool) > args.knn_k, \
                f"Task {task} ep {ep_idx}: pool {len(pool)} < knn_k {args.knn_k}"

            T = ep_to_T[ep_idx]
            # Random sample knn_k pairs per query frame
            # pool_idxs: (T, knn_k) int indices into pool
            pool_idxs = rng.integers(0, len(pool), size=(T, args.knn_k), dtype=np.int64)
            retrieved_indices = pool[pool_idxs]                   # (T, knn_k, 2) int32
            query_indices = np.stack(
                [np.full(T, ep_idx, dtype=np.int32), np.arange(T, dtype=np.int32)],
                axis=1,
            )                                                      # (T, 2)

            # Distances: all zeros (no-interp run never reads these; dataset still
            # asserts shape (T, knn_k+1) and normalizes by max_distance).
            distances = np.zeros((T, args.knn_k + 1), dtype=np.float64)

            out_path = os.path.join(dst_ep_fol, "indices_and_distances.npz")
            np.savez(
                out_path,
                retrieved_indices=retrieved_indices.astype(np.int32),
                query_indices=query_indices.astype(np.int32),
                distances=distances,
            )
            total_frames += T

    # Save top-level JSON mappings
    for name, data in [
        ("groups_to_ep_fols", groups_to_ep_fols_dst),
        ("ep_idxs_to_fol", ep_idxs_to_fol_dst),
        ("fols_to_ep_idxs", {k: int(v) for k, v in fols_to_ep_idxs_dst.items()}),
        ("groups_to_ep_idxs", {k: [int(v) for v in vs] for k, vs in groups_to_ep_idxs.items()}),
    ]:
        with open(os.path.join(dst, f"{name}.json"), "w") as f:
            json.dump(data, f, indent=2)

    # max_distance = 1.0 so that distance/max_distance = 0 (never used anyway)
    with open(os.path.join(dst, "max_distance.json"), "w") as f:
        json.dump({"max_distance": 1.0}, f, indent=2)

    logger.info(f"Done. Wrote {total_frames} frames of random indices to {dst}")
    logger.info(f"processed_demo.npz files are symlinks back to {src}")


if __name__ == "__main__":
    main()
