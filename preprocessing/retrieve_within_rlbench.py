"""
Build KNN retrieval indices for RLBench processed demos.

Groups = tasks. For each task, builds a KNN index from all episodes'
DINOv2 embeddings. For each episode, retrieves top-k nearest neighbors
from all OTHER episodes in the same task.

Outputs per episode:
  - indices_and_distances.npz (retrieved_indices, query_indices, distances)

Outputs per processed_dir:
  - collected_demos_infos.json (ep_idxs_to_fol, fols_to_ep_idxs, groups_to_ep_fols, groups_to_ep_idxs)
  - max_distance.json (global max distance for action interpolation normalization)

Usage:
    cd aha_ricl
    python preprocessing/retrieve_within_rlbench.py \
        --processed_dir ./processed_rlbench \
        --knn_k 100 --embedding_type top_image
"""

import argparse
import json
import logging
import os
from collections import defaultdict

import numpy as np
from autofaiss import build_index

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def create_idx_fol_mapping(processed_dir: str) -> dict:
    """Create episode index ↔ folder mappings, grouped by task.

    Directory structure expected:
        processed_dir/
        ├── pick_up_cup/
        │   ├── episode_0/processed_demo.npz
        │   ├── episode_1/processed_demo.npz
        │   └── ...
        ├── push_button_hard/
        │   └── ...
        └── ...
    """
    mapping_names = ["groups_to_ep_fols", "ep_idxs_to_fol", "fols_to_ep_idxs", "groups_to_ep_idxs"]
    mappings = {
        "groups_to_ep_fols": {},
        "ep_idxs_to_fol": {},
        "fols_to_ep_idxs": {},
        "groups_to_ep_idxs": defaultdict(list),
    }

    count = 100000  # Start from 100k (RICL convention)

    # Discover tasks (groups)
    tasks = sorted([
        d for d in os.listdir(processed_dir)
        if os.path.isdir(os.path.join(processed_dir, d)) and not d.startswith(".")
    ])

    for task in tasks:
        task_dir = os.path.join(processed_dir, task)
        episodes = sorted(
            [d for d in os.listdir(task_dir) if d.startswith("episode_")],
            key=lambda x: int(x.split("_")[1]),
        )

        ep_fols = [os.path.join(task_dir, ep) for ep in episodes]
        mappings["groups_to_ep_fols"][task] = ep_fols

        for ep_fol in ep_fols:
            mappings["ep_idxs_to_fol"][count] = ep_fol
            mappings["fols_to_ep_idxs"][ep_fol] = count
            mappings["groups_to_ep_idxs"][task].append(count)
            count += 1

    # Save as JSON
    for name in mapping_names:
        path = os.path.join(processed_dir, f"{name}.json")
        data = dict(mappings[name])  # convert defaultdict
        # JSON requires string keys
        if name in ("ep_idxs_to_fol", "fols_to_ep_idxs"):
            data = {str(k): v for k, v in data.items()}
        elif name == "groups_to_ep_idxs":
            data = {str(k): [int(v) for v in vals] for k, vals in data.items()}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    logger.info(f"Created mappings: {count - 100000} episodes across {len(tasks)} tasks")
    return mappings


def retrieval_preprocessing(
    groups_to_ep_idxs: dict,
    ep_idxs_to_fol: dict,
    knn_k: int,
    embedding_type: str,
    nb_cores: int = 8,
):
    """Build KNN indices and compute retrieval for each episode within its task group."""
    global_max_distance = 0.0

    for group_idx, (task, ep_idxs) in enumerate(groups_to_ep_idxs.items()):
        num_episodes = len(ep_idxs)
        logger.info(f"[{group_idx+1}/{len(groups_to_ep_idxs)}] Task: {task} ({num_episodes} episodes)")

        # Load all embeddings for this task
        all_embeddings = []
        all_embeddings_map = {}
        all_indices = []

        for ep_idx in ep_idxs:
            ep_fol = ep_idxs_to_fol[ep_idx]
            npz = np.load(os.path.join(ep_fol, "processed_demo.npz"))
            ep_embeddings = npz[f"{embedding_type}_embeddings"]
            all_embeddings.append(ep_embeddings)
            all_embeddings_map[ep_idx] = ep_embeddings
            num_steps = len(ep_embeddings)
            all_indices.extend([[ep_idx, step_idx] for step_idx in range(num_steps)])

        all_embeddings = np.concatenate(all_embeddings, axis=0)
        all_indices = np.array(all_indices)
        embedding_dim = all_embeddings.shape[1]
        num_total = len(all_embeddings)
        logger.info(f"  Total frames: {num_total}, embedding_dim: {embedding_dim}")

        # For each episode, retrieve from all other episodes in same task
        for ep_count, ep_idx in enumerate(ep_idxs):
            ep_fol = ep_idxs_to_fol[ep_idx]
            output_path = os.path.join(ep_fol, "indices_and_distances.npz")

            if os.path.exists(output_path):
                logger.info(f"  [{ep_count+1}/{num_episodes}] Episode {ep_idx} — skipped (exists)")
                # Still need to load distances for max_distance computation
                saved = np.load(output_path)
                ep_max = np.max(saved["distances"])
                global_max_distance = max(global_max_distance, ep_max)
                continue

            # Separate this episode from others
            other_mask = np.array([idx[0] != ep_idx for idx in all_indices])
            this_mask = ~other_mask

            other_embeddings = all_embeddings[other_mask]
            other_indices = all_indices[other_mask]
            this_embeddings = all_embeddings[this_mask]
            this_indices = all_indices[this_mask]
            num_query = len(this_embeddings)
            num_retrieval = len(other_embeddings)

            logger.info(f"  [{ep_count+1}/{num_episodes}] Episode {ep_idx}: "
                        f"{num_query} query frames, {num_retrieval} retrieval frames")

            # Build KNN index from other episodes
            effective_k = min(knn_k, num_retrieval)
            knn_index, _ = build_index(
                embeddings=other_embeddings.astype(np.float32),
                save_on_disk=False,
                min_nearest_neighbors_to_retrieve=effective_k + 5,
                max_index_query_time_ms=10,
                max_index_memory_usage="25G",
                current_memory_available="50G",
                metric_type="l2",
                nb_cores=nb_cores,
            )

            # Search
            topk_distances, topk_indices = knn_index.search(this_embeddings, min(2 * effective_k, num_retrieval))

            # Remove -1 padding and crop to knn_k
            cleaned_indices = []
            for indices in topk_indices:
                valid = [idx for idx in indices if idx != -1][:effective_k]
                # Pad with -1 if not enough results
                while len(valid) < effective_k:
                    valid.append(-1)
                cleaned_indices.append(valid)
            topk_indices = np.array(cleaned_indices)

            # Map back to (ep_idx, step_idx) pairs
            # Handle -1 indices
            valid_mask = topk_indices >= 0
            safe_indices = np.where(valid_mask, topk_indices, 0)
            retrieved_indices = other_indices[safe_indices]
            # Zero out invalid entries
            retrieved_indices[~valid_mask] = -1

            assert retrieved_indices.shape == (num_query, effective_k, 2)

            # Compute distances to first retrieved embedding (for action interpolation)
            all_distances = []
            for ct in range(num_query):
                row = retrieved_indices[ct]
                if row[0][0] < 0:
                    # No valid retrieval
                    all_distances.append([0.0] * (effective_k + 1))
                    continue

                first_emb = all_embeddings_map[row[0][0]][row[0][1]]
                distances = [0.0]
                for k_idx in range(1, effective_k):
                    if row[k_idx][0] < 0:
                        distances.append(0.0)
                    else:
                        distances.append(float(np.linalg.norm(
                            all_embeddings_map[row[k_idx][0]][row[k_idx][1]] - first_emb
                        )))

                # Query distance to first retrieved
                query_ep, query_step = this_indices[ct]
                distances.append(float(np.linalg.norm(
                    all_embeddings_map[query_ep][query_step] - first_emb
                )))
                all_distances.append(distances)

            all_distances = np.array(all_distances)
            assert all_distances.shape == (num_query, effective_k + 1)

            # Track max distance
            ep_max = np.max(all_distances)
            global_max_distance = max(global_max_distance, ep_max)

            # Save
            np.savez(
                output_path,
                retrieved_indices=retrieved_indices.astype(np.int32),
                query_indices=this_indices.astype(np.int32),
                distances=all_distances,
            )
            logger.info(f"    Saved: {output_path}")

    return global_max_distance


def main():
    parser = argparse.ArgumentParser(description="Build KNN retrieval for RLBench demos")
    parser.add_argument("--processed_dir", type=str, default="./processed_rlbench")
    parser.add_argument("--knn_k", type=int, default=100)
    parser.add_argument("--embedding_type", type=str, default="top_image")
    parser.add_argument("--nb_cores", type=int, default=8)
    args = parser.parse_args()

    # Step 1: Create index-folder mappings
    logger.info("Creating episode index mappings...")
    mappings = create_idx_fol_mapping(args.processed_dir)

    # Step 2: Build retrieval indices
    logger.info("Building retrieval indices...")
    max_distance = retrieval_preprocessing(
        groups_to_ep_idxs=mappings["groups_to_ep_idxs"],
        ep_idxs_to_fol=mappings["ep_idxs_to_fol"],
        knn_k=args.knn_k,
        embedding_type=args.embedding_type,
        nb_cores=args.nb_cores,
    )

    # Step 3: Save max_distance for action interpolation normalization
    max_dist_path = os.path.join(args.processed_dir, "max_distance.json")
    with open(max_dist_path, "w") as f:
        json.dump({"max_distance": float(max_distance)}, f, indent=2)
    logger.info(f"Saved max_distance={max_distance:.4f} to {max_dist_path}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
