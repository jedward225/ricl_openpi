"""
Build RANDOM retrieval indices for RLBench processed demos (baseline).

Groups = tasks. For each task, uses all episodes' DINOv2 embeddings.
For each episode, randomly samples top-k neighbors from all OTHER episodes
in the same task.

Outputs per episode:
  - indices_and_distances_random.npz
      (retrieved_indices, query_indices, distances)

Outputs per processed_dir:
  - groups_to_ep_fols.json
  - ep_idxs_to_fol.json
  - fols_to_ep_idxs.json
  - groups_to_ep_idxs.json
  - max_distance_random.json

Distances are TRUE distances with same convention as KNN version:
  anchor = first sampled item
  distances =
    [0,
     dist(sample_2, anchor),
     ...,
     dist(sample_k, anchor),
     dist(query, anchor)]

Usage:
    cd aha_ricl
    python preprocessing/retrieve_within_rlbench_random.py \
        --processed_dir ./processed_rlbench \
        --knn_k 100 --embedding_type top_image
"""

import argparse
import json
import logging
import os
from collections import defaultdict

import numpy as np

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def create_idx_fol_mapping(processed_dir: str) -> dict:
    mapping_names = [
        "groups_to_ep_fols",
        "ep_idxs_to_fol",
        "fols_to_ep_idxs",
        "groups_to_ep_idxs",
    ]

    mappings = {
        "groups_to_ep_fols": {},
        "ep_idxs_to_fol": {},
        "fols_to_ep_idxs": {},
        "groups_to_ep_idxs": defaultdict(list),
    }

    count = 100000

    tasks = sorted(
        [
            d
            for d in os.listdir(processed_dir)
            if os.path.isdir(os.path.join(processed_dir, d))
            and not d.startswith(".")
        ]
    )

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

    # save jsons
    for name in mapping_names:
        path = os.path.join(processed_dir, f"{name}.json")
        data = dict(mappings[name])

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
    nb_cores: int = 8,   # kept for CLI compatibility
):
    """
    Random-sample baseline version.
    Same API as original KNN version.
    """
    global_max_distance = 0.0

    for group_idx, (task, ep_idxs) in enumerate(groups_to_ep_idxs.items()):
        num_episodes = len(ep_idxs)
        logger.info(
            f"[{group_idx+1}/{len(groups_to_ep_idxs)}] "
            f"Task: {task} ({num_episodes} episodes)"
        )

        # -------------------------------------------------
        # Load all embeddings for this task
        # -------------------------------------------------
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
        all_indices = np.array(all_indices, dtype=np.int64)

        embedding_dim = all_embeddings.shape[1]
        num_total = len(all_embeddings)

        logger.info(f"  Total frames: {num_total}, embedding_dim: {embedding_dim}")

        # -------------------------------------------------
        # Process each episode
        # -------------------------------------------------
        for ep_count, ep_idx in enumerate(ep_idxs):
            ep_fol = ep_idxs_to_fol[ep_idx]
            output_path = os.path.join(
                ep_fol, "indices_and_distances_random.npz"
            )

            if os.path.exists(output_path):
                logger.info(
                    f"  [{ep_count+1}/{num_episodes}] "
                    f"Episode {ep_idx} — skipped (exists)"
                )
                saved = np.load(output_path)
                ep_max = np.max(saved["distances"])
                global_max_distance = max(global_max_distance, ep_max)
                continue

            # split this episode vs others
            other_mask = np.array([idx[0] != ep_idx for idx in all_indices])
            this_mask = ~other_mask

            other_indices = all_indices[other_mask]
            this_indices = all_indices[this_mask]

            num_query = len(this_indices)
            num_retrieval = len(other_indices)

            effective_k = min(knn_k, num_retrieval)

            logger.info(
                f"  [{ep_count+1}/{num_episodes}] Episode {ep_idx}: "
                f"{num_query} query frames, "
                f"{num_retrieval} retrieval frames"
            )

            # -------------------------------------------------
            # RANDOM SAMPLE
            # -------------------------------------------------
            retrieved_indices = np.stack(
                [
                    other_indices[
                        np.random.choice(
                            num_retrieval,
                            size=effective_k,
                            replace=False,
                        )
                    ]
                    for _ in range(num_query)
                ],
                axis=0,
            ).astype(np.int32)

            # -------------------------------------------------
            # TRUE DISTANCES
            # -------------------------------------------------
            all_distances = []

            for ct in range(num_query):
                row = retrieved_indices[ct]

                anchor_ep, anchor_step = row[0]
                anchor_emb = all_embeddings_map[anchor_ep][anchor_step]

                distances = [0.0]

                for k_idx in range(1, effective_k):
                    e_idx, s_idx = row[k_idx]

                    dist = float(
                        np.linalg.norm(
                            all_embeddings_map[e_idx][s_idx] - anchor_emb
                        )
                    )
                    distances.append(dist)

                query_ep, query_step = this_indices[ct]

                query_dist = float(
                    np.linalg.norm(
                        all_embeddings_map[query_ep][query_step] - anchor_emb
                    )
                )

                distances.append(query_dist)
                all_distances.append(distances)

            all_distances = np.array(all_distances, dtype=np.float32)

            assert retrieved_indices.shape == (num_query, effective_k, 2)
            assert all_distances.shape == (num_query, effective_k + 1)

            ep_max = np.max(all_distances)
            global_max_distance = max(global_max_distance, ep_max)

            # -------------------------------------------------
            # Save
            # -------------------------------------------------
            np.savez(
                output_path,
                retrieved_indices=retrieved_indices.astype(np.int32),
                query_indices=this_indices.astype(np.int32),
                distances=all_distances,
            )

            logger.info(f"    Saved: {output_path}")

    return global_max_distance


def main():
    parser = argparse.ArgumentParser(
        description="Build RANDOM retrieval for RLBench demos"
    )

    parser.add_argument(
        "--processed_dir",
        type=str,
        default="./processed_rlbench",
    )
    parser.add_argument(
        "--knn_k",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--embedding_type",
        type=str,
        default="top_image",
    )
    parser.add_argument(
        "--nb_cores",
        type=int,
        default=8,
    )

    args = parser.parse_args()

    # Step 1
    logger.info("Creating episode index mappings...")
    mappings = create_idx_fol_mapping(args.processed_dir)

    # Step 2
    logger.info("Building RANDOM retrieval indices...")
    max_distance = retrieval_preprocessing(
        groups_to_ep_idxs=mappings["groups_to_ep_idxs"],
        ep_idxs_to_fol=mappings["ep_idxs_to_fol"],
        knn_k=args.knn_k,
        embedding_type=args.embedding_type,
        nb_cores=args.nb_cores,
    )

    # Step 3
    max_dist_path = os.path.join(
        args.processed_dir,
        "max_distance_random.json",
    )

    with open(max_dist_path, "w") as f:
        json.dump(
            {"max_distance": float(max_distance)},
            f,
            indent=2,
        )

    logger.info(
        f"Saved max_distance={max_distance:.4f} "
        f"to {max_dist_path}"
    )

    logger.info("Done!")


if __name__ == "__main__":
    main()