"""Compute normalization statistics from processed RLBench demos.

Reads state (8D) and actions (7D) from processed_demo.npz files
and computes mean/std/q01/q99 for normalization.

Usage:
    cd aha_ricl
    python scripts/compute_norm_stats_rlbench.py \
        --processed_dir ./processed_rlbench \
        --output_dir ./assets/pi0_fast_rlbench_ricl/rlbench
"""

import argparse
import json
import os

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Compute RLBench norm stats")
    parser.add_argument("--processed_dir", type=str, default="./processed_rlbench")
    parser.add_argument("--output_dir", type=str, default="./assets/pi0_fast_rlbench_ricl/rlbench")
    args = parser.parse_args()

    all_states = []
    all_actions = []
    count = 0

    # Walk through all tasks and episodes
    for task in sorted(os.listdir(args.processed_dir)):
        task_dir = os.path.join(args.processed_dir, task)
        if not os.path.isdir(task_dir) or task.startswith("."):
            continue

        for ep in sorted(os.listdir(task_dir)):
            ep_dir = os.path.join(task_dir, ep)
            npz_path = os.path.join(ep_dir, "processed_demo.npz")
            if not os.path.exists(npz_path):
                continue

            data = np.load(npz_path)
            all_states.append(data["state"])
            all_actions.append(data["actions"])
            count += 1

    print(f"Loaded {count} episodes")

    states = np.concatenate(all_states, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    print(f"States: {states.shape}, Actions: {actions.shape}")

    def compute_stats(arr):
        return {
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "q01": np.percentile(arr, 1, axis=0).tolist(),
            "q99": np.percentile(arr, 99, axis=0).tolist(),
        }

    norm_stats = {
        "norm_stats": {
            "state": compute_stats(states),
            "actions": compute_stats(actions),
        }
    }

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "norm_stats.json")
    with open(output_path, "w") as f:
        json.dump(norm_stats, f, indent=2)
    print(f"Saved norm stats to {output_path}")

    # Print summary
    for key in ["state", "actions"]:
        stats = norm_stats["norm_stats"][key]
        print(f"\n{key}:")
        print(f"  mean: {[f'{v:.4f}' for v in stats['mean']]}")
        print(f"  std:  {[f'{v:.4f}' for v in stats['std']]}")


if __name__ == "__main__":
    main()
