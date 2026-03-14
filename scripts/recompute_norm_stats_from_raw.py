"""Recompute norm stats directly from raw RLBench data (no GPU needed).

Reads pkl files, applies fixed extract_delta_action (with euler wrapping),
computes mean/std/q01/q99, and saves norm_stats.json.

Usage:
    cd aha_ricl
    python scripts/recompute_norm_stats_from_raw.py \
        --data_root /home/ruoqu/jjliu/AHA/data_v1 \
        --num_episodes 25
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "shared"))
from rlbench_io import (
    VLA_TASK_DESCRIPTIONS,
    extract_delta_action,
    extract_state,
    load_episode_data,
)

ALL_TASKS = list(VLA_TASK_DESCRIPTIONS.keys())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/home/ruoqu/jjliu/AHA/data_v1")
    parser.add_argument("--output_dir", type=str, default="./assets/pi0_fast_rlbench_ricl/rlbench")
    parser.add_argument("--num_episodes", type=int, default=25)
    parser.add_argument("--tasks", nargs="+", default=["all"])
    args = parser.parse_args()

    tasks = ALL_TASKS if "all" in args.tasks else args.tasks

    all_states = []
    all_actions = []
    total_frames = 0

    for task in tasks:
        success_dir = os.path.join(args.data_root, task, "success", task, "success", "episodes")
        if not os.path.exists(success_dir):
            print(f"  [SKIP] {task}: not found")
            continue

        eps = sorted(
            [d for d in os.listdir(success_dir) if d.startswith("episode")],
            key=lambda x: int(x.replace("episode", "")),
        )[: args.num_episodes]

        task_frames = 0
        for ep_name in eps:
            ep_path = os.path.join(success_dir, ep_name)
            try:
                ep_data = load_episode_data(ep_path)
                obs = ep_data["observations"]
                n = len(obs)

                states = np.stack([extract_state(o) for o in obs], axis=0)  # (T, 8)
                actions = []
                for i in range(n - 1):
                    actions.append(extract_delta_action(obs[i], obs[i + 1]))
                actions.append(actions[-1].copy())  # pad last frame
                actions = np.stack(actions, axis=0)  # (T, 7)

                all_states.append(states)
                all_actions.append(actions)
                task_frames += n
            except Exception as e:
                print(f"  [WARN] {task}/{ep_name}: {e}")

        print(f"  {task}: {len(eps)} eps, {task_frames} frames")
        total_frames += task_frames

    states = np.concatenate(all_states, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    print(f"\nTotal: {total_frames} frames, states={states.shape}, actions={actions.shape}")

    def compute_stats(arr):
        return {
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "q01": np.percentile(arr, 1, axis=0).tolist(),
            "q99": np.percentile(arr, 99, axis=0).tolist(),
        }

    stats = {
        "norm_stats": {
            "state": compute_stats(states),
            "actions": compute_stats(actions),
        }
    }

    # Print detailed comparison
    dim_names_s = ["x", "y", "z", "rx", "ry", "rz", "gripper", "pad"]
    dim_names_a = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]

    print("\n=== NEW Action Norm Stats ===")
    a_stats = stats["norm_stats"]["actions"]
    for i, name in enumerate(dim_names_a):
        print(f"  {name:>8s}: q01={a_stats['q01'][i]:>10.6f}, q99={a_stats['q99'][i]:>10.6f}, "
              f"range={a_stats['q99'][i]-a_stats['q01'][i]:>10.6f}, "
              f"mean={a_stats['mean'][i]:>10.6f}, std={a_stats['std'][i]:>10.6f}")

    # Load old stats for comparison
    old_path = os.path.join(args.output_dir, "norm_stats.json")
    if os.path.exists(old_path):
        with open(old_path) as f:
            old = json.load(f)["norm_stats"]["actions"]
        print("\n=== OLD Action Norm Stats (for comparison) ===")
        for i, name in enumerate(dim_names_a):
            print(f"  {name:>8s}: q01={old['q01'][i]:>10.6f}, q99={old['q99'][i]:>10.6f}, "
                  f"range={old['q99'][i]-old['q01'][i]:>10.6f}")

        print("\n=== DIFF (new - old) ===")
        for i, name in enumerate(dim_names_a):
            dq01 = a_stats['q01'][i] - old['q01'][i]
            dq99 = a_stats['q99'][i] - old['q99'][i]
            new_range = a_stats['q99'][i] - a_stats['q01'][i]
            old_range = old['q99'][i] - old['q01'][i]
            print(f"  {name:>8s}: Δq01={dq01:>+10.6f}, Δq99={dq99:>+10.6f}, "
                  f"range {old_range:.6f} → {new_range:.6f} ({new_range/old_range:.2f}x)")

    # Test: what % of actions fall in [-1,1] with new stats
    print("\n=== Normalized value coverage with NEW stats ===")
    new_q01 = np.array(a_stats["q01"])
    new_q99 = np.array(a_stats["q99"])
    actions_norm = (actions - new_q01) / (new_q99 - new_q01 + 1e-6) * 2.0 - 1.0
    for i, name in enumerate(dim_names_a):
        vals = actions_norm[:, i]
        in_range = np.sum((vals >= -1) & (vals <= 1)) / len(vals) * 100
        print(f"  {name:>8s}: in[-1,1]={in_range:.1f}%, min={vals.min():.2f}, max={vals.max():.2f}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "norm_stats.json")
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
