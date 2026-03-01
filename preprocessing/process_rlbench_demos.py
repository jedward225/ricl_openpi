"""
Process RLBench success demos into RICL-compatible processed_demo.npz format.

Converts RLBench raw data (pkl + PNG) to the format expected by RICL:
  - state: (T, 8) float32  [x,y,z,rx,ry,rz,gripper,0]
  - actions: (T, 7) float32  [dx,dy,dz,drx,dry,drz,gripper]
  - top_image: (T, 224, 224, 3) uint8  [front_rgb resized]
  - right_image: (T, 224, 224, 3) uint8  [overhead_rgb resized]
  - wrist_image: (T, 224, 224, 3) uint8  [wrist_rgb resized]
  - top_image_embeddings: (T, 49152) float32  [DINOv2 64-patch]
  - prompt: str  [task description]

Usage:
    cd aha_ricl
    python preprocessing/process_rlbench_demos.py \
        --data_root /home/ruoqu/jjliu/AHA/data_v1 \
        --output_dir ./processed_rlbench \
        --tasks all --num_episodes 25
"""

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image

# Add shared module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "shared"))
from rlbench_io import (
    VLA_TASK_DESCRIPTIONS,
    extract_delta_action,
    extract_state,
    load_episode_data,
)

# Add src to path for openpi imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from openpi.policies.utils import EMBED_DIM, embed_with_batches, load_dinov2

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

ALL_TASKS = list(VLA_TASK_DESCRIPTIONS.keys())

# Camera mapping: RLBench camera name → RICL field name
CAMERA_MAP = {
    "front_rgb": "top_image",
    "overhead_rgb": "right_image",
    "wrist_rgb": "wrist_image",
}

# Only compute DINOv2 embeddings for the main camera (used for KNN retrieval)
EMBEDDING_CAMERA = "front_rgb"


def discover_success_episodes(data_root: str, task: str) -> list[str]:
    """Find all success episode paths for a task."""
    success_dir = os.path.join(data_root, task, "success", task, "success", "episodes")
    if not os.path.exists(success_dir):
        logger.warning(f"Success dir not found: {success_dir}")
        return []

    episodes = sorted(
        [os.path.join(success_dir, d) for d in os.listdir(success_dir) if d.startswith("episode")],
        key=lambda p: int(os.path.basename(p).replace("episode", "")),
    )
    return episodes


def _load_single_frame(args):
    """Load and resize a single frame (for parallel loading)."""
    img_path, target_size = args
    if os.path.exists(img_path):
        img = Image.open(img_path).convert("RGB")
        img = img.resize((target_size, target_size), Image.BILINEAR)
        return np.array(img, dtype=np.uint8)
    return np.zeros((target_size, target_size, 3), dtype=np.uint8)


def load_all_frames(episode_path: str, camera: str, num_frames: int, target_size: int = 224, num_workers: int = 8) -> np.ndarray:
    """Load all frames for a camera, resized to target_size × target_size. Uses thread pool for I/O."""
    camera_dir = os.path.join(episode_path, camera)
    args_list = [(os.path.join(camera_dir, f"{i}.png"), target_size) for i in range(num_frames)]

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        frames = list(pool.map(_load_single_frame, args_list))

    return np.stack(frames, axis=0)  # (T, H, W, 3)


def process_episode(
    episode_path: str,
    task: str,
    dinov2,
    embed_batch_size: int = 256,
    num_workers: int = 8,
) -> dict:
    """Process a single RLBench episode into RICL format."""
    t0 = time.time()

    # Load episode observations
    episode_data = load_episode_data(episode_path)
    observations = episode_data["observations"]
    num_frames = len(observations)
    t_pkl = time.time()

    # Extract states: (T, 8)
    states = np.stack([extract_state(obs) for obs in observations], axis=0)
    assert states.shape == (num_frames, 8), f"states shape: {states.shape}"

    # Extract delta actions: (T-1, 7), then pad last frame
    actions_list = []
    for i in range(num_frames - 1):
        action = extract_delta_action(observations[i], observations[i + 1])
        actions_list.append(action)
    # Last frame: repeat the last action (or zero)
    if actions_list:
        actions_list.append(actions_list[-1].copy())
    else:
        actions_list.append(np.zeros(7, dtype=np.float32))
    actions = np.stack(actions_list, axis=0)
    assert actions.shape == (num_frames, 7), f"actions shape: {actions.shape}"

    processed = {
        "state": states.astype(np.float32),
        "actions": actions.astype(np.float32),
    }

    # Load images for all cameras (parallel I/O)
    t_img_start = time.time()
    for rlbench_camera, ricl_key in CAMERA_MAP.items():
        frames = load_all_frames(episode_path, rlbench_camera, num_frames, target_size=224, num_workers=num_workers)
        assert frames.shape == (num_frames, 224, 224, 3), f"{ricl_key} shape: {frames.shape}"
        processed[ricl_key] = frames
    t_img = time.time()

    # Compute DINOv2 embeddings only for the main camera
    embeddings = embed_with_batches(processed["top_image"], dinov2, batch_size=embed_batch_size)
    assert embeddings.shape == (num_frames, EMBED_DIM), f"embeddings shape: {embeddings.shape}"
    processed["top_image_embeddings"] = embeddings
    t_emb = time.time()

    # Task prompt
    processed["prompt"] = VLA_TASK_DESCRIPTIONS.get(task, task.replace("_", " "))

    logger.info(f"    {num_frames} frames | pkl={t_pkl-t0:.1f}s, images={t_img-t_img_start:.1f}s, embed={t_emb-t_img:.1f}s, total={t_emb-t0:.1f}s")
    return processed


def main():
    parser = argparse.ArgumentParser(description="Process RLBench demos for RICL")
    parser.add_argument("--data_root", type=str, required=True, help="Path to data_v1 root")
    parser.add_argument("--output_dir", type=str, default="./processed_rlbench", help="Output directory")
    parser.add_argument("--tasks", nargs="+", default=["all"], help="Tasks to process (or 'all')")
    parser.add_argument("--num_episodes", type=int, default=25, help="Number of episodes per task")
    parser.add_argument("--embed_batch_size", type=int, default=256, help="DINOv2 embedding batch size")
    parser.add_argument("--skip_existing", action="store_true", help="Skip already processed episodes")
    parser.add_argument("--num_workers", type=int, default=8, help="Threads for parallel image loading")
    args = parser.parse_args()

    tasks = ALL_TASKS if "all" in args.tasks else args.tasks

    # GPU check
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
    else:
        logger.warning("NO GPU — DINOv2 will run on CPU (very slow!)")

    # Load DINOv2 model
    logger.info("Loading DINOv2 model...")
    dinov2 = load_dinov2()
    device = next(dinov2.parameters()).device
    logger.info(f"DINOv2 loaded on {device}")

    total_processed = 0
    total_skipped = 0

    for task in tasks:
        logger.info(f"=== Processing task: {task} ===")
        episodes = discover_success_episodes(args.data_root, task)

        if not episodes:
            logger.warning(f"No episodes found for {task}, skipping")
            continue

        # Limit to num_episodes
        episodes = episodes[: args.num_episodes]
        logger.info(f"Found {len(episodes)} episodes (using first {args.num_episodes})")

        for ep_idx, episode_path in enumerate(episodes):
            ep_name = os.path.basename(episode_path)
            output_dir = os.path.join(args.output_dir, task, f"episode_{ep_idx}")
            output_path = os.path.join(output_dir, "processed_demo.npz")

            if args.skip_existing and os.path.exists(output_path):
                logger.info(f"  [{ep_idx+1}/{len(episodes)}] {ep_name} — skipped (exists)")
                total_skipped += 1
                continue

            logger.info(f"  [{ep_idx+1}/{len(episodes)}] {ep_name} — processing...")

            try:
                processed = process_episode(
                    episode_path, task, dinov2,
                    embed_batch_size=args.embed_batch_size,
                    num_workers=args.num_workers,
                )

                os.makedirs(output_dir, exist_ok=True)
                np.savez(output_path, **processed)

                num_frames = processed["state"].shape[0]
                logger.info(f"    Saved: {num_frames} frames, {output_path}")
                total_processed += 1
            except Exception as e:
                logger.error(f"    Failed: {e}")
                raise

    logger.info(f"Done! Processed: {total_processed}, Skipped: {total_skipped}")


if __name__ == "__main__":
    main()
