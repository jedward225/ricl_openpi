#!/usr/bin/env python3
"""
Evaluate RICL (success-only retrieval) on RLBench.

Loads a trained Pi0-FAST-RICL checkpoint and evaluates on RLBench tasks.
At each step: DINOv2 embed → KNN retrieve top-4 demos → RICL inference → action chunk.

Usage:
    cd aha_ricl

    # Single task
    python scripts/eval_ricl_rlbench.py \
        --checkpoint checkpoints/pi0_fast_rlbench_ricl/rlbench_ricl_v2/20000 \
        --demos_dir ./processed_rlbench \
        --task pick_up_cup --episodes 25

    # All tasks
    python scripts/eval_ricl_rlbench.py \
        --checkpoint checkpoints/pi0_fast_rlbench_ricl/rlbench_ricl_v2/20000 \
        --demos_dir ./processed_rlbench \
        --task all --episodes 25

Environment:
    export COPPELIASIM_ROOT=/path/to/CoppeliaSim
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
    export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

# Add src to path for OpenPI imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
# Add shared to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))

from rlbench_io import VLA_TASK_DESCRIPTIONS, TASK_MAX_STEPS

ALL_TASKS = list(VLA_TASK_DESCRIPTIONS.keys())


def quat_to_euler(quat: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(quat).as_euler('xyz')


def euler_to_quat(euler: np.ndarray) -> np.ndarray:
    return Rotation.from_euler('xyz', euler).as_quat()


def load_ricl_policy(checkpoint_dir: str, demos_dir: str):
    """Load trained RICL policy."""
    from openpi.policies import policy_config
    from openpi.training import config as train_config

    config = train_config.get_config("pi0_fast_rlbench_ricl")
    print(f"Loading RICL policy from: {checkpoint_dir}")
    print(f"Demos dir: {demos_dir}")
    policy = policy_config.create_trained_ricl_policy(config, checkpoint_dir, demos_dir)
    print("RICL policy loaded.")
    return policy


def get_observation_dict(obs, task_name: str) -> dict:
    """Convert RLBench observation to RICL input format."""
    gripper_pose = obs.gripper_pose
    gripper_open = float(obs.gripper_open)
    euler = quat_to_euler(gripper_pose[3:])
    state = np.concatenate([
        gripper_pose[:3], euler, [gripper_open], [0.0]
    ]).astype(np.float32)

    # Resize images from 256x256 to 224x224 for RICL
    front = np.array(Image.fromarray(obs.front_rgb).resize((224, 224)), dtype=np.uint8)
    overhead = np.array(Image.fromarray(obs.overhead_rgb).resize((224, 224)), dtype=np.uint8)
    wrist = np.array(Image.fromarray(obs.wrist_rgb).resize((224, 224)), dtype=np.uint8)

    prompt = VLA_TASK_DESCRIPTIONS.get(task_name, task_name.replace("_", " "))

    return {
        "query_top_image": front,
        "query_right_image": overhead,
        "query_wrist_image": wrist,
        "query_state": state,
        "query_prompt": prompt,
        "prefix": f"eval_{task_name}",
    }


def apply_delta_action(env, action: np.ndarray, current_pose: np.ndarray):
    """Apply delta EEF action."""
    delta_pos = action[:3]
    delta_euler = action[3:6]
    gripper = action[6]

    new_pos = current_pose[:3] + delta_pos
    current_euler = quat_to_euler(current_pose[3:])
    new_euler = current_euler + delta_euler
    new_quat = euler_to_quat(new_euler)

    gripper_action = 1.0 if gripper > 0.5 else 0.0
    rlbench_action = np.concatenate([new_pos, new_quat, [gripper_action]])

    try:
        obs, reward, terminate = env.step(rlbench_action)
        return obs, reward, terminate
    except Exception as e:
        print(f"  Action failed: {e}")
        return None, 0, True


class VideoRecorder:
    def __init__(self, output_path: str, fps: int = 10):
        self.output_path = output_path
        self.fps = fps
        self.frames = []

    def add_frame(self, obs):
        front = obs.front_rgb
        wrist = obs.wrist_rgb
        overhead = obs.overhead_rgb
        for img_ref in [front, wrist, overhead]:
            if img_ref.shape[:2] != (256, 256):
                pass  # images should already be 256x256 from RLBench
        # 3-camera side by side
        frame = np.concatenate([front, overhead, wrist], axis=1)
        self.frames.append(frame)

    def save(self):
        if not self.frames:
            return
        try:
            import imageio
            imageio.mimsave(self.output_path, self.frames, fps=self.fps)
            print(f"  Video saved: {self.output_path}")
        except ImportError:
            images = [Image.fromarray(f) for f in self.frames]
            gif_path = self.output_path.replace('.mp4', '.gif')
            images[0].save(gif_path, save_all=True, append_images=images[1:],
                           duration=1000 // self.fps, loop=0)

    def clear(self):
        self.frames = []


def evaluate_episode(
    env,
    policy,
    task_name: str,
    max_steps: int = 200,
    replan_steps: int = 10,
    video_recorder=None,
    debug: bool = False,
) -> dict:
    """Evaluate a single episode with RICL policy."""
    descriptions, obs = env.reset()

    if video_recorder:
        video_recorder.add_frame(obs)

    steps = 0
    success = False
    action_buffer = []
    action_idx = 0
    start_time = time.time()
    num_inferences = 0

    for step in range(max_steps):
        # Replan: get new action chunk
        if action_idx >= len(action_buffer):
            obs_dict = get_observation_dict(obs, task_name)
            result = policy.infer(obs_dict, debug=debug)
            actions = result["query_actions"]  # (action_horizon, 7)
            action_buffer = actions
            action_idx = 0
            num_inferences += 1

        action = action_buffer[action_idx]
        action_idx += 1

        # Execute action
        current_pose = obs.gripper_pose
        new_obs, reward, done = apply_delta_action(env, action, current_pose)

        if new_obs is None:
            break

        obs = new_obs
        steps += 1

        if video_recorder:
            video_recorder.add_frame(obs)

        if done or reward > 0:
            success = reward > 0
            break

    elapsed = time.time() - start_time
    return {
        "success": success,
        "steps": steps,
        "time": elapsed,
        "num_inferences": num_inferences,
    }


def run_evaluation(args):
    """Run RICL evaluation on RLBench tasks."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)

    tasks = ALL_TASKS if args.task == "all" else [t.strip() for t in args.task.split(",")]

    # Load RICL policy
    policy = load_ricl_policy(args.checkpoint, args.demos_dir)

    # Setup RLBench
    print("Setting up RLBench environment...")
    from rlbench.environment import Environment
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.observation_config import ObservationConfig
    import importlib

    def get_task_class(task_name):
        class_name = ''.join(word.capitalize() for word in task_name.split('_'))
        task_module = importlib.import_module(f'rlbench.tasks.{task_name}')
        return getattr(task_module, class_name)

    obs_config = ObservationConfig()
    obs_config.front_camera.set_all(True)
    obs_config.wrist_camera.set_all(True)
    obs_config.overhead_camera.set_all(True)
    obs_config.front_camera.image_size = (256, 256)
    obs_config.wrist_camera.image_size = (256, 256)
    obs_config.overhead_camera.image_size = (256, 256)

    action_mode = MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaPlanning(),
        gripper_action_mode=Discrete()
    )

    env = Environment(
        action_mode=action_mode,
        obs_config=obs_config,
        headless=args.headless,
    )
    env.launch()

    # Evaluate
    all_results = {}
    results_path = os.path.join(args.output_dir, f"ricl_{timestamp}.json")

    for task_name in tasks:
        max_steps = TASK_MAX_STEPS.get(task_name, 200)
        print(f"\n{'='*60}")
        print(f"Task: {task_name} | Episodes: {args.episodes} | Max steps: {max_steps}")
        print(f"{'='*60}")

        task_class = get_task_class(task_name)
        task = env.get_task(task_class)

        successes = 0
        episodes = []

        for ep in range(args.episodes):
            video_recorder = None
            if args.save_video:
                video_path = os.path.join(
                    args.output_dir,
                    f"ricl_{task_name}_ep{ep:03d}_{timestamp}.mp4"
                )
                video_recorder = VideoRecorder(video_path)

            print(f"  Episode {ep + 1}/{args.episodes}", end="")

            try:
                ep_result = evaluate_episode(
                    env=task,
                    policy=policy,
                    task_name=task_name,
                    max_steps=max_steps,
                    replan_steps=args.replan_steps,
                    video_recorder=video_recorder,
                    debug=args.debug,
                )
            except Exception as e:
                print(f" ERROR: {e}")
                import traceback
                traceback.print_exc()
                ep_result = {"success": False, "steps": 0, "time": 0, "num_inferences": 0}

            if video_recorder:
                video_recorder.save()

            episodes.append(ep_result)
            if ep_result["success"]:
                successes += 1
                print(f" -> SUCCESS ({ep_result['steps']} steps, {ep_result['num_inferences']} inferences)")
            else:
                print(f" -> FAILED ({ep_result['steps']} steps)")

        success_rate = successes / args.episodes
        task_result = {
            "task": task_name,
            "num_episodes": args.episodes,
            "success_rate": success_rate,
            "successes": successes,
            "episodes": episodes,
        }
        all_results[task_name] = task_result

        print(f"\n  {task_name}: {success_rate*100:.1f}% ({successes}/{args.episodes})")

        # Save incrementally
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # Summary
    print(f"\n{'='*60}")
    print(f"RICL EVALUATION SUMMARY")
    print(f"{'='*60}")
    for task_name, result in all_results.items():
        sr = result["success_rate"]
        print(f"  {task_name}: {sr*100:.1f}% ({result['successes']}/{result['num_episodes']})")

    if all_results:
        avg_sr = np.mean([r["success_rate"] for r in all_results.values()])
        print(f"  ---")
        print(f"  Average: {avg_sr*100:.1f}%")

    print(f"\nResults saved to: {results_path}")
    env.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Evaluate RICL on RLBench")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained RICL checkpoint dir")
    parser.add_argument("--demos_dir", type=str, default="./processed_rlbench",
                        help="Path to processed demo dir (for KNN retrieval)")
    parser.add_argument("--task", type=str, default="all",
                        help="Task name or 'all'")
    parser.add_argument("--episodes", type=int, default=25,
                        help="Episodes per task")
    parser.add_argument("--replan_steps", type=int, default=10,
                        help="Re-plan interval (default=action_horizon=10)")
    parser.add_argument("--save_video", action="store_true",
                        help="Save evaluation videos")
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                        help="Output directory")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--debug", action="store_true",
                        help="Save obs/tokenized inputs for debugging")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    if args.display:
        args.headless = False

    np.random.seed(args.seed)
    import random
    random.seed(args.seed)

    run_evaluation(args)


if __name__ == "__main__":
    main()
