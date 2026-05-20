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

import random
import numpy as np
from PIL import Image

# Add src to path for OpenPI imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
# Add shared to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))

from rlbench_io import (
    VLA_TASK_DESCRIPTIONS, TASK_MAX_STEPS, TASK_VARIATIONS,
    quat_to_euler, euler_to_quat, apply_delta_action, setup_episode,
)

ALL_TASKS = list(VLA_TASK_DESCRIPTIONS.keys())


class PlannerCache:
    """Cache RLBench arm planner outcomes within an episode.

    This removes OMPL resampling as a source of retry lift: if the same robot
    joint state requests the same target pose again, it gets the same path or
    the same planner failure as the first attempt.
    """

    def __init__(self, precision: int = 2):
        self._cache = {}
        self._precision = precision
        self._hits = 0
        self._misses = 0
        self._cached_failures = 0

    def _make_key(self, arm, position, euler=None, quaternion=None):
        joints = tuple(np.round(arm.get_joint_positions(), self._precision))
        target = tuple(np.round(position, self._precision))
        if quaternion is not None:
            target += tuple(np.round(quaternion, self._precision))
        elif euler is not None:
            target += tuple(np.round(euler, self._precision))
        return joints, target

    def clear(self):
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        self._cached_failures = 0

    def stats(self) -> str:
        total = self._hits + self._misses
        if total == 0:
            return "no calls"
        return (
            f"{self._hits} hits / {self._misses} misses "
            f"({self._hits / total * 100:.0f}%), cached_failures={self._cached_failures}"
        )

    def install(self, task_env):
        arm = task_env._scene.robot.arm
        original_get_path = arm.get_path
        cache = self

        def cached_get_path(position, euler=None, quaternion=None, **kwargs):
            key = cache._make_key(arm, position, euler=euler, quaternion=quaternion)
            if key in cache._cache:
                cache._hits += 1
                kind, payload = cache._cache[key]
                if kind == "error":
                    cache._cached_failures += 1
                    raise RuntimeError(payload)
                from pyrep.robots.configuration_paths.arm_configuration_path import ArmConfigurationPath

                return ArmConfigurationPath(arm, payload.copy())
            cache._misses += 1
            try:
                path = original_get_path(position, euler=euler, quaternion=quaternion, **kwargs)
            except Exception as e:
                cache._cache[key] = ("error", f"{type(e).__name__}: {e}")
                raise
            cache._cache[key] = ("path", path._path_points.copy())
            return path

        arm.get_path = cached_get_path


_planner_cache = PlannerCache()


def load_ricl_policy(checkpoint_dir: str, demos_dir: str, no_interpolation: bool = False, config_name: str = None, random: bool = False):
    """Load trained RICL policy."""
    from openpi.policies import policy_config
    from openpi.training import config as train_config
    import dataclasses

    config = train_config.get_config(config_name or "pi0_fast_rlbench_ricl")

    if no_interpolation:
        config = dataclasses.replace(
            config,
            model=dataclasses.replace(config.model, use_action_interpolation=False)
        )
        print("Action interpolation: DISABLED")
    else:
        print(f"Action interpolation: {config.model.use_action_interpolation}")

    print(f"Loading RICL policy from: {checkpoint_dir}")
    print(f"Demos dir: {demos_dir}")
    if random:
        print("Using random sampling instead of KNN retrieval.")
        policy = policy_config.create_trained_ricl_random_policy(config, checkpoint_dir, demos_dir)
    else:
        policy = policy_config.create_trained_ricl_policy(config, checkpoint_dir, demos_dir)
    print("RICL policy loaded.")
    return policy


def get_observation_dict(obs, task_name: str, prompt_override: str = None) -> dict:
    """Convert RLBench observation to RICL input format."""
    gripper_pose = obs.gripper_pose
    gripper_open = float(obs.gripper_open)
    euler = quat_to_euler(gripper_pose[3:])
    state = np.concatenate([
        gripper_pose[:3], euler, [gripper_open], [0.0]
    ]).astype(np.float32)

    # Resize images from 256x256 to 224x224 for RICL (2 cameras only)
    front = np.array(Image.fromarray(obs.front_rgb).resize((224, 224)), dtype=np.uint8)
    wrist = np.array(Image.fromarray(obs.wrist_rgb).resize((224, 224)), dtype=np.uint8)

    prompt = prompt_override or VLA_TASK_DESCRIPTIONS.get(task_name, task_name.replace("_", " "))

    return {
        "query_top_image": front,
        "query_wrist_image": wrist,
        "query_state": state,
        "query_prompt": prompt,
        "prefix": f"eval_{task_name}",
    }


    # apply_delta_action imported from shared/rlbench_io.py


class VideoRecorder:
    def __init__(self, output_path: str, fps: int = 10):
        self.output_path = output_path
        self.fps = fps
        self.frames = []
        self._context_images = None  # (4, 224, 224, 3) retrieved context

    def set_context(self, policy_result: dict):
        """Extract retrieved context images from policy inference result."""
        ctx_imgs = []
        for i in range(4):
            key = f"retrieved_{i}_top_image"
            if key in policy_result:
                img = policy_result[key]
                # Resize from 224x224 to 256x256 to match obs
                if img.shape[:2] != (256, 256):
                    img = np.array(Image.fromarray(img).resize((256, 256)))
                ctx_imgs.append(img)
        self._context_images = ctx_imgs if ctx_imgs else None

    def add_frame(self, obs):
        front = obs.front_rgb  # (256, 256, 3)
        wrist = obs.wrist_rgb
        overhead = obs.overhead_rgb
        # Top row: front + overhead + wrist (live observation)
        top_row = np.concatenate([front, overhead, wrist], axis=1)  # (256, 768, 3)
        # Bottom row: retrieved context images (or black placeholder)
        if self._context_images:
            ctx_imgs = list(self._context_images[:3])
            while len(ctx_imgs) < 3:
                ctx_imgs.append(np.zeros_like(front))
            ctx_row = np.concatenate(ctx_imgs, axis=1)
        else:
            ctx_row = np.zeros_like(top_row)
        separator = np.zeros((4, top_row.shape[1], 3), dtype=np.uint8)
        frame = np.concatenate([top_row, separator, ctx_row], axis=0)
        self.frames.append(frame)

    def save(self):
        if not self.frames:
            return
        try:
            import imageio
            imageio.mimsave(self.output_path, self.frames, fps=self.fps)
            print(f"  Video saved: {self.output_path} ({len(self.frames)} frames, {self.fps} fps)")
        except ImportError:
            images = [Image.fromarray(f) for f in self.frames]
            gif_path = self.output_path.replace('.mp4', '.gif')
            images[0].save(gif_path, save_all=True, append_images=images[1:],
                           duration=1000 // self.fps, loop=0)

    def clear(self):
        self.frames = []
        self._context_images = None


def evaluate_episode(
    env,
    policy,
    task_name: str,
    max_steps: int = 200,
    replan_steps: int = 10,
    video_recorder=None,
    debug: bool = False,
    prompt_override: str = None,
) -> dict:
    """Evaluate a single episode with RICL policy."""
    descriptions, obs = env.reset()
    # Use variation-specific description from RLBench
    episode_prompt = prompt_override or (descriptions[0] if descriptions else None)

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
            obs_dict = get_observation_dict(obs, task_name, prompt_override=episode_prompt)
            result = policy.infer(obs_dict, debug=debug)
            actions = result["query_actions"]  # (action_horizon, 7)
            action_buffer = actions
            action_idx = 0
            num_inferences += 1
            # Update context visualization on each replan
            if video_recorder:
                video_recorder.set_context(result)

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
    policy = load_ricl_policy(
        args.checkpoint, args.demos_dir,
        no_interpolation=getattr(args, 'no_interpolation', False),
        config_name=getattr(args, 'config_name', None),
        random=getattr(args, 'random', False),
    )

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
    planner_cache_installed = False

    for task_name in tasks:
        # Rebuild KNN index with only this task's demos (within-task retrieval).
        # RiclRandomPolicy does not have rebuild_index (random sampling is data-driven).
        task_demos_dir = os.path.join(args.demos_dir, task_name)
        if hasattr(policy, "rebuild_index"):
            if os.path.isdir(task_demos_dir):
                policy.rebuild_index(task_demos_dir)
            else:
                print(f"WARNING: task demos dir not found: {task_demos_dir}, using global index")

        max_steps = TASK_MAX_STEPS.get(task_name, 200)
        print(f"\n{'='*60}")
        print(f"Task: {task_name} | Episodes: {args.episodes} | Max steps: {max_steps}")
        print(f"{'='*60}")

        task_class = get_task_class(task_name)
        task = env.get_task(task_class)
        if not planner_cache_installed:
            _planner_cache.install(task)
            planner_cache_installed = True
            print("Planner cache installed")

        successes = 0
        episodes = []
        max_attempts = 1 + args.blind_retry
        cumulative_successes = [0] * max_attempts

        for ep in range(args.episodes):
            _planner_cache.clear()
            # Fix scene determinism: seed per-episode so all models see same scene.
            # For retry, reuse the same seed/variation before each attempt. RICL
            # receives success-demo context as usual and never receives failure slots.
            ep_seed = args.seed * 10000 + ep

            # Set variation for this episode (matching eval_baseline.py)
            var_indices = TASK_VARIATIONS.get(task_name, [0])
            var_idx = var_indices[ep % len(var_indices)]

            video_recorder = None
            if args.save_video:
                video_path = os.path.join(
                    args.output_dir,
                    f"ricl_{task_name}_ep{ep:03d}_{timestamp}.mp4"
                )
                video_recorder = VideoRecorder(video_path)

            print(f"  Episode {ep + 1}/{args.episodes} (var={var_idx})", end="")

            attempt_results = []
            for attempt in range(max_attempts):
                if attempt > 0:
                    print(f" [retry {attempt}]", end="")
                    if video_recorder:
                        sep = np.full((256, 768, 3), 40, dtype=np.uint8)
                        sep[:4, :, 0] = 200
                        video_recorder.frames.append(sep)

                np.random.seed(ep_seed)
                random.seed(ep_seed)
                task.set_variation(var_idx)

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

                attempt_results.append(ep_result)
                if ep_result["success"]:
                    break

            if video_recorder:
                video_recorder.save()

            final_success = any(r["success"] for r in attempt_results)
            success_at_attempt = next((i for i, r in enumerate(attempt_results) if r["success"]), -1)
            final_result = {
                "success": final_success,
                "steps": sum(r["steps"] for r in attempt_results),
                "time": sum(r["time"] for r in attempt_results),
                "num_inferences": sum(r["num_inferences"] for r in attempt_results),
                "attempts": len(attempt_results),
                "success_at_attempt": success_at_attempt,
                "attempt_results": attempt_results,
            }

            episodes.append(final_result)
            if final_success:
                successes += 1
                for k in range(success_at_attempt, max_attempts):
                    cumulative_successes[k] += 1
                print(f" -> SUCCESS (attempt {success_at_attempt}, {final_result['steps']} total steps, {final_result['num_inferences']} inferences)")
            else:
                print(f" -> FAILED ({max_attempts} attempts, {final_result['steps']} total steps)")
            print(f"    Planner cache: {_planner_cache.stats()}")

        success_rate = successes / args.episodes
        cumulative_sr = [c / args.episodes for c in cumulative_successes]
        task_result = {
            "task": task_name,
            "num_episodes": args.episodes,
            "success_rate": success_rate,
            "successes": successes,
            "cumulative_sr": cumulative_sr,
            "episodes": episodes,
        }
        all_results[task_name] = task_result

        sr_str = f"{task_name}: SR@0={cumulative_sr[0]*100:.1f}%"
        for k in range(1, max_attempts):
            sr_str += f"  SR@{k}={cumulative_sr[k]*100:.1f}%"
        print(f"\n  {sr_str} ({successes}/{args.episodes} solved within {max_attempts} attempts)")

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
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config file (overrides CLI args)")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--demos_dir", type=str, default="./processed_rlbench")
    parser.add_argument("--task", type=str, default="all")
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--replan_steps", type=int, default=10)
    parser.add_argument("--no_interpolation", action="store_true")
    parser.add_argument("--config_name", type=str, default=None)
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./eval_results")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--random", action="store_true", default=False)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--blind_retry", type=int, default=0,
                        help="Number of additional attempts after failure. RICL keeps success-demo context and receives no failure slots.")

    # Two-pass parsing: load YAML as defaults, CLI overrides
    args_first, _ = parser.parse_known_args()

    if args_first.config:
        import yaml
        with open(args_first.config) as f:
            yaml_config = yaml.safe_load(f)
        model_cfg = yaml_config.get("model", {})
        eval_cfg = yaml_config.get("eval", {})
        output_cfg = yaml_config.get("output", {})
        env_cfg = yaml_config.get("environment", {})
        retry_cfg = yaml_config.get("retry", {})

        yaml_defaults = {}
        if "checkpoint" in model_cfg: yaml_defaults["checkpoint"] = model_cfg["checkpoint"]
        if "config_name" in model_cfg: yaml_defaults["config_name"] = model_cfg["config_name"]
        if "demos_dir" in model_cfg: yaml_defaults["demos_dir"] = model_cfg["demos_dir"]
        if "no_interpolation" in model_cfg and model_cfg["no_interpolation"]: yaml_defaults["no_interpolation"] = True
        if "tasks" in eval_cfg: yaml_defaults["task"] = ",".join(eval_cfg["tasks"])
        if "episodes_per_task" in eval_cfg: yaml_defaults["episodes"] = eval_cfg["episodes_per_task"]
        if "seed" in eval_cfg: yaml_defaults["seed"] = eval_cfg["seed"]
        if "dir" in output_cfg: yaml_defaults["output_dir"] = output_cfg["dir"]
        if "save_video" in output_cfg and output_cfg["save_video"]: yaml_defaults["save_video"] = True
        if "headless" in env_cfg: yaml_defaults["headless"] = env_cfg["headless"]
        if "max_k" in retry_cfg: yaml_defaults["blind_retry"] = retry_cfg["max_k"]
        if "blind_retry" in eval_cfg: yaml_defaults["blind_retry"] = eval_cfg["blind_retry"]

        parser.set_defaults(**yaml_defaults)

    args = parser.parse_args()

    if not args.checkpoint:
        parser.error("--checkpoint is required (or set model.checkpoint in YAML config)")

    if args.display:
        args.headless = False

    np.random.seed(args.seed)
    random.seed(args.seed)

    run_evaluation(args)


if __name__ == "__main__":
    main()
