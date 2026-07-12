#!/usr/bin/env python3
"""Matched RICL retry evaluation for the RoboRetry paper tables.

This runner intentionally differs from ``eval_ricl_rlbench.py`` in one key
respect: a retry attempt restores the exact scene snapshot from attempt 0
instead of calling ``env.reset()`` again.  That matches the RF-F Table-1 retry
protocol used by ``aha_openpi/scripts/eval_multi_condition.py``.
"""

import argparse
import json
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

_SCRIPT_DIR = Path(__file__).resolve().parent
_RICL_DIR = _SCRIPT_DIR.parent
_REPO_ROOT = _RICL_DIR.parent
sys.path.insert(0, str(_RICL_DIR / "src"))
sys.path.insert(0, str(_REPO_ROOT / "shared"))

from eval_ricl_rlbench import PlannerCache, load_ricl_policy  # noqa: E402
from rlbench_io import (  # noqa: E402
    TASK_MAX_STEPS,
    TASK_VARIATIONS,
    VLA_TASK_DESCRIPTIONS,
    apply_delta_action,
    quat_to_euler,
)

TASKS_12 = [
    "push_button",
    "push_buttons",
    "meat_on_grill",
    "pick_up_cup",
    "open_wine_bottle",
    "turn_tap",
    "close_box",
    "close_microwave",
    "take_lid_off_saucepan",
    "open_drawer",
    "lamp_on",
    "pick_and_lift",
]

DEFAULT_SEEDS = [21, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 39, 40, 41, 42, 43]
_planner_cache = PlannerCache()


class SceneManager:
    """Save and restore the exact initial task scene for deterministic retry."""

    def __init__(self):
        self.arm_config = None
        self.gripper_config = None
        self.arm_joints = None
        self.gripper_joints = None
        self.task_state = None
        self.initial_obs = None

    def save_initial_state(self, task_env, obs):
        scene = task_env._scene
        try:
            canonical_obs = scene.get_observation()
        except Exception:
            canonical_obs = obs
        self.arm_config = scene.robot.arm.get_configuration_tree()
        self.gripper_config = scene.robot.gripper.get_configuration_tree()
        self.arm_joints = scene.robot.arm.get_joint_positions()
        self.gripper_joints = scene.robot.gripper.get_joint_positions()
        self.task_state = scene.task.get_state()
        self.initial_obs = deepcopy(canonical_obs)

    def _restore_robot(self, scene):
        scene.pyrep.set_configuration_tree(self.arm_config)
        scene.pyrep.set_configuration_tree(self.gripper_config)
        scene.robot.arm.set_joint_positions(self.arm_joints, disable_dynamics=True)
        scene.robot.arm.set_joint_target_velocities([0] * len(scene.robot.arm.joints))
        scene.robot.arm.set_joint_target_positions(self.arm_joints)
        scene.robot.gripper.set_joint_positions(self.gripper_joints, disable_dynamics=True)
        scene.robot.gripper.set_joint_target_velocities([0] * len(scene.robot.gripper.joints))
        scene.robot.gripper.set_joint_target_positions(self.gripper_joints)

    def restore_full(self, task_env):
        scene = task_env._scene
        scene.robot.gripper.release()
        scene.task.cleanup_()
        scene.task.restore_state(self.task_state)
        scene.task.set_initial_objects_in_scene()

        for cond_list_attr in ("_success_conditions", "_fail_conditions"):
            cond_sets = getattr(scene.task, cond_list_attr, [])
            for cs in cond_sets:
                sub_conds = getattr(cs, "_conditions", [cs])
                for cond in sub_conds:
                    if hasattr(cond, "_original_pos") and hasattr(cond, "_joint"):
                        cond._original_pos = cond._joint.get_joint_position()

        self._restore_robot(scene)
        for _ in range(10):
            scene.pyrep.step()
        self._restore_robot(scene)
        return deepcopy(self.initial_obs)


def parse_seed_spec(spec: str) -> list[int]:
    if not spec:
        return list(DEFAULT_SEEDS)
    seeds = []
    for part in spec.replace(",", " ").split():
        if "-" in part:
            a, b = part.split("-", 1)
            seeds.extend(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    return sorted(dict.fromkeys(seeds))


def parse_config(path: str | None) -> dict:
    cfg = {}
    if path:
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    return cfg


def resolve_config_path(value: str | None, base: Path = _RICL_DIR) -> str | None:
    """Resolve config paths that were historically written relative to aha_ricl."""
    if not value:
        return value
    path = Path(value).expanduser()
    if path.is_absolute() or path.exists():
        return str(path)
    candidate = base / path
    if candidate.exists():
        return str(candidate)
    return str(path)


def get_task_class(task_name: str):
    import importlib

    class_name = "".join(word.capitalize() for word in task_name.split("_"))
    task_module = importlib.import_module(f"rlbench.tasks.{task_name}")
    return getattr(task_module, class_name)


def setup_rlbench(headless: bool = True):
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.environment import Environment
    from rlbench.observation_config import ObservationConfig

    obs_config = ObservationConfig()
    obs_config.front_camera.set_all(True)
    obs_config.wrist_camera.set_all(True)
    obs_config.overhead_camera.set_all(True)
    obs_config.front_camera.image_size = (256, 256)
    obs_config.wrist_camera.image_size = (256, 256)
    obs_config.overhead_camera.image_size = (256, 256)

    action_mode = MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaPlanning(),
        gripper_action_mode=Discrete(),
    )
    env = Environment(action_mode=action_mode, obs_config=obs_config, headless=headless)
    env.launch()
    return env


def matched_variation(task_name: str, seed: int) -> int:
    var_indices = TASK_VARIATIONS.get(task_name, [0])
    if seed < 24:
        return var_indices[0]
    return var_indices[(seed - 24) % len(var_indices)]


def get_observation_dict(obs, task_name: str, prompt_override: str | None = None) -> dict:
    gripper_pose = obs.gripper_pose
    gripper_open = float(obs.gripper_open)
    euler = quat_to_euler(gripper_pose[3:])
    state = np.concatenate([gripper_pose[:3], euler, [gripper_open], [0.0]]).astype(np.float32)

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


def run_single_attempt(task_env, policy, task_name: str, obs, max_steps: int, prompt: str | None, debug: bool) -> dict:
    steps = 0
    success = False
    action_buffer = []
    action_idx = 0
    num_inferences = 0
    start_time = time.time()

    for _ in range(max_steps):
        if action_idx >= len(action_buffer):
            obs_dict = get_observation_dict(obs, task_name, prompt_override=prompt)
            result = policy.infer(obs_dict, debug=debug)
            actions = result["query_actions"]
            action_buffer = actions
            action_idx = 0
            num_inferences += 1

        action = action_buffer[action_idx]
        action_idx += 1
        current_pose = obs.gripper_pose
        new_obs, reward, done = apply_delta_action(task_env, action, current_pose)
        if new_obs is None:
            break
        obs = new_obs
        steps += 1
        if done or reward > 0:
            success = reward > 0
            break

    return {
        "success": bool(success),
        "steps": int(steps),
        "time": float(time.time() - start_time),
        "num_inferences": int(num_inferences),
    }


def evaluate_retry_episode(task_env, policy, task_name: str, obs, scene_mgr: SceneManager, prompt: str | None,
                           max_k: int, max_steps: int, debug: bool) -> dict:
    initial_rng = getattr(policy, "_rng", None)
    attempt_results = []

    for attempt in range(max_k + 1):
        if initial_rng is not None and hasattr(policy, "_rng"):
            policy._rng = initial_rng
        if attempt > 0:
            obs = scene_mgr.restore_full(task_env)

        print(f"    Attempt {attempt}", end="")
        result = run_single_attempt(task_env, policy, task_name, obs, max_steps, prompt, debug)
        attempt_result = {
            "attempt": attempt,
            "success": result["success"],
            "steps": result["steps"],
            "time": result["time"],
            "num_inferences": result["num_inferences"],
        }
        attempt_results.append(attempt_result)

        if result["success"]:
            print(f" -> SUCCESS in {result['steps']} steps")
            break
        print(f" -> FAILED after {result['steps']} steps")

    success_at_attempt = next((r["attempt"] for r in attempt_results if r["success"]), -1)
    return {
        "success": success_at_attempt >= 0,
        "final_success": success_at_attempt >= 0,
        "success_at_attempt": success_at_attempt,
        "total_attempts": len(attempt_results),
        "attempts": attempt_results,
        "attempt_results": attempt_results,
        "steps": sum(r["steps"] for r in attempt_results),
        "time": sum(r["time"] for r in attempt_results),
        "num_inferences": sum(r["num_inferences"] for r in attempt_results),
    }


def is_seed_complete(path: Path, tasks: list[str]) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
    except Exception:
        return False
    return all(task in data and data[task].get("episodes") for task in tasks)


def save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def run_seed(args, cfg: dict, seed: int):
    model_cfg = cfg.get("model", {})
    eval_cfg = cfg.get("eval", {})
    retry_cfg = cfg.get("retry", {})
    env_cfg = cfg.get("environment", {})

    checkpoint = args.checkpoint or model_cfg.get("checkpoint")
    config_name = args.config_name or model_cfg.get("config_name")
    demos_dir = resolve_config_path(args.demos_dir or model_cfg.get("demos_dir", "./processed_rlbench_v4"))
    no_interpolation = bool(args.no_interpolation or model_cfg.get("no_interpolation", False))
    random_retrieval = bool(args.random or model_cfg.get("random", False))
    max_k = args.max_k if args.max_k is not None else int(retry_cfg.get("max_k", 2))
    tasks = args.tasks or eval_cfg.get("tasks") or TASKS_12
    headless = args.headless if args.headless is not None else bool(env_cfg.get("headless", True))
    output_dir = Path(args.output_dir)
    output_path = output_dir / f"ricl_matched_seed{seed}.json"

    if not checkpoint:
        raise ValueError("Missing checkpoint. Pass --checkpoint or set model.checkpoint in YAML.")
    if args.resume and is_seed_complete(output_path, tasks):
        print(f"skip seed {seed}, complete: {output_path}")
        return

    print("=" * 72)
    print(f"RICL matched retry seed {seed}")
    print(f"checkpoint: {checkpoint}")
    print(f"demos_dir:  {demos_dir}")
    print(f"output:     {output_path}")
    print(f"tasks:      {len(tasks)}")
    print(f"max_k:      {max_k}")
    print("=" * 72)

    policy = load_ricl_policy(
        checkpoint,
        demos_dir,
        no_interpolation=no_interpolation,
        config_name=config_name,
        random=random_retrieval,
    )

    env = setup_rlbench(headless=headless)
    planner_cache_installed = False
    all_results = {}
    if output_path.exists():
        try:
            all_results = json.loads(output_path.read_text())
        except Exception:
            all_results = {}

    try:
        for task_name in tasks:
            if args.resume and task_name in all_results and all_results[task_name].get("episodes"):
                print(f"skip seed {seed} task {task_name}, exists")
                continue

            if hasattr(policy, "rebuild_index"):
                task_demos_dir = os.path.join(demos_dir, task_name)
                if os.path.isdir(task_demos_dir):
                    policy.rebuild_index(task_demos_dir)
                else:
                    print(f"WARNING: task demos dir not found: {task_demos_dir}, using global index")

            task_class = get_task_class(task_name)
            task_env = env.get_task(task_class)
            if not planner_cache_installed:
                _planner_cache.install(task_env)
                planner_cache_installed = True
                print("Planner cache installed")

            max_steps = TASK_MAX_STEPS.get(task_name, 200)
            ep_seed = seed * 10000
            var_idx = matched_variation(task_name, seed)
            np.random.seed(ep_seed)
            random.seed(ep_seed)
            task_env.set_variation(var_idx)

            print(f"\nTask: {task_name} | seed={seed} | var={var_idx} | ep_seed={ep_seed}")
            descriptions, obs = task_env.reset()
            prompt = descriptions[0] if descriptions else None
            scene_mgr = SceneManager()
            scene_mgr.save_initial_state(task_env, obs)
            _planner_cache.clear()

            result = evaluate_retry_episode(
                task_env,
                policy,
                task_name,
                obs,
                scene_mgr,
                prompt,
                max_k=max_k,
                max_steps=max_steps,
                debug=args.debug,
            )

            cumulative_successes = [0] * (max_k + 1)
            if result["success_at_attempt"] >= 0:
                for k in range(result["success_at_attempt"], max_k + 1):
                    cumulative_successes[k] = 1

            all_results[task_name] = {
                "task": task_name,
                "num_episodes": 1,
                "success_rate": 1.0 if result["final_success"] else 0.0,
                "successes": 1 if result["final_success"] else 0,
                "cumulative_sr": cumulative_successes,
                "variation": var_idx,
                "episodes": [result],
            }
            save_json(output_path, all_results)
            print(f"    Planner cache: {_planner_cache.stats()}")
    finally:
        env.shutdown()

    print(f"\nSaved: {output_path}")


def success_by_k(ep: dict, k: int) -> bool:
    if not ep.get("final_success", ep.get("success", False)):
        return False
    sat = int(ep.get("success_at_attempt", -1))
    return 0 <= sat <= k


def aggregate(input_dir: Path, seeds: list[int], output: Path | None):
    episodes_by_task = {task: [] for task in TASKS_12}
    missing = []
    for seed in seeds:
        path = input_dir / f"ricl_matched_seed{seed}.json"
        if not path.exists():
            missing.append(seed)
            continue
        data = json.loads(path.read_text())
        for task in TASKS_12:
            td = data.get(task)
            if not td:
                continue
            for ep in td.get("episodes", []):
                ep = dict(ep)
                ep["seed"] = seed
                ep["task"] = task
                episodes_by_task[task].append(ep)

    episodes = [ep for eps in episodes_by_task.values() for ep in eps]
    if not episodes:
        raise RuntimeError(f"No episodes found under {input_dir}")

    print("RICL matched retry aggregate")
    print(f"Seeds loaded: {[s for s in seeds if s not in missing]}")
    print(f"Missing seeds: {missing}")
    print(f"N task-seed: {len(episodes)}")
    print()
    print(f"{'Task':28s} {'N':>3s} {'SR@1':>8s} {'SR@2':>8s} {'SR@3':>8s}")
    print("-" * 62)

    per_task = {}
    for task in TASKS_12:
        eps = episodes_by_task[task]
        if not eps:
            continue
        vals = [sum(success_by_k(ep, k) for ep in eps) / len(eps) for k in range(3)]
        per_task[task] = {"n": len(eps), "sr1": vals[0], "sr2": vals[1], "sr3": vals[2]}
        print(f"{task:28s} {len(eps):3d} {vals[0]*100:7.1f}% {vals[1]*100:7.1f}% {vals[2]*100:7.1f}%")

    sr = [sum(success_by_k(ep, k) for ep in episodes) / len(episodes) for k in range(3)]
    summary = {
        "seeds": [s for s in seeds if s not in missing],
        "missing_seeds": missing,
        "n_task_seed": len(episodes),
        "sr1": sr[0],
        "sr2": sr[1],
        "sr3": sr[2],
        "lift": sr[2] - sr[0],
        "per_task": per_task,
    }

    print("-" * 62)
    print(f"{'Macro':28s} {len(episodes):3d} {sr[0]*100:7.1f}% {sr[1]*100:7.1f}% {sr[2]*100:7.1f}%")
    print(f"Lift SR@3-SR@1: {(sr[2] - sr[0]) * 100:+.1f}%")

    if output:
        save_json(output, summary)
        print(f"\nSaved: {output}")


def main():
    parser = argparse.ArgumentParser(description="Matched 20-seed RICL blind-retry evaluation")
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config_name", default=None)
    parser.add_argument("--demos_dir", default=None)
    parser.add_argument("--no_interpolation", action="store_true")
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--max_k", type=int, default=None)
    parser.add_argument("--output_dir", default="aha_ricl/eval_results/ricl_matched_retry_20seed")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--input_dir", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    seeds = parse_seed_spec(args.seeds)

    if args.aggregate:
        input_dir = Path(args.input_dir or args.output_dir)
        output = Path(args.output) if args.output else input_dir / "aggregated_current.json"
        aggregate(input_dir, seeds, output)
        return

    cfg = parse_config(args.config)
    for seed in seeds:
        run_seed(args, cfg, seed)

    aggregate(Path(args.output_dir), seeds, Path(args.output_dir) / "aggregated_current.json")


if __name__ == "__main__":
    main()
