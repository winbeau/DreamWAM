import inspect
import math
import random
import re
from pathlib import Path
from typing import Protocol

import numpy as np
import torch


class ActionPolicy(Protocol):
    def predict_action(
        self,
        *,
        images: dict[str, np.ndarray],
        state: np.ndarray,
        instruction: str,
    ) -> np.ndarray:
        ...


def _libero_images(observation: dict) -> dict[str, np.ndarray]:
    return {
        "agentview": np.ascontiguousarray(
            observation["agentview_image"][::-1, ::-1]
        ),
        "wrist": np.ascontiguousarray(
            observation["robot0_eye_in_hand_image"][::-1, ::-1]
        ),
    }


def _quat_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64).copy()
    quaternion[3] = np.clip(quaternion[3], -1.0, 1.0)
    denominator = math.sqrt(max(1.0 - quaternion[3] ** 2, 0.0))
    if math.isclose(denominator, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (
        quaternion[:3] * 2.0 * math.acos(quaternion[3]) / denominator
    ).astype(np.float32)


def _robot_state(observation: dict) -> np.ndarray:
    return np.concatenate(
        [
            observation["robot0_eef_pos"],
            _quat_to_axis_angle(observation["robot0_eef_quat"]),
            observation["robot0_gripper_qpos"],
        ]
    ).astype(np.float32)


def _resolve_init_states_path(task) -> Path:
    from libero.libero import get_libero_path

    root = Path(get_libero_path("init_states"))
    filename = str(task.init_states_file)
    suite_root = root / task.problem_folder
    candidates = [suite_root / filename]
    suffix = Path(filename).suffix
    if "_language_" in filename:
        candidates.append(suite_root / f"{filename.split('_language_')[0]}{suffix}")
    if "_view_" in filename:
        candidates.append(suite_root / f"{filename.split('_view_')[0]}{suffix}")
    if "_table_" in filename:
        candidates.append(suite_root / re.sub(r"_table_\d+", "", filename))
    if "_tb_" in filename:
        candidates.append(suite_root / re.sub(r"_tb_\d+", "", filename))
    if "_light_" in filename:
        candidates.append(suite_root / f"{filename.split('_light_')[0]}{suffix}")
    if "_add_" in filename or "_level" in filename:
        candidates.append(root / "libero_newobj" / task.problem_folder / filename)

    for candidate in dict.fromkeys(candidates):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"cannot resolve init states for task {task.name!r}: "
        f"{[str(path) for path in candidates]}"
    )


def _load_init_states(task_suite, task_id: int) -> torch.Tensor:
    task = task_suite.get_task(task_id)
    load_kwargs = {}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = False
    states = torch.load(_resolve_init_states_path(task), **load_kwargs)
    if "_add_" in str(task.init_states_file) or "_level" in str(
        task.init_states_file
    ):
        states = states.reshape(1, -1)
    return states


def _make_environment(task, resolution: int, seed: int):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl_file = (
        Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    if not bddl_file.is_file():
        raise FileNotFoundError(f"missing BDDL file: {bddl_file}")
    environment = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=resolution,
        camera_widths=resolution,
    )
    environment.seed(seed)
    return environment


def _set_evaluation_seed(seed: int) -> None:
    if seed < 0 or seed >= np.iinfo(np.uint32).max:
        raise ValueError("seed must be inside the uint32 range.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _max_steps(suite_name: str) -> int:
    limits = {
        "libero_spatial": 400,
        "libero_object": 400,
        "libero_goal": 400,
        "libero_10": 700,
        "libero_90": 700,
    }
    if suite_name not in limits:
        raise ValueError(f"unsupported LIBERO suite: {suite_name}")
    return limits[suite_name]


def _run_episode(
    *,
    environment,
    initial_state: torch.Tensor,
    instruction: str,
    policy: ActionPolicy,
    max_steps: int,
    replan_steps: int,
    wait_steps: int,
) -> bool:
    environment.reset()
    observation = environment.set_init_state(initial_state)
    for _ in range(wait_steps):
        observation, _, _, _ = environment.step(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
        )

    pending_actions = np.empty((0, 7), dtype=np.float32)
    for _ in range(max_steps):
        if len(pending_actions) == 0:
            pending_actions = np.asarray(
                policy.predict_action(
                    images=_libero_images(observation),
                    state=_robot_state(observation),
                    instruction=instruction,
                ),
                dtype=np.float32,
            )
            if pending_actions.ndim != 2 or pending_actions.shape[1] != 7:
                raise ValueError(
                    "policy action chunk must be [T,7] in LIBERO simulator units, "
                    f"got {tuple(pending_actions.shape)}."
                )
            if len(pending_actions) < replan_steps:
                raise ValueError(
                    f"policy returned {len(pending_actions)} actions, "
                    f"fewer than replan_steps={replan_steps}."
                )
            if not np.isfinite(pending_actions).all():
                raise FloatingPointError("policy returned non-finite actions.")
            pending_actions = pending_actions[:replan_steps]

        action = pending_actions[0]
        pending_actions = pending_actions[1:]
        observation, _, done, _ = environment.step(action)
        if done:
            return True
    return False


def evaluate_benchmark(
    policy: ActionPolicy,
    *,
    suite_name: str,
    trials_per_task: int = 50,
    task_ids: tuple[int, ...] | None = None,
    seed: int = 0,
    resolution: int = 256,
    replan_steps: int = 10,
    wait_steps: int = 30,
) -> dict:
    if trials_per_task <= 0:
        raise ValueError("trials_per_task must be positive.")
    if resolution <= 0 or replan_steps <= 0 or wait_steps < 0:
        raise ValueError("resolution/replan_steps must be positive; wait_steps cannot be negative.")

    from libero.libero import benchmark

    suites = benchmark.get_benchmark_dict()
    if suite_name not in suites:
        raise ValueError(f"unknown LIBERO suite: {suite_name}")
    task_suite = suites[suite_name]()
    if task_ids is None:
        task_ids = tuple(range(task_suite.n_tasks))
    if not task_ids:
        raise ValueError("task_ids cannot be empty.")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("task_ids cannot contain duplicates.")

    results = {}
    total_successes = 0
    total_episodes = 0
    for task_id in task_ids:
        if task_id < 0 or task_id >= task_suite.n_tasks:
            raise ValueError(
                f"task_id {task_id} outside [0,{task_suite.n_tasks})."
            )
        task = task_suite.get_task(task_id)
        initial_states = _load_init_states(task_suite, task_id)
        if len(initial_states) == 0:
            raise ValueError(f"task {task_id} has no initial states.")
        _set_evaluation_seed(seed)
        environment = _make_environment(task, resolution, seed)
        successes = 0
        try:
            for trial_id in range(trials_per_task):
                successes += int(
                    _run_episode(
                        environment=environment,
                        initial_state=initial_states[trial_id % len(initial_states)],
                        instruction=task.language,
                        policy=policy,
                        max_steps=_max_steps(suite_name),
                        replan_steps=replan_steps,
                        wait_steps=wait_steps,
                    )
                )
        finally:
            environment.close()
        results[task_id] = {
            "name": task.name,
            "instruction": task.language,
            "successes": successes,
            "episodes": trials_per_task,
            "success_rate": successes / trials_per_task,
        }
        total_successes += successes
        total_episodes += trials_per_task

    return {
        "suite": suite_name,
        "available_tasks": int(task_suite.n_tasks),
        "tasks": results,
        "successes": total_successes,
        "episodes": total_episodes,
        "success_rate": total_successes / total_episodes,
    }
