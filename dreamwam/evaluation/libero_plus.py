import json
import inspect
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from .rollout import ActionPolicy, evaluate_benchmark


LIBERO_PLUS_SUITES = (
    "libero_10",
    "libero_goal",
    "libero_spatial",
    "libero_object",
)
LIBERO_PLUS_TASK_COUNTS = {
    "libero_10": 2519,
    "libero_goal": 2591,
    "libero_spatial": 2402,
    "libero_object": 2518,
}
LIBERO_PLUS_PERTURBATIONS = (
    ("Camera", "Camera Viewpoints"),
    ("Robot", "Robot Initial States"),
    ("Language", "Language Instructions"),
    ("Light", "Light Conditions"),
    ("Background", "Background Textures"),
    ("Noise", "Sensor Noise"),
    ("Layout", "Objects Layout"),
)
LIBERO_PLUS_PERTURBATION_COUNTS = {
    "Camera": 1599,
    "Robot": 1550,
    "Language": 1537,
    "Light": 1142,
    "Background": 1076,
    "Noise": 1601,
    "Layout": 1525,
}


def configure_libero_plus(
    *,
    repository_root: str | Path,
    config_root: str | Path,
    dataset_root: str | Path,
) -> None:
    repository_root = Path(repository_root).resolve()
    package_root = repository_root / "libero"
    benchmark_root = package_root / "libero"
    required_directories = {
        "package": package_root,
        "bddl_files": benchmark_root / "bddl_files",
        "init_states": benchmark_root / "init_files",
        "assets": benchmark_root / "assets",
    }
    classification_path = benchmark_root / "benchmark" / "task_classification.json"
    missing_directories = {
        name: path
        for name, path in required_directories.items()
        if not path.is_dir()
    }
    if missing_directories or not classification_path.is_file():
        raise FileNotFoundError(
            "Incomplete LIBERO Plus installation: "
            f"missing_directories={missing_directories}, "
            f"task_classification={classification_path}"
        )
    _preload_noise_dependencies()

    config_root = Path(config_root).resolve()
    config_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(required_directories["bddl_files"]),
        "init_states": str(required_directories["init_states"]),
        "datasets": str(Path(dataset_root).resolve()),
        "assets": str(required_directories["assets"]),
    }
    rendered = yaml.safe_dump(payload, sort_keys=False)
    config_path = config_root / "config.yaml"
    if config_path.exists():
        if config_path.read_text() != rendered:
            raise FileExistsError(
                f"Existing LIBERO Plus config does not match this checkout: {config_path}"
            )
    else:
        config_path.write_text(rendered)
    os.environ["LIBERO_CONFIG_PATH"] = str(config_root)
    source_path = str(repository_root)
    if source_path not in sys.path:
        sys.path.insert(0, source_path)


def _validate_libero_plus_import(repository_root: str | Path) -> None:
    import libero.libero as libero_core

    package_path = Path(inspect.getfile(libero_core)).resolve()
    repository_root = Path(repository_root).resolve()
    if not package_path.is_relative_to(repository_root):
        raise RuntimeError(
            f"LIBERO Plus evaluation imported the wrong package: {package_path}"
        )


def _preload_noise_dependencies() -> None:
    try:
        from skimage.filters import gaussian as _gaussian
        from wand.api import library as _wand_library
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "LIBERO Plus sensor-noise evaluation requires scikit-image, Wand, "
            "and a loadable ImageMagick MagickWand runtime."
        ) from exc
    if _gaussian is None or _wand_library is None:
        raise RuntimeError("Failed to load LIBERO Plus sensor-noise dependencies.")


def _validate_noise_operators() -> None:
    from libero.libero.envs.env_wrapper import gaussian_blur, motion_blur
    from PIL import Image

    image = Image.fromarray(np.full((224, 224, 3), 127, dtype=np.uint8))
    gaussian_blur(image, severity=1)
    motion_blur(image, severity=1)


def _load_task_classification(
    repository_root: str | Path,
) -> dict[str, dict[int, dict]]:
    classification_path = (
        Path(repository_root).resolve()
        / "libero"
        / "libero"
        / "benchmark"
        / "task_classification.json"
    )
    if not classification_path.is_file():
        raise FileNotFoundError(
            f"Missing LIBERO Plus task classification: {classification_path}"
        )
    payload = json.loads(classification_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("LIBERO Plus task classification must be a JSON object.")

    official_categories = {
        official_name for _, official_name in LIBERO_PLUS_PERTURBATIONS
    }
    category_totals: Counter[str] = Counter()
    classification: dict[str, dict[int, dict]] = {}
    for suite_name, expected_count in LIBERO_PLUS_TASK_COUNTS.items():
        rows = payload.get(suite_name)
        if not isinstance(rows, list) or len(rows) != expected_count:
            raise ValueError(
                f"LIBERO Plus classification for {suite_name} must contain "
                f"{expected_count} tasks, got "
                f"{len(rows) if isinstance(rows, list) else type(rows).__name__}."
            )
        indexed_rows: dict[int, dict] = {}
        for row in rows:
            if not isinstance(row, dict):
                raise TypeError(
                    f"LIBERO Plus classification row in {suite_name} is not an object."
                )
            classification_id = row.get("id")
            if isinstance(classification_id, bool) or not isinstance(
                classification_id, int
            ):
                raise TypeError(
                    f"Invalid classification id in {suite_name}: {classification_id!r}."
                )
            if classification_id in indexed_rows:
                raise ValueError(
                    f"Duplicate classification id {classification_id} in {suite_name}."
                )
            category = row.get("category")
            if category not in official_categories:
                raise ValueError(
                    f"Unknown LIBERO Plus category in {suite_name}: {category!r}."
                )
            if not isinstance(row.get("name"), str) or not row["name"]:
                raise ValueError(
                    f"Missing task name for classification id {classification_id} "
                    f"in {suite_name}."
                )
            indexed_rows[classification_id] = row
            category_totals[category] += 1

        expected_ids = set(range(1, expected_count + 1))
        if set(indexed_rows) != expected_ids:
            raise ValueError(
                f"LIBERO Plus classification ids for {suite_name} must cover "
                f"[1,{expected_count}]."
            )
        classification[suite_name] = indexed_rows

    short_name_by_category = {
        official_name: short_name
        for short_name, official_name in LIBERO_PLUS_PERTURBATIONS
    }
    actual_category_counts = {
        short_name_by_category[category]: count
        for category, count in category_totals.items()
    }
    if actual_category_counts != LIBERO_PLUS_PERTURBATION_COUNTS:
        raise ValueError(
            "LIBERO Plus perturbation counts do not match the 10,030-task "
            f"protocol: {actual_category_counts}."
        )
    return classification


def _checked_count(value, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{field} must be an integer, got {value!r}.")
    value = int(value)
    if value < 0:
        raise ValueError(f"{field} cannot be negative, got {value}.")
    return value


def _summarize_libero_plus_results(
    suite_results: dict[str, dict],
    *,
    classification: dict[str, dict[int, dict]],
    trials_per_task: int,
) -> dict:
    if not suite_results:
        raise ValueError("LIBERO Plus results cannot be empty.")
    if trials_per_task != 1:
        raise ValueError(
            "The official LIBERO Plus protocol uses exactly one trial per task."
        )

    category_name_map = dict(LIBERO_PLUS_PERTURBATIONS)
    official_to_short = {
        official_name: short_name
        for short_name, official_name in LIBERO_PLUS_PERTURBATIONS
    }
    category_stats = {
        short_name: {"successes": 0, "episodes": 0, "tasks": 0}
        for short_name in category_name_map
    }
    summarized_suites: dict[str, dict] = {}
    all_tasks_complete = True
    total_successes = 0
    total_episodes = 0
    total_tasks = 0

    for suite_name, suite_result in suite_results.items():
        if suite_name not in LIBERO_PLUS_TASK_COUNTS:
            raise ValueError(f"Unsupported LIBERO Plus suite: {suite_name}.")
        if not isinstance(suite_result, dict):
            raise TypeError(f"Result for {suite_name} must be an object.")
        available_tasks = _checked_count(
            suite_result.get("available_tasks"),
            field=f"{suite_name}.available_tasks",
        )
        expected_tasks = LIBERO_PLUS_TASK_COUNTS[suite_name]
        if available_tasks != expected_tasks:
            raise ValueError(
                f"{suite_name} exposes {available_tasks} tasks; the official "
                f"LIBERO Plus protocol requires {expected_tasks}."
            )
        raw_tasks = suite_result.get("tasks")
        if not isinstance(raw_tasks, dict) or not raw_tasks:
            raise ValueError(f"{suite_name}.tasks must be a non-empty mapping.")

        tasks: dict[int, dict] = {}
        suite_successes = 0
        suite_episodes = 0
        for raw_task_id, raw_task_result in raw_tasks.items():
            if isinstance(raw_task_id, bool):
                raise TypeError(f"Invalid task id in {suite_name}: {raw_task_id!r}.")
            try:
                task_id = int(raw_task_id)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"Invalid task id in {suite_name}: {raw_task_id!r}."
                ) from exc
            if task_id in tasks or task_id < 0 or task_id >= expected_tasks:
                raise ValueError(f"Invalid or duplicate task id {task_id} in {suite_name}.")
            if not isinstance(raw_task_result, dict):
                raise TypeError(f"Result for {suite_name}/task{task_id} is not an object.")

            metadata = classification[suite_name][task_id + 1]
            task_name = raw_task_result.get("name")
            if task_name != metadata["name"]:
                raise ValueError(
                    f"Task metadata mismatch for {suite_name}/task{task_id}: "
                    f"benchmark={task_name!r}, classification={metadata['name']!r}."
                )
            successes = _checked_count(
                raw_task_result.get("successes"),
                field=f"{suite_name}/task{task_id}.successes",
            )
            episodes = _checked_count(
                raw_task_result.get("episodes"),
                field=f"{suite_name}/task{task_id}.episodes",
            )
            if episodes == 0 or successes > episodes:
                raise ValueError(
                    f"Invalid outcome for {suite_name}/task{task_id}: "
                    f"{successes}/{episodes}."
                )
            if episodes != trials_per_task:
                raise ValueError(
                    f"{suite_name}/task{task_id} contains {episodes} episodes; "
                    f"expected {trials_per_task}."
                )
            short_category = official_to_short[metadata["category"]]
            category_stats[short_category]["successes"] += successes
            category_stats[short_category]["episodes"] += episodes
            category_stats[short_category]["tasks"] += 1
            suite_successes += successes
            suite_episodes += episodes
            tasks[task_id] = {
                **raw_task_result,
                "category": short_category,
                "difficulty_level": metadata.get("difficulty_level"),
                "success_rate": successes / episodes,
            }

        expected_task_ids = set(range(expected_tasks))
        suite_complete = set(tasks) == expected_task_ids
        all_tasks_complete = all_tasks_complete and suite_complete
        if suite_successes != suite_result.get("successes") or suite_episodes != suite_result.get(
            "episodes"
        ):
            raise ValueError(
                f"Aggregate outcome mismatch in {suite_name}: derived "
                f"{suite_successes}/{suite_episodes}, reported "
                f"{suite_result.get('successes')}/{suite_result.get('episodes')}."
            )
        summarized_suites[suite_name] = {
            "tasks": tasks,
            "evaluated_tasks": len(tasks),
            "expected_tasks": expected_tasks,
            "complete": suite_complete,
            "successes": suite_successes,
            "episodes": suite_episodes,
            "success_rate": suite_successes / suite_episodes,
        }
        total_successes += suite_successes
        total_episodes += suite_episodes
        total_tasks += len(tasks)

    perturbations = {}
    for short_name, official_name in LIBERO_PLUS_PERTURBATIONS:
        stats = category_stats[short_name]
        perturbations[short_name] = {
            "official_category": official_name,
            **stats,
            "success_rate": (
                stats["successes"] / stats["episodes"]
                if stats["episodes"] > 0
                else None
            ),
        }

    selected_suites = tuple(suite_results)
    full_suite_set = set(selected_suites) == set(LIBERO_PLUS_SUITES)
    official_protocol = (
        full_suite_set and all_tasks_complete and trials_per_task == 1
    )
    if official_protocol:
        actual_counts = {
            name: stats["tasks"] for name, stats in perturbations.items()
        }
        if actual_counts != LIBERO_PLUS_PERTURBATION_COUNTS:
            raise ValueError(
                "Evaluated perturbation counts do not match the official "
                f"LIBERO Plus protocol: {actual_counts}."
            )

    suite_rates = [
        suite["success_rate"] for suite in summarized_suites.values()
    ]
    perturbation_rates = [
        perturbation["success_rate"]
        for perturbation in perturbations.values()
        if perturbation["success_rate"] is not None
    ]
    return {
        "benchmark": "LIBERO-Plus",
        "official_protocol": official_protocol,
        "protocol": {
            "suite_order": list(selected_suites),
            "trials_per_task": trials_per_task,
            "evaluated_tasks": total_tasks,
            "expected_official_tasks": sum(LIBERO_PLUS_TASK_COUNTS.values()),
        },
        "suites": summarized_suites,
        "suite_average_success_rate": float(np.mean(suite_rates)),
        "successes": total_successes,
        "episodes": total_episodes,
        "weighted_success_rate": total_successes / total_episodes,
        "perturbations": perturbations,
        "perturbation_average_success_rate": float(
            np.mean(perturbation_rates)
        ),
    }


def summarize_libero_plus_results(
    suite_results: dict[str, dict],
    *,
    repository_root: str | Path,
    trials_per_task: int = 1,
) -> dict:
    classification = _load_task_classification(repository_root)
    return _summarize_libero_plus_results(
        suite_results,
        classification=classification,
        trials_per_task=trials_per_task,
    )


def evaluate_libero_plus(
    policy: ActionPolicy,
    *,
    suite_names: tuple[str, ...] = LIBERO_PLUS_SUITES,
    trials_per_task: int = 1,
    task_ids: tuple[int, ...] | None = None,
    seed: int = 42,
    resolution: int = 256,
    replan_steps: int = 10,
    wait_steps: int = 30,
    repository_root: str | Path,
) -> dict:
    if trials_per_task != 1:
        raise ValueError(
            "The official LIBERO Plus protocol uses exactly one trial per task."
        )
    if not suite_names or len(set(suite_names)) != len(suite_names):
        raise ValueError("suite_names must be non-empty and contain no duplicates.")
    unknown_suites = sorted(set(suite_names) - set(LIBERO_PLUS_SUITES))
    if unknown_suites:
        raise ValueError(f"Unsupported LIBERO Plus suites: {unknown_suites}.")
    if task_ids is not None and len(suite_names) != 1:
        raise ValueError("task_ids can only be used with exactly one suite.")

    _preload_noise_dependencies()
    _validate_libero_plus_import(repository_root)
    _validate_noise_operators()
    classification = _load_task_classification(repository_root)
    suite_results = {}
    for suite_name in suite_names:
        suite_results[suite_name] = evaluate_benchmark(
            policy,
            suite_name=suite_name,
            trials_per_task=trials_per_task,
            task_ids=task_ids,
            seed=seed,
            resolution=resolution,
            replan_steps=replan_steps,
            wait_steps=wait_steps,
        )
    summary = _summarize_libero_plus_results(
        suite_results,
        classification=classification,
        trials_per_task=trials_per_task,
    )
    summary["protocol"].update(
        {
            "seed": seed,
            "resolution": resolution,
            "replan_steps": replan_steps,
            "wait_steps": wait_steps,
        }
    )
    return summary
