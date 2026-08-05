#!/usr/bin/env python3
import argparse
import json

try:
    from wand.api import library as _wand_library
except (ImportError, OSError) as exc:
    raise RuntimeError(
        "LIBERO Plus evaluation requires Wand and the ImageMagick "
        "MagickWand runtime."
    ) from exc

from dreamwam.config import load_release_config
from dreamwam.evaluation.libero_plus import (
    LIBERO_PLUS_SUITES,
    configure_libero_plus,
    evaluate_libero_plus,
)
from dreamwam.policy import build_policy


if _wand_library is None:
    raise RuntimeError("Failed to load the ImageMagick MagickWand runtime.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--suite",
        action="append",
        choices=LIBERO_PLUS_SUITES,
        help="Run selected suites only. Omit for the official four-suite protocol.",
    )
    parser.add_argument("--task-id", type=int, action="append")
    args = parser.parse_args()
    config = load_release_config(args.config)
    suite_names = tuple(args.suite or LIBERO_PLUS_SUITES)
    if len(set(suite_names)) != len(suite_names):
        parser.error("--suite cannot contain duplicates")
    if args.task_id is not None and len(suite_names) != 1:
        parser.error("--task-id requires exactly one --suite")
    if args.task_id is not None:
        if len(set(args.task_id)) != len(args.task_id):
            parser.error("--task-id cannot contain duplicates")
        if any(task_id < 0 for task_id in args.task_id):
            parser.error("--task-id cannot be negative")

    if args.task_id is None:
        suffix = "" if suite_names == LIBERO_PLUS_SUITES else "-" + "-".join(
            suite_names
        )
    elif len(args.task_id) <= 8:
        suffix = "-" + suite_names[0] + "-tasks-" + "-".join(
            str(task_id) for task_id in args.task_id
        )
    else:
        suffix = (
            f"-{suite_names[0]}-tasks-{args.task_id[0]}-{args.task_id[-1]}"
            f"-n{len(args.task_id)}"
        )
    output = config.paths.output_dir / "evaluation" / f"libero-plus{suffix}.json"
    if output.exists():
        raise FileExistsError(f"Evaluation output already exists: {output}")

    if not config.paths.libero_plus_root.is_dir():
        raise FileNotFoundError(config.paths.libero_plus_root)
    configure_libero_plus(
        repository_root=config.paths.libero_plus_root,
        config_root=config.paths.output_dir / "libero-plus-config",
        dataset_root=config.paths.libero_plus_data_root,
    )
    policy = build_policy(config, device=args.device)
    evaluation = config.evaluation
    result = evaluate_libero_plus(
        policy,
        suite_names=suite_names,
        trials_per_task=1,
        task_ids=None if args.task_id is None else tuple(args.task_id),
        seed=int(evaluation["seed"]),
        resolution=int(evaluation["resolution"]),
        replan_steps=int(evaluation["replan_steps"]),
        wait_steps=int(evaluation["wait_steps"]),
        repository_root=config.paths.libero_plus_root,
    )
    rendered = json.dumps(result, indent=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
