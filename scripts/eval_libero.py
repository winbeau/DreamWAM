#!/usr/bin/env python3
import argparse
import json

from dreamwam.config import load_release_config
from dreamwam.evaluation import evaluate_libero
from dreamwam.policy import build_policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--suite", required=True)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--task-id", type=int, action="append")
    args = parser.parse_args()
    config = load_release_config(args.config)
    policy = build_policy(config, device=args.device)
    evaluation = config.evaluation
    result = evaluate_libero(
        policy,
        suite_name=args.suite,
        trials_per_task=args.trials,
        task_ids=None if args.task_id is None else tuple(args.task_id),
        seed=int(evaluation["seed"]),
        resolution=int(evaluation["resolution"]),
        replan_steps=int(evaluation["replan_steps"]),
        wait_steps=int(evaluation["wait_steps"]),
    )
    rendered = json.dumps(result, indent=2)
    output = config.paths.output_dir / "evaluation" / f"{args.suite}.json"
    if output.exists():
        raise FileExistsError(f"Evaluation output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
