from .rollout import ActionPolicy, evaluate_benchmark


def evaluate_libero(
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
    return evaluate_benchmark(
        policy,
        suite_name=suite_name,
        trials_per_task=trials_per_task,
        task_ids=task_ids,
        seed=seed,
        resolution=resolution,
        replan_steps=replan_steps,
        wait_steps=wait_steps,
    )
