import json
from collections.abc import Iterator
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import torch
from torchvision.transforms import functional as transforms_F


SUITES = (
    "libero_spatial_no_noops_lerobot",
    "libero_object_no_noops_lerobot",
    "libero_goal_no_noops_lerobot",
    "libero_10_no_noops_lerobot",
)
PROMPT_TEMPLATE = (
    "A video recorded from a robot's point of view executing the following "
    "instruction: {task}"
)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _decode_video(path: Path) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(path)
    frames = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            frames.append(torch.from_numpy(frame.to_ndarray(format="rgb24")))
    if not frames:
        raise ValueError(f"Video contains no frames: {path}")
    return torch.stack(frames).permute(0, 3, 1, 2).contiguous()


def _resize(frames: torch.Tensor, size: int) -> torch.Tensor:
    return transforms_F.resize(
        frames.float().div(255.0),
        size=[size, size],
        interpolation=transforms_F.InterpolationMode.BILINEAR,
        antialias=True,
    )


def iter_libero_windows(
    dataset_root: str | Path,
    *,
    frame_span: int = 33,
    video_stride: int = 4,
    image_size: int = 224,
    sample_stride: int = 1,
) -> Iterator[dict]:
    root = Path(dataset_root)
    if frame_span != 33 or video_stride != 4:
        raise ValueError("The released LIBERO setup requires 33 steps and stride 4.")
    if image_size <= 0 or sample_stride <= 0:
        raise ValueError("image_size and sample_stride must be positive.")
    video_offsets = tuple(range(0, frame_span, video_stride))
    if len(video_offsets) != 9:
        raise RuntimeError("DreamWAM preprocessing requires exactly 9 RGB frames.")

    sample_index = 0
    for suite_name in SUITES:
        suite_root = root / suite_name
        info_path = suite_root / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"Missing official LIBERO suite: {info_path}")
        info = json.loads(info_path.read_text())
        tasks = {
            int(item["task_index"]): str(item["task"])
            for item in _read_jsonl(suite_root / "meta" / "tasks.jsonl")
        }
        total_episodes = int(info["total_episodes"])
        for episode_index in range(total_episodes):
            parquet_path = (
                suite_root
                / "data"
                / "chunk-000"
                / f"episode_{episode_index:06d}.parquet"
            )
            table = pq.read_table(
                parquet_path,
                columns=[
                    "action",
                    "observation.state",
                    "task_index",
                    "frame_index",
                    "episode_index",
                ],
            )
            actions = torch.tensor(
                np.asarray(table["action"].to_pylist()), dtype=torch.float32
            )
            states = torch.tensor(
                np.asarray(table["observation.state"].to_pylist()),
                dtype=torch.float32,
            )
            task_indices = table["task_index"].to_numpy()
            frame_indices = table["frame_index"].to_numpy()
            episode_indices = table["episode_index"].to_numpy()
            head = _decode_video(
                suite_root
                / "videos"
                / "chunk-000"
                / "observation.images.image"
                / f"episode_{episode_index:06d}.mp4"
            )
            wrist = _decode_video(
                suite_root
                / "videos"
                / "chunk-000"
                / "observation.images.wrist_image"
                / f"episode_{episode_index:06d}.mp4"
            )
            lengths = {
                "action": len(actions),
                "state": len(states),
                "task_index": len(task_indices),
                "frame_index": len(frame_indices),
                "episode_index": len(episode_indices),
                "head_video": len(head),
                "wrist_video": len(wrist),
            }
            if len(set(lengths.values())) != 1:
                raise ValueError(
                    f"Episode stream lengths differ in {parquet_path}: {lengths}"
                )
            frame_count = len(actions)
            if frame_count <= 0:
                raise ValueError(f"Episode is empty: {parquet_path}")
            expected_frames = np.arange(frame_count)
            if not np.array_equal(frame_indices, expected_frames):
                raise ValueError(f"Non-contiguous frame_index in {parquet_path}.")
            if not np.all(episode_indices == episode_index):
                raise ValueError(f"episode_index mismatch in {parquet_path}.")

            for start in range(0, frame_count, sample_stride):
                requested_frame_ids = [start + offset for offset in video_offsets]
                image_is_pad = torch.tensor(
                    [index >= frame_count for index in requested_frame_ids],
                    dtype=torch.bool,
                )
                frame_ids = [min(index, frame_count - 1) for index in requested_frame_ids]
                head_clip = _resize(head[frame_ids], image_size)
                wrist_clip = _resize(wrist[frame_ids], image_size)
                video = torch.cat([head_clip, wrist_clip], dim=3)
                video = video.permute(1, 0, 2, 3).sub(0.5).div(0.5)

                requested_action_ids = list(range(start, start + frame_span - 1))
                action_is_pad = torch.tensor(
                    [index >= frame_count for index in requested_action_ids],
                    dtype=torch.bool,
                )
                action_ids = [
                    min(index, frame_count - 1) for index in requested_action_ids
                ]
                action = actions[action_ids].clone()
                action[action_is_pad, :6] = 0.0
                proprio = states[action_ids].clone()

                task_index = int(task_indices[start])
                if task_index not in tasks:
                    raise KeyError(f"Unknown task_index={task_index} in {suite_name}.")
                task = tasks[task_index]
                yield {
                    "sample_index": sample_index,
                    "suite": suite_name,
                    "episode_index": episode_index,
                    "start": start,
                    "video": video.contiguous(),
                    "image_is_pad": image_is_pad,
                    "action": action.contiguous(),
                    "action_is_pad": action_is_pad,
                    "proprio": proprio.contiguous(),
                    "instruction": task,
                    "prompt": PROMPT_TEMPLATE.format(task=task),
                }
                sample_index += 1
