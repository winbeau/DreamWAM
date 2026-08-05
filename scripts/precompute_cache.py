#!/usr/bin/env python3
import argparse
import json
from itertools import islice
from pathlib import Path

import torch

from dreamwam.components import build_precompute_components
from dreamwam.config import load_release_config
from dreamwam.preprocessing import iter_libero_windows, load_rank8_projection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int)
    return parser.parse_args()


def batched(iterator, batch_size: int):
    if batch_size <= 0:
        raise ValueError("precompute_batch_size must be positive.")
    while batch := list(islice(iterator, batch_size)):
        yield batch


def main() -> None:
    args = parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("max-samples must be positive when provided.")

    config = load_release_config(args.config)
    components = build_precompute_components(config, device=args.device)
    preprocessor = components.preprocessor
    encode_text = components.encode_text
    cache_root = config.paths.cache_root
    index_path = cache_root / "index.jsonl"
    if index_path.exists():
        raise FileExistsError(
            f"Cache index already exists; choose a new cache_root: {index_path}"
        )
    cache_root.mkdir(parents=True, exist_ok=True)
    samples_root = cache_root / "samples"
    samples_root.mkdir(exist_ok=True)

    iterator_kwargs = {
        "dataset_root": config.paths.dataset_root,
        "frame_span": int(config.preprocessing["frame_span"]),
        "video_stride": int(config.preprocessing["video_stride"]),
        "image_size": int(config.preprocessing["image_size"]),
        "sample_stride": int(config.preprocessing["sample_stride"]),
    }
    projections = {
        "dino": load_rank8_projection(
            config.preprocessing["dino_projection"],
            input_dim=768,
        ),
        "depth": load_rank8_projection(
            config.preprocessing["depth_projection"],
            input_dim=256,
        ),
    }
    torch.save(projections, cache_root / "projections.pt")

    context_ids = {}
    contexts = []
    batch_size = int(config.preprocessing.get("precompute_batch_size", 16))
    with index_path.open("x") as index_file:
        windows = iter_libero_windows(**iterator_kwargs)
        if args.max_samples is not None:
            windows = islice(windows, args.max_samples)
        for items in batched(windows, batch_size):
            unknown_prompts = list(
                dict.fromkeys(
                    item["prompt"]
                    for item in items
                    if item["prompt"] not in context_ids
                )
            )
            if unknown_prompts:
                encoded_contexts, encoded_masks = encode_text(unknown_prompts)
                if encoded_contexts.shape[0] != len(unknown_prompts):
                    raise ValueError("Text encoder returned the wrong batch size.")
                for prompt, context, context_mask in zip(
                    unknown_prompts,
                    encoded_contexts,
                    encoded_masks,
                    strict=True,
                ):
                    context_id = len(contexts)
                    context_ids[prompt] = context_id
                    contexts.append(
                        {
                            "context": context.detach().cpu().contiguous(),
                            "context_mask": context_mask
                            .detach()
                            .bool()
                            .cpu()
                            .contiguous(),
                        }
                    )
            item_context_ids = [context_ids[item["prompt"]] for item in items]
            context = torch.stack(
                [contexts[index]["context"] for index in item_context_ids]
            )
            context_mask = torch.stack(
                [contexts[index]["context_mask"] for index in item_context_ids]
            )
            batch = preprocessor.preprocess_batch(
                videos=torch.stack([item["video"] for item in items]),
                action=torch.stack([item["action"] for item in items]),
                proprio=torch.stack([item["proprio"] for item in items]),
                context=context,
                context_mask=context_mask,
                projections=projections,
                action_is_pad=torch.stack(
                    [item["action_is_pad"] for item in items]
                ),
                image_is_pad=torch.stack(
                    [item["image_is_pad"] for item in items]
                ),
            )
            for position, (item, context_id) in enumerate(
                zip(items, item_context_ids, strict=True)
            ):
                sample = {
                    key: value[position].contiguous()
                    for key, value in batch.items()
                    if key not in {"context", "context_mask"}
                }
                relative = Path("samples") / f"{item['sample_index']:09d}.pt"
                torch.save(sample, cache_root / relative)
                entry = {
                    "path": relative.as_posix(),
                    "suite": item["suite"],
                    "episode_index": item["episode_index"],
                    "start": item["start"],
                    "instruction": item["instruction"],
                    "context_id": context_id,
                }
                index_file.write(json.dumps(entry, ensure_ascii=True) + "\n")
                index_file.flush()
    torch.save(contexts, cache_root / "contexts.pt")


if __name__ == "__main__":
    main()
