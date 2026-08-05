#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader

from dreamwam.config import load_release_config
from dreamwam.data import CachedDreamWAMDataset
from dreamwam.initialization import (
    initialize_from_fastwam_pretrained,
    validate_pretrained_initialization_files,
)
from dreamwam.runtime import build_model, load_model_checkpoint
from dreamwam.train import (
    DreamWAMTrainingWrapper,
    build_learning_rate_scheduler,
    train_steps,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int)
    parser.add_argument(
        "--initial-checkpoint",
        help="Optional full DreamWAM checkpoint for continuation training.",
    )
    args = parser.parse_args()

    config = load_release_config(args.config)
    steps = args.steps or int(config.training["steps"])
    if steps <= 0:
        raise ValueError("Training steps must be positive.")
    initial_checkpoint = (
        None if args.initial_checkpoint is None else Path(args.initial_checkpoint)
    )
    if initial_checkpoint is None:
        validate_pretrained_initialization_files(config.initialization)
    elif not initial_checkpoint.is_file():
        raise FileNotFoundError(
            f"Missing initial DreamWAM checkpoint: {initial_checkpoint}"
        )
    accelerator = Accelerator(
        gradient_accumulation_steps=int(
            config.training["gradient_accumulation_steps"]
        ),
        mixed_precision="bf16",
        step_scheduler_with_optimizer=False,
    )
    # FastWAM creates every rank from the same seed before pretrained loading.
    set_seed(int(config.training["seed"]), device_specific=False)
    requested_device = torch.device(args.device)
    if requested_device.type != accelerator.device.type:
        raise ValueError(
            f"Requested {requested_device}, but Accelerate selected {accelerator.device}."
        )
    device = accelerator.device
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    core_model = build_model(config, device=device, dtype=dtype)
    if initial_checkpoint is not None:
        load_model_checkpoint(core_model, initial_checkpoint, strict=True)
        if accelerator.is_main_process:
            print(f"Initialized from full DreamWAM checkpoint: {initial_checkpoint}")
    else:
        report = initialize_from_fastwam_pretrained(
            core_model,
            config.initialization,
        )
        if accelerator.is_main_process:
            print(
                "Initialized with FastWAM's pretrained component policy: "
                f"Wan VideoDiT tensors={report.video_tensors}, "
                f"ActionDiT backbone tensors={report.action_backbone_tensors}, "
                f"Flow channel scale={report.flow_channel_init_scale}."
            )
    model = DreamWAMTrainingWrapper(core_model)
    dataset = CachedDreamWAMDataset(
        config.paths.cache_root,
        model_config=config.model,
        action_horizon=int(config.evaluation["action_horizon"]),
        video_frames=int(config.evaluation["video_frames"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config.training["batch_size"]),
        shuffle=True,
        num_workers=int(config.training["num_workers"]),
        pin_memory=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.training["learning_rate"]),
        weight_decay=float(config.training["weight_decay"]),
        betas=(0.9, 0.95),
    )
    scheduler = build_learning_rate_scheduler(
        optimizer,
        num_steps=steps,
        warmup_ratio=float(config.training["warmup_ratio"]),
    )
    model, optimizer, loader, scheduler = accelerator.prepare(
        model,
        optimizer,
        loader,
        scheduler,
    )
    metrics = train_steps(
        model,
        loader,
        optimizer,
        accelerator,
        num_steps=steps,
        max_grad_norm=float(config.training["max_grad_norm"]),
        learning_rate_scheduler=scheduler,
    )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        config.paths.output_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        torch.save(
            {"model": unwrapped.model.state_dict(), "metrics": metrics},
            config.paths.output_dir / "final.pt",
        )
        print(metrics)


if __name__ == "__main__":
    main()
