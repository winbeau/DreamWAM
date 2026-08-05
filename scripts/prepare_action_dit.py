#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch

from dreamwam.config import load_release_config
from dreamwam.initialization import build_action_dit_backbone_payload


def _dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare FastWAM-style ActionDiT weights from Wan2.2 VideoDiT."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    args = parser.parse_args()

    config = load_release_config(args.config)
    output = config.initialization.action_dit_checkpoint
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite ActionDiT checkpoint: {output}")
    payload, report = build_action_dit_backbone_payload(
        config.model,
        config.initialization.wan_dit_root,
        device=torch.device(args.device),
        dtype=_dtype(args.dtype),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(
        f"Saved {output} (copied={report.copied_tensors}, "
        f"interpolated={report.interpolated_tensors})."
    )


if __name__ == "__main__":
    main()
