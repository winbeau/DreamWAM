import json
from pathlib import Path

import torch


class LiberoNormalizer:
    range_tolerance = 1.0e-4

    def __init__(self, stats_path: str | Path):
        path = Path(stats_path)
        if not path.is_file():
            raise FileNotFoundError(f"Missing LIBERO statistics: {path}")
        payload = json.loads(path.read_text())
        if set(payload) != {"state", "action"}:
            raise ValueError("LIBERO statistics must contain state and action exactly.")

        self.parameters = {}
        for name, expected_width in (("state", 8), ("action", 7)):
            stats = payload[name]
            if not isinstance(stats, dict) or set(stats) != {"min", "max"}:
                raise ValueError(f"{name} statistics must contain min and max exactly.")
            minimum = torch.tensor(stats["min"], dtype=torch.float32)
            maximum = torch.tensor(stats["max"], dtype=torch.float32)
            if minimum.shape != (expected_width,) or maximum.shape != (expected_width,):
                raise ValueError(
                    f"{name} statistics must have width {expected_width}."
                )
            input_range = maximum - minimum
            ignored = input_range < self.range_tolerance
            adjusted_range = input_range.clone()
            adjusted_range[ignored] = 2.0
            scale = 2.0 / adjusted_range
            offset = -1.0 - scale * minimum
            offset[ignored] = -minimum[ignored]
            self.parameters[name] = (scale, offset)

    def _forward(self, name: str, value: torch.Tensor) -> torch.Tensor:
        scale, offset = self.parameters[name]
        if value.shape[-1] != scale.numel():
            raise ValueError(
                f"{name} width must be {scale.numel()}, got {value.shape[-1]}."
            )
        scale = scale.to(device=value.device, dtype=value.dtype)
        offset = offset.to(device=value.device, dtype=value.dtype)
        return (value * scale + offset).clamp(-5.0, 5.0)

    def _backward(self, name: str, value: torch.Tensor) -> torch.Tensor:
        scale, offset = self.parameters[name]
        if value.shape[-1] != scale.numel():
            raise ValueError(
                f"{name} width must be {scale.numel()}, got {value.shape[-1]}."
            )
        scale = scale.to(device=value.device, dtype=value.dtype)
        offset = offset.to(device=value.device, dtype=value.dtype)
        return (value - offset) / scale

    def normalize_state(self, value: torch.Tensor) -> torch.Tensor:
        return self._forward("state", value)

    def normalize_action(self, value: torch.Tensor) -> torch.Tensor:
        return self._forward("action", value)

    def denormalize_action(self, value: torch.Tensor) -> torch.Tensor:
        return self._backward("action", value)
