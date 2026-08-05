from collections.abc import Iterable

import torch
from accelerate import Accelerator
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from .model import DreamWAMJoint


class DreamWAMTrainingWrapper(nn.Module):
    def __init__(self, model: DreamWAMJoint):
        super().__init__()
        self.model = model

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        progress: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.model.training_loss(batch, progress=progress)


def build_learning_rate_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    num_steps: int,
    warmup_ratio: float,
):
    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    if not 0.0 <= warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must be in [0,1).")
    warmup_steps = min(int(num_steps * warmup_ratio), num_steps - 1)
    remaining_steps = max(num_steps - warmup_steps, 1)
    learning_rate = float(optimizer.param_groups[0]["lr"])
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=remaining_steps,
        eta_min=learning_rate * 0.01,
    )
    if warmup_steps == 0:
        return cosine
    warmup = LinearLR(
        optimizer,
        start_factor=1.0 / warmup_steps,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )


def _move_batch(
    batch: dict,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    return {
        key: value.to(
            device=device,
            dtype=dtype if value.is_floating_point() else value.dtype,
            non_blocking=True,
        )
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def train_steps(
    model: nn.Module,
    dataloader: Iterable[dict],
    optimizer: torch.optim.Optimizer,
    accelerator: Accelerator,
    *,
    num_steps: int,
    max_grad_norm: float = 1.0,
    learning_rate_scheduler=None,
) -> dict[str, float]:
    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    if max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive.")
    try:
        first_parameter = next(model.parameters())
    except StopIteration as error:
        raise ValueError("model has no trainable parameters.") from error
    device = first_parameter.device
    dtype = first_parameter.dtype

    model.train()
    optimizer.zero_grad(set_to_none=True)
    iterator = iter(dataloader)
    last_metrics: dict[str, float] = {}
    for step in range(num_steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            try:
                batch = next(iterator)
            except StopIteration as error:
                raise ValueError("dataloader is empty.") from error
        if not isinstance(batch, dict):
            raise TypeError(f"dataloader must yield dict batches, got {type(batch)}.")

        batch = _move_batch(batch, device, dtype)
        with accelerator.accumulate(model):
            progress = step / num_steps
            loss, components = model(batch, progress=progress)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite training loss: {loss.detach().item()}"
                )
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            if learning_rate_scheduler is not None and accelerator.sync_gradients:
                learning_rate_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        reduced_loss = accelerator.gather(loss.detach().reshape(1)).float().mean()
        reduced_components = {
            name: accelerator.gather(value.detach().reshape(1)).float().mean()
            for name, value in components.items()
        }
        last_metrics = {
            "loss": float(reduced_loss),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **{name: float(value) for name, value in reduced_components.items()},
        }
    return last_metrics
