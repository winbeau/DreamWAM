import torch


class ContinuousFlowMatchScheduler:
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 5.0,
        eps: float = 1.0e-10,
    ):
        if num_train_timesteps <= 0:
            raise ValueError("num_train_timesteps must be positive.")
        if shift <= 0:
            raise ValueError("shift must be positive.")
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.eps = float(eps)
        self._weight_min, self._weight_mean = self._training_weight_stats()

    @staticmethod
    def _shift_time(value: torch.Tensor, shift: float) -> torch.Tensor:
        return shift * value / (1.0 + (shift - 1.0) * value)

    def _training_weight_stats(self) -> tuple[float, float]:
        steps = self.num_train_timesteps
        grid = torch.linspace(
            1.0,
            0.0,
            steps + 1,
            dtype=torch.float64,
            device="cpu",
        )[:-1]
        timestep = self._shift_time(grid, self.shift) * float(steps)
        weight = torch.exp(-2.0 * ((timestep - steps / 2.0) / steps) ** 2)
        weight_min = float(weight.min())
        return weight_min, float((weight - weight_min).mean())

    def sample_training_t(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        progress = torch.rand(batch_size, device=device, dtype=torch.float32)
        timestep = self._shift_time(progress, self.shift)
        return (timestep * self.num_train_timesteps).to(dtype=dtype)

    def training_weight(self, timestep: torch.Tensor) -> torch.Tensor:
        timestep = timestep.float()
        steps = float(self.num_train_timesteps)
        weight = torch.exp(-2.0 * ((timestep - steps / 2.0) / steps) ** 2)
        return (weight - self._weight_min) / (self._weight_mean + self.eps)

    def add_noise(
        self,
        clean: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        if clean.shape != noise.shape:
            raise ValueError(
                f"clean/noise shape mismatch: {tuple(clean.shape)} vs {tuple(noise.shape)}"
            )
        sigma = (timestep / self.num_train_timesteps).to(
            device=clean.device,
            dtype=clean.dtype,
        )
        sigma = sigma.view(-1, *([1] * (clean.ndim - 1)))
        return (1.0 - sigma) * clean + sigma * noise

    @staticmethod
    def training_target(clean: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return noise - clean

    def inference_schedule(
        self,
        num_steps: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if num_steps <= 0:
            raise ValueError("num_steps must be positive.")
        progress = torch.linspace(
            1.0,
            0.0,
            num_steps + 1,
            device=device,
            dtype=torch.float32,
        )
        sigma = self._shift_time(progress, self.shift)
        timesteps = sigma[:-1] * self.num_train_timesteps
        deltas = sigma[1:] - sigma[:-1]
        return timesteps.to(dtype=dtype), deltas.to(dtype=dtype)

    @staticmethod
    def step(
        prediction: torch.Tensor,
        delta: torch.Tensor,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        delta = delta.to(device=sample.device, dtype=sample.dtype)
        return sample + prediction * delta
