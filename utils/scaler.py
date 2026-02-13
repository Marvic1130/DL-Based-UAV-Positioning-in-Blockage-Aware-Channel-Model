from __future__ import annotations

import numpy as np
import torch


class MinMaxScaler:
    """A lightweight Min-Max scaler for both NumPy and torch.

    Goals for this repo:
    - Training loop can call inverse_transform on CUDA tensors without CPU round-trips.
    - Data prep can call transform on NumPy arrays.

    Notes:
    - If input is a CUDA tensor, return is a CUDA tensor.
    - If input is a CPU tensor, return is a NumPy array (convenient for sklearn utils).
    - If input is array-like/NumPy, return is a NumPy array.
    """

    def __init__(self, data_min: np.ndarray, data_max: np.ndarray):
        self.data_min = np.asarray(data_min, dtype=np.float32)
        self.data_max = np.asarray(data_max, dtype=np.float32)

        if self.data_min.shape != self.data_max.shape:
            raise ValueError(f"data_min shape {self.data_min.shape} != data_max shape {self.data_max.shape}")

        self._torch_cache: dict[tuple[str, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}

    def _torch_minmax(self, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        key = (str(device), dtype)
        cached = self._torch_cache.get(key)
        if cached is not None:
            return cached

        data_min_t = torch.as_tensor(self.data_min, device=device, dtype=dtype)
        data_max_t = torch.as_tensor(self.data_max, device=device, dtype=dtype)
        self._torch_cache[key] = (data_min_t, data_max_t)
        return data_min_t, data_max_t

    def transform(self, data):
        if isinstance(data, torch.Tensor):
            if data.device.type == 'cpu':
                return self.transform(data.detach().cpu().numpy())
            data_min_t, data_max_t = self._torch_minmax(data.device, data.dtype)
            return (data - data_min_t) / (data_max_t - data_min_t)

        data_np = np.asarray(data, dtype=np.float32)
        return (data_np - self.data_min) / (self.data_max - self.data_min)

    def inverse_transform(self, data):
        if isinstance(data, torch.Tensor):
            if data.device.type == 'cpu':
                return self.inverse_transform(data.detach().cpu().numpy())
            data_min_t, data_max_t = self._torch_minmax(data.device, data.dtype)
            return data * (data_max_t - data_min_t) + data_min_t

        data_np = np.asarray(data, dtype=np.float32)
        return data_np * (self.data_max - self.data_min) + self.data_min
