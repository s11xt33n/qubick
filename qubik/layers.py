"""QuantumLayer — обучаемый квантовый слой как обычный модуль PyTorch."""
from __future__ import annotations

import math

import torch
from torch import nn

from .backends import make_backend
from .circuits import build_circuit


def init_weights(n: int, init: str = "uniform", scale: float = 0.1, generator=None) -> torch.Tensor:
    if init == "uniform":
        return torch.rand(n, generator=generator) * 2 * math.pi
    if init == "small":
        return torch.randn(n, generator=generator) * scale
    if init == "zero":
        return torch.zeros(n)
    raise ValueError(f"Неизвестная инициализация: {init}")


class QuantumLayer(nn.Module):
    """Параметризованная квантовая схема: R^in -> R^n_qubits (значения <Z_k>).

    Args:
        n_qubits:    число кубитов.
        n_layers:    число вариационных слоёв (глубина схемы).
        n_inputs:    размерность входа (по умолчанию = n_qubits).
        encoding:    "angle" | "angle_x" | "dense".
        ansatz:      "strong" | "basic" | "hea".
        reupload:    повторное кодирование данных перед каждым слоем.
        backend:     "torch" (собственный симулятор) | "pennylane".
        diff_method: "backprop" | "parameter-shift".
        shots:       None — точные ожидания; число — оценка по измерениям.
        init:        инициализация весов: "uniform" — U[0, 2π) (стандарт),
                     "small" — N(0, init_scale²) (схема близка к тождественной,
                     смягчает barren plateaus, Grant et al., 2019), "zero".
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int = 2,
        n_inputs: int | None = None,
        encoding: str = "angle",
        ansatz: str = "strong",
        reupload: bool = False,
        backend: str = "torch",
        diff_method: str = "backprop",
        shots: int | None = None,
        init: str = "uniform",
        init_scale: float = 0.1,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.circuit = build_circuit(n_qubits, n_layers, n_inputs, encoding, ansatz, reupload)
        self.backend = make_backend(backend, self.circuit, diff_method, shots)
        self.weights = nn.Parameter(init_weights(self.circuit.n_weights, init, init_scale))
        self.config = dict(n_qubits=n_qubits, n_layers=n_layers, encoding=encoding,
                           ansatz=ansatz, reupload=reupload, backend=backend,
                           diff_method=diff_method, shots=shots, init=init)

    @property
    def out_features(self) -> int:
        return self.n_qubits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angles = self.circuit.build_angles(x, self.weights)
        return self.backend(angles)

    def extra_repr(self) -> str:
        s = self.circuit.depth_summary()
        return ", ".join(f"{k}={v}" for k, v in {**self.config, **s}.items())
