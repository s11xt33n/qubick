"""QuantumLayer — обучаемый квантовый слой как обычный модуль PyTorch."""
from __future__ import annotations

import math

import torch
from torch import nn

from .backends import make_backend
from .circuits import build_circuit


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
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.circuit = build_circuit(n_qubits, n_layers, n_inputs, encoding, ansatz, reupload)
        self.backend = make_backend(backend, self.circuit, diff_method, shots)
        self.weights = nn.Parameter(torch.rand(self.circuit.n_weights) * 2 * math.pi)
        self.config = dict(n_qubits=n_qubits, n_layers=n_layers, encoding=encoding,
                           ansatz=ansatz, reupload=reupload, backend=backend,
                           diff_method=diff_method, shots=shots)

    @property
    def out_features(self) -> int:
        return self.n_qubits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angles = self.circuit.build_angles(x, self.weights)
        return self.backend(angles)

    def extra_repr(self) -> str:
        s = self.circuit.depth_summary()
        return ", ".join(f"{k}={v}" for k, v in {**self.config, **s}.items())
