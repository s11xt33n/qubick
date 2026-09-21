"""Модели для сравнения.

classical         — классический многослойный перцептрон (сильный baseline).
classical_matched — MLP с числом параметров, подобранным под гибридную модель.
bottleneck        — та же архитектура, что hybrid, но квантовый слой заменён
                    классическим слоем n_q -> n_q (абляция: «даёт ли что-то
                    именно квантовая часть?»).
quantum           — чистая квантовая сеть: признаки напрямую кодируются в
                    кубиты, классы получаются линейным считыванием <Z_k>.
hybrid            — классический encoder -> QuantumLayer -> классический head.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .layers import QuantumLayer


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ScaledTanh(nn.Module):
    """Отображает выход encoder'а в диапазон углов (-π, π)."""

    def forward(self, x):
        return math.pi * torch.tanh(x)


class ClassicalMLP(nn.Sequential):
    def __init__(self, in_dim: int, n_classes: int, hidden=(32, 16)):
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, n_classes))
        super().__init__(*layers)


class QuantumNet(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, n_qubits: int = 4, **qkw):
        super().__init__()
        self.q = QuantumLayer(n_qubits, n_inputs=in_dim, **qkw)
        self.readout = nn.Linear(n_qubits, n_classes)

    def forward(self, x):
        return self.readout(self.q(x))


class HybridNet(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, n_qubits: int = 4, **qkw):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(in_dim, n_qubits), ScaledTanh())
        self.q = QuantumLayer(n_qubits, **qkw)
        self.head = nn.Linear(n_qubits, n_classes)

    def forward(self, x):
        return self.head(self.q(self.encoder(x)))


class BottleneckNet(nn.Module):
    """Hybrid без квантовой части: encoder -> Linear(n_q, n_q)+tanh -> head."""

    def __init__(self, in_dim: int, n_classes: int, n_qubits: int = 4, **_):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(in_dim, n_qubits), ScaledTanh())
        self.mid = nn.Sequential(nn.Linear(n_qubits, n_qubits), nn.Tanh())
        self.head = nn.Linear(n_qubits, n_classes)

    def forward(self, x):
        return self.head(self.mid(self.encoder(x)))


def matched_hidden(in_dim: int, n_classes: int, target: int) -> int:
    """Ширина скрытого слоя MLP in->h->C с числом параметров ≈ target."""
    # (in+1)h + (h+1)C = target  =>  h = (target - C) / (in + 1 + C)
    return max(1, round((target - n_classes) / (in_dim + 1 + n_classes)))


def build_model(name: str, in_dim: int, n_classes: int, n_qubits: int = 4,
                n_layers: int = 2, hidden=(32, 16), **qkw) -> nn.Module:
    qkw = dict(n_layers=n_layers, **qkw)
    if name == "classical":
        return ClassicalMLP(in_dim, n_classes, hidden)
    if name == "quantum":
        return QuantumNet(in_dim, n_classes, n_qubits, **qkw)
    if name == "hybrid":
        return HybridNet(in_dim, n_classes, n_qubits, **qkw)
    if name == "bottleneck":
        return BottleneckNet(in_dim, n_classes, n_qubits)
    if name == "classical_matched":
        target = n_params(HybridNet(in_dim, n_classes, n_qubits, **qkw))
        return ClassicalMLP(in_dim, n_classes, (matched_hidden(in_dim, n_classes, target),))
    raise ValueError(f"Неизвестная модель: {name}")


MODEL_NAMES = ["classical", "classical_matched", "bottleneck", "quantum", "hybrid"]
