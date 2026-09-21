"""Собственный батчевый симулятор вектора состояния на PyTorch.

Состояние n кубитов — тензор формы (B, 2, ..., 2) из 2**n комплексных
амплитуд для каждого объекта батча. Ось 1 соответствует кубиту 0
(старший бит), как в PennyLane и в учебнике Нильсена–Чанга.

Все операции — обычные дифференцируемые операции PyTorch, поэтому
градиенты по углам вентилей считаются штатным autograd (режим backprop),
а схемы для разных объектов батча исполняются одновременно, в том числе
на GPU.
"""
from __future__ import annotations

import math
from functools import lru_cache

import torch

from .circuits import Circuit, PARAM_GATES

CDTYPE = torch.complex64


# ---------------------------------------------------------------------------
# Матрицы вентилей. theta: (B,) -> (B, 2, 2)
# ---------------------------------------------------------------------------
def rx(theta: torch.Tensor) -> torch.Tensor:
    c = torch.cos(theta / 2).to(CDTYPE)
    s = torch.sin(theta / 2).to(CDTYPE)
    return torch.stack([torch.stack([c, -1j * s], -1), torch.stack([-1j * s, c], -1)], -2)


def ry(theta: torch.Tensor) -> torch.Tensor:
    c = torch.cos(theta / 2).to(CDTYPE)
    s = torch.sin(theta / 2).to(CDTYPE)
    return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)


def rz(theta: torch.Tensor) -> torch.Tensor:
    e = torch.exp(-0.5j * theta.to(CDTYPE))
    z = torch.zeros_like(e)
    return torch.stack([torch.stack([e, z], -1), torch.stack([z, e.conj()], -1)], -2)


GATE_FN = {"RX": rx, "RY": ry, "RZ": rz}
H_MAT = torch.tensor([[1, 1], [1, -1]], dtype=CDTYPE) / math.sqrt(2)


# ---------------------------------------------------------------------------
# Предвычисленные перестановки и знаки для двухкубитных вентилей и измерений
# ---------------------------------------------------------------------------
def _bit(idx: torch.Tensor, q: int, n: int) -> torch.Tensor:
    return (idx >> (n - 1 - q)) & 1


@lru_cache(maxsize=None)
def _cnot_perm(n: int, c: int, t: int) -> torch.Tensor:
    idx = torch.arange(2**n)
    flip = _bit(idx, c, n) << (n - 1 - t)
    return idx ^ flip


@lru_cache(maxsize=None)
def _cz_sign(n: int, a: int, b: int) -> torch.Tensor:
    idx = torch.arange(2**n)
    return 1 - 2 * (_bit(idx, a, n) & _bit(idx, b, n))


@lru_cache(maxsize=None)
def z_signs(n: int) -> torch.Tensor:
    """Матрица (2**n, n): собственные значения Z_k (+1/-1) для базисных состояний."""
    idx = torch.arange(2**n).unsqueeze(1)
    q = torch.arange(n).unsqueeze(0)
    return (1 - 2 * ((idx >> (n - 1 - q)) & 1)).float()


# ---------------------------------------------------------------------------
# Применение вентилей
# ---------------------------------------------------------------------------
def apply_1q(state: torch.Tensor, mat: torch.Tensor, q: int) -> torch.Tensor:
    """state: (B, 2, ..., 2); mat: (B, 2, 2) или (2, 2)."""
    psi = state.movedim(q + 1, -1)
    if mat.dim() == 2:
        psi = psi @ mat.T
    else:
        shape = psi.shape
        psi = psi.reshape(shape[0], -1, 2) @ mat.transpose(-1, -2)
        psi = psi.reshape(shape)
    return psi.movedim(-1, q + 1)


def apply_perm(state: torch.Tensor, perm: torch.Tensor, n: int) -> torch.Tensor:
    flat = state.reshape(state.shape[0], -1)
    return flat[:, perm.to(state.device)].reshape(state.shape)


def apply_sign(state: torch.Tensor, sign: torch.Tensor, n: int) -> torch.Tensor:
    flat = state.reshape(state.shape[0], -1)
    return (flat * sign.to(state.device)).reshape(state.shape)


def run_circuit(circuit: Circuit, angles: torch.Tensor) -> torch.Tensor:
    """Исполняет схему для батча углов (B, G), возвращает состояние (B, 2**n)."""
    n = circuit.n_qubits
    batch = angles.shape[0]
    state = torch.zeros(batch, 2**n, dtype=CDTYPE, device=angles.device)
    state[:, 0] = 1.0
    state = state.reshape(batch, *([2] * n))
    k = 0
    for g in circuit.gates:
        if g.name in PARAM_GATES:
            state = apply_1q(state, GATE_FN[g.name](angles[:, k]), g.wires[0])
            k += 1
        elif g.name == "H":
            state = apply_1q(state, H_MAT.to(angles.device), g.wires[0])
        elif g.name == "CNOT":
            state = apply_perm(state, _cnot_perm(n, *g.wires), n)
        elif g.name == "CZ":
            state = apply_sign(state, _cz_sign(n, *g.wires), n)
        else:
            raise ValueError(g.name)
    return state.reshape(batch, -1)


def probabilities(state: torch.Tensor) -> torch.Tensor:
    return state.real**2 + state.imag**2


def expval_z(circuit: Circuit, angles: torch.Tensor) -> torch.Tensor:
    """<Z_k> для каждого кубита, форма (B, n)."""
    p = probabilities(run_circuit(circuit, angles))
    return p @ z_signs(circuit.n_qubits).to(p.device)


def expval_zz(circuit: Circuit, angles: torch.Tensor, a: int = 0, b: int = 1) -> torch.Tensor:
    """<Z_a Z_b>, форма (B,). Используется в эксперименте с barren plateaus."""
    p = probabilities(run_circuit(circuit, angles))
    s = z_signs(circuit.n_qubits).to(p.device)
    return p @ (s[:, a] * s[:, b])


def sample_expval(ev: torch.Tensor, shots: int, generator=None) -> torch.Tensor:
    """Оценка <Z> по конечному числу измерений (shots), как на реальном устройстве.

    Для каждого кубита число исходов «0» ~ Binomial(shots, (1+<Z>)/2).
    """
    p0 = ((1 + ev) / 2).clamp(0, 1)
    n0 = torch.binomial(torch.full_like(p0, float(shots)), p0, generator=generator)
    return 2 * n0 / shots - 1
