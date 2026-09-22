"""Бэкенды исполнения схем и способы вычисления градиентов.

Все бэкенды имеют одинаковый интерфейс: `backend(angles) -> <Z_k>` с формой
(B, n_qubits), где angles — углы вентилей формы (B, G).

- TorchBackend      — собственный симулятор (qubik.simulator);
                      градиенты: backprop или parameter-shift.
- PennyLaneBackend  — эталон PennyLane default.qubit, для проверки
                      корректности и сравнения скорости.
"""
from __future__ import annotations

import math

import torch

from . import simulator as sim
from .circuits import Circuit


class _ParameterShift(torch.autograd.Function):
    """Градиент по правилу сдвига параметра (parameter-shift rule).

    Для вентилей вида exp(-i θ P / 2), где P — оператор Паули:
        d<O>/dθ = ( <O>(θ + π/2) - <O>(θ - π/2) ) / 2.
    Правило даёт точный градиент и применимо на реальном квантовом
    устройстве, где backprop через амплитуды невозможен.
    """

    @staticmethod
    def forward(ctx, angles, circuit, shots):
        ctx.circuit, ctx.shots = circuit, shots
        ctx.save_for_backward(angles)
        with torch.no_grad():
            ev = sim.expval_z(circuit, angles)
            if shots:
                ev = sim.sample_expval(ev, shots)
        return ev

    @staticmethod
    def backward(ctx, grad_out):
        (angles,) = ctx.saved_tensors
        circuit, shots = ctx.circuit, ctx.shots
        b, g = angles.shape
        shift = (math.pi / 2) * torch.eye(g, device=angles.device, dtype=angles.dtype)
        with torch.no_grad():
            shifted = torch.cat(
                [angles[:, None, :] + shift, angles[:, None, :] - shift], dim=1
            ).reshape(b * 2 * g, g)
            ev = sim.expval_z(circuit, shifted)
            if shots:
                ev = sim.sample_expval(ev, shots)
            ev = ev.reshape(b, 2, g, -1)
            d_ev = (ev[:, 0] - ev[:, 1]) / 2  # (B, G, n)
            grad_angles = torch.einsum("bgn,bn->bg", d_ev, grad_out)
        return grad_angles, None, None


class TorchBackend:
    name = "torch"

    def __init__(self, circuit: Circuit, diff_method: str = "backprop", shots: int | None = None):
        if diff_method not in ("backprop", "parameter-shift"):
            raise ValueError(diff_method)
        if shots and diff_method == "backprop":
            raise ValueError("При конечном числе shots используйте parameter-shift")
        self.circuit, self.diff_method, self.shots = circuit, diff_method, shots

    def __call__(self, angles: torch.Tensor) -> torch.Tensor:
        if self.diff_method == "parameter-shift":
            return _ParameterShift.apply(angles, self.circuit, self.shots)
        return sim.expval_z(self.circuit, angles)


class PennyLaneBackend:
    name = "pennylane"

    def __init__(self, circuit: Circuit, diff_method: str = "backprop", shots: int | None = None):
        import pennylane as qml

        self.circuit = circuit
        dev = qml.device("default.qubit", wires=circuit.n_qubits)
        gates = {"RX": qml.RX, "RY": qml.RY, "RZ": qml.RZ}

        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def qnode(angles):
            k = 0
            for g in circuit.gates:
                if g.name in gates:
                    gates[g.name](angles[..., k], wires=g.wires[0])
                    k += 1
                elif g.name == "H":
                    qml.Hadamard(wires=g.wires[0])
                elif g.name == "CNOT":
                    qml.CNOT(wires=list(g.wires))
                elif g.name == "CZ":
                    qml.CZ(wires=list(g.wires))
            return [qml.expval(qml.PauliZ(q)) for q in range(circuit.n_qubits)]

        self.qnode = qml.set_shots(qnode, shots) if shots else qnode
        # parameter-shift в PennyLane не поддерживает батч по обучаемым углам,
        # поэтому объекты батча исполняются по одному
        self.per_sample = diff_method != "backprop"

    def __call__(self, angles: torch.Tensor) -> torch.Tensor:
        if self.per_sample:
            return torch.stack([self._run(a) for a in angles]).to(angles.dtype)
        return self._run(angles).to(angles.dtype)

    def _run(self, angles):
        return torch.stack(list(self.qnode(angles)), dim=-1)


BACKENDS = {"torch": TorchBackend, "pennylane": PennyLaneBackend}


def make_backend(name: str, circuit: Circuit, diff_method: str = "backprop", shots=None):
    return BACKENDS[name](circuit, diff_method=diff_method, shots=shots)
