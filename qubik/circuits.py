"""Описание параметризованных квантовых схем.

Схема хранится как список вентилей `Gate`. Каждый параметризованный вентиль
берёт угол из одного источника:
    ("x", i) — i-й входной признак (кодирование данных),
    ("w", j) — j-й обучаемый вес.
Такое представление не зависит от бэкенда: одну и ту же схему исполняет
собственный симулятор на PyTorch, PennyLane или экспорт в OpenQASM.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

PARAM_GATES = {"RX", "RY", "RZ"}
FIXED_GATES = {"H", "CNOT", "CZ"}


@dataclass(frozen=True)
class Gate:
    name: str
    wires: tuple[int, ...]
    src: tuple[str, int] | None = None  # источник угла для RX/RY/RZ


@dataclass
class Circuit:
    n_qubits: int
    n_inputs: int
    gates: list[Gate] = field(default_factory=list)
    n_weights: int = 0

    # ---- построение -------------------------------------------------------
    def add(self, name: str, wires, src=None) -> None:
        self.gates.append(Gate(name, tuple(wires), src))

    def new_weight(self) -> tuple[str, int]:
        self.n_weights += 1
        return ("w", self.n_weights - 1)

    # ---- служебное --------------------------------------------------------
    @property
    def param_gates(self) -> list[int]:
        """Индексы вентилей, у которых есть угол."""
        return [k for k, g in enumerate(self.gates) if g.name in PARAM_GATES]

    def angle_index(self) -> torch.Tensor:
        """Для каждого параметризованного вентиля — номер столбца в [x | w]."""
        cols = []
        for k in self.param_gates:
            kind, i = self.gates[k].src
            cols.append(i if kind == "x" else self.n_inputs + i)
        return torch.tensor(cols, dtype=torch.long)

    def build_angles(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Углы всех параметризованных вентилей, форма (B, G).

        Операция дифференцируема, поэтому градиент по углам автоматически
        раскладывается на градиенты по входам и весам (правило цепочки).
        """
        batch = x.shape[0]
        full = torch.cat([x, w.unsqueeze(0).expand(batch, -1)], dim=1)
        return full[:, self.angle_index().to(x.device)]

    def depth_summary(self) -> dict:
        n_2q = sum(len(g.wires) == 2 for g in self.gates)
        return {
            "gates": len(self.gates),
            "param_gates": len(self.param_gates),
            "two_qubit_gates": n_2q,
            "weights": self.n_weights,
        }


# ---------------------------------------------------------------------------
# Кодирование данных
# ---------------------------------------------------------------------------
def _encode(c: Circuit, kind: str) -> None:
    n = c.n_qubits
    for q in range(n):
        i = q % c.n_inputs  # если признаков меньше, чем кубитов, — повторяем
        if kind == "angle":  # RY(x)
            c.add("RY", [q], ("x", i))
        elif kind == "angle_x":  # RX(x), как AngleEmbedding в PennyLane
            c.add("RX", [q], ("x", i))
        elif kind == "dense":  # H, затем RZ(x) и RY(x): больше нелинейности
            c.add("H", [q])
            c.add("RZ", [q], ("x", i))
            c.add("RY", [q], ("x", i))
        else:
            raise ValueError(f"Неизвестное кодирование: {kind}")


# ---------------------------------------------------------------------------
# Вариационные слои (анзацы)
# ---------------------------------------------------------------------------
def _entangle_ring(c: Circuit, gate: str, r: int = 1) -> None:
    n = c.n_qubits
    if n == 1:
        return
    if n == 2:
        c.add(gate, [0, 1])
        return
    for q in range(n):
        c.add(gate, [q, (q + r) % n])


def _layer(c: Circuit, ansatz: str, layer_idx: int) -> None:
    n = c.n_qubits
    if ansatz == "strong":
        # аналог StronglyEntanglingLayers: Rot = RZ·RY·RZ и CNOT-кольцо
        for q in range(n):
            c.add("RZ", [q], c.new_weight())
            c.add("RY", [q], c.new_weight())
            c.add("RZ", [q], c.new_weight())
        r = (layer_idx % (n - 1)) + 1 if n > 1 else 1
        _entangle_ring(c, "CNOT", r)
    elif ansatz == "basic":
        for q in range(n):
            c.add("RY", [q], c.new_weight())
        _entangle_ring(c, "CNOT")
    elif ansatz == "hea":  # hardware-efficient: RY, RZ и цепочка CZ
        for q in range(n):
            c.add("RY", [q], c.new_weight())
            c.add("RZ", [q], c.new_weight())
        for q in range(n - 1):
            c.add("CZ", [q, q + 1])
    else:
        raise ValueError(f"Неизвестный анзац: {ansatz}")


def build_circuit(
    n_qubits: int,
    n_layers: int,
    n_inputs: int | None = None,
    encoding: str = "angle",
    ansatz: str = "strong",
    reupload: bool = False,
) -> Circuit:
    """Собирает схему: кодирование -> L вариационных слоёв.

    При reupload=True данные кодируются заново перед каждым слоем
    (data re-uploading, Pérez-Salinas et al., 2020): это повышает
    выразительность схемы без увеличения числа кубитов.
    """
    c = Circuit(n_qubits=n_qubits, n_inputs=n_inputs or n_qubits)
    for layer in range(n_layers):
        if layer == 0 or reupload:
            _encode(c, encoding)
        _layer(c, ansatz, layer)
    return c


def to_qasm(c: Circuit, x: torch.Tensor, w: torch.Tensor) -> str:
    """Экспорт схемы для одного объекта в OpenQASM 2.0.

    Нужен для запуска на реальном квантовом железе (IBM Quantum и др.)
    и для визуальной проверки схемы.
    """
    angles = c.build_angles(x.reshape(1, -1), w)[0].tolist()
    lines = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{c.n_qubits}];",
             f"creg c[{c.n_qubits}];"]
    k = 0
    for g in c.gates:
        if g.name in PARAM_GATES:
            lines.append(f"{g.name.lower()}({angles[k]:.8f}) q[{g.wires[0]}];")
            k += 1
        elif g.name == "H":
            lines.append(f"h q[{g.wires[0]}];")
        elif g.name == "CNOT":
            lines.append(f"cx q[{g.wires[0]}],q[{g.wires[1]}];")
        elif g.name == "CZ":
            lines.append(f"cz q[{g.wires[0]}],q[{g.wires[1]}];")
    lines.append("measure q -> c;")
    return "\n".join(lines)
