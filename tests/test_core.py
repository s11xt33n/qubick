"""Проверка корректности: собственный симулятор против PennyLane,
parameter-shift против backprop, базовые свойства схем и моделей."""
import itertools
import math

import pytest
import torch

from qhnn import QuantumLayer, build_circuit, build_model, n_params, to_qasm
from qhnn import simulator as sim
from qhnn.backends import PennyLaneBackend, TorchBackend

torch.manual_seed(0)
CONFIGS = list(itertools.product([1, 2, 3, 5], [1, 3], ["angle", "angle_x", "dense"],
                                 ["strong", "basic", "hea"], [False, True]))


@pytest.mark.parametrize("n,L,enc,ans,re", CONFIGS)
def test_matches_pennylane(n, L, enc, ans, re):
    c = build_circuit(n, L, n_inputs=3, encoding=enc, ansatz=ans, reupload=re)
    x, w = torch.randn(4, 3), torch.rand(c.n_weights) * 2 * math.pi
    angles = c.build_angles(x, w).double()
    ours = TorchBackend(c)(angles.float())
    ref = PennyLaneBackend(c)(angles)
    assert torch.allclose(ours.double(), ref, atol=1e-5)


def test_state_is_normalized():
    c = build_circuit(6, 4, encoding="dense")
    angles = torch.randn(8, len(c.param_gates))
    p = sim.probabilities(sim.run_circuit(c, angles))
    assert torch.allclose(p.sum(1), torch.ones(8), atol=1e-5)


@pytest.mark.parametrize("n,L", [(2, 1), (4, 2), (5, 3)])
def test_parameter_shift_equals_backprop(n, L):
    c = build_circuit(n, L, reupload=True)
    x, w = torch.randn(3, n), torch.rand(c.n_weights) * 6
    grads = []
    for method in ("backprop", "parameter-shift"):
        xi, wi = x.clone().requires_grad_(), w.clone().requires_grad_()
        out = TorchBackend(c, diff_method=method)(c.build_angles(xi, wi))
        (out * torch.arange(1, n + 1)).sum().backward()
        grads.append((xi.grad, wi.grad))
    assert torch.allclose(grads[0][0], grads[1][0], atol=1e-4)
    assert torch.allclose(grads[0][1], grads[1][1], atol=1e-4)


def test_shots_estimate_is_close():
    c = build_circuit(3, 2)
    angles = torch.randn(2, len(c.param_gates))
    exact = TorchBackend(c)(angles)
    noisy = TorchBackend(c, "parameter-shift", shots=200_000)(angles)
    assert torch.allclose(exact, noisy, atol=0.02)


def test_known_states():
    # RY(π) переводит |0> в |1>: <Z> = -1; RY(π/2): <Z> = 0
    c = build_circuit(1, 1, ansatz="basic")
    x = torch.tensor([[math.pi], [math.pi / 2]])
    w = torch.zeros(1)
    ev = TorchBackend(c)(c.build_angles(x, w))
    assert torch.allclose(ev[:, 0], torch.tensor([-1.0, 0.0]), atol=1e-6)


def test_bell_state_zz():
    # H + CNOT: <Z0 Z1> = 1, <Z0> = 0
    from qhnn.circuits import Circuit
    c = Circuit(n_qubits=2, n_inputs=1)
    c.add("H", [0]); c.add("CNOT", [0, 1])
    angles = torch.zeros(1, 0)
    assert torch.allclose(sim.expval_zz(c, angles), torch.ones(1), atol=1e-6)
    assert torch.allclose(sim.expval_z(c, angles), torch.zeros(1, 2), atol=1e-6)


@pytest.mark.parametrize("name", ["classical", "classical_matched", "bottleneck",
                                  "quantum", "hybrid"])
def test_models_train_step(name):
    m = build_model(name, in_dim=13, n_classes=3, n_qubits=4, n_layers=2)
    x, y = torch.randn(16, 13), torch.randint(0, 3, (16,))
    loss = torch.nn.functional.cross_entropy(m(x), y)
    loss.backward()
    assert all(p.grad is not None for p in m.parameters())


def test_matched_param_count_close():
    h = n_params(build_model("hybrid", 30, 2, n_qubits=4, n_layers=2))
    c = n_params(build_model("classical_matched", 30, 2, n_qubits=4, n_layers=2))
    assert abs(h - c) / h < 0.25


def test_qasm_export():
    layer = QuantumLayer(3, 1)
    q = to_qasm(layer.circuit, torch.zeros(3), layer.weights.detach())
    assert q.startswith("OPENQASM 2.0;") and "cx q[0],q[1];" in q
