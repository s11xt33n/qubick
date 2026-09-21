"""Эксперимент: barren plateaus (затухание градиентов).

Для случайных параметров схемы оценивается дисперсия производной функции
стоимости по одному параметру, Var[dC/dθ], в зависимости от числа кубитов n
и глубины L. Сравниваются две функции стоимости (Cerezo et al., 2021):
    local  — C = <Z_0 Z_1>          (наблюдаемая на двух кубитах),
    global — C = <Z_0 Z_1 ... Z_n-1> (наблюдаемая на всех кубитах).
Ожидание (McClean et al., 2018; Cerezo et al., 2021): для global-стоимости
дисперсия убывает экспоненциально по n уже при малой глубине; для
local-стоимости при малой глубине убывание гораздо медленнее.

Производная считается по правилу сдвига параметра — это требует только
двух прямых проходов и почти не расходует память даже для 14+ кубитов.
"""
from __future__ import annotations

import argparse
import math
import time

import pandas as pd
import torch

from qhnn import build_circuit
from qhnn import simulator as sim
from qhnn.experiments.runner import ROOT


def cost(c, angles, kind):
    p = sim.probabilities(sim.run_circuit(c, angles))
    s = sim.z_signs(c.n_qubits, p.device)
    obs = s[:, 0] * s[:, 1] if kind == "local" else s.prod(1)
    return p @ obs


def sample_angles(c, b, init, scale, device):
    """Углы вентилей: вход (кодирование) ~ U[0, 2π), веса — по стратегии init."""
    G = len(c.param_gates)
    a = torch.rand(b, G, device=device) * 2 * math.pi
    is_w = torch.tensor([c.gates[k].src[0] == "w" for k in c.param_gates], device=device)
    if init == "small":
        a[:, is_w] = torch.randn(b, int(is_w.sum()), device=device) * scale
    elif init == "zero":
        a[:, is_w] = 0.0
    return a


@torch.no_grad()
def grad_variance(n, L, kind, samples, ansatz, device, param_idx=None, chunk=None,
                  init="uniform", scale=0.1):
    c = build_circuit(n, L, n_inputs=n, encoding="angle", ansatz=ansatz)
    if chunk is None:
        # батч подбирается так, чтобы состояние занимало ~256 МБ (2^25 амплитуд)
        chunk = max(1, min(1024, 2 ** 25 // 2 ** n))
    k = n if param_idx is None else param_idx  # первый обучаемый вес (после кодирования)
    grads = []
    for i in range(0, samples, chunk):
        b = min(chunk, samples - i)
        a = sample_angles(c, b, init, scale, device)
        plus, minus = a.clone(), a.clone()
        plus[:, k] += math.pi / 2
        minus[:, k] -= math.pi / 2
        grads.append((cost(c, plus, kind) - cost(c, minus, kind)) / 2)
    g = torch.cat(grads)
    return g.var().item(), g.abs().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qubits", type=int, nargs="+", default=[2, 4, 6, 8, 10, 12])
    ap.add_argument("--layers", type=int, nargs="+", default=[1, 5, 20])
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--ansatz", default="hea")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--inits", nargs="+", default=["uniform"],
                    help="стратегии инициализации весов: uniform small zero")
    ap.add_argument("--init-scale", type=float, default=0.1)
    ap.add_argument("--out", default=str(ROOT / "results" / "barren.csv"))
    a = ap.parse_args()
    rows = []
    for init in a.inits:
        for kind in ("local", "global"):
            for L in a.layers:
                for n in a.qubits:
                    t0 = time.time()
                    var, mean_abs = grad_variance(n, L, kind, a.samples, a.ansatz, a.device,
                                                  init=init, scale=a.init_scale)
                    rows.append(dict(init=init, cost=kind, n_layers=L, n_qubits=n,
                                     ansatz=a.ansatz, grad_var=var, grad_mean_abs=mean_abs,
                                     samples=a.samples))
                    print(f"{init:>7} {kind:>6} L={L:<3} n={n:<3} Var={var:.3e}  "
                          f"({time.time() - t0:.1f} с)", flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print("->", a.out)


if __name__ == "__main__":
    main()
