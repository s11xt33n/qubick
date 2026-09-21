"""Эксперимент: barren plateaus (затухание градиентов).

Для случайных параметров схемы оценивается дисперсия производной функции
стоимости по одному параметру, Var[dC/dθ], в зависимости от числа кубитов n
и глубины L. Сравниваются две функции стоимости (Cerezo et al., 2021):
    local  — C = <Z_0 Z_1>          (наблюдаемая на двух кубитах),
    global — C = <Z_0 Z_1 ... Z_n-1> (наблюдаемая на всех кубитах).
Ожидание (McClean et al., 2018; Cerezo et al., 2021): для global-стоимости
дисперсия убывает экспоненциально по n уже при малой глубине; для
local-стоимости при малой глубине убывание гораздо медленнее.

Дополнительно сравниваются стратегии инициализации весов (--inits):
uniform — U[0, 2π), small — N(0, σ²), zero — нули (Grant et al., 2019).

Производная считается по правилу сдвига параметра — это требует только
двух прямых проходов и почти не расходует память даже для 20 кубитов.
Обе функции стоимости вычисляются по одним и тем же прогонам схемы.
Каждая точка сразу дописывается в CSV; повторный запуск продолжает с места.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import pandas as pd
import torch

from qhnn import build_circuit
from qhnn import simulator as sim
from qhnn.experiments.runner import ROOT


def costs(c, angles):
    """Локальная и глобальная стоимость по одному прогону схемы: (B,), (B,)."""
    p = sim.probabilities(sim.run_circuit(c, angles))
    s = sim.z_signs(c.n_qubits, p.device)
    return p @ (s[:, 0] * s[:, 1]), p @ s.prod(1)


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
def grad_variance(n, L, samples, ansatz, device, param_idx=None, chunk=None,
                  init="uniform", scale=0.1):
    """Возвращает {"local": (var, mean_abs), "global": (var, mean_abs)}."""
    c = build_circuit(n, L, n_inputs=n, encoding="angle", ansatz=ansatz)
    if chunk is None:
        # батч подбирается так, чтобы состояние занимало ~256 МБ (2^25 амплитуд)
        chunk = max(1, min(1024, 2 ** 25 // 2 ** n))
    k = n if param_idx is None else param_idx  # первый обучаемый вес (после кодирования)
    g_loc, g_glob = [], []
    for i in range(0, samples, chunk):
        b = min(chunk, samples - i)
        a = sample_angles(c, b, init, scale, device)
        plus, minus = a.clone(), a.clone()
        plus[:, k] += math.pi / 2
        minus[:, k] -= math.pi / 2
        lp, gp = costs(c, plus)
        lm, gm = costs(c, minus)
        g_loc.append((lp - lm) / 2)
        g_glob.append((gp - gm) / 2)
    out = {}
    for name, g in (("local", torch.cat(g_loc)), ("global", torch.cat(g_glob))):
        out[name] = (g.var().item(), g.abs().mean().item())
    return out


def main():
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
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

    out = Path(a.out)
    done = set()
    if out.exists():
        prev = pd.read_csv(out)
        if "init" not in prev:
            prev["init"] = "uniform"
        done = {(r.init, r.n_layers, r.n_qubits) for r in prev.itertuples()}
    # сначала все стратегии на малом числе кубитов, затем больше — графики растут равномерно
    for L in a.layers:
        for n in a.qubits:
            for init in a.inits:
                if (init, L, n) in done:
                    continue
                t0 = time.time()
                res = grad_variance(n, L, a.samples, a.ansatz, a.device,
                                    init=init, scale=a.init_scale)
                rows = [dict(init=init, cost=cost, n_layers=L, n_qubits=n, ansatz=a.ansatz,
                             grad_var=v, grad_mean_abs=m, samples=a.samples)
                        for cost, (v, m) in res.items()]
                pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)
                print(f"{init:>7} L={L:<3} n={n:<3} Var local={res['local'][0]:.3e} "
                      f"global={res['global'][0]:.3e}  ({time.time() - t0:.1f} с)", flush=True)
    print("->", out)


if __name__ == "__main__":
    main()
