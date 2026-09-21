"""Эксперимент: скорость собственного симулятора против PennyLane.

Измеряется время одного шага обучения (прямой + обратный проход) для батча
объектов в зависимости от числа кубитов и способа вычисления градиента.
"""
from __future__ import annotations

import argparse
import statistics
import time

import pandas as pd
import torch

from qhnn import QuantumLayer
from qhnn.experiments.runner import ROOT


def step_time(layer, x, repeats):
    times = []
    for r in range(repeats + 1):
        if x.is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        layer.zero_grad()
        layer(x).sum().backward()
        if x.is_cuda:
            torch.cuda.synchronize()
        if r:  # первый прогон — прогрев, не учитываем
            times.append(time.perf_counter() - t0)
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qubits", type=int, nargs="+", default=[2, 4, 6, 8, 10, 12])
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--devices", nargs="+", default=["cpu"])
    ap.add_argument("--pl-max-qubits", type=int, default=10)
    ap.add_argument("--pl-ps-max-qubits", type=int, default=6)
    ap.add_argument("--ps-max-qubits", type=int, default=10,
                    help="parameter-shift дорог (2G прогонов схемы): ограничиваем размер")
    ap.add_argument("--out", default=str(ROOT / "results" / "speed.csv"))
    a = ap.parse_args()
    torch.set_num_threads(4)
    combos = [("torch", "backprop"), ("torch", "parameter-shift"),
              ("pennylane", "backprop"), ("pennylane", "parameter-shift")]
    rows = []
    for n in a.qubits:
        for backend, method in combos:
            devs = a.devices if backend == "torch" else ["cpu"]
            if backend == "pennylane" and n > a.pl_max_qubits:
                continue
            if backend == "pennylane" and method != "backprop" and n > a.pl_ps_max_qubits:
                continue
            if method == "parameter-shift" and n > a.ps_max_qubits:
                continue
            for dev in devs:
                torch.manual_seed(0)
                layer = QuantumLayer(n, a.layers, backend=backend, diff_method=method).to(dev)
                dtype = torch.float64 if backend == "pennylane" else torch.float32
                layer = layer.to(dtype)
                x = torch.randn(a.batch, n, device=dev, dtype=dtype)
                t = step_time(layer, x, a.repeats)
                rows.append(dict(n_qubits=n, backend=backend, diff_method=method, device=dev,
                                 batch=a.batch, n_layers=a.layers, step_time=t))
                print(f"n={n:<3} {backend:>9} {method:>15} {dev:>5}: {t * 1000:9.1f} мс", flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print("->", a.out)


if __name__ == "__main__":
    main()
