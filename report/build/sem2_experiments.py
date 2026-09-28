# -*- coding: utf-8 -*-
"""Эксперименты для отчёта за 2 семестр: классические нейросети и гибридная сеть на PennyLane.

Серии (по 10 seed на конфигурацию, табличные наборы iris, wine, breast_cancer, moons, circles):
    arch — многослойный перцептрон разной ширины и глубины;
    lr   — перцептрон 32–16: скорость обучения × функция активации;
    pl   — сравнение перцептронов с гибридной и чисто квантовой сетью на PennyLane
           (AngleEmbedding + StronglyEntanglingLayers, qml.qnn.TorchLayer);
    reg  — переобучение перцептрона 64–32–16: ранняя остановка × L2-регуляризация (weight decay);
    curves — кривые обучения (потери по эпохам) той же сети без ранней остановки;
    time — время эпохи гибридной сети PennyLane от числа кубитов.

    python sem2_experiments.py <root> [--workers 10] [--series arch lr pl time]
Результаты: results/sem2/<серия>.jsonl
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

DATASETS = ["iris", "wine", "breast_cancer", "moons", "circles"]
SEEDS = range(10)


# ------------------------------------------------------------------ модели
def mlp(in_dim, n_classes, hidden, act="relu"):
    from torch import nn
    A = {"relu": nn.ReLU, "tanh": nn.Tanh}[act]
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), A()]
        d = h
    layers.append(nn.Linear(d, n_classes))
    return nn.Sequential(*layers)


def pl_layer(n_qubits=4, n_layers=2, diff_method="backprop"):
    import pennylane as qml
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]

    return qml.qnn.TorchLayer(circuit, {"weights": (n_layers, n_qubits, 3)})


def pl_hybrid(in_dim, n_classes, n_qubits=4, n_layers=2):
    import torch
    from torch import nn

    class Angles(nn.Module):  # (π/2)·tanh: углы кодирования в (−π/2, π/2)
        def forward(self, x):
            return 0.5 * math.pi * torch.tanh(x)

    return nn.Sequential(nn.Linear(in_dim, n_qubits), Angles(), pl_layer(n_qubits, n_layers),
                         nn.Linear(n_qubits, n_classes))


def pl_quantum(n_classes, n_qubits=4, n_layers=2):
    from torch import nn
    return nn.Sequential(pl_layer(n_qubits, n_layers), nn.Linear(n_qubits, n_classes))


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def matched_hidden(in_dim, n_classes, target):
    return max(1, round((target - n_classes) / (in_dim + 1 + n_classes)))


# ------------------------------------------------------------------ один запуск
def run(p: dict) -> dict:
    import torch
    torch.set_num_threads(1)
    from qubik.data import load_split
    from qubik.training import fit

    seed = p["seed"]
    model_name = p["model"]
    nq = p.get("n_qubits", 4)
    data = load_split(p["dataset"], seed, reduce_to=nq if model_name == "pl_quantum" else None)
    torch.manual_seed(seed)
    if model_name == "mlp":
        m = mlp(data.in_dim, data.n_classes, tuple(p["hidden"]), p.get("act", "relu"))
    elif model_name == "mlp_matched":
        target = n_params(pl_hybrid(data.in_dim, data.n_classes, nq))
        m = mlp(data.in_dim, data.n_classes, (matched_hidden(data.in_dim, data.n_classes, target),))
    elif model_name == "pl_hybrid":
        m = pl_hybrid(data.in_dim, data.n_classes, nq)
    elif model_name == "pl_quantum":
        m = pl_quantum(data.n_classes, nq)
    else:
        raise ValueError(model_name)
    epochs = p.get("epochs", 300)
    es = p.get("es", True)  # ранняя остановка с возвратом лучших по валидации весов
    res = fit(m, data, epochs=epochs, patience=p.get("patience", 30) if es else 10**9, lr=p.get("lr", 0.01),
              batch_size=32, seed=seed, weight_decay=p.get("wd", 0.0), restore_best=es,
              log_history=p.get("curves", False))
    if not p.get("curves"):
        res.pop("history", None)
    return {**p, "n_params": n_params(m), "in_dim": data.in_dim, **res}


# ------------------------------------------------------------------ серии
def series(name):
    runs = []
    if name == "arch":
        for ds, hidden, s in itertools.product(DATASETS, [(8,), (32,), (128,), (32, 16), (64, 32), (64, 32, 16)], SEEDS):
            runs.append({"dataset": ds, "model": "mlp", "hidden": list(hidden), "seed": s})
    elif name == "lr":
        for ds, lr, act, s in itertools.product(DATASETS, [0.001, 0.01, 0.1], ["relu", "tanh"], SEEDS):
            runs.append({"dataset": ds, "model": "mlp", "hidden": [32, 16], "lr": lr, "act": act, "seed": s})
    elif name == "pl":
        for ds, model, s in itertools.product(DATASETS, ["mlp", "mlp_matched", "pl_hybrid", "pl_quantum"], SEEDS):
            r = {"dataset": ds, "model": model, "seed": s}
            if model == "mlp":
                r["hidden"] = [32, 16]
            runs.append(r)
    elif name == "reg":
        # переобучение и регуляризация: большая сеть 64–32–16, 300 эпох с ранней остановкой и без
        for ds, es, wd, s in itertools.product(DATASETS, [False, True], [0.0, 1e-3, 1e-2], SEEDS):
            runs.append({"dataset": ds, "model": "mlp", "hidden": [64, 32, 16], "es": es, "wd": wd, "seed": s})
    elif name == "curves":
        # кривые обучения: потери по эпохам для сети 64–32–16 без ранней остановки
        for ds, s in itertools.product(["moons", "breast_cancer"], SEEDS):
            runs.append({"dataset": ds, "model": "mlp", "hidden": [64, 32, 16], "es": False, "curves": True, "seed": s})
    elif name == "time":
        # фиксированные 20 эпох без ранней остановки: сравнивается только время
        for nq, s in itertools.product([2, 4, 6, 8, 10], range(3)):
            runs.append({"dataset": "breast_cancer", "model": "pl_hybrid", "n_qubits": nq, "seed": s,
                         "epochs": 20, "patience": 1000})
        for s in range(3):
            runs.append({"dataset": "breast_cancer", "model": "mlp", "hidden": [32, 16], "seed": s,
                         "epochs": 20, "patience": 1000})
    return runs


KEY = ("dataset", "model", "hidden", "lr", "act", "seed", "n_qubits", "epochs", "es", "wd", "curves")


def key(r):
    """Параметры запуска без результатов: по ним повторный запуск пропускает уже посчитанное."""
    return json.dumps({k: r[k] for k in KEY if k in r}, sort_keys=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--series", nargs="+", default=["arch", "lr", "pl", "time"])
    a = ap.parse_args()
    root = pathlib.Path(a.root)
    sys.path.insert(0, str(root))
    out_dir = root / "results" / "sem2"
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout.reconfigure(encoding="utf-8")
    for name in a.series:
        out = out_dir / f"{name}.jsonl"
        done = set()
        if out.exists():
            done = {key(json.loads(line)) for line in out.read_text(encoding="utf-8").splitlines()}
        todo = [r for r in series(name) if key(r) not in done]
        # сначала долгие (PennyLane), чтобы пул не простаивал в конце
        todo.sort(key=lambda r: (r["model"].startswith("pl"), r.get("n_qubits", 4)), reverse=True)
        print(f"[{name}] к запуску {len(todo)}", flush=True)
        t0 = time.time()
        with ProcessPoolExecutor(a.workers) as ex, open(out, "a", encoding="utf-8") as f:
            futs = [ex.submit(run, r) for r in todo]
            for i, fu in enumerate(as_completed(futs), 1):
                r = fu.result()
                f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
                if i % 25 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)}  {time.time() - t0:.0f} с", flush=True)


if __name__ == "__main__":
    main()
