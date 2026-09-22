"""Запуск серии экспериментов по YAML-конфигу.

Пример конфига (configs/tabular.yaml):
    name: tabular
    seeds: [0, 1, 2, 3, 4]
    grid:                      # декартово произведение значений
      dataset: [iris, wine]
      model: [classical, hybrid]
    fixed:                     # общие параметры всех запусков
      n_qubits: 4
      n_layers: 2
      lr: 0.01

Каждый запуск идентифицируется хешем параметров; уже посчитанные
запуски пропускаются, поэтому серию можно прерывать и продолжать.
Запуски выполняются параллельно в нескольких процессах.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import torch
import yaml

from qubik.data import load_split
from qubik.models import build_model, n_params
from qubik.training import fit

ROOT = Path(__file__).resolve().parents[2]
CLASSICAL = {"classical"}  # модели, не зависящие от параметров квантовой схемы
ENCODER_MODELS = {"hybrid", "bottleneck"}  # модели с классическим encoder'ом (ключ enc)
QUANTUM_KEYS = ("n_qubits", "n_layers", "encoding", "ansatz", "reupload",
                "backend", "diff_method", "shots", "init", "init_scale")
TRAIN_KEYS = ("epochs", "batch_size", "lr", "weight_decay", "patience")


def load_results(path) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def run_id(p: dict) -> str:
    return hashlib.md5(json.dumps(p, sort_keys=True, default=str).encode()).hexdigest()[:12]


def expand(cfg: dict) -> list[dict]:
    grid, fixed = cfg.get("grid", {}), cfg.get("fixed", {})
    keys = list(grid)
    runs, seen = [], set()
    for values in itertools.product(*(grid[k] for k in keys)):
        for seed in cfg.get("seeds", [0]):
            p = {**fixed, **dict(zip(keys, values)), "seed": seed}
            if p["model"] in CLASSICAL:  # у классической MLP нет кубитов
                for k in QUANTUM_KEYS:
                    p.pop(k, None)
            if p["model"] not in ENCODER_MODELS:  # enc влияет только на hybrid/bottleneck
                p.pop("enc", None)
            rid = run_id(p)
            if rid not in seen:
                seen.add(rid)
                runs.append({**p, "run_id": rid})
    return runs


LOCKS = ROOT / "results" / "locks"


def claim(rid: str) -> bool:
    """Бронирует запуск (lock-файл), чтобы несколько параллельных runner'ов
    не считали одно и то же."""
    LOCKS.mkdir(parents=True, exist_ok=True)
    try:
        os.close(os.open(LOCKS / rid, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False


def run_one(p: dict) -> dict | None:
    torch.set_num_threads(1)
    p = dict(p)
    if not claim(p["run_id"]):
        return None
    rid, seed, device = p.pop("run_id"), p["seed"], p.pop("device", "cpu")
    log_history = p.pop("log_history", False)
    qkw = {k: p[k] for k in QUANTUM_KEYS if k in p}
    n_q = qkw.get("n_qubits", 4)
    data = load_split(p["dataset"], seed, p.get("n_train"),
                      reduce_to=n_q if p["model"] == "quantum" else None)
    torch.manual_seed(seed)
    model = build_model(p["model"], data.in_dim, data.n_classes,
                        hidden=tuple(p.get("hidden", (32, 16))), enc=p.get("enc", "pi2"), **qkw)
    res = fit(model, data, seed=seed, device=device, log_history=log_history,
              **{k: p[k] for k in TRAIN_KEYS if k in p})
    history = res.pop("history")
    row = {"run_id": rid, **p, "in_dim": data.in_dim, "n_train_actual": len(data.y_train),
           "n_params": n_params(model), "data_note": data.note, **res}
    return {"row": row, "history": [{"run_id": rid, **h} for h in history]}


def _init_worker():
    torch.set_num_threads(1)


def estimated_cost(p: dict) -> float:
    """Грубая оценка длительности запуска — длинные запускаются первыми."""
    if p["model"] == "classical":
        return 1.0
    n_q, L = p.get("n_qubits", 4), p.get("n_layers", 2)
    cost = (2 ** n_q) * n_q * L * (p.get("n_train") or 300)
    if p.get("diff_method") == "parameter-shift":
        cost *= 6 * n_q * L
    return cost


def main(argv=None):
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Запуск серий экспериментов Qubik")
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=None, help="запустить только N первых")
    ap.add_argument("--gpu-min-qubits", type=int, default=None,
                    help="запуски с n_qubits >= N отправлять на GPU (отдельный пул)")
    ap.add_argument("--gpu-workers", type=int, default=4)
    ap.add_argument("--min-qubits", type=int, default=None,
                    help="брать только запуски с n_qubits >= N (классика считается 0)")
    ap.add_argument("--max-qubits", type=int, default=None,
                    help="брать только запуски с n_qubits <= N")
    ap.add_argument("--tag", default="",
                    help="суффикс файла результатов, напр. .gpu -> <name>.gpu.jsonl")
    args = ap.parse_args(argv)
    # в Windows ProcessPoolExecutor допускает не более 61 процесса
    if os.name == "nt":
        args.workers = min(args.workers, 60)

    todo, outs = [], {}
    for path in args.configs:
        cfg = yaml.safe_load(open(path, encoding="utf-8"))
        name = cfg["name"]
        out = ROOT / "results" / f"{name}{args.tag}.jsonl"
        outs[name] = (out, ROOT / "results" / f"{name}_history{args.tag}.jsonl")
        out.parent.mkdir(parents=True, exist_ok=True)
        runs = expand(cfg)
        done = set()
        for f in (ROOT / "results").glob(f"{name}*.jsonl"):
            if "_history" not in f.name:
                done |= set(load_results(f)["run_id"])
        q = lambda r: r.get("n_qubits", 0)
        runs = [r for r in runs
                if (args.min_qubits is None or q(r) >= args.min_qubits)
                and (args.max_qubits is None or q(r) <= args.max_qubits)]
        new = [r for r in runs if r["run_id"] not in done]
        for r in new:
            r["device"] = args.device
            r["log_history"] = cfg.get("log_history", False)
            r["_series"] = name
        print(f"[{name}] всего {len(runs)}, уже есть {len(done)}, к запуску {len(new)}",
              flush=True)
        todo += new
    todo.sort(key=estimated_cost, reverse=True)
    todo = todo[: args.limit]
    for r in todo:
        if args.gpu_min_qubits and r.get("n_qubits", 0) >= args.gpu_min_qubits:
            r["device"] = "cuda"
    print(f"Итого к запуску: {len(todo)} (процессов: {args.workers})", flush=True)

    t0 = time.time()

    def save(series, res):
        out, hist_out = outs[series]
        # JSONL: одна строка — один запуск; набор полей у моделей может различаться
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(res["row"], ensure_ascii=False, default=str) + "\n")
        if res["history"]:
            with open(hist_out, "a", encoding="utf-8") as f:
                for h in res["history"]:
                    f.write(json.dumps(h) + "\n")

    def report(i, series, r):
        row = r["row"]
        print(f"  {i}/{len(todo)} [{series}] {row['dataset']:>16} {row['model']:>17} "
              f"q={row.get('n_qubits', '-')} L={row.get('n_layers', '-')} "
              f"n={row.get('n_train', '-')} seed={row['seed']} "
              f"acc={row['accuracy']:.3f} ep={row['epochs']} "
              f"t={row['train_time']:.1f}s  [{time.time() - t0:.0f}s]", flush=True)

    def strip(r):
        return {k: v for k, v in r.items() if k != "_series"}

    if args.workers <= 1:
        for i, r in enumerate(todo, 1):
            res = run_one(strip(r))
            if res is not None:
                save(r["_series"], res); report(i, r["_series"], res)
    else:
        cpu = ProcessPoolExecutor(args.workers, initializer=_init_worker)
        gpu = ProcessPoolExecutor(args.gpu_workers, initializer=_init_worker)             if any(r["device"] != "cpu" for r in todo) else None
        with cpu:
            futs = {(gpu if r["device"] != "cpu" else cpu).submit(run_one, strip(r)): r
                    for r in todo}
            for i, f in enumerate(as_completed(futs), 1):
                r = futs[f]
                try:
                    res = f.result()
                except Exception:
                    (LOCKS / r["run_id"]).unlink(missing_ok=True)  # чтобы перезапуск досчитал
                    print("ОШИБКА в запуске", r, traceback.format_exc(), flush=True)
                    continue
                if res is None:  # запуск уже взял другой runner
                    continue
                save(r["_series"], res); report(i, r["_series"], res)
        if gpu:
            gpu.shutdown()
    print(f"Готово за {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main()
