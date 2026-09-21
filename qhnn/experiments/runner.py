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

from qhnn.data import load_split
from qhnn.models import build_model, n_params
from qhnn.training import fit

ROOT = Path(__file__).resolve().parents[2]
CLASSICAL = {"classical"}  # модели, не зависящие от параметров квантовой схемы
QUANTUM_KEYS = ("n_qubits", "n_layers", "encoding", "ansatz", "reupload",
                "backend", "diff_method", "shots")
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
            rid = run_id(p)
            if rid not in seen:
                seen.add(rid)
                runs.append({**p, "run_id": rid})
    return runs


def run_one(p: dict) -> dict:
    torch.set_num_threads(1)
    p = dict(p)
    rid, seed, device = p.pop("run_id"), p["seed"], p.pop("device", "cpu")
    log_history = p.pop("log_history", False)
    qkw = {k: p[k] for k in QUANTUM_KEYS if k in p}
    n_q = qkw.get("n_qubits", 4)
    data = load_split(p["dataset"], seed, p.get("n_train"),
                      reduce_to=n_q if p["model"] == "quantum" else None)
    torch.manual_seed(seed)
    model = build_model(p["model"], data.in_dim, data.n_classes,
                        hidden=tuple(p.get("hidden", (32, 16))), **qkw)
    res = fit(model, data, seed=seed, device=device, log_history=log_history,
              **{k: p[k] for k in TRAIN_KEYS if k in p})
    history = res.pop("history")
    row = {"run_id": rid, **p, "in_dim": data.in_dim, "n_train_actual": len(data.y_train),
           "n_params": n_params(model), "data_note": data.note, **res}
    return {"row": row, "history": [{"run_id": rid, **h} for h in history]}


def _init_worker():
    torch.set_num_threads(1)


def main(argv=None):
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Запуск серии экспериментов qhnn")
    ap.add_argument("config")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=None, help="запустить только N первых")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    out = ROOT / "results" / f"{cfg['name']}.jsonl"
    hist_out = ROOT / "results" / f"{cfg['name']}_history.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    runs = expand(cfg)
    done = set(load_results(out)["run_id"]) if out.exists() else set()
    todo = [r for r in runs if r["run_id"] not in done][: args.limit]
    for r in todo:
        r["device"] = args.device
        r["log_history"] = cfg.get("log_history", False)
    print(f"[{cfg['name']}] всего {len(runs)}, уже есть {len(done)}, к запуску {len(todo)}"
          f" (процессов: {args.workers})", flush=True)

    t0 = time.time()

    def save(res):
        # JSONL: одна строка — один запуск; набор полей у моделей может различаться
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(res["row"], ensure_ascii=False, default=str) + "\n")
        if res["history"]:
            with open(hist_out, "a", encoding="utf-8") as f:
                for h in res["history"]:
                    f.write(json.dumps(h) + "\n")

    def report(i, r):
        row = r["row"]
        print(f"  {i}/{len(todo)} {row['dataset']:>14} {row['model']:>18} "
              f"q={row.get('n_qubits', '-')} L={row.get('n_layers', '-')} seed={row['seed']} "
              f"acc={row['accuracy']:.3f} f1={row['f1']:.3f} ep={row['epochs']} "
              f"t={row['train_time']:.1f}s  [{time.time() - t0:.0f}s]", flush=True)

    if args.workers <= 1:
        for i, r in enumerate(todo, 1):
            res = run_one(r); save(res); report(i, res)
    else:
        with ProcessPoolExecutor(args.workers, initializer=_init_worker) as ex:
            futs = {ex.submit(run_one, r): r for r in todo}
            for i, f in enumerate(as_completed(futs), 1):
                try:
                    res = f.result()
                except Exception:
                    print("ОШИБКА в запуске", futs[f], traceback.format_exc(), flush=True)
                    continue
                save(res); report(i, res)
    print(f"[{cfg['name']}] готово за {time.time() - t0:.0f} с -> {out}")


if __name__ == "__main__":
    main()
