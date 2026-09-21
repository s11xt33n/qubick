"""Экспорт результатов в JSON для сайта (site/data/*.json).

    python -m qhnn.experiments.export_site [--out site/data]

Сайт — статический: страница читает эти файлы и рисует графики. На сервере
экспорт запускается по расписанию, и сайт показывает прогресс в реальном времени.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from qhnn.experiments.analyze import RES, _load, mean_std, paired_tests
from qhnn.experiments.runner import ROOT, expand

SERIES = ["tabular", "sweep", "ablation", "shots", "vision", "vision_q12", "init"]


def _clean(o):
    """NaN/inf -> None, numpy -> python, чтобы JSON был валидным."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if math.isnan(f) or math.isinf(f) else round(f, 6)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def _records(df: pd.DataFrame) -> list[dict]:
    return _clean(df.to_dict(orient="records"))


def progress() -> dict:
    out = {"updated": time.strftime("%Y-%m-%d %H:%M:%S"), "series": {}}
    for name in SERIES:
        cfg_path = ROOT / "configs" / f"{name}.yaml"
        if not cfg_path.exists():
            continue
        total = len(expand(yaml.safe_load(open(cfg_path, encoding="utf-8"))))
        df = _load(name)
        done = 0 if df is None else df.run_id.nunique()
        out["series"][name] = {"done": int(min(done, total)), "total": total}
    out["extra"] = {"barren": (RES / "barren.csv").exists(), "speed": (RES / "speed.csv").exists()}
    mon = RES / "monitor.csv"
    if mon.exists():
        m = pd.read_csv(mon, encoding="utf-8-sig").tail(480)  # ~2 часа при шаге 15 с
        out["monitor"] = _records(m)
    return out


ENCS = ["bn", "pi2", "pi"]


def enc_tests(df, keys):
    """Hybrid против остальных отдельно для каждого варианта encoder'а."""
    out = []
    for e in ENCS:
        sub = df[(df.enc == "-") | (df.enc == e)]
        if not (sub.model == "hybrid").any():
            continue
        t = paired_tests(sub, keys)
        if not t.empty:
            t["enc"] = e
            out.append(t)
    return _records(pd.concat(out)) if out else []


def tabular():
    df = _load("tabular")
    if df is None:
        return None
    return {"summary": _records(mean_std(df, ["dataset", "model", "enc"])),
            "tests": enc_tests(df, ["dataset"]),
            "runs": int(len(df))}


def sweep():
    df = _load("sweep")
    if df is None:
        return None
    s = mean_std(df, ["dataset", "model", "enc", "n_qubits", "n_layers"],
                 cols=("accuracy", "f1", "train_time", "time_per_epoch", "epochs", "n_params"))
    return {"summary": _records(s), "runs": int(len(df))}


def ablation():
    df = _load("ablation")
    if df is None:
        return None
    s = mean_std(df, ["dataset", "model", "enc", "encoding", "ansatz", "reupload"])
    return {"summary": _records(s), "runs": int(len(df))}


def shots():
    df = _load("shots")
    if df is None:
        return None
    df = df.copy()
    df["shots"] = df["shots"].fillna(0).astype(int)
    return {"summary": _records(mean_std(df, ["dataset", "enc", "shots"])), "runs": int(len(df))}


def vision():
    df = _load("vision")
    if df is None:
        return None
    df = df.copy()
    df["n_qubits"] = df.get("n_qubits", pd.Series(dtype=float)).fillna(0).astype(int)
    df["n_train"] = df[["n_train", "n_train_actual"]].min(axis=1)
    df = df.drop_duplicates(["dataset", "model", "enc", "n_qubits", "n_train", "seed"])
    s = mean_std(df, ["dataset", "model", "enc", "n_qubits", "n_train"])
    tests = []
    for q in (4, 8):
        for t in enc_tests(df[df.n_qubits.isin([0, q])], ["dataset", "n_train"]):
            t["n_qubits"] = q
            tests.append(t)
    return {"summary": _records(s), "tests": tests, "runs": int(len(df))}


def vision_q12():
    df = _load("vision_q12")
    if df is None:
        return None
    df = df.copy()
    df["n_train"] = df[["n_train", "n_train_actual"]].min(axis=1)
    return {"summary": _records(mean_std(df, ["dataset", "model", "enc", "n_train"])), "runs": int(len(df))}


def init_exp():
    df = _load("init")
    if df is None:
        return None
    s = mean_std(df, ["dataset", "model", "enc", "n_qubits", "n_layers", "init"])
    return {"summary": _records(s), "runs": int(len(df))}


def encoder():
    rows = []
    for name in ("tabular", "vision"):
        df = _load(name)
        if df is None:
            continue
        df = df[df.model.isin(["hybrid", "bottleneck"])].copy()
        if "n_qubits" in df:
            df = df[df.n_qubits.fillna(4).astype(int) == 4]
        if "n_train" in df:
            df["n_train"] = df[["n_train", "n_train_actual"]].min(axis=1)
            df = df[df.n_train.isna() | (df.n_train >= 1000)]
        g = df.groupby(["dataset", "model", "enc"])
        s = g[["accuracy", "class_coverage"]].agg(["mean", "std"])
        s.columns = [f"{a}_{b}" for a, b in s.columns]
        s["runs"] = g.size()
        rows.append(s.reset_index())
    return {"summary": _records(pd.concat(rows))} if rows else None


def csv_records(name):
    p = RES / f"{name}.csv"
    return _records(pd.read_csv(p)) if p.exists() else None


FIG_TITLES = {
    "tabular_accuracy": "Табличные данные: accuracy пяти моделей",
    "tabular_curves": "Табличные данные: кривые обучения",
    "vision_lowdata_q4": "Изображения: малые выборки, 4 кубита",
    "vision_lowdata_q8": "Изображения: малые выборки, 8 кубитов",
    "vision_q12": "Изображения: 4, 8 и 12 кубитов",
    "sweep_heatmap": "Кубиты × глубина: accuracy",
    "sweep_time": "Кубиты × глубина: время эпохи",
    "ablation": "Абляция квантового слоя",
    "shots": "Конечное число измерений",
    "barren": "Barren plateaus",
    "barren_init": "Barren plateaus: стратегии инициализации",
    "init": "Инициализация и точность глубоких схем",
    "encoder": "Находка: масштаб encoder'а",
    "speed": "Скорость симулятора против PennyLane",
}


def figures(site_dir: Path):
    """Копирует PNG из results/figures в site/figures и возвращает их список."""
    import shutil
    src = RES / "figures"
    dst = site_dir / "figures"
    dst.mkdir(parents=True, exist_ok=True)
    items = []
    for key, title in FIG_TITLES.items():
        f = src / f"{key}.png"
        if f.exists():
            shutil.copy2(f, dst / f.name)
            items.append({"file": f.name, "title": title, "mtime": int(f.stat().st_mtime)})
    return items or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "site" / "data"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    parts = {"progress": progress(), "tabular": tabular(), "sweep": sweep(),
             "ablation": ablation(), "shots": shots(), "vision": vision(),
             "vision_q12": vision_q12(), "init": init_exp(), "encoder": encoder(),
             "barren": csv_records("barren"), "barren_init": csv_records("barren_init"),
             "speed": csv_records("speed"), "figures": figures(Path(a.out).parent)}
    for name, data in parts.items():
        if data is None:
            continue
        tmp = out / f"{name}.json.tmp"
        tmp.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        tmp.replace(out / f"{name}.json")
    print("exported:", ", ".join(k for k, v in parts.items() if v is not None))


if __name__ == "__main__":
    main()
