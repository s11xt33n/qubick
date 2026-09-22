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
        # 6 значащих цифр (а не знаков после запятой): дисперсии градиентов бывают ~1e-7
        return None if math.isnan(f) or math.isinf(f) else float(f"{f:.6g}")
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
        ids = {r["run_id"] for r in expand(yaml.safe_load(open(cfg_path, encoding="utf-8")))}
        df = _load(name)
        # засчитываются только запуски текущей версии конфига (старые варианты не в счёт)
        done = 0 if df is None else len(ids & set(df.run_id))
        out["series"][name] = {"done": int(done), "total": len(ids)}
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


def _monitor_frame():
    """Журнал нагрузки (results/monitor.csv и архивные копии) с полными временными метками."""
    frames = []
    for f in [RES / "old_logs" / "monitor.csv", RES / "monitor.csv"]:
        if f.exists():
            try:
                frames.append(pd.read_csv(f, encoding="utf-8-sig", on_bad_lines="skip"))
            except Exception:
                pass
    if not frames:
        return None
    m = pd.concat(frames, ignore_index=True)
    day = time.strftime("%Y-%m-%d", time.localtime((RES / "monitor.csv").stat().st_mtime)) \
        if (RES / "monitor.csv").exists() else time.strftime("%Y-%m-%d")
    full = m["time"].astype(str).str.len() > 8
    day0 = m.loc[full, "time"].astype(str).str[:10].min() if full.any() else day
    m["ts"] = pd.to_datetime(m["time"].where(full, day0 + " " + m["time"].astype(str)), errors="coerce")
    for c in ("gpu_temp", "gpu_util", "gpu_power", "gpu_mem", "cpu_load", "python_procs"):
        m[c] = pd.to_numeric(m.get(c), errors="coerce")
    return m.dropna(subset=["ts"]).sort_values("ts").drop_duplicates("ts")


def server():
    """История нагрузки за весь прогон (средние по минутам) и итоговые цифры."""
    m = _monitor_frame()
    out = {"updated": time.strftime("%Y-%m-%d %H:%M:%S")}
    if m is not None and len(m):
        h = m.set_index("ts")[["cpu_load", "gpu_util", "gpu_temp", "gpu_power", "gpu_mem", "python_procs"]]
        h = h.resample("1min").mean().dropna(how="all").round(1)
        out["history"] = [{"t": t.strftime("%d.%m %H:%M"), **{k: (None if pd.isna(v) else float(v)) for k, v in r.items()}}
                          for t, r in h.iterrows()]
        hours = (m.ts.max() - m.ts.min()).total_seconds() / 3600
        out["totals"] = {
            "start": m.ts.min().strftime("%d.%m.%Y %H:%M"), "end": m.ts.max().strftime("%d.%m.%Y %H:%M"),
            "wall_hours": round(hours, 2),
            "cpu_avg": round(float(m.cpu_load.mean()), 1), "gpu_util_avg": round(float(m.gpu_util.mean()), 1),
            "gpu_temp_max": float(m.gpu_temp.max()), "gpu_power_avg": round(float(m.gpu_power.mean()), 1),
            "gpu_power_max": float(m.gpu_power.max()),
            "gpu_energy_kwh": round(float(m.gpu_power.mean()) * hours / 1000, 2),
            "procs_max": int(m.python_procs.max()),
        }
    runs, core_h, gpu_runs = 0, 0.0, 0
    per = {}
    for name in SERIES:
        parts = [f for f in RES.glob(f"{name}*.jsonl") if "_history" not in f.name]
        n = 0
        for f in parts:
            try:
                d = pd.read_json(f, lines=True)
            except Exception:
                continue
            n += len(d)
            core_h += float(d.get("train_time", pd.Series(dtype=float)).sum()) / 3600
            if ".gpu" in f.name:
                gpu_runs += len(d)
        per[name] = n
        runs += n
    barren_pts = 0
    for f in RES.glob("barren*.csv"):
        try:
            barren_pts += len(pd.read_csv(f))
        except Exception:
            pass
    out.setdefault("totals", {}).update({"runs": runs, "train_core_hours": round(core_h, 1),
                                         "gpu_runs": gpu_runs, "barren_points": barren_pts, "per_series": per})
    return out


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
    "barren_ansatz": "Barren plateaus: три типа схем",
    "init": "Инициализация и точность глубоких схем",
    "encoder": "Находка: масштаб encoder'а",
    "speed": "Скорость симулятора против PennyLane",
}


FIG_SERIES = {
    "tabular_accuracy": "tabular", "tabular_curves": "tabular", "vision_lowdata_q4": "vision",
    "vision_lowdata_q8": "vision", "vision_q12": "vision_q12", "sweep_heatmap": "sweep", "sweep_time": "sweep",
    "ablation": "ablation", "shots": "shots", "init": "init", "encoder": "vision",
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
            items.append({"file": f.name, "title": title, "mtime": int(f.stat().st_mtime),
                          "series": FIG_SERIES.get(key)})
    return items or None


def _raise_priority():
    """На загруженном сервере экспорт для сайта не должен ждать своей очереди за экспериментами."""
    import os
    if os.name == "nt" and os.environ.get("QHNN_HIGH_PRIORITY"):
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x80)


def main():
    _raise_priority()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "site" / "data"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    parts = {"progress": progress(), "tabular": tabular(), "sweep": sweep(),
             "ablation": ablation(), "shots": shots(), "vision": vision(),
             "vision_q12": vision_q12(), "init": init_exp(), "encoder": encoder(),
             "barren": csv_records("barren"), "barren_init": csv_records("barren_init"),
             "barren_strong": csv_records("barren_strong"), "barren_basic": csv_records("barren_basic"),
             "speed": csv_records("speed"), "figures": figures(Path(a.out).parent),
             "server": server()}
    for name, data in parts.items():
        if data is None:
            continue
        tmp = out / f"{name}.json.tmp"
        tmp.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        tmp.replace(out / f"{name}.json")
    print("exported:", ", ".join(k for k, v in parts.items() if v is not None))


if __name__ == "__main__":
    main()
