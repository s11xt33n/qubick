"""Анализ результатов: сводные таблицы, статистические тесты и графики.

    python -m qhnn.experiments.analyze

Читает results/*.csv, пишет таблицы в results/tables и рисунки в
results/figures. Каждый раздел строится, только если есть его данные.
"""
from __future__ import annotations

import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from qhnn.experiments.runner import ROOT

RES = ROOT / "results"
FIG = RES / "figures"
TAB = RES / "tables"
MODEL_ORDER = ["classical", "classical_matched", "bottleneck", "quantum", "hybrid"]
MODEL_LABEL = {"classical": "Classical MLP", "classical_matched": "MLP (равн. парам.)",
               "bottleneck": "Bottleneck (без КС)", "quantum": "Quantum QNN",
               "hybrid": "Hybrid QNN"}
DS_LABEL = {"iris": "Iris", "wine": "Wine", "breast_cancer": "Breast Cancer",
            "moons": "Moons", "circles": "Circles", "vision:mnist": "MNIST",
            "vision:fashion": "Fashion-MNIST", "vision:pneumonia": "PneumoniaMNIST",
            "vision:breast": "BreastMNIST"}
COLORS = {"classical": "#4C72B0", "classical_matched": "#64B5CD", "bottleneck": "#8C8C8C",
          "quantum": "#C44E52", "hybrid": "#8172B3"}

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3})
warnings.filterwarnings("ignore")


ENC_MODELS = {"hybrid", "bottleneck"}


def with_enc(df: pd.DataFrame) -> pd.DataFrame:
    """Столбец enc: у hybrid/bottleneck без enc (первые запуски) — "pi", у остальных — "-"."""
    df = df.copy()
    if "enc" not in df:
        df["enc"] = None
    is_enc = df["model"].isin(ENC_MODELS)
    df.loc[is_enc & df["enc"].isna(), "enc"] = "pi"
    df.loc[~is_enc, "enc"] = "-"
    return df


PRIMARY_ENCS = ("bn", "pi2", "pi")


def primary(df: pd.DataFrame) -> pd.DataFrame:
    """Оставляет модели без encoder'а и один (основной из доступных) вариант encoder'а."""
    if df is None or "enc" not in df:
        return df
    avail = set(df.loc[df.enc != "-", "enc"])
    e = next((x for x in PRIMARY_ENCS if x in avail), None)
    return df[(df.enc == "-") | (df.enc == e)]


def _load(name):
    """Результаты серии: <name>.jsonl и <name>.gpu.jsonl (если считалось на GPU)."""
    parts = [pd.read_json(p, lines=True) for p in sorted(RES.glob(f"{name}*.jsonl"))
             if "_history" not in p.name or name.endswith("_history")]
    if parts:
        df = pd.concat(parts, ignore_index=True)
        if name.endswith("_history"):
            return df
        df = df.drop_duplicates("run_id")
        return with_enc(df) if "model" in df else df
    p = RES / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print("  рисунок:", FIG / f"{name}.png")


def _table(df, name):
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / f"{name}.csv", index=False)
    print("  таблица:", TAB / f"{name}.csv")


def mean_std(df, keys, cols=("accuracy", "f1", "train_time", "epochs", "n_params")):
    g = df.groupby(keys)
    out = g[list(cols)].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out["runs"] = g.size()
    return out.reset_index()


def paired_tests(df, keys, ref="hybrid", metric="accuracy"):
    """Парный тест Уилкоксона: ref против каждой другой модели (пары по seed)."""
    rows = []
    for grp, sub in df.groupby(keys):
        grp = grp if isinstance(grp, tuple) else (grp,)
        piv = sub.pivot_table(index="seed", columns="model", values=metric)
        if ref not in piv:
            continue
        for m in piv.columns:
            if m == ref:
                continue
            pair = piv[[ref, m]].dropna()
            diff = pair[ref] - pair[m]
            p = wilcoxon(pair[ref], pair[m]).pvalue if (diff != 0).any() and len(pair) > 1 else 1.0
            rows.append({**dict(zip(keys, grp)), "vs": m, "mean_diff": diff.mean(),
                         "wins": int((diff > 0).sum()), "losses": int((diff < 0).sum()),
                         "p_value": p, "significant": p < 0.05})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def tabular():
    df = primary(_load("tabular"))
    if df is None:
        return
    print("[tabular]")
    s = mean_std(df, ["dataset", "model"])
    _table(s, "tabular_summary")
    _table(paired_tests(df, ["dataset"]), "tabular_tests")

    datasets = [d for d in DS_LABEL if d in df.dataset.unique()]
    models = [m for m in MODEL_ORDER if m in df.model.unique()]
    fig, ax = plt.subplots(figsize=(10, 4))
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        r = s[s.model == m].set_index("dataset").reindex(datasets)
        ax.bar(np.arange(len(datasets)) + i * w - 0.4 + w / 2, r.accuracy_mean,
               w, yerr=r.accuracy_std, capsize=2, label=MODEL_LABEL[m], color=COLORS[m])
    ax.set_xticks(range(len(datasets)), [DS_LABEL[d] for d in datasets])
    ax.set_ylabel("Accuracy (среднее ± std)")
    ax.set_ylim(0.4, 1.02)
    ax.legend(ncol=5, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    _save(fig, "tabular_accuracy")

    h = _load("tabular_history")
    if h is not None:
        h = h.merge(df[["run_id", "dataset", "model"]], on="run_id")
        show = [d for d in ("breast_cancer", "moons", "circles") if d in datasets]
        fig, axes = plt.subplots(1, len(show), figsize=(4 * len(show), 3.2), squeeze=False)
        for ax, d in zip(axes[0], show):
            for m in models:
                cur = h[(h.dataset == d) & (h.model == m)].groupby("epoch").val_loss.mean()
                cur = cur[cur.index <= 150]
                ax.plot(cur.index, cur.values, label=MODEL_LABEL[m], color=COLORS[m])
            ax.set_title(DS_LABEL[d]); ax.set_xlabel("Эпоха"); ax.set_ylabel("Loss (валидация)")
            ax.set_yscale("log")
        axes[0][0].legend(fontsize=7)
        _save(fig, "tabular_curves")


def sweep():
    df = primary(_load("sweep"))
    if df is None:
        return
    print("[sweep]")
    s = mean_std(df, ["dataset", "model", "n_qubits", "n_layers"])
    _table(s, "sweep_summary")
    datasets = [d for d in DS_LABEL if d in df.dataset.unique()]
    models = [m for m in ("quantum", "hybrid") if m in df.model.unique()]
    fig, axes = plt.subplots(len(models), len(datasets),
                             figsize=(3.4 * len(datasets), 2.9 * len(models)), squeeze=False)
    for i, m in enumerate(models):
        for j, d in enumerate(datasets):
            piv = s[(s.model == m) & (s.dataset == d)].pivot(
                index="n_layers", columns="n_qubits", values="accuracy_mean")
            ax = axes[i][j]
            im = ax.imshow(piv.values, cmap="viridis", vmin=0.5, vmax=1.0, origin="lower")
            ax.set_xticks(range(piv.shape[1]), piv.columns)
            ax.set_yticks(range(piv.shape[0]), piv.index)
            for (a, b), v in np.ndenumerate(piv.values):
                ax.text(b, a, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if v < 0.8 else "black")
            ax.set_title(f"{MODEL_LABEL[m]} — {DS_LABEL[d]}", fontsize=9)
            ax.set_xlabel("Кубиты"); ax.set_ylabel("Слои")
            ax.grid(False)
    fig.colorbar(im, ax=axes, shrink=0.8, label="Accuracy")
    fig.savefig(FIG / "sweep_heatmap.png", bbox_inches="tight"); plt.close(fig)
    print("  рисунок:", FIG / "sweep_heatmap.png")

    fig, ax = plt.subplots(figsize=(5, 3.4))
    for m in models:
        for L, ls in zip(sorted(df.n_layers.unique()), ["-", "--", ":"]):
            r = s[(s.model == m) & (s.n_layers == L)].groupby("n_qubits").time_per_epoch_mean.mean() \
                if "time_per_epoch_mean" in s else None
            if r is None:
                r = df[(df.model == m) & (df.n_layers == L)].groupby("n_qubits").time_per_epoch.mean()
            ax.plot(r.index, r.values, ls, marker="o", color=COLORS[m],
                    label=f"{MODEL_LABEL[m]}, L={L}")
    ax.set_yscale("log"); ax.set_xlabel("Число кубитов"); ax.set_ylabel("Время эпохи, с")
    ax.legend(fontsize=7)
    _save(fig, "sweep_time")


def ablation():
    df = primary(_load("ablation"))
    if df is None:
        return
    print("[ablation]")
    s = mean_std(df, ["dataset", "model", "encoding", "ansatz", "reupload"])
    _table(s.sort_values(["dataset", "model", "accuracy_mean"], ascending=[True, True, False]),
           "ablation_summary")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    for ax, factor in zip(axes, ["encoding", "ansatz", "reupload"]):
        g = df.groupby(["model", factor]).accuracy.agg(["mean", "std"]).reset_index()
        levels = list(dict.fromkeys(g[factor]))
        for i, m in enumerate(["quantum", "hybrid"]):
            r = g[g.model == m].set_index(factor).reindex(levels)
            ax.bar(np.arange(len(levels)) + (i - 0.5) * 0.38, r["mean"], 0.38, yerr=r["std"],
                   capsize=2, color=COLORS[m], label=MODEL_LABEL[m])
        ax.set_xticks(range(len(levels)), [str(l) for l in levels])
        ax.set_title({"encoding": "Кодирование", "ansatz": "Анзац",
                      "reupload": "Re-uploading"}[factor])
        ax.set_ylim(0.5, 1.0)
    axes[0].set_ylabel("Accuracy (по всем наборам)")
    axes[0].legend(fontsize=8)
    _save(fig, "ablation")


def shots():
    df = primary(_load("shots"))
    if df is None:
        return
    print("[shots]")
    df["shots_label"] = df.shots.fillna(0).astype(int).map(lambda v: "точно" if v == 0 else str(v))
    s = mean_std(df, ["dataset", "shots_label"])
    _table(s, "shots_summary")
    order = ["100", "1000", "10000", "точно"]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for d in df.dataset.unique():
        r = s[s.dataset == d].set_index("shots_label").reindex(order)
        ax.errorbar(range(len(order)), r.accuracy_mean, r.accuracy_std, marker="o",
                    capsize=3, label=DS_LABEL[d])
    ax.set_xticks(range(len(order)), order)
    ax.set_xlabel("Число измерений (shots)"); ax.set_ylabel("Accuracy")
    ax.legend()
    _save(fig, "shots")


def vision():
    df = primary(_load("vision"))
    if df is None:
        return
    print("[vision]")
    df["n_qubits"] = df.get("n_qubits", pd.Series(dtype=float)).fillna(0).astype(int)
    # если выборка меньше запрошенного n_train (BreastMNIST), берём фактический размер
    df["n_train"] = df[["n_train", "n_train_actual"]].min(axis=1)
    df = df.drop_duplicates(["dataset", "model", "n_qubits", "n_train", "seed"])
    s = mean_std(df, ["dataset", "model", "n_qubits", "n_train"])
    _table(s, "vision_summary")
    _table(paired_tests(df[df.n_qubits.isin([0, 4])], ["dataset", "n_train"]), "vision_tests_q4")
    _table(paired_tests(df[df.n_qubits.isin([0, 8])], ["dataset", "n_train"]), "vision_tests_q8")
    datasets = [d for d in DS_LABEL if d in df.dataset.unique()]
    for q in sorted(x for x in df.n_qubits.unique() if x):
        fig, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), 3.4),
                                 squeeze=False)
        for ax, d in zip(axes[0], datasets):
            for m in MODEL_ORDER:
                r = s[(s.dataset == d) & (s.model == m) & (s.n_qubits.isin([0, q]))]
                if r.empty:
                    continue
                ax.errorbar(r.n_train, r.accuracy_mean, r.accuracy_std, marker="o",
                            ms=3, capsize=2, color=COLORS[m], label=MODEL_LABEL[m])
            ax.set_xscale("log"); ax.set_title(DS_LABEL[d])
            ax.set_xlabel("Размер обучающей выборки"); ax.set_ylabel("Accuracy")
        axes[0][0].legend(fontsize=7)
        fig.suptitle(f"ResNet18 + голова, {q} кубитов", fontsize=10)
        _save(fig, f"vision_lowdata_q{q}")


def barren():
    df = _load("barren")
    if df is None:
        return
    print("[barren]")
    _table(df, "barren")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, cost in zip(axes, ["local", "global"]):
        for L in sorted(df.n_layers.unique()):
            r = df[(df.cost == cost) & (df.n_layers == L)]
            ax.plot(r.n_qubits, r.grad_var, marker="o", label=f"L = {L}")
        ax.set_yscale("log"); ax.set_xlabel("Число кубитов")
        ax.set_title("Локальная стоимость ⟨Z₀Z₁⟩" if cost == "local"
                     else "Глобальная стоимость ⟨Z₀…Zₙ₋₁⟩")
    axes[0].set_ylabel("Var[∂C/∂θ]")
    axes[0].legend()
    _save(fig, "barren")


def speed():
    df = _load("speed")
    if df is None:
        return
    print("[speed]")
    _table(df, "speed")
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for (b, m, d), r in df.groupby(["backend", "diff_method", "device"]):
        name = ("qhnn" if b == "torch" else "PennyLane") + f", {m}" + (f", {d}" if b == "torch" else "")
        ls = "-" if b == "torch" else "--"
        ax.plot(r.n_qubits, r.step_time * 1000, ls, marker="o", label=name)
    ax.set_yscale("log"); ax.set_xlabel("Число кубитов"); ax.set_ylabel("Шаг обучения, мс")
    ax.legend(fontsize=7)
    _save(fig, "speed")


def encoder():
    rows = []
    for name in ("tabular", "vision"):
        df = _load(name)
        if df is None:
            continue
        df = df[df.model == "hybrid"].copy()
        if "n_qubits" in df:
            df = df[df.n_qubits.fillna(4).astype(int) == 4]
        if "n_train" in df:
            df["n_train"] = df[["n_train", "n_train_actual"]].min(axis=1)
            df = df[df.n_train.isna() | (df.n_train >= 1000)]
        rows.append(df)
    if not rows:
        return
    print("[encoder]")
    df = pd.concat(rows)
    g = df.groupby(["dataset", "enc"]).accuracy.agg(["mean", "std"]).reset_index()
    _table(g, "encoder_summary")
    datasets = [d for d in DS_LABEL if d in set(g.dataset)]
    encs = [e for e in ("pi", "pi2", "bn") if e in set(g.enc)]
    lab = {"pi": "π·tanh (исходный)", "pi2": "(π/2)·tanh", "bn": "BN + (π/2)·tanh"}
    col = {"pi": "#e34948", "pi2": "#86b6ef", "bn": "#2a78d6"}
    fig, ax = plt.subplots(figsize=(10, 3.8))
    w = 0.8 / len(encs)
    for i, e in enumerate(encs):
        r = g[g.enc == e].set_index("dataset").reindex(datasets)
        ax.bar(np.arange(len(datasets)) + i * w - 0.4 + w / 2, r["mean"], w, yerr=r["std"],
               capsize=2, color=col[e], label=lab[e])
    ax.set_xticks(range(len(datasets)), [DS_LABEL[d] for d in datasets], rotation=20)
    ax.set_ylabel("Accuracy (Hybrid QNN)"); ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    _save(fig, "encoder")


def barren_init():
    p = RES / "barren_init.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    print("[barren_init]")
    lab = {"uniform": "uniform U[0, 2π)", "small": "small N(0, 0.1²)", "zero": "zero"}
    Ls = sorted(df.n_layers.unique())
    fig, axes = plt.subplots(len(Ls), 2, figsize=(9, 3.2 * len(Ls)), squeeze=False, sharey="row")
    for i, L in enumerate(Ls):
        for j, cost in enumerate(("local", "global")):
            ax = axes[i][j]
            for init in df.init.unique():
                r = df[(df.cost == cost) & (df.n_layers == L) & (df.init == init)].sort_values("n_qubits")
                ax.plot(r.n_qubits, r.grad_var, marker="o", label=lab.get(init, init))
            ax.set_yscale("log"); ax.set_xlabel("Число кубитов")
            ax.set_title(f"{'Локальная' if cost == 'local' else 'Глобальная'} стоимость, L = {L}", fontsize=10)
        axes[i][0].set_ylabel("Var[∂C/∂θ]")
    axes[0][0].legend(fontsize=8)
    _save(fig, "barren_init")


def barren_ansatz():
    """Сравнение анзацев: hea (barren_init), strong, basic — локальная и глобальная стоимость, L = 20."""
    parts = []
    for f, name in (("barren_init.csv", "hea"), ("barren_strong.csv", "strong"), ("barren_basic.csv", "basic")):
        p = RES / f
        if p.exists():
            d = pd.read_csv(p)
            d["ansatz"] = name
            parts.append(d)
    if len(parts) < 2:
        return
    print("[barren_ansatz]")
    df = pd.concat(parts)
    df = df[df.init.isin(["uniform", "small"])]
    _table(df, "barren_ansatz")
    lab = {"hea": "HEA (RY·RZ + CZ)", "strong": "Strongly entangling", "basic": "RY + CNOT"}
    col = {"hea": "#2a78d6", "strong": "#eb6834", "basic": "#1baf7a"}
    L = df.n_layers.max()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, cost in zip(axes, ("local", "global")):
        for a in ("hea", "strong", "basic"):
            for init, ls in (("uniform", "-"), ("small", "--")):
                r = df[(df.ansatz == a) & (df.cost == cost) & (df.n_layers == L) & (df.init == init)].sort_values("n_qubits")
                if len(r):
                    ax.plot(r.n_qubits, r.grad_var, ls, marker="o", ms=3, color=col[a],
                            label=f"{lab[a]}, {init}")
        ax.set_yscale("log"); ax.set_xlabel("Число кубитов")
        ax.set_title(f"{'Локальная' if cost == 'local' else 'Глобальная'} стоимость, L = {L}", fontsize=10)
    axes[0].set_ylabel("Var[∂C/∂θ]")
    axes[0].legend(fontsize=7)
    _save(fig, "barren_ansatz")


def init_exp():
    df = primary(_load("init"))
    if df is None:
        return
    print("[init]")
    s = mean_std(df, ["dataset", "model", "n_qubits", "n_layers", "init"])
    _table(s, "init_summary")
    cfgs = sorted({(q, L) for q, L in zip(s.n_qubits, s.n_layers)})
    datasets = [d for d in DS_LABEL if d in set(s.dataset)]
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), 3.4), squeeze=False, sharey=True)
    for ax, d in zip(axes[0], datasets):
        for i, init in enumerate(("uniform", "small")):
            r = s[(s.dataset == d) & (s.model == "hybrid") & (s.init == init)]
            r = r.set_index(["n_qubits", "n_layers"]).reindex(cfgs)
            ax.bar(np.arange(len(cfgs)) + (i - 0.5) * 0.38, r.accuracy_mean, 0.38, yerr=r.accuracy_std,
                   capsize=2, label=init, color=["#2a78d6", "#eb6834"][i])
        ax.set_xticks(range(len(cfgs)), [f"{q}q, L={L}" for q, L in cfgs], fontsize=8)
        ax.set_title(f"Hybrid — {DS_LABEL[d]}", fontsize=10); ax.set_ylim(0.4, 1.0)
    axes[0][0].set_ylabel("Accuracy"); axes[0][0].legend(fontsize=8)
    _save(fig, "init")


def vision_q12():
    q12 = primary(_load("vision_q12"))
    v = primary(_load("vision"))
    if q12 is None:
        return
    print("[vision_q12]")
    df = pd.concat([v, q12]) if v is not None else q12
    df = df[df.model == "hybrid"].copy()
    df["n_qubits"] = df["n_qubits"].fillna(0).astype(int)
    df["n_train"] = df[["n_train", "n_train_actual"]].min(axis=1)
    ns = sorted(set(q12.n_train))
    df = df[df.n_train.isin(ns)]
    s = df.groupby(["dataset", "n_qubits", "n_train"]).accuracy.agg(["mean", "std"]).reset_index()
    _table(s, "vision_q12_summary")
    datasets = [d for d in DS_LABEL if d in set(s.dataset)]
    fig, axes = plt.subplots(1, len(datasets), figsize=(4 * len(datasets), 3.3), squeeze=False)
    for ax, d in zip(axes[0], datasets):
        for n, c in zip(ns, ["#86b6ef", "#2a78d6", "#104281"]):
            r = s[(s.dataset == d) & (s.n_train == n)].sort_values("n_qubits")
            ax.errorbar(r.n_qubits, r["mean"], r["std"], marker="o", capsize=2, color=c, label=f"{n} примеров")
        ax.set_title(DS_LABEL[d], fontsize=10); ax.set_xlabel("Кубиты"); ax.set_xticks([4, 8, 12])
    axes[0][0].set_ylabel("Accuracy (Hybrid QNN)"); axes[0][0].legend(fontsize=8)
    _save(fig, "vision_q12")


def _raise_priority():
    """На загруженном сервере экспорт для сайта не должен ждать своей очереди за экспериментами."""
    import os
    if os.name == "nt" and os.environ.get("QHNN_HIGH_PRIORITY"):
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x80)


def main():
    _raise_priority()
    sys.stdout.reconfigure(encoding="utf-8")
    for f in (tabular, sweep, ablation, shots, vision, vision_q12, init_exp, encoder,
              barren, barren_init, barren_ansatz, speed):
        try:
            f()
        except Exception as e:  # один сломанный раздел не должен ронять остальные
            print(f"  ! {f.__name__}: {e}")


if __name__ == "__main__":
    main()
