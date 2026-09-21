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
            "vision:fashion": "Fashion-MNIST", "vision:pneumonia": "PneumoniaMNIST"}
COLORS = {"classical": "#4C72B0", "classical_matched": "#64B5CD", "bottleneck": "#8C8C8C",
          "quantum": "#C44E52", "hybrid": "#8172B3"}

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3})
warnings.filterwarnings("ignore")


def _load(name):
    for ext, reader in ((".jsonl", lambda f: pd.read_json(f, lines=True)), (".csv", pd.read_csv)):
        p = RES / f"{name}{ext}"
        if p.exists():
            return reader(p)
    return None


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
    df = _load("tabular")
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
    df = _load("sweep")
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
    df = _load("ablation")
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
    df = _load("shots")
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
    df = _load("vision")
    if df is None:
        return
    print("[vision]")
    df["n_qubits"] = df.get("n_qubits", pd.Series(dtype=float)).fillna(0).astype(int)
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


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    for f in (tabular, sweep, ablation, shots, vision, barren, speed):
        f()


if __name__ == "__main__":
    main()
