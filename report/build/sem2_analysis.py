# -*- coding: utf-8 -*-
"""Таблицы и рисунки для отчёта за 2 семестр по results/sem2/*.jsonl.

    python sem2_analysis.py <root>
Рисунки: report/sem2/fig/, сводка: results/sem2/summary.json
"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.stdout.reconfigure(encoding="utf-8")
root = pathlib.Path(sys.argv[1])
R = root / "results" / "sem2"
FIG = root / "report" / "sem2" / "fig"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 12, "legend.fontsize": 9.5,
                     "savefig.dpi": 220, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3})
DS = {"iris": "Iris", "wine": "Wine", "breast_cancer": "Breast Cancer", "moons": "Moons", "circles": "Circles"}


def load(name):
    return pd.read_json(R / f"{name}.jsonl", lines=True)


def fmt(x, d=3):
    return f"{x:.{d}f}".replace(".", ",")


out = {}

# ---------------------------------------------------------------- архитектура перцептрона
a = load("arch")
a["arch"] = a.hidden.apply(lambda h: "–".join(map(str, h)))
order = ["8", "32", "128", "32–16", "64–32", "64–32–16"]
acc = a.pivot_table(index="arch", columns="dataset", values="accuracy", aggfunc="mean").loc[order, list(DS)]
acc["mean"] = acc.mean(axis=1)
params = a[a.dataset == "breast_cancer"].groupby("arch").n_params.first().loc[order]
ep = a.groupby("arch").epochs.mean().loc[order]
out["arch"] = [{"arch": k, "params_bc": int(params[k]), **{d: round(acc.loc[k, d], 4) for d in list(DS) + ["mean"]},
                "epochs": round(ep[k], 1)} for k in order]

fig, ax = plt.subplots(figsize=(9.5, 4.0))
w = 0.13
cols = plt.cm.viridis(np.linspace(0.1, 0.85, len(order)))
for i, k in enumerate(order):
    ax.bar(np.arange(len(DS)) + (i - 2.5) * w, [acc.loc[k, d] for d in DS], w, color=cols[i], label=f"скрытые слои {k}")
ax.set_xticks(range(len(DS))); ax.set_xticklabels(DS.values())
ax.set_ylim(0.8, 1.005); ax.set_ylabel("Accuracy (среднее по 10 запускам)")
ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
fig.tight_layout(); fig.savefig(FIG / "fig_arch.png", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- скорость обучения × активация
l = load("lr")
g = l.groupby(["act", "lr"]).agg(acc=("accuracy", "mean"), epochs=("epochs", "mean")).reset_index()
per = l.pivot_table(index=["act", "lr"], columns="dataset", values="accuracy", aggfunc="mean")
out["lr"] = [{"act": r.act, "lr": r.lr, "mean": round(r.acc, 4), "epochs": round(r.epochs, 1),
              **{d: round(per.loc[(r.act, r.lr), d], 4) for d in DS}} for r in g.itertuples()]

# ---------------------------------------------------------------- сравнение с PennyLane
p = load("pl")
MODELS = {"mlp": "Classical MLP 32–16", "mlp_matched": "MLP равного размера", "pl_hybrid": "Hybrid QNN (PennyLane)",
          "pl_quantum": "Quantum QNN (PennyLane)"}
rows = []
for d in DS:
    for m in MODELS:
        s = p[(p.dataset == d) & (p.model == m)]
        rows.append({"dataset": d, "model": m, "acc": round(s.accuracy.mean(), 4), "acc_std": round(s.accuracy.std(), 4),
                     "precision": round(s.precision.mean(), 4), "recall": round(s.recall.mean(), 4),
                     "f1": round(s.f1.mean(), 4), "epoch_ms": round(1000 * s.time_per_epoch.mean(), 1),
                     "train_s": round(s.train_time.mean(), 2), "epochs": round(s.epochs.mean(), 1),
                     "params": int(s.n_params.median()), "runs": len(s)})
out["pl"] = rows
tests = []
for d in DS:
    base = p[(p.dataset == d) & (p.model == "pl_hybrid")].sort_values("seed").accuracy.values
    for other in ("mlp", "mlp_matched", "pl_quantum"):
        o = p[(p.dataset == d) & (p.model == other)].sort_values("seed").accuracy.values
        diff = base - o
        pv = 1.0 if np.allclose(diff, 0) else float(wilcoxon(base, o).pvalue)
        tests.append({"dataset": d, "vs": other, "diff_pp": round(100 * diff.mean(), 1), "p": round(pv, 4)})
out["tests"] = tests

fig, ax = plt.subplots(figsize=(9.5, 4.0))
w = 0.2
colors = {"mlp": "#2a78d6", "mlp_matched": "#46b3d1", "pl_hybrid": "#7b5fb8", "pl_quantum": "#c44e52"}
for i, m in enumerate(MODELS):
    vals = [next(r for r in rows if r["dataset"] == d and r["model"] == m) for d in DS]
    ax.bar(np.arange(len(DS)) + (i - 1.5) * w, [v["acc"] for v in vals], w, yerr=[v["acc_std"] for v in vals],
           capsize=2, color=colors[m], label=MODELS[m])
ax.set_xticks(range(len(DS))); ax.set_xticklabels(DS.values())
ax.set_ylim(0.4, 1.02); ax.set_ylabel("Accuracy (среднее ± std)")
ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
fig.tight_layout(); fig.savefig(FIG / "fig_pl_acc.png", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- время эпохи от числа кубитов
t = load("time")
th = t[t.model == "pl_hybrid"].groupby("n_qubits").time_per_epoch.mean()
tm = t[t.model == "mlp"].time_per_epoch.mean()
out["time"] = {"pl_hybrid": {int(k): round(v, 4) for k, v in th.items()}, "mlp": round(tm, 5)}
fig, ax = plt.subplots(figsize=(7.5, 3.8))
ax.plot(th.index, th.values, "o-", color="#7b5fb8", lw=2, label="Hybrid QNN (PennyLane)")
ax.axhline(tm, color="#2a78d6", ls="--", lw=1.8, label="Classical MLP 32–16")
ax.set_yscale("log"); ax.set_xticks(th.index)
ax.set_xlabel("Число кубитов"); ax.set_ylabel("Время одной эпохи, с"); ax.legend()
fig.tight_layout(); fig.savefig(FIG / "fig_time.png", bbox_inches="tight"); plt.close(fig)

(R / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=1))
