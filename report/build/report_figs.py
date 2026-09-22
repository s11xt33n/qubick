"""Рисунки для отчёта (A4, книжная): крупные шрифты, сетка 2×2, чистые оси."""
import sys, glob, pathlib
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = pathlib.Path(sys.argv[1]); R = root / "results"; OUT = root / "report" / "fig"; OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                     "legend.fontsize": 9, "savefig.dpi": 220, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3})
COL = {"hybrid": "#7b5fb8", "quantum": "#c44e52", "classical": "#2a78d6", "classical_matched": "#46b3d1", "bottleneck": "#8c8c8c"}
LAB = {"hybrid": "Hybrid QNN", "quantum": "Quantum QNN", "classical": "Classical MLP",
       "classical_matched": "MLP (равное число параметров)", "bottleneck": "Bottleneck (без квантовой части)"}
DS = {"vision:mnist": "MNIST", "vision:fashion": "Fashion-MNIST", "vision:pneumonia": "PneumoniaMNIST", "vision:breast": "BreastMNIST"}


def load(name):
    fs = [f for f in glob.glob(str(R / f"{name}*.jsonl")) if "_history" not in f]
    df = pd.concat([pd.read_json(f, lines=True) for f in fs]).drop_duplicates("run_id")
    if "enc" not in df: df["enc"] = None
    m = df.model.isin(["hybrid", "bottleneck"])
    df.loc[m & df.enc.isna(), "enc"] = "pi"; df.loc[~m, "enc"] = "-"
    return df


def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT / name, bbox_inches="tight"); plt.close(fig); print("->", name)


# ---- изображения, 4 кубита, энкодер BN: 2×2
v = load("vision"); v["n_qubits"] = v.n_qubits.fillna(0).astype(int)
v["n_train"] = v[["n_train", "n_train_actual"]].min(axis=1)
v = v[v.n_qubits.isin([0, 4]) & v.enc.isin(["-", "bn"])]
fig, axes = plt.subplots(2, 2, figsize=(10, 7.4))
for ax, (ds, title) in zip(axes.flat, DS.items()):
    sub = v[v.dataset == ds]
    for m in ("classical", "classical_matched", "bottleneck", "quantum", "hybrid"):
        g = sub[sub.model == m].groupby("n_train").accuracy.agg(["mean", "std"]).sort_index()
        ax.errorbar(g.index, g["mean"], g["std"], marker="o", ms=4, capsize=2, lw=1.8, color=COL[m], label=LAB[m])
    ns = sorted(sub.n_train.unique())
    ax.set_xscale("log"); ax.set_xticks(ns); ax.set_xticklabels([str(int(n)) for n in ns]); ax.minorticks_off()
    ax.set_title(title); ax.set_xlabel("Размер обучающей выборки"); ax.set_ylabel("Accuracy")
h, l = axes[0][0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06), frameon=False)
save(fig, "fig_vision.png")

# ---- barren plateaus: HEA, L = 20, три инициализации; локальная и глобальная стоимость
b = pd.read_csv(R / "barren_init.csv")
fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
lab = {"uniform": "uniform U[0, 2π)", "small": "small N(0, 0.1²)", "zero": "zero"}
cc = {"uniform": "#2a78d6", "small": "#eb6834", "zero": "#1baf7a"}
for ax, cost in zip(axes, ("local", "global")):
    for init in ("uniform", "small", "zero"):
        r = b[(b.cost == cost) & (b.n_layers == 20) & (b.init == init)].sort_values("n_qubits")
        ax.plot(r.n_qubits, r.grad_var, marker="o", ms=4, lw=1.8, color=cc[init], label=lab[init])
    ax.set_yscale("log"); ax.set_xlabel("Число кубитов"); ax.set_xticks(range(2, 21, 2))
    ax.set_title("Локальная стоимость ⟨Z₀Z₁⟩" if cost == "local" else "Глобальная стоимость ⟨Z₀…Zₙ₋₁⟩")
axes[0].set_ylabel("Var[∂C/∂θ]"); axes[0].legend()
save(fig, "fig_barren.png")

# ---- скорость
s = pd.read_csv(R / "speed.csv")
fig, ax = plt.subplots(figsize=(8.5, 4.3))
style = {("torch", "backprop", "cpu"): ("Qubik, backprop, CPU", "#2a78d6", "-"),
         ("torch", "backprop", "cuda"): ("Qubik, backprop, Tesla V100", "#1baf7a", "-"),
         ("torch", "parameter-shift", "cpu"): ("Qubik, parameter-shift, CPU", "#7b5fb8", "-"),
         ("torch", "parameter-shift", "cuda"): ("Qubik, parameter-shift, Tesla V100", "#e87ba4", "-"),
         ("pennylane", "backprop", "cpu"): ("PennyLane, backprop", "#2a78d6", "--"),
         ("pennylane", "parameter-shift", "cpu"): ("PennyLane, parameter-shift", "#eb6834", "--")}
for (bk, m, dv), (l, c, ls) in style.items():
    r = s[(s.backend == bk) & (s.diff_method == m) & (s.device == dv)].sort_values("n_qubits")
    if len(r): ax.plot(r.n_qubits, r.step_time * 1000, ls, marker="o", ms=4, lw=1.8, color=c, label=l)
ax.set_yscale("log"); ax.set_xlabel("Число кубитов"); ax.set_ylabel("Время шага обучения, мс"); ax.set_xticks(range(2, 17, 2))
ax.legend(ncol=2, fontsize=8.5)
save(fig, "fig_speed.png")

# ---- готовые рисунки анализа
import shutil
for f in ("tabular_accuracy.png", "encoder.png", "sweep_heatmap.png"):
    shutil.copy2(R / "figures" / f, OUT / ("fig_" + f)); print("->", "fig_" + f)
