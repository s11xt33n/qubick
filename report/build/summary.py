import sys, glob
import pandas as pd, numpy as np
from scipy.stats import wilcoxon
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width", 220); pd.set_option("display.max_rows", 400)
R = "results_server"
def load(name):
    fs = [f for f in glob.glob(f"{R}/{name}*.jsonl") if "_history" not in f]
    df = pd.concat([pd.read_json(f, lines=True) for f in fs]).drop_duplicates("run_id")
    if "enc" not in df: df["enc"] = None
    enc_m = df.model.isin(["hybrid", "bottleneck"])
    df.loc[enc_m & df.enc.isna(), "enc"] = "pi"; df.loc[~enc_m, "enc"] = "-"
    return df
def tests(df, keys, ref_enc):
    rows = []
    for g, sub in df.groupby(keys):
        h = sub[(sub.model == "hybrid") & (sub.enc == ref_enc)].set_index("seed").accuracy
        for m, e in (("classical_matched", "-"), ("classical", "-"), ("bottleneck", ref_enc), ("quantum", "-")):
            o = sub[(sub.model == m) & (sub.enc == e)].set_index("seed").accuracy
            j = h.index.intersection(o.index)
            if len(j) < 5: continue
            d = h[j] - o[j]
            p = wilcoxon(h[j], o[j]).pvalue if (d != 0).any() else 1.0
            rows.append({**dict(zip(keys, g if isinstance(g, tuple) else (g,))), "vs": m, "diff": round(d.mean() * 100, 1), "p": round(p, 4), "sig": p < 0.05})
    return pd.DataFrame(rows)

print("=== TABULAR: accuracy mean ===")
t = load("tabular")
print(t.groupby(["dataset", "model", "enc"]).accuracy.mean().unstack([1, 2]).round(3).to_string())
for e in ("bn", "pi2"):
    tt = tests(t, ["dataset"], e); print(f"--- hybrid[{e}] vs others (sig only):"); print(tt[tt.sig].to_string(index=False))

print("\n=== VISION: hybrid vs others by dataset/n_train/q (bn) ===")
v = load("vision"); v["n_qubits"] = v.n_qubits.fillna(0).astype(int)
v["n_train"] = v[["n_train", "n_train_actual"]].min(axis=1)
for q in (4, 8):
    sub = v[v.n_qubits.isin([0, q])]
    piv = sub.groupby(["dataset", "n_train", "model", "enc"]).accuracy.mean().unstack([2, 3]).round(3)
    cols = [c for c in [("hybrid", "bn"), ("hybrid", "pi2"), ("hybrid", "pi"), ("bottleneck", "bn"), ("classical_matched", "-"), ("classical", "-"), ("quantum", "-")] if c in piv.columns]
    print(f"--- q={q}"); print(piv[cols].to_string())
    for e in ("bn", "pi2"):
        tt = tests(sub, ["dataset", "n_train"], e)
        s = tt[tt.sig]
        print(f"   sig [{e}] wins:", len(s[s["diff"] > 0]), "losses:", len(s[s["diff"] < 0]), "of", len(tt))
        print(s[s["diff"] > 0].to_string(index=False))
print("\n=== VISION_Q12 ===")
q = load("vision_q12"); q["n_train"] = q[["n_train", "n_train_actual"]].min(axis=1)
print(q.groupby(["dataset", "n_train", "model", "enc"]).accuracy.mean().unstack([2, 3]).round(3).to_string())
print("\n=== INIT ===")
i = load("init")
print(i[i.model == "hybrid"].groupby(["dataset", "n_qubits", "n_layers", "init"]).accuracy.agg(["mean", "std"]).round(3).unstack("init").to_string())
print(i[i.model == "quantum"].groupby(["dataset", "n_qubits", "n_layers", "init"]).accuracy.mean().round(3).unstack("init").to_string())
print("\n=== SHOTS ===")
s = load("shots"); s["shots"] = s.shots.fillna(0).astype(int)
print(s.groupby(["dataset", "enc", "shots"]).accuracy.agg(["mean", "std"]).round(3).to_string())
print("\n=== ABLATION (hybrid pi2 / quantum) ===")
a = load("ablation")
for f in ("encoding", "ansatz", "reupload"):
    print(a[a.enc.isin(["pi2", "-"])].groupby(["model", f]).accuracy.mean().round(3).unstack(f).to_string())
print("\n=== SWEEP (pi2 hybrid & quantum mean over datasets) ===")
w = load("sweep"); w = w[w.enc.isin(["pi2", "-"])]
print(w.groupby(["model", "n_layers", "n_qubits"]).accuracy.mean().round(3).unstack("n_qubits").to_string())
print(w.groupby(["model", "n_qubits"]).time_per_epoch.mean().round(3).unstack("n_qubits").to_string())
print("\n=== BARREN ===")
for f in ("barren", "barren_init", "barren_strong", "barren_basic"):
    try:
        b = pd.read_csv(f"{R}/{f}.csv")
    except Exception as e:
        print(f, e); continue
    if "init" not in b: b["init"] = "uniform"
    print(f"--- {f}")
    print(b.pivot_table(index=["init", "cost", "n_layers"], columns="n_qubits", values="grad_var").map(lambda x: f"{x:.1e}").to_string())
