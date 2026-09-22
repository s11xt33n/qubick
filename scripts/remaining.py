"""Сколько запусков осталось: всего / готово / в работе (lock) / ещё не начато — по сериям."""
import glob, os, sys
import yaml
sys.stdout.reconfigure(encoding="utf-8")
from qhnn.experiments.runner import ROOT, expand, load_results

res = ROOT / "results"
locks = {p.name for p in (res / "locks").glob("*")} if (res / "locks").exists() else set()
tot = [0, 0, 0, 0]
for cfg in sorted(glob.glob(str(ROOT / "configs" / "*.yaml"))):
    c = yaml.safe_load(open(cfg, encoding="utf-8"))
    if c["name"] == "smoke":
        continue
    ids = {r["run_id"]: r for r in expand(c)}
    done = set()
    for f in res.glob(f"{c['name']}*.jsonl"):
        if "_history" not in f.name:
            done |= set(load_results(f)["run_id"])
    d = len(set(ids) & done)
    running = len((set(ids) - done) & locks)
    todo = [r for k, r in ids.items() if k not in done and k not in locks]
    heavy = sum(1 for r in todo if r.get("n_qubits", 0) >= 10)
    print(f"{c['name']:>11}: всего {len(ids):5d}  готово {d:5d}  в работе {running:4d}  не начато {len(todo):5d} (из них 10+ кубитов: {heavy})")
    for i, v in enumerate((len(ids), d, running, len(todo))):
        tot[i] += v
print(f"{'ИТОГО':>11}: всего {tot[0]:5d}  готово {tot[1]:5d}  в работе {tot[2]:4d}  не начато {tot[3]:5d}")
