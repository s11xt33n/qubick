import pymupdf as f, sys, json, glob, os
from PIL import Image
sys.stdout.reconfigure(encoding="utf-8")
pdf, out = sys.argv[1], sys.argv[2]
d = f.open(pdf); print("pages", d.page_count)
keys = {"intro": "ВВЕДЕНИЕ", "ch1": "Переход от анализа", "ch2": "Архитектура и реализация библиотеки Qubik",
        "ch3": "Данные, модели и методика экспериментов", "ch4": "Результаты экспериментов",
        "concl": "ЗАКЛЮЧЕНИЕ", "refs": "СПИСОК ИСПОЛЬЗОВАННЫХ ИСТОЧНИКОВ", "apx": "ПРИЛОЖЕНИЯ"}
toc = {}
for i, p in enumerate(d):
    if i < 2: continue
    lines = [l.strip() for l in p.get_text().splitlines() if l.strip()][1:4]
    for k, v in keys.items():
        if k not in toc and any(l.startswith(v) for l in lines):
            toc[k] = i + 1
print(json.dumps(toc))
json.dump(toc, open(os.path.join(out, "..", "toc.json"), "w"))
for fn in glob.glob(os.path.join(out, "*.png")): os.remove(fn)
ims = [Image.frombytes("RGB", (pm.width, pm.height), pm.samples) for pm in (p.get_pixmap(dpi=60) for p in d)]
w, h = ims[0].size
for k in range(0, len(ims), 8):
    sheet = Image.new("RGB", (w * 4 + 30, h * 2 + 10), "gray")
    for j, im in enumerate(ims[k:k + 8]): sheet.paste(im, ((j % 4) * (w + 10), (j // 4) * (h + 10)))
    sheet.save(os.path.join(out, f"sheet{k // 8 + 1}.png"))
