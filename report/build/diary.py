import sys, pathlib
import docx
sys.stdout.reconfigure(encoding="utf-8")
root = pathlib.Path(sys.argv[1])
src = root / "src" / "Дневник_студента_09_535_2_сем_Филадельов .docx"
out_dir = root / "report"; out_dir.mkdir(exist_ok=True)
out = out_dir / "Дневник_студента_09_535_2_сем_Филадельфов.docx"

# номер строки таблицы -> новое содержание (даты и аудиторные занятия не трогаем)
NEW = {
    12: "Реализация модуля загрузки и предварительной обработки табличных наборов данных Iris, Wine, Breast Cancer, Moons и Circles: стратифицированное разбиение, масштабирование признаков, PCA.",
    14: "Реализация базового интерфейса модели для обучения, предсказания и оценки качества (accuracy, precision, recall, F1-score).",
    16: "Реализация классической многослойной нейронной сети Classical MLP с настраиваемым числом и шириной скрытых слоёв.",
    18: "Реализация квантовой нейронной сети Quantum QNN на основе параметризованной квантовой схемы в PennyLane (AngleEmbedding, StronglyEntanglingLayers).",
    20: "Реализация гибридной квантово-классической модели Hybrid QNN: классический энкодер, квантовый слой TorchLayer и выходной слой.",
    22: "Настройка алгоритма обучения: оптимизатор Adam, функция потерь, ранняя остановка по валидационной выборке; исследование влияния скорости обучения и функции активации.",
    24: "Проведение серий экспериментов на пяти наборах данных по 10 запусков: исследование архитектуры перцептрона, сравнение классических моделей с квантовой и гибридной сетями.",
    26: "Анализ результатов экспериментов: accuracy, precision, recall, F1-score и время обучения, проверка значимости различий критерием Уилкоксона.",
    28: "Построение графиков и таблиц результатов, исследование зависимости времени обучения от числа кубитов, оформление отчёта по практике.",
}

d = docx.Document(src)
t = d.tables[0]
for i, text in NEW.items():
    cell = t.rows[i].cells[1]
    p = cell.paragraphs[0]
    runs = p.runs
    runs[0].text = text
    for r in runs[1:]:
        r._element.getparent().remove(r._element)
    for extra in cell.paragraphs[1:]:
        extra._element.getparent().remove(extra._element)

# опечатка в отчестве
fixed = 0
for p in d.paragraphs:
    for r in p.runs:
        if "Геогиевич" in r.text:
            r.text = r.text.replace("Геогиевич", "Георгиевич"); fixed += 1
d.save(out)
print("saved", out, "rows changed:", len(NEW), "typo fixed:", fixed)
for i, r in enumerate(t.rows):
    print(i, r.cells[0].text.strip(), "|", r.cells[1].text.strip()[:110])
