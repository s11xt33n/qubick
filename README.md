# qhnn — гибридные квантово-классические нейронные сети на PyTorch

Библиотека для построения, обучения и исследования гибридных
квантово-классических нейронных сетей. Разработана в рамках НИР
магистранта КФУ ИВМиИТ (гр. 09-535, «Машинное обучение и компьютерное зрение»).

## Возможности

- **Собственный батчевый симулятор вектора состояния** на PyTorch
  (`qhnn/simulator.py`): все объекты батча исполняются одновременно,
  работает на CPU и GPU, градиенты — через autograd.
- **`QuantumLayer`** — обучаемый квантовый слой как обычный `nn.Module`:
  выбор кодирования (`angle`, `angle_x`, `dense`), анзаца (`strong`, `basic`,
  `hea`), data re-uploading, числа кубитов и слоёв.
- **Два способа вычисления градиентов**: `backprop` и `parameter-shift`
  (применим на реальном квантовом устройстве), режим конечного числа
  измерений `shots`.
- **Два бэкенда** с одним интерфейсом: собственный (`torch`) и `pennylane`
  (эталон для проверки корректности).
- **Экспорт схемы в OpenQASM 2.0** (`qhnn.to_qasm`) для запуска на
  реальном оборудовании (IBM Quantum и др.).
- **Модели для честного сравнения**: `classical`, `classical_matched`
  (MLP с тем же числом параметров), `bottleneck` (гибрид без квантовой
  части — абляция), `quantum`, `hybrid`.
- **Воспроизводимые эксперименты** по YAML-конфигам: серия запусков
  с разными seed, параллельное выполнение, продолжение прерванной серии,
  сводные таблицы, тест Уилкоксона, графики.

## Пример

```python
import torch
from torch import nn
from qhnn import QuantumLayer

model = nn.Sequential(
    nn.Linear(30, 4), nn.Tanh(),            # классический encoder
    QuantumLayer(n_qubits=4, n_layers=2),   # квантовый слой: 4 -> 4
    nn.Linear(4, 2),                        # классический head
)
logits = model(torch.randn(8, 30))
logits.sum().backward()                    # градиенты проходят через квантовый слой
```

## Установка

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126  # GPU; для CPU: .../whl/cpu
.venv/Scripts/python -m pip install -e ".[pennylane,vision,dev]"
.venv/Scripts/python -m pytest
```

## Эксперименты

| Конфиг / скрипт | Что исследуется |
|---|---|
| `configs/tabular.yaml` | 5 моделей × 5 табличных наборов × 10 seed |
| `configs/sweep.yaml` | влияние числа кубитов (2–8) и глубины (1–4) |
| `configs/ablation.yaml` | кодирование × анзац × re-uploading |
| `configs/shots.yaml` | обучение при конечном числе измерений |
| `configs/vision.yaml` | ResNet18 + голова, режим малых данных (MNIST, Fashion-MNIST, PneumoniaMNIST) |
| `qhnn.experiments.barren` | barren plateaus: Var[∂C/∂θ] от числа кубитов |
| `qhnn.experiments.speed` | скорость симулятора против PennyLane |

```bash
python -m qhnn.experiments.runner configs/tabular.yaml --workers 8
python -m qhnn.features mnist fashion pneumonia --device cuda --n-train 10000 --n-test 2000
python -m qhnn.experiments.runner configs/vision.yaml --workers 30
python -m qhnn.experiments.barren --device cuda
python -m qhnn.experiments.speed --devices cpu cuda
python -m qhnn.experiments.analyze        # таблицы -> results/tables, рисунки -> results/figures
```

## Структура

```
qhnn/
  circuits.py     описание схем: кодирование, анзацы, экспорт в OpenQASM
  simulator.py    батчевый симулятор вектора состояния на PyTorch
  backends.py     бэкенды torch / pennylane, parameter-shift, shots
  layers.py       QuantumLayer
  models.py       модели для сравнения
  data.py         загрузка и подготовка данных
  features.py     извлечение признаков изображений (ResNet18)
  training.py     обучение с ранней остановкой, метрики
  experiments/    запуск серий, barren plateaus, скорость, анализ
configs/          YAML-конфиги экспериментов
tests/            тесты (сверка с PennyLane, parameter-shift = backprop и др.)
```
