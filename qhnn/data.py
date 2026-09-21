"""Загрузка и подготовка данных.

Табличные наборы: iris, wine, breast_cancer, moons, circles.
Признаки изображений: vision:<name> — заранее извлечённые признаки
(см. qhnn/features.py), хранятся в results/features/<name>.npz.

Разбиение: стратифицированное train/val/test = 60/20/20 (для табличных),
seed управляет и разбиением, и инициализацией модели. Масштабирование и
PCA обучаются только на train, чтобы не было утечки информации из теста.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn import datasets as skd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURES_DIR = Path(__file__).resolve().parent.parent / "results" / "features"


@dataclass
class Split:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    n_classes: int
    note: str = ""

    @property
    def in_dim(self) -> int:
        return self.X_train.shape[1]


def load_tabular(name: str, seed: int = 0):
    if name == "iris":
        d = skd.load_iris(); return d.data, d.target
    if name == "wine":
        d = skd.load_wine(); return d.data, d.target
    if name == "breast_cancer":
        d = skd.load_breast_cancer(); return d.data, d.target
    if name == "moons":
        return skd.make_moons(n_samples=400, noise=0.25, random_state=seed)
    if name == "circles":
        return skd.make_circles(n_samples=400, noise=0.1, factor=0.5, random_state=seed)
    raise ValueError(f"Неизвестный набор данных: {name}")


def _subsample(X, y, n, seed):
    """Стратифицированная подвыборка из n объектов (режим малых данных)."""
    if n is None or n >= len(y):
        return X, y
    X, _, y, _ = train_test_split(X, y, train_size=n, stratify=y, random_state=seed)
    return X, y


def load_split(dataset: str, seed: int = 0, n_train: int | None = None,
               reduce_to: int | None = None) -> Split:
    """Возвращает подготовленное разбиение.

    n_train   — ограничить обучающую выборку (для экспериментов с малыми данными).
    reduce_to — привести размерность к заданной с помощью PCA (для чистой QNN,
                где число признаков должно совпадать с числом кубитов).
    """
    if dataset.startswith("vision:"):
        f = np.load(FEATURES_DIR / f"{dataset.split(':', 1)[1]}.npz")
        X_tr, y_tr, X_te, y_te = f["X_train"], f["y_train"], f["X_test"], f["y_test"]
        X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.2,
                                                  stratify=y_tr, random_state=seed)
    else:
        X, y = load_tabular(dataset, seed)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y,
                                                  random_state=seed)
        X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.25,
                                                  stratify=y_tr, random_state=seed)
    if n_train:
        X_tr, y_tr = _subsample(X_tr, y_tr, n_train, seed)
        n_val = max(len(np.unique(y_tr)) * 2, min(len(y_va), n_train // 4))
        X_va, y_va = _subsample(X_va, y_va, n_val, seed)

    scaler = StandardScaler().fit(X_tr)
    X_tr, X_va, X_te = (scaler.transform(a) for a in (X_tr, X_va, X_te))
    note = "standard scaling"
    if reduce_to and X_tr.shape[1] > reduce_to:
        pca = PCA(n_components=reduce_to, random_state=seed).fit(X_tr)
        X_tr, X_va, X_te = (pca.transform(a) for a in (X_tr, X_va, X_te))
        # после PCA снова нормируем, чтобы углы кодирования были сопоставимы
        s = StandardScaler().fit(X_tr)
        X_tr, X_va, X_te = (s.transform(a) for a in (X_tr, X_va, X_te))
        note += f"; PCA->{reduce_to} (explained var={pca.explained_variance_ratio_.sum():.3f})"

    n_classes = int(max(y_tr.max(), y_te.max()) + 1)
    f32 = lambda a: np.asarray(a, dtype=np.float32)
    i64 = lambda a: np.asarray(a, dtype=np.int64)
    return Split(f32(X_tr), i64(y_tr), f32(X_va), i64(y_va), f32(X_te), i64(y_te),
                 n_classes, note)
