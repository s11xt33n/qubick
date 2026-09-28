"""Обучение и оценка моделей."""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn

from .data import Split


def classification_metrics(y_true, y_pred) -> dict:
    kw = dict(average="macro", zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, **kw),
        "recall": recall_score(y_true, y_pred, **kw),
        "f1": f1_score(y_true, y_pred, **kw),
        # доля классов, которые модель вообще предсказывает (выявляет «схлопывание»)
        "class_coverage": len(np.unique(y_pred)) / len(np.unique(y_true)),
    }


@torch.no_grad()
def predict(model: nn.Module, X: torch.Tensor, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    out = [model(X[i:i + batch_size]).argmax(1) for i in range(0, len(X), batch_size)]
    return torch.cat(out).cpu().numpy()


def fit(model: nn.Module, data: Split, epochs: int = 300, batch_size: int = 32,
        lr: float = 0.01, weight_decay: float = 0.0, patience: int = 30,
        seed: int = 0, device: str = "cpu", log_history: bool = False,
        restore_best: bool = True) -> dict:
    """Adam + кросс-энтропия, ранняя остановка по loss на валидации.

    Возвращаются метрики на тесте для весов с лучшей валидацией (restore_best=False —
    для весов после последней эпохи), время обучения, число эпох и (опционально)
    история по эпохам.
    """
    torch.manual_seed(seed)
    model.to(device)
    t = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=device)
    Xtr, ytr = t(data.X_train), t(data.y_train, torch.long)
    Xva, yva = t(data.X_val), t(data.y_val, torch.long)
    Xte = t(data.X_test)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(seed)
    best, best_state, bad, history = float("inf"), None, 0, []

    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr), generator=gen).to(device)
        tr_loss = 0.0
        for i in range(0, len(Xtr), batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) < 2:  # BatchNorm не обучается на батче из одного объекта
                continue
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            logits = model(Xva)
            va_loss = loss_fn(logits, yva).item()
            va_acc = (logits.argmax(1) == yva).float().mean().item()
        if log_history:
            history.append({"epoch": epoch, "train_loss": tr_loss / len(Xtr),
                            "val_loss": va_loss, "val_acc": va_acc})
        if va_loss < best - 1e-4:
            best, best_state, bad = va_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    train_time = time.perf_counter() - t0

    if restore_best:
        model.load_state_dict(best_state)
    y_pred = predict(model, Xte)
    train_acc = float((predict(model, Xtr) == data.y_train).mean())
    return {
        **classification_metrics(data.y_test, y_pred),
        "train_accuracy": train_acc,
        "val_loss": best,
        "epochs": epoch,
        "train_time": train_time,
        "time_per_epoch": train_time / epoch,
        "history": history,
    }
