"""Извлечение признаков изображений предобученной CNN (hybrid transfer learning).

Схема из Mari et al., 2020 («Transfer learning in hybrid classical-quantum
neural networks»): замороженная ResNet18, обученная на ImageNet, превращает
изображение в вектор из 512 признаков, а обучается только «голова» —
классическая, квантовая или гибридная.

Признаки считаются один раз и сохраняются в results/features/<name>.npz.

    python -m qhnn.features mnist fashion pneumonia --device cuda
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .data import FEATURES_DIR

DATA_ROOT = FEATURES_DIR.parent / "raw"


def _dataset(name: str, train: bool, tf):
    from torchvision import datasets
    if name == "mnist":
        return datasets.MNIST(DATA_ROOT, train=train, download=True, transform=tf)
    if name == "fashion":
        return datasets.FashionMNIST(DATA_ROOT, train=train, download=True, transform=tf)
    if name in ("pneumonia", "breast", "blood", "derma", "organa"):
        import medmnist
        cls = {"pneumonia": medmnist.PneumoniaMNIST, "breast": medmnist.BreastMNIST,
               "blood": medmnist.BloodMNIST, "derma": medmnist.DermaMNIST,
               "organa": medmnist.OrganAMNIST}[name]
        DATA_ROOT.mkdir(parents=True, exist_ok=True)
        return cls(split="train" if train else "test", transform=tf, download=True,
                   root=str(DATA_ROOT))
    raise ValueError(name)


def _subset(ds, n, seed):
    if n is None or n >= len(ds):
        return ds
    idx = np.random.default_rng(seed).permutation(len(ds))[:n]
    return Subset(ds, idx.tolist())


@torch.no_grad()
def extract(name: str, device: str = "cpu", size: int = 224, n_train=None, n_test=None,
            batch_size: int = 256, seed: int = 0) -> None:
    from torchvision import models, transforms

    weights = models.ResNet18_Weights.IMAGENET1K_V1
    net = models.resnet18(weights=weights)
    net.fc = torch.nn.Identity()  # берём 512-мерный вектор перед классификатором
    net.eval().to(device)
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    out = {}
    for split, n in (("train", n_train), ("test", n_test)):
        ds = _subset(_dataset(name, split == "train", tf), n, seed)
        dl = DataLoader(ds, batch_size=batch_size, num_workers=0)
        feats, labels, t0 = [], [], time.time()
        for x, y in dl:
            feats.append(net(x.to(device)).cpu())
            labels.append(torch.as_tensor(y).reshape(-1))
        out[f"X_{split}"] = torch.cat(feats).numpy().astype(np.float32)
        out[f"y_{split}"] = torch.cat(labels).numpy().astype(np.int64)
        print(f"{name}/{split}: {len(out[f'y_{split}'])} изображений за {time.time() - t0:.0f} с")
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FEATURES_DIR / f"{name}.npz", **out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-test", type=int, default=None)
    a = ap.parse_args()
    for n in a.names:
        extract(n, a.device, a.size, a.n_train, a.n_test)


if __name__ == "__main__":
    main()
