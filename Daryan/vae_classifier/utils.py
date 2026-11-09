"""utils.py - seeding, device selection, metrics and small I/O helpers."""
from __future__ import annotations

import json
import math
import os
import random

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, precision_recall_curve,
    precision_recall_fscore_support, roc_auc_score,
)

LN2 = math.log(2.0)


def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds CUDA and MPS generators
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def binary_metrics(y_true, prob, threshold: float = 0.5) -> dict:
    """Accuracy, precision, recall, F1 (oil = positive class), ROC-AUC, PR-AUC and the
    confusion matrix [[TN, FP], [FN, TP]]. AUCs are NaN if only one class is present."""
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=np.float64)
    pred = (prob >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    both = len(np.unique(y_true)) == 2
    return {
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(y_true, prob)) if both else float("nan"),
        "pr_auc": float(average_precision_score(y_true, prob)) if both else float("nan"),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),
    }


def cross_entropy(y_true, prob) -> dict:
    """Mean binary cross-entropy -log q(true label) without class weights, in nats and bits."""
    y = np.asarray(y_true, dtype=np.float64)
    q = np.clip(np.asarray(prob, dtype=np.float64), 1e-7, 1 - 1e-7)
    nats = float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))
    return {"ce_nats": nats, "ce_bits": nats / LN2}


def best_f1_threshold(y_true, prob) -> tuple[float, float]:
    """Threshold that maximises F1 (to be computed on the validation split only)."""
    y_true = np.asarray(y_true).astype(int)
    if y_true.sum() == 0:
        return 0.5, 0.0
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    i = int(np.argmax(f1))
    return float(thresholds[i]), float(f1[i])


def to_python(obj):
    """Convert numpy scalars and arrays so they serialise to JSON (and load with weights_only=True)."""
    if isinstance(obj, dict):
        return {str(k): to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_python(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (str, os.PathLike)):  # also str subclasses such as torch.__version__
        return str(obj)
    return obj


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_python(obj), f, indent=2)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
