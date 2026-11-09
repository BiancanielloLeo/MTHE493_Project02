"""engine.py - run a model over a DataLoader (shared by train.py and evaluate.py)."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch


@torch.no_grad()
def predict(model, loader, device, loss_fn=None, beta=None, keep_mu=False):
    """Eval-mode pass (z = mu). Returns dict with labels, probs, idx, optionally mu,
    and per-image means of each loss term when loss_fn is given."""
    model.eval()
    decode = loss_fn is not None and loss_fn.uses_decoder
    labels, probs, idx, mus = [], [], [], []
    sums, n = defaultdict(float), 0
    for x, y, i in loader:
        x, y = x.to(device), y.to(device)
        out = model(x, sample=False, decode=decode)
        if loss_fn is not None:
            _, terms = loss_fn(out, x, y, beta)
            for k, v in terms.items():
                sums[k] += float(v) * len(y)
        n += len(y)
        labels.append(y.cpu())
        probs.append(torch.sigmoid(out["logit"]).float().cpu())
        idx.append(i)
        if keep_mu:
            mus.append(out["mu"].float().cpu())
    res = {
        "labels": torch.cat(labels).numpy().astype(int),
        "probs": torch.cat(probs).numpy().astype(np.float64),
        "idx": torch.cat(idx).numpy(),
        "loss": {k: v / max(n, 1) for k, v in sums.items()},
    }
    if keep_mu:
        res["mu"] = torch.cat(mus).numpy()
    return res


def active_units(mu: np.ndarray, threshold: float = 0.01) -> int:
    """Number of latent dimensions whose posterior mean varies across inputs
    (Var_x[mu_d] > threshold). A low count means posterior collapse."""
    return int((mu.var(axis=0) > threshold).sum())


def encoder_params(model):
    return [p for m in (model.encoder, model.fc_mu, model.fc_logvar) for p in m.parameters()]


def term_grad_norms(model, terms) -> dict:
    """L2 norm of the gradient of each weighted loss term with respect to the shared encoder.
    Shows which term actually drives the encoder, independent of the raw loss magnitudes."""
    params = encoder_params(model)
    out = {}
    for name in ("w_recon", "w_kl", "w_cls"):
        t = terms[name]
        if not t.requires_grad:
            out[name] = 0.0
            continue
        grads = torch.autograd.grad(t, params, retain_graph=True, allow_unused=True)
        sq = [(g.float() ** 2).sum() for g in grads if g is not None]
        out[name] = float(torch.stack(sq).sum().sqrt()) if sq else 0.0
    return out
