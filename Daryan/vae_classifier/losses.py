"""
losses.py - the joint objective.

    L = recon_weight * Recon(x, x_hat) + beta * KL(q(z|x) || N(0, I)) + lambda * BCE(logit, y)

Units: every term is per image, averaged over the batch.
  - Recon (bce): pixel-wise binary cross-entropy summed over all pixels, in nats per image.
  - Recon (mse): squared error summed over all pixels, per image.
  - KL: summed over the latent dimensions, in nats per image.
  - BCE: classification cross-entropy with pos_weight, in nats per image.
With recon_weight = 1, beta = 1 and bce reconstruction, the first two terms are the usual
negative ELBO. With recon_weight = 0 the decoder is never run, and the objective is a
variational information bottleneck: lambda * CE + beta * KL.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def recon_loss(recon_logits, x, kind: str = "bce"):
    """Per-image reconstruction loss, shape [B]."""
    if kind == "bce":
        per_pixel = F.binary_cross_entropy_with_logits(recon_logits, x, reduction="none")
    elif kind == "mse":
        per_pixel = (torch.sigmoid(recon_logits) - x) ** 2
    else:
        raise ValueError(f"unknown reconstruction loss {kind!r}")
    return per_pixel.flatten(1).sum(1)


def kl_divergence(mu, logvar):
    """KL(N(mu, sigma^2) || N(0, I)) per image, shape [B], in nats."""
    return 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(1)


class JointLoss(nn.Module):
    def __init__(self, recon_kind="bce", recon_weight=1.0, beta=1.0, lam=1.0, pos_weight=1.0):
        super().__init__()
        self.recon_kind = recon_kind
        self.recon_weight = float(recon_weight)
        self.beta = float(beta)  # target beta; the current (warm-up) beta is passed to forward()
        self.lam = float(lam)
        self.register_buffer("pos_weight", torch.tensor([float(pos_weight)]))

    @property
    def uses_decoder(self) -> bool:
        return self.recon_weight > 0

    def forward(self, out, x, y, beta=None):
        """Returns (total, terms). `terms` holds the raw and weighted batch means as tensors."""
        beta = self.beta if beta is None else float(beta)
        if self.uses_decoder:
            recon = recon_loss(out["recon_logits"], x, self.recon_kind).mean()
        else:
            recon = x.new_zeros(())
        kl = kl_divergence(out["mu"], out["logvar"]).mean()
        cls = F.binary_cross_entropy_with_logits(out["logit"], y, pos_weight=self.pos_weight)

        w_recon, w_kl, w_cls = self.recon_weight * recon, beta * kl, self.lam * cls
        total = w_recon + w_kl + w_cls
        terms = {
            "recon": recon, "kl": kl, "cls": cls,
            "w_recon": w_recon, "w_kl": w_kl, "w_cls": w_cls,
            "total": total,
        }
        return total, terms
