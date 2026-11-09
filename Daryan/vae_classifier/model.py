"""
model.py - joint VAE + classifier.

    image x --encoder--> (mu, logvar) --reparameterize--> z --decoder--> reconstruction of x
                                                           \--classifier--> one "oil" logit

During training z is sampled (z = mu + sigma * eps). In eval mode z = mu, so predictions
are deterministic.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class VAEClassifier(nn.Module):
    def __init__(
        self,
        img_size: int = 128,
        latent_dim: int = 32,
        channels: tuple[int, ...] = (32, 64, 128, 256),
        cls_hidden: int = 64,
        cls_dropout: float = 0.0,
        pool_features: bool = True,
    ):
        super().__init__()
        n_down = len(channels)
        if img_size % (2 ** n_down) != 0:
            raise ValueError(f"img_size ({img_size}) must be divisible by 2^{n_down} = {2 ** n_down}")

        # Saved in checkpoints so the model can be rebuilt without the training command line.
        self.kwargs = dict(
            img_size=img_size, latent_dim=latent_dim, channels=list(channels),
            cls_hidden=cls_hidden, cls_dropout=cls_dropout, pool_features=pool_features,
        )
        self.pool_features = pool_features
        self.latent_dim = latent_dim
        self.feat_ch = channels[-1]
        self.feat_hw = img_size // (2 ** n_down)
        flat = self.feat_ch * self.feat_hw * self.feat_hw
        # The mu / logvar heads see the flattened feature map (spatial layout, needed by the
        # decoder) plus, by default, its global average and max over positions. The pooled
        # part tells z "is there a slick-like feature anywhere", which a flatten-only head has
        # to learn separately for every position (slow and data-hungry on ~4k images).
        head_in = flat + (2 * self.feat_ch if pool_features else 0)

        # Encoder: each stride-2 conv halves the height and width.
        enc, in_ch = [], 1
        for ch in channels:
            enc += [nn.Conv2d(in_ch, ch, 4, stride=2, padding=1), nn.BatchNorm2d(ch), nn.LeakyReLU(0.2)]
            in_ch = ch
        self.encoder = nn.Sequential(*enc)
        self.fc_mu = nn.Linear(head_in, latent_dim)
        self.fc_logvar = nn.Linear(head_in, latent_dim)

        # Decoder: mirror of the encoder. Outputs logits; sigmoid gives the image in [0, 1].
        self.dec_fc = nn.Sequential(nn.Linear(latent_dim, flat), nn.LeakyReLU(0.2))
        dec, rev = [], list(reversed(channels))
        for c_in, c_out in zip(rev[:-1], rev[1:]):
            dec += [nn.ConvTranspose2d(c_in, c_out, 4, stride=2, padding=1), nn.BatchNorm2d(c_out), nn.LeakyReLU(0.2)]
        dec.append(nn.ConvTranspose2d(rev[-1], 1, 4, stride=2, padding=1))
        self.decoder = nn.Sequential(*dec)

        # Classifier head on the latent code.
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, cls_hidden), nn.LeakyReLU(0.2), nn.Dropout(cls_dropout), nn.Linear(cls_hidden, 1),
        )

    def encode(self, x):
        f = self.encoder(x)
        h = f.flatten(1)
        if self.pool_features:
            h = torch.cat([h, f.mean(dim=(2, 3)), f.amax(dim=(2, 3))], dim=1)
        # Clamp keeps exp(logvar) finite early in training; the range is far wider than what is used.
        return self.fc_mu(h), self.fc_logvar(h).clamp(-12.0, 8.0)

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def decode_logits(self, z):
        h = self.dec_fc(z).view(-1, self.feat_ch, self.feat_hw, self.feat_hw)
        return self.decoder(h)

    def forward(self, x, sample: bool | None = None, decode: bool = True):
        """Returns a dict with mu, logvar, z, logit (shape [B]) and recon_logits (or None)."""
        if sample is None:
            sample = self.training
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) if sample else mu
        return {
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "logit": self.classifier(z).squeeze(1),
            "recon_logits": self.decode_logits(z) if decode else None,
        }


def load_checkpoint(path, device="cpu"):
    """Rebuild a VAEClassifier from a checkpoint written by train.py. Returns (model, checkpoint dict)."""
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # very old torch without weights_only
        ckpt = torch.load(path, map_location=device)
    model = VAEClassifier(**ckpt["model_kwargs"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt
