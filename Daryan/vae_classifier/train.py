"""
train.py - train the joint VAE + classifier on the train split, select on the val split.

The test split is never loaded here; use evaluate.py for the final test evaluation.

Example (run from the repo root):
    python Daryan/vae_classifier/train.py --epochs 20
    python Daryan/vae_classifier/train.py --smoke-test
"""
from __future__ import annotations

import argparse
import csv
import platform
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data import (  # noqa: E402
    DEFAULT_SPLIT_FILE, REPO_ROOT, OilChips, SplitError, display_path, load_split, split_entries,
    stratified_subset,
)
from engine import active_units, predict, term_grad_norms  # noqa: E402
from losses import JointLoss  # noqa: E402
from model import VAEClassifier, load_checkpoint  # noqa: E402
from utils import binary_metrics, cross_entropy, pick_device, save_json, seed_everything, to_python  # noqa: E402

# Default loss weights (see README, "Choosing the loss weights").
DEFAULT_RECON_WEIGHT = {"bce": 1.0, "mse": 2.0}
DEFAULT_BETA = 1.0
DEFAULT_LAMBDA = 1000.0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("data")
    g.add_argument("--split-file", default=DEFAULT_SPLIT_FILE,
                   help="teammate's split JSON (relative to the current folder or the repo root)")
    g.add_argument("--img-size", type=int, default=128, help="images are resized to img_size x img_size")
    g.add_argument("--no-augment", action="store_true", help="disable flips / 90-degree rotations")
    g = p.add_argument_group("model")
    g.add_argument("--latent-dim", type=int, default=32)
    g.add_argument("--channels", default="32,64,128,256", help="encoder conv channels, comma-separated")
    g.add_argument("--cls-hidden", type=int, default=64)
    g.add_argument("--cls-dropout", type=float, default=0.0)
    g.add_argument("--no-pool-features", action="store_true",
                   help="mu/logvar heads see only the flattened feature map (no global avg/max pooling)")
    g = p.add_argument_group("loss: recon_weight * recon + beta * KL + lambda * BCE")
    g.add_argument("--recon-loss", choices=["bce", "mse"], default="bce")
    g.add_argument("--recon-weight", type=float, default=None,
                   help="0 gives a variational information bottleneck (no decoder). "
                        f"Default: {DEFAULT_RECON_WEIGHT['bce']} for bce, {DEFAULT_RECON_WEIGHT['mse']} for mse")
    g.add_argument("--beta", type=float, default=DEFAULT_BETA, help="target KL weight after warm-up")
    g.add_argument("--lam", "--lambda", dest="lam", type=float, default=DEFAULT_LAMBDA,
                   help="classification loss weight")
    g.add_argument("--kl-warmup-epochs", type=float, default=3,
                   help="beta ramps linearly from 0 to --beta over this many epochs (0 = no warm-up)")
    g = p.add_argument_group("optimisation")
    g.add_argument("--lr", type=float, default=3e-4)
    g.add_argument("--weight-decay", type=float, default=0.0)
    g.add_argument("--batch-size", type=int, default=64)
    g.add_argument("--epochs", type=int, default=30)
    g.add_argument("--patience", type=int, default=8,
                   help="stop after this many epochs without a better val F1 (counted after warm-up)")
    g = p.add_argument_group("misc")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    g.add_argument("--runs-dir", default=str(HERE / "runs"))
    g.add_argument("--run-name", default=None, help="default: timestamp")
    g.add_argument("--threads", type=int, default=8, help="threads used to decode images")
    g.add_argument("--smoke-test", action="store_true",
                   help="~200 train / 100 val images, 2 epochs: checks the whole pipeline quickly")
    args = p.parse_args(argv)

    if args.recon_weight is None:
        args.recon_weight = DEFAULT_RECON_WEIGHT[args.recon_loss]
    args.channels = [int(c) for c in str(args.channels).split(",") if c.strip()]
    if args.smoke_test:
        args.epochs = min(args.epochs, 2)
        args.kl_warmup_epochs = min(args.kl_warmup_epochs, 1)
    for name in ("recon_weight", "beta", "lam", "kl_warmup_epochs"):
        if getattr(args, name) < 0:
            p.error(f"--{name.replace('_', '-')} must be >= 0")
    return args


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                              text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


# ---------------------------------------------------------------- training loop

def train_one_epoch(model, loader, loss_fn, opt, device, beta_at, step0):
    model.train()
    sums, n, correct, gnorms, beta = defaultdict(float), 0, 0, {}, 0.0
    for b, (x, y, _) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        beta = beta_at(step0 + b)
        out = model(x, decode=loss_fn.uses_decoder)
        total, terms = loss_fn(out, x, y, beta)
        if b == 0:  # diagnostic: which term drives the encoder (one batch per epoch)
            gnorms = term_grad_norms(model, terms)
        opt.zero_grad(set_to_none=True)
        total.backward()
        opt.step()
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite loss at step {step0 + b}: { {k: float(v) for k, v in terms.items()} }")
        bs = len(y)
        n += bs
        for k, v in terms.items():
            sums[k] += float(v) * bs
        correct += int(((out["logit"] > 0).float() == y).sum())
    stats = {k: v / n for k, v in sums.items()}
    stats["acc"] = correct / n
    return stats, gnorms, beta, step0 + len(loader)


def plot_curves(rows, path, warmup_epochs):
    ep = [r["epoch"] for r in rows]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.ravel()

    def pair(ax, key, title, logy=False):
        ax.plot(ep, [r[f"train_{key}"] for r in rows], "o-", ms=3, label="train")
        ax.plot(ep, [r[f"val_{key}"] for r in rows], "s-", ms=3, label="val")
        ax.set_title(title)
        if logy:
            ax.set_yscale("log")
        ax.legend()

    pair(axes[0], "total", "total loss")
    for k, lab in (("w_recon", "recon_weight*recon"), ("w_kl", "beta*KL"), ("w_cls", "lambda*BCE")):
        axes[1].plot(ep, [max(r[f"train_{k}"], 1e-6) for r in rows], "o-", ms=3, label=lab)
    axes[1].set_yscale("log")
    axes[1].set_title("weighted terms (train)")
    axes[1].legend()
    pair(axes[2], "recon", "reconstruction (per image)")
    pair(axes[3], "kl", "KL (nats per image)")
    ax2 = axes[3].twinx()
    ax2.plot(ep, [r["beta"] for r in rows], "k--", lw=1, label="beta")
    ax2.set_ylabel("beta")
    pair(axes[4], "cls", "classification BCE (pos-weighted)")
    axes[5].plot(ep, [r["train_acc"] for r in rows], "o-", ms=3, label="train acc")
    axes[5].plot(ep, [r["val_acc"] for r in rows], "s-", ms=3, label="val acc")
    axes[5].plot(ep, [r["val_f1"] for r in rows], "^-", ms=3, label="val F1")
    axes[5].set_title("accuracy / F1")
    axes[5].legend()
    for k, lab in (("gnorm_recon", "recon"), ("gnorm_kl", "KL"), ("gnorm_cls", "BCE")):
        axes[6].plot(ep, [max(r[k], 1e-8) for r in rows], "o-", ms=3, label=lab)
    axes[6].set_yscale("log")
    axes[6].set_title("encoder gradient norm per weighted term")
    axes[6].legend()
    axes[7].plot(ep, [r["val_active_units"] for r in rows], "o-", ms=3)
    axes[7].set_title("active latent units (val)")
    for ax in axes:
        ax.set_xlabel("epoch")
        if warmup_epochs > 0:
            ax.axvline(warmup_epochs, color="grey", lw=0.8, ls=":")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    seed_everything(args.seed)
    device = pick_device(args.device)

    try:
        split, split_info = load_split(args.split_file)
    except SplitError as e:
        sys.exit(f"ERROR: {e}")
    train_items, val_items = split["train"], split["val"]  # test split deliberately unused
    if args.smoke_test:
        train_items = stratified_subset(train_items, 200, args.seed)
        val_items = stratified_subset(val_items, 100, args.seed)

    n_pos = sum(l for _, l in train_items)
    n_neg = len(train_items) - n_pos
    pos_weight = n_neg / n_pos

    run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S") + ("_smoke" if args.smoke_test else "")
    run_dir = Path(args.runs_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    print(f"device={device}  run_dir={display_path(run_dir)}")
    print(f"split={split_info['path']} sha256={split_info['sha256'][:12]}  "
          f"train={len(train_items)} ({n_pos} oil)  val={len(val_items)}  pos_weight={pos_weight:.3f}")

    t0 = time.time()
    train_ds = OilChips(split_entries(train_items), args.img_size, augment=not args.no_augment, threads=args.threads)
    val_ds = OilChips(split_entries(val_items), args.img_size, threads=args.threads)
    print(f"loaded images in {time.time() - t0:.1f}s")
    gen = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=gen)
    val_loader = DataLoader(val_ds, batch_size=256)

    model = VAEClassifier(args.img_size, args.latent_dim, tuple(args.channels), args.cls_hidden, args.cls_dropout,
                          pool_features=not args.no_pool_features).to(device)
    loss_fn = JointLoss(args.recon_loss, args.recon_weight, args.beta, args.lam, pos_weight).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = len(train_loader)
    warmup_steps = int(round(args.kl_warmup_epochs * steps_per_epoch))

    def beta_at(step):
        return args.beta if warmup_steps == 0 else args.beta * min(1.0, (step + 1) / warmup_steps)

    config = {
        "args": vars(args),
        "run_dir": display_path(run_dir),
        "device": str(device),
        "split_file": split_info["path"],
        "split_sha256": split_info["sha256"],
        "split_counts": split_info["counts"],
        "n_train_used": len(train_items),
        "n_val_used": len(val_items),
        "pos_weight": pos_weight,
        "model_kwargs": model.kwargs,
        "n_params": sum(p.numel() for p in model.parameters()),
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
        "platform": platform.platform(),
        "git_commit": git_commit(),
        "started": datetime.now().isoformat(timespec="seconds"),
    }
    save_json(config, run_dir / "config.json")

    csv_path, ckpt_path = run_dir / "metrics.csv", run_dir / "best.pt"
    rows, best, bad_epochs, step = [], None, 0, 0
    print(f"{'ep':>3} {'beta':>5} | {'train: recon':>12} {'KL':>7} {'BCE':>6} {'acc':>5} | "
          f"{'val: recon':>10} {'KL':>7} {'BCE':>6} {'acc':>5} {'F1':>5} {'AU':>3} | {'grad r/k/c':>20} {'s':>4}")
    for epoch in range(1, args.epochs + 1):
        t_ep = time.time()
        tr, gn, beta_now, step = train_one_epoch(model, train_loader, loss_fn, opt, device, beta_at, step)
        # Validation uses the target beta so val totals are comparable across epochs.
        va = predict(model, val_loader, device, loss_fn, beta=args.beta, keep_mu=True)
        vm = binary_metrics(va["labels"], va["probs"])
        row = {
            "epoch": epoch, "beta": beta_now, "lr": opt.param_groups[0]["lr"],
            **{f"train_{k}": v for k, v in tr.items()},
            **{f"val_{k}": v for k, v in va["loss"].items()},
            "val_acc": vm["accuracy"], "val_precision": vm["precision"], "val_recall": vm["recall"],
            "val_f1": vm["f1"], "val_roc_auc": vm["roc_auc"], "val_ce_nats": cross_entropy(va["labels"], va["probs"])["ce_nats"],
            "val_active_units": active_units(va["mu"]),
            "gnorm_recon": gn["w_recon"], "gnorm_kl": gn["w_kl"], "gnorm_cls": gn["w_cls"],
            "seconds": time.time() - t_ep,
        }
        rows.append(row)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            w.writeheader()
            w.writerows(rows)
        plot_curves(rows, run_dir / "loss_curves.png", args.kl_warmup_epochs)

        # Model selection on val F1, only once beta has reached its target.
        msg = ""
        if epoch >= args.kl_warmup_epochs:
            better = best is None or vm["f1"] > best["val_f1"] + 1e-4 or (
                abs(vm["f1"] - best["val_f1"]) <= 1e-4 and va["loss"]["total"] < best["val_total"])
            if better:
                best = {"epoch": epoch, "val_f1": vm["f1"], "val_acc": vm["accuracy"], "val_total": va["loss"]["total"],
                        "val_prob_mean": float(va["probs"].mean())}
                torch.save(to_python({
                    "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "model_kwargs": model.kwargs,
                    "epoch": epoch,
                    "val_metrics": {k: v for k, v in vm.items()},
                    "best": best,
                    "config": config,
                }), ckpt_path)
                bad_epochs, msg = 0, "  * saved"
            else:
                bad_epochs += 1
        else:
            msg = "  (warm-up)"
        print(f"{epoch:>3} {beta_now:>5.2f} | {tr['recon']:>12.1f} {tr['kl']:>7.1f} {tr['cls']:>6.3f} {tr['acc']:>5.3f} | "
              f"{va['loss']['recon']:>10.1f} {va['loss']['kl']:>7.1f} {va['loss']['cls']:>6.3f} {vm['accuracy']:>5.3f} "
              f"{vm['f1']:>5.3f} {row['val_active_units']:>3} | "
              f"{gn['w_recon']:>6.2g}/{gn['w_kl']:>6.2g}/{gn['w_cls']:>6.2g} {row['seconds']:>4.0f}{msg}", flush=True)
        if bad_epochs >= args.patience:
            print(f"early stopping: no val F1 improvement for {args.patience} epochs")
            break

    if best is None:
        sys.exit("ERROR: no checkpoint saved (training ended before KL warm-up finished; "
                 "raise --epochs or lower --kl-warmup-epochs)")

    # Reload the saved checkpoint into a fresh model and check it reproduces the val results.
    reloaded, ckpt = load_checkpoint(ckpt_path, device)
    rv = predict(reloaded, val_loader, device)
    rm = binary_metrics(rv["labels"], rv["probs"])
    ok = abs(rm["f1"] - best["val_f1"]) < 1e-3 and abs(float(rv["probs"].mean()) - best["val_prob_mean"]) < 1e-3
    print(f"checkpoint reload check: {'OK' if ok else 'MISMATCH'} "
          f"(val F1 {rm['f1']:.4f} vs {best['val_f1']:.4f} at save time)")

    config.update({"best": best, "epochs_run": len(rows), "finished": datetime.now().isoformat(timespec="seconds"),
                   "reload_check_ok": ok})
    save_json(config, run_dir / "config.json")
    print(f"best epoch {best['epoch']}: val F1 {best['val_f1']:.4f}, val acc {best['val_acc']:.4f}")
    print(f"checkpoint: {display_path(ckpt_path)}")
    if not ok:
        sys.exit("ERROR: reloaded checkpoint does not reproduce the saved validation results")
    return run_dir


if __name__ == "__main__":
    main()
