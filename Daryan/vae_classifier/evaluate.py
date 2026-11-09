"""
evaluate.py - final evaluation of a trained checkpoint on the held-out test split.

Reports accuracy, precision, recall, F1, ROC-AUC, PR-AUC and mean cross-entropy (nats, bits)
at threshold 0.5 and at the threshold that maximises F1 on the validation split.
Also saves the confusion matrix, reconstructions, a 2-D view of the latent means, and
(optionally) metrics vs severity for simulated channel damage.

Examples (run from the repo root):
    python Daryan/vae_classifier/evaluate.py --checkpoint Daryan/vae_classifier/runs/<run>/best.pt
    python Daryan/vae_classifier/evaluate.py --checkpoint <...> --corruption all
    python Daryan/vae_classifier/evaluate.py --checkpoint <...> --data-dir path/to/decoded_images
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
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
    CORRUPTIONS, DEFAULT_SPLIT_FILE, OilChips, SplitError, corruption_levels, display_path, folder_items,
    load_split, remap_to_data_dir, resolve_user_path, split_entries, stratified_subset,
)
from engine import active_units, predict  # noqa: E402
from model import load_checkpoint  # noqa: E402
from utils import best_f1_threshold, binary_metrics, cross_entropy, pick_device, save_json, seed_everything  # noqa: E402

BANNER = "!" * 78


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="best.pt, or the run folder that contains it")
    p.add_argument("--split-file", default=DEFAULT_SPLIT_FILE)
    p.add_argument("--data-dir", default=None,
                   help="folder with 'Oil/' and 'No Oil/' subfolders (e.g. images after compression + channel). "
                        "Test-split images are looked up there by file name.")
    p.add_argument("--all-images", action="store_true",
                   help="with --data-dir: evaluate every image in the folder instead of only the test split")
    p.add_argument("--corruption", default="none", choices=["none", "all", *CORRUPTIONS],
                   help="simulated channel damage applied on the fly at several severities")
    p.add_argument("--out-dir", default=None, help="default: <run folder>/eval/<name>")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=0, help="seed for corruptions and plots")
    p.add_argument("--no-tsne", action="store_true", help="skip t-SNE (PCA is always drawn)")
    p.add_argument("--smoke-test", action="store_true", help="100 val / 100 test images only")
    args = p.parse_args(argv)
    if args.all_images and not args.data_dir:
        p.error("--all-images needs --data-dir")
    return args


# ---------------------------------------------------------------- plots

def plot_confusion(results, path):
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4.4))
    axes = np.atleast_1d(axes)
    for ax, (title, m) in zip(axes, results):
        cm = np.array(m["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v}\n({v / cm.sum():.1%})", ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black", fontsize=11)
        ax.set_xticks([0, 1], ["No Oil", "Oil"])
        ax.set_yticks([0, 1], ["No Oil", "Oil"])
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{title}\nacc {m['accuracy']:.3f}  F1 {m['f1']:.3f}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


@torch.no_grad()
def plot_reconstructions(model, ds, probs, device, path, seed, n_per_class=6):
    rng = np.random.default_rng(seed)
    labels = ds.labels.numpy().astype(int)
    picks = []
    for lab in (1, 0):
        idx = np.flatnonzero(labels == lab)
        picks.append(rng.choice(idx, size=min(n_per_class, len(idx)), replace=False))
    n_cols = max(len(p) for p in picks)
    fig, axes = plt.subplots(4, n_cols, figsize=(2 * n_cols, 8.6), squeeze=False)
    model.eval()
    for r, (lab, idx) in enumerate(zip(("Oil", "No Oil"), picks)):
        x = torch.stack([ds[i][0] for i in idx]).to(device)
        recon = torch.sigmoid(model(x, sample=False)["recon_logits"]).cpu()
        for c in range(n_cols):
            for k, img in ((0, x), (1, recon)):
                ax = axes[2 * r + k, c]
                ax.axis("off")
                if c < len(idx):
                    ax.imshow(img[c, 0].cpu(), cmap="gray", vmin=0, vmax=1)
                    if k == 0:
                        ax.set_title(f"{lab}  p(oil)={probs[idx[c]]:.2f}", fontsize=8)
        axes[2 * r, 0].text(-0.1, 0.5, "original", transform=axes[2 * r, 0].transAxes, rotation=90, va="center", ha="right")
        axes[2 * r + 1, 0].text(-0.1, 0.5, "reconstruction", transform=axes[2 * r + 1, 0].transAxes, rotation=90,
                                va="center", ha="right")
    fig.suptitle("Test images: original vs VAE reconstruction (z = mu)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


_TSNE_CODE = """
import sys, numpy as np
from sklearn.manifold import TSNE
mu = np.load(sys.argv[1]); seed = int(sys.argv[3])
kw = dict(n_components=2, init="pca", perplexity=min(30.0, (len(mu) - 1) / 3), random_state=seed)
try:
    ts = TSNE(learning_rate="auto", **kw)
except TypeError:
    ts = TSNE(learning_rate=200.0, **kw)
np.save(sys.argv[2], ts.fit_transform(mu))
"""


def tsne_in_subprocess(mu, seed):
    """t-SNE in a separate process that never imports torch. On some macOS/conda setups,
    torch's and scikit-learn's OpenMP libraries crash when loaded together."""
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "mu.npy", Path(tmp) / "tsne.npy"
        np.save(src, mu.astype(np.float64))
        r = subprocess.run([sys.executable, "-c", _TSNE_CODE, str(src), str(dst), str(seed)],
                           capture_output=True, text=True)
        if r.returncode != 0 or not dst.exists():
            print(f"t-SNE skipped (exit code {r.returncode}): {r.stderr.strip().splitlines()[-1:]}")
            return None
        return np.load(dst)


def plot_latent(mu, labels, path, seed, tsne=True):
    from sklearn.decomposition import PCA

    panels = []
    pca = PCA(n_components=2, random_state=seed)
    pts = pca.fit_transform(mu)
    ev = pca.explained_variance_ratio_
    panels.append((pts, f"PCA of test mu ({ev[0]:.0%} + {ev[1]:.0%} of variance)"))
    if tsne and len(mu) > 10:
        pts = tsne_in_subprocess(mu, seed)
        if pts is not None:
            panels.append((pts, "t-SNE of test mu"))
    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 5.5), squeeze=False)
    for ax, (pts, title) in zip(axes[0], panels):
        for lab, name, color in ((0, "No Oil", "#4C72B0"), (1, "Oil", "#DD8452")):
            m = labels == lab
            ax.scatter(pts[m, 0], pts[m, 1], s=8, alpha=0.6, c=color, label=f"{name} ({m.sum()})")
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_corruption_examples(entry, names, path, seed):
    from data import _load_gray  # full-resolution loader
    rows = [(n, corruption_levels(n)) for n in names]
    n_cols = max(len(lv) for _, lv in rows)
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(2.2 * n_cols, 2.4 * len(rows)), squeeze=False)
    img = _load_gray(entry[0])
    for r, (name, levels) in enumerate(rows):
        param = CORRUPTIONS[name][0]
        for c, (sev, val, fn) in enumerate(levels):
            ax = axes[r, c]
            out = img if fn is None else fn(img, np.random.default_rng([seed, 0]))
            ax.imshow(np.asarray(out), cmap="gray", vmin=0, vmax=255)
            ax.set_title("clean" if fn is None else f"{param}={val}", fontsize=8)
            ax.axis("off")
    fig.suptitle(f"corruption levels on {Path(entry[2]).name}")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_corruption_curves(rows, names, baseline_acc, path):
    fig, axes = plt.subplots(1, len(names), figsize=(5.2 * len(names), 4.2), squeeze=False)
    for ax, name in zip(axes[0], names):
        rs = [r for r in rows if r["corruption"] == name]
        sev = [r["severity"] for r in rs]
        ax.plot(sev, [r["accuracy"] for r in rs], "o-", label="accuracy (t=0.5)")
        ax.plot(sev, [r["f1"] for r in rs], "s-", label="F1 (t=0.5)")
        ax.plot(sev, [r["accuracy_valthr"] for r in rs], "o--", alpha=0.6, label="accuracy (val-opt t)")
        ax.plot(sev, [r["f1_valthr"] for r in rs], "s--", alpha=0.6, label="F1 (val-opt t)")
        ax.axhline(baseline_acc, color="grey", ls=":", label="always 'No Oil' accuracy")
        ax.set_xticks(sev, ["clean" if r["param_value"] in ("", None) else str(r["param_value"]) for r in rs])
        ax.set_xlabel(f"{CORRUPTIONS[name][0]} (severity 0-{len(sev) - 1})")
        ax.set_ylim(0, 1)
        ax.set_title(name)
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- main

def summarise(labels, probs, threshold):
    return {**binary_metrics(labels, probs, threshold), **cross_entropy(labels, probs)}


def main(argv=None):
    args = parse_args(argv)
    seed_everything(args.seed)
    device = pick_device(args.device)

    ckpt_path = resolve_user_path(args.checkpoint)
    if ckpt_path.is_dir():
        ckpt_path = ckpt_path / "best.pt"
    if not ckpt_path.is_file():
        sys.exit(f"ERROR: checkpoint not found: {ckpt_path}")
    model, ckpt = load_checkpoint(ckpt_path, device)
    cfg = ckpt["config"]
    img_size = ckpt["model_kwargs"]["img_size"]
    uses_decoder = cfg["args"]["recon_weight"] > 0

    try:
        split, split_info = load_split(args.split_file)
    except SplitError as e:
        sys.exit(f"ERROR: {e}")
    warnings_list = []
    if split_info["sha256"] != cfg.get("split_sha256"):
        msg = (f"WARNING: this checkpoint was trained with split {cfg.get('split_file')} "
               f"(sha256 {str(cfg.get('split_sha256'))[:12]}), but you are evaluating with {split_info['path']} "
               f"(sha256 {split_info['sha256'][:12]}). Test images may overlap the training data.")
        warnings_list.append(msg)
        print(f"{BANNER}\n{msg}\n{BANNER}")

    val_items, test_items = split["val"], split["test"]
    if args.smoke_test:
        val_items = stratified_subset(val_items, 100, args.seed)
        test_items = stratified_subset(test_items, 100, args.seed)

    if args.data_dir:
        data_dir = resolve_user_path(args.data_dir)
        if args.all_images:
            test_entries = folder_items(data_dir, split)
            if args.smoke_test:
                test_entries = stratified_subset([(e, e[1]) for e in test_entries], 100, args.seed)
                test_entries = [e for e, _ in test_entries]
            source = f"all images in {display_path(data_dir)}"
        else:
            test_entries = remap_to_data_dir(test_items, data_dir)
            source = f"test split, images from {display_path(data_dir)}"
        tag = ("all_" if args.all_images else "test_") + data_dir.name.replace(" ", "_")
    else:
        test_entries = split_entries(test_items)
        source = "test split, original images"
        tag = "test"
    if args.smoke_test:
        tag += "_smoke"
    out_dir = Path(args.out_dir) if args.out_dir else ckpt_path.parent / "eval" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  checkpoint={display_path(ckpt_path)} (epoch {ckpt['epoch']})")
    print(f"evaluating on {source}: {len(test_entries)} images -> {display_path(out_dir)}")

    # 1. Threshold from the clean validation split only (never from test).
    val_ds = OilChips(split_entries(val_items), img_size, threads=args.threads)
    val = predict(model, DataLoader(val_ds, batch_size=args.batch_size), device)
    val_thr, val_best_f1 = best_f1_threshold(val["labels"], val["probs"])

    # 2. Test split.
    test_ds = OilChips(test_entries, img_size, threads=args.threads)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)
    test = predict(model, test_loader, device, keep_mu=True)
    labels, probs = test["labels"], test["probs"]
    at_half = summarise(labels, probs, 0.5)
    at_val = summarise(labels, probs, val_thr)
    majority_acc = float((labels == 0).mean())

    plot_confusion([("threshold 0.5", at_half), (f"val-optimal threshold {val_thr:.3f}", at_val)],
                   out_dir / "confusion_matrix.png")
    with open(out_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "label", "prob_oil", "pred_t0.5", f"pred_t{val_thr:.4f}"])
        for k in range(len(labels)):
            w.writerow([test_ds.names[test["idx"][k]], labels[k], f"{probs[k]:.6f}",
                        int(probs[k] >= 0.5), int(probs[k] >= val_thr)])
    if uses_decoder:
        plot_reconstructions(model, test_ds, probs, device, out_dir / "reconstructions.png", args.seed)
    t0 = time.time()
    plot_latent(test["mu"], labels, out_dir / "latent_projection.png", args.seed, tsne=not args.no_tsne)
    latent_seconds = time.time() - t0

    # 3. Simulated channel damage.
    corr_rows = []
    names = list(CORRUPTIONS) if args.corruption == "all" else ([] if args.corruption == "none" else [args.corruption])
    for name in names:
        param = CORRUPTIONS[name][0]
        for sev, val_, fn in corruption_levels(name):
            if fn is None:
                p_ = probs
            else:
                ds = OilChips(test_entries, img_size, corruption=fn, seed=args.seed, threads=args.threads)
                p_ = predict(model, DataLoader(ds, batch_size=args.batch_size), device)["probs"]
            m5, mv = summarise(labels, p_, 0.5), summarise(labels, p_, val_thr)
            corr_rows.append({
                "corruption": name, "severity": sev, "param_name": param, "param_value": "" if val_ is None else val_,
                **{k: m5[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "ce_nats", "ce_bits")},
                "accuracy_valthr": mv["accuracy"], "f1_valthr": mv["f1"],
            })
            print(f"  {name:>8} severity {sev} ({param}={'clean' if val_ is None else val_}): "
                  f"acc {m5['accuracy']:.3f}  F1 {m5['f1']:.3f}  AUC {m5['roc_auc']:.3f}", flush=True)
    if corr_rows:
        with open(out_dir / "corruption_results.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(corr_rows[0]))
            w.writeheader()
            w.writerows(corr_rows)
        plot_corruption_curves(corr_rows, names, majority_acc, out_dir / "corruption_curves.png")
        plot_corruption_examples(test_entries[0], names, out_dir / "corruption_examples.png", args.seed)

    results = {
        "checkpoint": display_path(ckpt_path),
        "checkpoint_epoch": ckpt["epoch"],
        "evaluated_on": source,
        "n_test": int(len(labels)),
        "n_test_oil": int(labels.sum()),
        "split_file": split_info["path"],
        "split_sha256": split_info["sha256"],
        "trained_split_sha256": cfg.get("split_sha256"),
        "warnings": warnings_list,
        "majority_baseline_accuracy": majority_acc,
        "val_optimal_threshold": val_thr,
        "val_f1_at_that_threshold": val_best_f1,
        "test_at_threshold_0.5": at_half,
        "test_at_val_optimal_threshold": at_val,
        "test_active_units": active_units(test["mu"]),
        "loss_weights": {k: cfg["args"][k] for k in ("recon_loss", "recon_weight", "beta", "lam")},
        "latent_plot_seconds": latent_seconds,
    }
    save_json(results, out_dir / "metrics.json")

    print(f"\n{'':>24}{'t = 0.5':>12}{f't = {val_thr:.3f} (val)':>20}")
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "ce_nats", "ce_bits"):
        print(f"{k:>24}{at_half[k]:>12.4f}{at_val[k]:>20.4f}")
    print(f"{'always No Oil accuracy':>24}{majority_acc:>12.4f}")
    print(f"confusion matrix [[TN FP] [FN TP]] at 0.5: {at_half['confusion_matrix']}")
    print(f"outputs in {display_path(out_dir)}")
    for msg in warnings_list:
        print(msg)
    return results


if __name__ == "__main__":
    main()
