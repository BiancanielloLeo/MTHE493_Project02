"""
audit_data.py - sanity checks on the images listed in the split file (read-only).

Checks:
  1. size / mode, and whether the 3 RGB channels are identical (true grayscale)
  2. exact pixel duplicates
  3. near-duplicates, including flipped / rotated copies (normalised correlation of
     thumbnails under the 8 flips/rotations, confirmed at 128x128)
  4. low-information chips (almost constant, or mostly no-data padding at the swath edge)
For each duplicate pair it records whether the two images sit in different splits
(leakage) or carry different labels (conflict).

    python Daryan/vae_classifier/audit_data.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data import DEFAULT_SPLIT_FILE, SPLITS, SplitError, display_path, load_split, repo_path  # noqa: E402
from utils import save_json  # noqa: E402


def dihedral(a):
    """The 8 flips/rotations of the last two axes."""
    out = []
    for k in range(4):
        r = np.rot90(a, k, axes=(-2, -1))
        out += [r, r[..., ::-1]]
    return out


def normalise(a):
    a = a.reshape(len(a), -1).astype(np.float32)
    a = a - a.mean(1, keepdims=True)
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-6)


def nodata_mask(g):
    """Swath-edge padding: saturated white (or pure black) pixels with no radar data."""
    return (g >= 250) | (g <= 2)


def thumb(g, size):
    """Thumbnail with padding replaced by the mean of the valid pixels, so padding does not
    make unrelated chips look alike. Also returns the thumbnail's valid-pixel mask."""
    mask = nodata_mask(g)
    fill = g[~mask].mean() if (~mask).any() else 128.0
    f = np.where(mask, fill, g).astype(np.float32)
    t = np.asarray(Image.fromarray(f, mode="F").resize((size, size), Image.BILINEAR), dtype=np.float32)
    m = np.asarray(Image.fromarray((~mask).astype(np.uint8) * 255).resize((size, size), Image.BILINEAR)) > 250
    return t, m


def inspect(path):
    with Image.open(path) as im:
        size, mode = im.size, im.mode
        rgb = np.asarray(im.convert("RGB"))
    gray_ok = bool((rgb[..., 0] == rgb[..., 1]).all() and (rgb[..., 1] == rgb[..., 2]).all())
    g = rgb[..., 0]
    valid = ~nodata_mask(g)
    t32, _ = thumb(g, 32)
    t128, m128 = thumb(g, 128)
    return {
        "size": size, "mode": mode, "gray": gray_ok,
        "md5": hashlib.md5(g.tobytes()).hexdigest(),
        "std": float(g[valid].std()) if valid.any() else 0.0,
        "mean": float(g[valid].mean()) if valid.any() else 0.0,
        "nodata_frac": float(1 - valid.mean()),
        "t32": t32, "t128": t128, "m128": m128,
        "thumb": np.asarray(Image.fromarray(g).resize((128, 128), Image.BILINEAR)),
    }


def masked_corr(a, b, m):
    a, b = a[m], b[m]
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-6))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split-file", default=DEFAULT_SPLIT_FILE)
    p.add_argument("--out-dir", default=str(HERE / "runs" / "data_audit"))
    p.add_argument("--candidate-r", type=float, default=0.90, help="32x32 correlation to shortlist a pair")
    p.add_argument("--confirm-r", type=float, default=0.95,
                   help="128x128 correlation on non-padding pixels to call it a near-duplicate")
    p.add_argument("--min-overlap", type=float, default=0.3,
                   help="fraction of pixels that must be valid (non-padding) in both images")
    args = p.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        split, info = load_split(args.split_file)
    except SplitError as e:
        sys.exit(f"ERROR: {e}")
    items = [(rel, lab, name) for name in SPLITS for rel, lab in split[name]]
    print(f"auditing {len(items)} images from {info['path']}")
    with ThreadPoolExecutor(8) as ex:
        recs = list(ex.map(lambda it: inspect(repo_path(it[0])), items))

    report = {
        "n_images": len(items),
        "sizes": {str(k): v for k, v in Counter(r["size"] for r in recs).items()},
        "modes": dict(Counter(r["mode"] for r in recs)),
        "not_pure_grayscale": [items[i][0] for i, r in enumerate(recs) if not r["gray"]],
    }

    # Exact duplicates.
    by_md5 = {}
    for i, r in enumerate(recs):
        by_md5.setdefault(r["md5"], []).append(i)
    exact = [g for g in by_md5.values() if len(g) > 1]

    # Near-duplicates under flips / rotations.
    n = len(items)
    t32 = np.stack([r["t32"] for r in recs])
    base = normalise(t32)
    best_r = np.full((n, n), -1.0, dtype=np.float32)
    best_k = np.zeros((n, n), dtype=np.int8)
    for k, tk in enumerate(dihedral(t32)):
        g = base @ normalise(np.ascontiguousarray(tk)).T
        upd = g > best_r
        best_r[upd], best_k[upd] = g[upd], k
    iu = np.triu_indices(n, 1)
    cand = np.flatnonzero(best_r[iu] >= args.candidate_r)
    pairs = []
    for c in cand:
        i, j = int(iu[0][c]), int(iu[1][c])
        k = int(best_k[i, j])
        tb = np.ascontiguousarray(dihedral(recs[j]["t128"])[k])
        mb = np.ascontiguousarray(dihedral(recs[j]["m128"])[k])
        m = recs[i]["m128"] & mb
        if m.mean() < args.min_overlap:  # too little real sea surface in common to judge
            continue
        r128 = masked_corr(recs[i]["t128"], tb, m)
        if r128 >= args.confirm_r:
            pairs.append({
                "image_a": items[i][0], "split_a": items[i][2], "label_a": items[i][1],
                "image_b": items[j][0], "split_b": items[j][2], "label_b": items[j][1],
                "r_32": float(best_r[i, j]), "r_128": r128, "transform": k, "overlap": float(m.mean()),
                "cross_split": items[i][2] != items[j][2], "label_conflict": items[i][1] != items[j][1],
                "_i": i, "_j": j,
            })
    pairs.sort(key=lambda d: -d["r_128"])
    del best_r, best_k

    # Low-information chips.
    low_info = [
        {"image": items[i][0], "split": items[i][2], "label": items[i][1], "std": r["std"], "nodata_frac": r["nodata_frac"]}
        for i, r in enumerate(recs) if r["std"] < 4 or r["nodata_frac"] > 0.5
    ]

    involved = {k for d in pairs for k in (d["_i"], d["_j"])}
    report.update({
        "exact_duplicate_groups": [[items[i][0] for i in g] for g in exact],
        "near_duplicate_pairs": len(pairs),
        "near_duplicate_images_involved": len(involved),
        "near_duplicate_pairs_cross_split": sum(d["cross_split"] for d in pairs),
        "near_duplicate_pairs_cross_split_involving_test": sum(
            d["cross_split"] and "test" in (d["split_a"], d["split_b"]) for d in pairs),
        "near_duplicate_pairs_label_conflict": sum(d["label_conflict"] for d in pairs),
        "near_duplicate_pairs_by_transform": dict(Counter(
            ["identity", "flip", "rot90", "rot90+flip", "rot180", "rot180+flip", "rot270", "rot270+flip"][d["transform"]]
            for d in pairs)),
        "low_information_chips": len(low_info),
        "low_information_by_label": dict(Counter(d["label"] for d in low_info)),
        "per_label_mean_intensity": {
            lab: float(np.mean([r["mean"] for r, it in zip(recs, items) if it[1] == lab])) for lab in (0, 1)},
        "per_label_mean_nodata_frac": {
            lab: float(np.mean([r["nodata_frac"] for r, it in zip(recs, items) if it[1] == lab])) for lab in (0, 1)},
        "per_label_share_with_padding": {
            lab: float(np.mean([r["nodata_frac"] > 0.01 for r, it in zip(recs, items) if it[1] == lab])) for lab in (0, 1)},
    })
    save_json(report, out / "audit_report.json")
    for fname, rows in (("near_duplicates.csv", [{k: v for k, v in d.items() if not k.startswith("_")} for d in pairs]),
                        ("low_information.csv", low_info)):
        if rows:
            with open(out / fname, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)

    # Montage: worst cases first (label conflicts, then cross-split pairs).
    show = sorted(pairs, key=lambda d: (not d["label_conflict"], not d["cross_split"], -d["r_128"]))[:10]
    if show:
        fig, axes = plt.subplots(2, len(show), figsize=(2.2 * len(show), 5), squeeze=False)
        for c, d in enumerate(show):
            for r, (k, s, lab) in enumerate(((d["_i"], d["split_a"], d["label_a"]), (d["_j"], d["split_b"], d["label_b"]))):
                axes[r, c].imshow(recs[k]["thumb"], cmap="gray", vmin=0, vmax=255)
                axes[r, c].set_title(f"{Path(items[k][0]).stem}\n{s} label={lab}", fontsize=7)
                axes[r, c].axis("off")
            axes[1, c].set_xlabel(f"r={d['r_128']:.3f}")
        fig.suptitle("near-duplicate pairs (top: image A, bottom: image B)")
        fig.tight_layout()
        fig.savefig(out / "near_duplicates.png", dpi=110)
        plt.close(fig)

    for k, v in report.items():
        if isinstance(v, list):
            print(f"{k}: {len(v)}")
        else:
            print(f"{k}: {v}")
    print(f"outputs in {display_path(out)}")


if __name__ == "__main__":
    main()
