"""
data.py - split file, in-memory datasets, augmentation and test-time corruptions.

The split file comes from a teammate (classifier_josh/split.json). This module only reads it;
it never creates or rewrites a split.
"""
from __future__ import annotations

import hashlib
import io
import json
import random
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT_FILE = "classifier_josh/split.json"
SPLITS = ("train", "val", "test")
CLASS_DIRS = {1: "Oil", 0: "No Oil"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class SplitError(RuntimeError):
    """The split file is missing or invalid. Training/evaluation must stop."""


# ---------------------------------------------------------------- paths and split file

def resolve_user_path(p) -> Path:
    """Absolute paths are used as-is. Relative paths are tried against the current
    directory first, then the repo root, so commands work from any folder."""
    p = Path(p).expanduser()
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    return (REPO_ROOT / p).resolve()


def repo_path(rel: str) -> Path:
    """Split-file paths use '/', relative to the repo root. Works on Windows and macOS."""
    return REPO_ROOT.joinpath(*PurePosixPath(rel.replace("\\", "/")).parts)


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def display_path(path: Path) -> str:
    """Repo-relative path when possible (portable in configs), else absolute."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def load_split(split_file=DEFAULT_SPLIT_FILE, check_files: bool = True):
    """Read and validate the split file.

    Returns (split, info). split maps 'train'/'val'/'test' to lists of (rel_path, label).
    info has the resolved path, its SHA-256 and per-split counts.
    Raises SplitError on any problem.
    """
    path = resolve_user_path(split_file)
    if not path.is_file():
        raise SplitError(
            f"Split file not found: {path}\n"
            f"The split is provided by a teammate ({DEFAULT_SPLIT_FILE} on the josh-classifier branch). "
            "This code never creates one; check out that file or pass --split-file."
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise SplitError(f"Split file {path} is not valid JSON: {e}") from e
    if not isinstance(raw, dict) or any(k not in raw for k in SPLITS):
        keys = list(raw) if isinstance(raw, dict) else type(raw).__name__
        raise SplitError(f"Split file {path} must be a JSON object with keys {list(SPLITS)}; found {keys}")

    split = {}
    for name in SPLITS:
        entries = raw[name]
        if not isinstance(entries, list) or not entries:
            raise SplitError(f"Split '{name}' in {path} must be a non-empty list of [path, label] pairs")
        items = []
        for i, e in enumerate(entries):
            ok = (
                isinstance(e, (list, tuple)) and len(e) == 2 and isinstance(e[0], str)
                and isinstance(e[1], int) and not isinstance(e[1], bool) and e[1] in (0, 1)
            )
            if not ok:
                raise SplitError(f"Malformed entry {name}[{i}] in {path}: {e!r} (expected [relative_path, 0 or 1])")
            items.append((e[0].replace("\\", "/"), e[1]))
        split[name] = items

    # No image may appear twice (within or across splits).
    seen, dups = {}, []
    for name in SPLITS:
        for rel, _ in split[name]:
            key = PurePosixPath(rel).as_posix()
            if key in seen:
                dups.append(f"{key} ({seen[key]} and {name})")
            seen[key] = name
    if dups:
        raise SplitError(f"{len(dups)} image(s) appear more than once in {path}, e.g.:\n  " + "\n  ".join(dups[:10]))

    if check_files:
        missing = [rel for name in SPLITS for rel, _ in split[name] if not repo_path(rel).is_file()]
        if missing:
            raise SplitError(
                f"{len(missing)} image(s) listed in {path} do not exist under {REPO_ROOT}, e.g.:\n  "
                + "\n  ".join(missing[:10])
            )

    # The file's labels are used as given; just flag entries whose folder disagrees.
    disagree = [
        rel for name in SPLITS for rel, lab in split[name]
        if f"/{CLASS_DIRS[1 - lab]}/" in f"/{rel}"
    ]
    if disagree:
        warnings.warn(f"{len(disagree)} split entries have a label that disagrees with their folder, e.g. {disagree[:3]}")

    info = {
        "path": display_path(path),
        "sha256": file_sha256(path),
        "counts": {n: {"total": len(v), "oil": sum(l for _, l in v)} for n, v in split.items()},
    }
    return split, info


def stratified_subset(items, n: int, seed: int = 0):
    """Class-balanced (in proportion) random subset, used by --smoke-test."""
    if n >= len(items):
        return list(items)
    rng = random.Random(seed)
    pos = [it for it in items if it[1] == 1]
    neg = [it for it in items if it[1] == 0]
    n_pos = max(1, round(n * len(pos) / len(items)))
    out = rng.sample(pos, min(n_pos, len(pos))) + rng.sample(neg, min(n - n_pos, len(neg)))
    rng.shuffle(out)
    return out


# ---------------------------------------------------------------- alternative data folders

def _index_folder(data_dir: Path):
    data_dir = Path(data_dir)
    index = {}
    for label, sub in CLASS_DIRS.items():
        d = data_dir / sub
        if not d.is_dir():
            raise FileNotFoundError(f"{d} not found: --data-dir must contain 'Oil/' and 'No Oil/' subfolders")
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in IMAGE_EXTS:
                index[(label, p.stem)] = p
    return index


def remap_to_data_dir(items, data_dir):
    """Map split entries (rel_path, label) to files in <data_dir>/<Oil|No Oil>/ with the same
    file name stem (any image extension, so .png output from the pipeline also works).
    Returns a list of (abs_path, label, name)."""
    index = _index_folder(data_dir)
    out, missing = [], []
    for rel, label in items:
        stem = PurePosixPath(rel).stem
        p = index.get((label, stem))
        if p is None:
            missing.append(f"{CLASS_DIRS[label]}/{stem}.*")
        else:
            out.append((p, label, rel))
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {len(items)} test images were not found in {data_dir}, e.g. {missing[:5]}.\n"
            "Use --all-images to evaluate every image in the folder instead of the test split."
        )
    return out


def folder_items(data_dir, split=None):
    """Every image in <data_dir>/<Oil|No Oil>/, labelled by folder. Warns if file names
    overlap the train/val splits (that would be training-data leakage)."""
    index = _index_folder(data_dir)
    out = [(p, label, f"{CLASS_DIRS[label]}/{p.name}") for (label, _), p in index.items()]
    if split is not None:
        seen = {PurePosixPath(rel).stem for name in ("train", "val") for rel, _ in split[name]}
        overlap = sum(1 for p, _, _ in out if p.stem in seen)
        if overlap:
            warnings.warn(f"{overlap} images in {data_dir} share a file name with train/val images; "
                          "metrics on them are not a held-out test.")
    return out


# ---------------------------------------------------------------- dataset

def _load_gray(path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("L")


class OilChips(Dataset):
    """Grayscale chips resized and cached in memory as uint8 (about 64 MB for the
    full train split at 128x128), so each epoch costs no disk I/O or JPEG decoding.

    entries: list of (abs_path, label, name).
    corruption: optional function (PIL L image, numpy Generator) -> PIL L image, applied at
    full resolution before resizing (seeded per image, so it is reproducible).
    __getitem__ returns (x in [0,1] of shape [1,S,S], label float, index).
    """

    def __init__(self, entries, img_size=128, augment=False, corruption=None, seed=0, threads=8):
        self.paths = [Path(p) for p, _, _ in entries]
        self.names = [n for _, _, n in entries]
        self.labels = torch.tensor([float(l) for _, l, _ in entries])
        self.augment = augment

        def load(i):
            img = _load_gray(self.paths[i])
            if corruption is not None:
                img = corruption(img, np.random.default_rng([seed, i]))
            if img.size != (img_size, img_size):
                img = img.resize((img_size, img_size), Image.BILINEAR)
            return np.asarray(img, dtype=np.uint8)

        with ThreadPoolExecutor(max_workers=threads) as ex:
            arrays = list(ex.map(load, range(len(entries))))
        self.images = torch.from_numpy(np.stack(arrays)).unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = self.images[i].float().div_(255.0)
        if self.augment:  # random dihedral transform: flips + 90-degree rotations
            if torch.rand(()) < 0.5:
                x = x.flip(-1)
            if torch.rand(()) < 0.5:
                x = x.flip(-2)
            k = int(torch.randint(0, 4, ()))
            if k:
                x = torch.rot90(x, k, dims=(-2, -1))
        return x, self.labels[i], i


def split_entries(items):
    """(rel_path, label) pairs -> (abs_path, label, name) entries."""
    return [(repo_path(rel), label, rel) for rel, label in items]


# ---------------------------------------------------------------- test-time corruptions
# Severity 0 is always the clean image, so plots start at the clean baseline.

def _gaussian(sigma):
    def f(img, rng):
        a = np.asarray(img, dtype=np.float32) / 255.0
        a = np.clip(a + rng.normal(0.0, sigma, a.shape).astype(np.float32), 0.0, 1.0)
        return Image.fromarray((a * 255.0 + 0.5).astype(np.uint8))
    return f


def _jpeg(quality):
    def f(img, rng):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(quality))
        buf.seek(0)
        with Image.open(buf) as im:
            return im.convert("L")
    return f


def _blocks(fraction, block=8):
    def f(img, rng):
        a = np.array(img, dtype=np.uint8)
        h, w = a.shape
        bh, bw = -(-h // block), -(-w // block)
        drop = rng.random((bh, bw)) < fraction
        mask = np.kron(drop, np.ones((block, block), dtype=bool))[:h, :w]
        a[mask] = 0
        return Image.fromarray(a)
    return f


CORRUPTIONS = {
    # name: (parameter name, values for severity 1..5, factory)
    "gaussian": ("noise_sigma", [0.05, 0.1, 0.2, 0.3, 0.5], _gaussian),
    "jpeg": ("jpeg_quality", [75, 50, 25, 10, 5], _jpeg),
    "blocks": ("block_drop_fraction", [0.02, 0.05, 0.1, 0.2, 0.4], _blocks),
}


def corruption_levels(name):
    """[(severity, parameter value, function or None)], starting with severity 0 = clean."""
    param, values, factory = CORRUPTIONS[name]
    return [(0, None, None)] + [(i + 1, v, factory(v)) for i, v in enumerate(values)]
