#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
balance_dataset.py — Class balancing for the SolarFCD dataset.

The raw SolarFCD collection is imbalanced because its class distribution
reflects the availability of the source imagery rather than a purposeful
sampling scheme. This script applies the two complementary procedures
described in the paper:

  1. Similarity-based subsampling (majority classes)
     Every image of an over-represented class is encoded with a perceptual
     hash (pHash). Candidates are streamed in a deterministic order and the
     pairwise Hamming distance of each candidate is computed against all
     images retained so far. A candidate is retained only if its highest
     similarity to any already-retained image stays below the threshold
     (default 0.30), so that near-duplicate and highly repetitive frames are
     discarded while morphologically distinct appearances -- different surface
     textures, lighting conditions and severity levels -- are preserved.

  2. Data augmentation (minority classes)
     Under-represented classes are expanded up to the target size with mild,
     label-preserving transforms (flips, small rotations, small translations,
     mild zoom, brightness / contrast / saturation jitter, light blur and
     Gaussian noise). Transforms are sampled from a seeded RNG, so the
     balanced dataset is reproducible.

Only the training split is balanced by default: validation and test splits are
left untouched so that the reported metrics remain comparable to the raw data
distribution.

The raw dataset is read from `--data-root`. If that folder does not exist (or
`--download` is passed) the archive is fetched from the DFKI Cloud share that
hosts SolarFCD, extracted, and verified before balancing starts:

    https://cloud.dfki.de/owncloud/index.php/s/SZYo97gdwY2k92Z

The balanced copy is written to `--out-root`, mirroring the original
`Train/ Valid/ Test/` layout with the same four class folders.

Usage
-----
    # download (if needed) + balance in one go
    python src/balance_dataset.py --data-root data/SolarFCD \
                                  --out-root  data/SolarFCD_balanced \
                                  --target-size 800 \
                                  --similarity-threshold 0.30 \
                                  --seed 42

    # force a fresh download of the archive
    python src/balance_dataset.py --data-root data/SolarFCD \
                                  --out-root  data/SolarFCD_balanced --download

    # only check that a local copy has the expected layout
    python src/balance_dataset.py --data-root data/SolarFCD \
                                  --out-root  data/SolarFCD_balanced --verify-only

Requirements: numpy, Pillow  (tqdm optional)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

try:  # optional progress bar
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #

# Fixed class order — identical to src/data_loader.py so that label indices
# are the same on every machine.
CLASSES: Tuple[str, ...] = (
    "Electrical_Fault",
    "Healthy",
    "Structural_Fault",
    "Surface_Obstruction",
)

IMG_EXTENSIONS: Tuple[str, ...] = (
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
)

# Public SolarFCD share on the DFKI Cloud (ownCloud). Appending "/download"
# to a public share link returns the archive itself.
DFKI_SHARE_URL: str = "https://cloud.dfki.de/owncloud/index.php/s/SZYo97gdwY2k92Z"
DFKI_DOWNLOAD_URL: str = DFKI_SHARE_URL + "/download"

SPLITS: Tuple[str, ...] = ("Train", "Valid", "Test")


# --------------------------------------------------------------------------- #
# Dataset acquisition — DFKI Cloud                                             #
# --------------------------------------------------------------------------- #

def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


def download_archive(url: str, dst: Path, chunk: int = 1 << 20) -> Path:
    """
    Download `url` to `dst`, resuming a partial file when the server supports
    HTTP range requests. Returns the path of the downloaded archive.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    done = dst.stat().st_size if dst.exists() else 0

    req = urllib.request.Request(url, headers={"User-Agent": "SolarFCD/1.0"})
    if done:
        req.add_header("Range", f"bytes={done}-")

    try:
        response = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as exc:
        if done and exc.code == 416:            # already complete
            print(f"  archive already complete ({human_bytes(done)})")
            return dst
        if done and exc.code in (400, 501):     # no range support -> restart
            print("  server refused resume, restarting download")
            dst.unlink(missing_ok=True)
            done = 0
            req = urllib.request.Request(url, headers={"User-Agent": "SolarFCD/1.0"})
            response = urllib.request.urlopen(req, timeout=60)
        else:
            raise

    resuming = response.status == 206 and done > 0
    if not resuming:
        done = 0
    remaining = int(response.headers.get("Content-Length") or 0)
    total = done + remaining if remaining else None

    print(f"  {'resuming' if resuming else 'downloading'} -> {dst} "
          f"({human_bytes(total) if total else 'unknown size'})")

    mode = "ab" if resuming else "wb"
    with response, open(dst, mode) as fh:
        bar = tqdm(total=total, initial=done, unit="B", unit_scale=True,
                   desc="  download", leave=False)
        while True:
            block = response.read(chunk)
            if not block:
                break
            fh.write(block)
            done += len(block)
            if hasattr(bar, "update"):
                bar.update(len(block))
        if hasattr(bar, "close"):
            bar.close()

    print(f"  downloaded {human_bytes(dst.stat().st_size)}")
    return dst


def extract_archive(archive: Path, out_dir: Path) -> None:
    """Extract a .zip / .tar.* archive into `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  extracting {archive.name} -> {out_dir}")

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            for member in tqdm(zf.infolist(), desc="  extract",
                               unit="file", leave=False):
                zf.extract(member, out_dir)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            tf.extractall(out_dir)
    else:
        raise ValueError(f"unsupported archive format: {archive}")


def find_dataset_root(search_dir: Path) -> Optional[Path]:
    """
    Locate the folder that directly contains Train/ Valid/ Test/ somewhere
    beneath `search_dir` (the archive may add one or more nesting levels).
    """
    if not search_dir.is_dir():
        return None
    candidates = [search_dir] + [p for p in search_dir.rglob("*") if p.is_dir()]
    for cand in candidates:
        if all((cand / s).is_dir() for s in SPLITS):
            return cand
    return None


def verify_layout(root: Path, strict: bool = True) -> Dict[str, Dict[str, int]]:
    """Print and return a per-split, per-class image count."""
    counts: Dict[str, Dict[str, int]] = {}
    print(f"\nVerifying dataset layout at {root}")
    for split in SPLITS:
        split_dir = root / split
        if not split_dir.is_dir():
            if strict:
                raise FileNotFoundError(f"missing split folder: {split_dir}")
            continue
        counts[split] = {}
        for cls in CLASSES:
            cls_dir = split_dir / cls
            if not cls_dir.is_dir():
                if strict:
                    raise FileNotFoundError(f"missing class folder: {cls_dir}")
                counts[split][cls] = 0
                continue
            counts[split][cls] = len(list_images(cls_dir))
        total = sum(counts[split].values())
        print(f"  {split:<6} {total:>6} images  " +
              "  ".join(f"{c}={counts[split][c]}" for c in CLASSES))
    return counts


def ensure_dataset(data_root: Path, url: str, force: bool,
                   keep_zip: bool) -> Path:
    """
    Make sure a usable SolarFCD copy exists at (or under) `data_root`,
    downloading it from the DFKI Cloud when necessary.

    Returns the folder that directly contains Train/ Valid/ Test/.
    """
    if not force:
        found = find_dataset_root(data_root)
        if found is not None:
            print(f"Found local dataset: {found}")
            return found

    print(f"\nFetching SolarFCD from the DFKI Cloud\n  {DFKI_SHARE_URL}")
    target = data_root if data_root.suffix == "" else data_root.parent
    target.mkdir(parents=True, exist_ok=True)
    archive = target / "SolarFCD.zip"

    download_archive(url, archive)
    extract_archive(archive, target)

    if not keep_zip:
        archive.unlink(missing_ok=True)
        print("  archive removed (pass --keep-zip to keep it)")

    found = find_dataset_root(target)
    if found is None:
        raise FileNotFoundError(
            f"could not locate a folder containing {'/'.join(SPLITS)} under {target}. "
            "Extract the archive manually and pass --data-root explicitly."
        )
    print(f"  dataset root: {found}")
    return found


# --------------------------------------------------------------------------- #
# Perceptual hash (pHash)                                                      #
# --------------------------------------------------------------------------- #

def _dct_matrix(n: int) -> np.ndarray:
    """Orthonormal DCT-II matrix of size n x n (avoids a SciPy dependency)."""
    k = np.arange(n).reshape(-1, 1)
    i = np.arange(n).reshape(1, -1)
    m = np.cos(np.pi * (2.0 * i + 1.0) * k / (2.0 * n))
    m *= np.sqrt(2.0 / n)
    m[0, :] *= 1.0 / np.sqrt(2.0)
    return m


def phash(path: Path, hash_size: int = 8, highfreq_factor: int = 4) -> np.ndarray:
    """
    Perceptual hash of an image, returned as a boolean bit vector of length
    hash_size ** 2 (64 bits by default).

    The image is converted to grayscale, resized to
    (hash_size * highfreq_factor)^2, transformed with a 2-D DCT, and the
    low-frequency block (excluding the DC term) is thresholded at its median.
    """
    size = hash_size * highfreq_factor
    with Image.open(path) as im:
        im = im.convert("L").resize((size, size), Image.Resampling.LANCZOS)
        arr = np.asarray(im, dtype=np.float64)

    d = _dct_matrix(size)
    dct = d @ arr @ d.T
    low = dct[:hash_size, :hash_size].flatten()

    median = np.median(low[1:])          # ignore the DC coefficient
    return low > median


def hamming_similarity(bits: np.ndarray, retained: np.ndarray) -> float:
    """
    Highest similarity between one hash and a stack of retained hashes.

    similarity = 1 - (Hamming distance / number of bits) in [0, 1].
    Returns 0.0 when nothing has been retained yet.
    """
    if retained.size == 0:
        return 0.0
    distances = np.count_nonzero(retained != bits, axis=1)
    return float(1.0 - distances.min() / bits.size)


# --------------------------------------------------------------------------- #
# 1) Similarity-based subsampling (majority classes)                           #
# --------------------------------------------------------------------------- #

def similarity_subsample(
    files: List[Path],
    target_size: int,
    threshold: float,
    hash_size: int,
    rng: random.Random,
    mode: str = "similarity",
) -> Tuple[List[Path], Dict[str, int]]:
    """
    Retain visually unique images from an over-represented class.

    Parameters
    ----------
    files        : candidate image paths.
    target_size  : stop once this many images have been retained.
    threshold    : decision threshold in [0, 1].
    hash_size    : pHash side length (8 -> 64-bit hash).
    rng          : seeded RNG used to shuffle the streaming order.
    mode         : 'similarity' keeps an image when its maximum similarity to
                   every retained image is BELOW `threshold` (the criterion as
                   written in the paper);
                   'distance'  keeps an image when its minimum normalised
                   Hamming DISTANCE to every retained image is ABOVE
                   `threshold` (the usual near-duplicate criterion, i.e.
                   equivalent to a similarity ceiling of 1 - threshold).

    Returns
    -------
    (retained paths, statistics dict)
    """
    order = list(files)
    rng.shuffle(order)                    # order-independent, seeded

    retained_paths: List[Path] = []
    retained_bits: List[np.ndarray] = []
    stack = np.empty((0, hash_size * hash_size), dtype=bool)
    n_unreadable = 0

    for path in tqdm(order, desc="  subsampling", unit="img", leave=False):
        if len(retained_paths) >= target_size:
            break
        try:
            bits = phash(path, hash_size=hash_size)
        except Exception:                 # corrupt / unreadable file
            n_unreadable += 1
            continue

        sim = hamming_similarity(bits, stack)
        keep = (sim < threshold) if mode == "similarity" else ((1.0 - sim) > threshold)

        if keep:
            retained_paths.append(path)
            retained_bits.append(bits)
            stack = np.asarray(retained_bits, dtype=bool)

    stats = {
        "candidates": len(files),
        "retained": len(retained_paths),
        "removed": len(files) - len(retained_paths) - n_unreadable,
        "unreadable": n_unreadable,
    }
    return retained_paths, stats


# --------------------------------------------------------------------------- #
# 2) Augmentation (minority classes)                                           #
# --------------------------------------------------------------------------- #

def augment_image(im: Image.Image, rng: random.Random) -> Image.Image:
    """Apply a random, mild, label-preserving transform chain to one image."""
    w, h = im.size

    # geometric ------------------------------------------------------------- #
    if rng.random() < 0.5:
        im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if rng.random() < 0.2:
        im = im.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    if rng.random() < 0.7:
        angle = rng.uniform(-15.0, 15.0)
        im = im.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False)
    if rng.random() < 0.5:
        dx = int(rng.uniform(-0.08, 0.08) * w)
        dy = int(rng.uniform(-0.08, 0.08) * h)
        im = im.transform(
            (w, h), Image.Transform.AFFINE, (1, 0, -dx, 0, 1, -dy),
            resample=Image.Resampling.BILINEAR,
        )
    if rng.random() < 0.4:                      # mild zoom-in (centre crop)
        s = rng.uniform(0.85, 0.98)
        cw, ch = int(w * s), int(h * s)
        x0, y0 = (w - cw) // 2, (h - ch) // 2
        im = im.crop((x0, y0, x0 + cw, y0 + ch)).resize(
            (w, h), Image.Resampling.BILINEAR
        )

    # photometric ----------------------------------------------------------- #
    if rng.random() < 0.7:
        im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.80, 1.20))
    if rng.random() < 0.7:
        im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.80, 1.20))
    if rng.random() < 0.4:
        im = ImageEnhance.Color(im).enhance(rng.uniform(0.85, 1.15))
    if rng.random() < 0.2:
        im = im.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 0.8)))
    if rng.random() < 0.2:                      # light sensor noise
        arr = np.asarray(im, dtype=np.float32)
        arr += np.random.normal(0.0, rng.uniform(2.0, 6.0), arr.shape)
        im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    return im


def augment_class(
    files: List[Path],
    dst_dir: Path,
    target_size: int,
    rng: random.Random,
    quality: int = 95,
) -> Dict[str, int]:
    """
    Copy the originals of an under-represented class and synthesise new
    samples until the class reaches `target_size`.
    """
    n_needed = max(0, target_size - len(files))
    n_created = 0
    round_idx = 0

    while n_created < n_needed:
        round_idx += 1
        pool = list(files)
        rng.shuffle(pool)
        for src in pool:
            if n_created >= n_needed:
                break
            try:
                with Image.open(src) as im:
                    im = im.convert("RGB")
                    out = augment_image(im, rng)
            except Exception:
                continue
            dst = dst_dir / f"{src.stem}_aug{round_idx}_{n_created:05d}.jpg"
            out.save(dst, quality=quality)
            n_created += 1

    return {
        "originals": len(files),
        "generated": n_created,
        "total": len(files) + n_created,
    }


# --------------------------------------------------------------------------- #
# Pipeline                                                                     #
# --------------------------------------------------------------------------- #

def list_images(folder: Path) -> List[Path]:
    """All image files in a folder, sorted for deterministic behaviour."""
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
    )


def resolve_target_size(counts: Dict[str, int], requested: int | None,
                        strategy: str) -> int:
    """Target number of images per class."""
    if requested is not None:
        return requested
    values = list(counts.values())
    if strategy == "median":
        return int(np.median(values))
    if strategy == "mean":
        return int(np.mean(values))
    if strategy == "min":
        return int(min(values))
    return int(max(values))


def balance_split(
    split_dir: Path,
    out_dir: Path,
    target_size: int | None,
    strategy: str,
    threshold: float,
    hash_size: int,
    mode: str,
    seed: int,
) -> Dict:
    """Balance one split (typically Train) and write it to `out_dir`."""
    counts = {}
    files_per_class: Dict[str, List[Path]] = {}
    for cls in CLASSES:
        cls_dir = split_dir / cls
        if not cls_dir.is_dir():
            raise FileNotFoundError(f"missing class folder: {cls_dir}")
        files_per_class[cls] = list_images(cls_dir)
        counts[cls] = len(files_per_class[cls])

    target = resolve_target_size(counts, target_size, strategy)
    print(f"\n[{split_dir.name}] raw counts: {counts}")
    print(f"[{split_dir.name}] target size per class: {target}")

    report: Dict[str, Dict] = {}
    for cls in CLASSES:
        files = files_per_class[cls]
        dst = out_dir / cls
        dst.mkdir(parents=True, exist_ok=True)
        rng = random.Random(seed + abs(hash(cls)) % 10_000)

        if len(files) > target:
            print(f"  {cls:<22} {len(files):>5} -> subsampling")
            kept, stats = similarity_subsample(
                files, target, threshold, hash_size, rng, mode
            )
            for src in kept:
                shutil.copy2(src, dst / src.name)
            stats["action"] = "subsampled"
            stats["final"] = len(kept)
            if len(kept) < target:
                print(f"    ! only {len(kept)}/{target} images passed the "
                      f"uniqueness filter for {cls}")
            report[cls] = stats

        elif len(files) < target:
            print(f"  {cls:<22} {len(files):>5} -> augmenting")
            for src in files:
                shutil.copy2(src, dst / src.name)
            stats = augment_class(files, dst, target, rng)
            stats["action"] = "augmented"
            stats["final"] = stats["total"]
            report[cls] = stats

        else:
            print(f"  {cls:<22} {len(files):>5} -> unchanged")
            for src in files:
                shutil.copy2(src, dst / src.name)
            report[cls] = {"action": "unchanged", "final": len(files)}

    return {
        "split": split_dir.name,
        "raw_counts": counts,
        "target_size": target,
        "per_class": report,
    }


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Balance SolarFCD: pHash subsampling for majority classes, "
                    "augmentation for minority classes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", required=True, type=Path,
                   help="folder containing Train/ Valid/ Test/; downloaded from "
                        "the DFKI Cloud if it does not exist yet")
    p.add_argument("--out-root", required=True, type=Path,
                   help="destination folder for the balanced dataset")
    p.add_argument("--download", action="store_true",
                   help="force a (re)download from the DFKI Cloud even if a "
                        "local copy is present")
    p.add_argument("--url", default=DFKI_DOWNLOAD_URL,
                   help="direct download URL of the SolarFCD archive")
    p.add_argument("--keep-zip", action="store_true",
                   help="keep the downloaded archive instead of deleting it")
    p.add_argument("--verify-only", action="store_true",
                   help="only download/verify the dataset layout, do not balance")
    p.add_argument("--splits", nargs="+", default=["Train"],
                   help="splits to balance (Valid/Test are copied as-is)")
    p.add_argument("--target-size", type=int, default=None,
                   help="images per class; if omitted it is derived from --strategy")
    p.add_argument("--strategy", default="median",
                   choices=["median", "mean", "min", "max"],
                   help="how to derive the target size when --target-size is unset")
    p.add_argument("--similarity-threshold", type=float, default=0.30,
                   help="uniqueness threshold in [0, 1]")
    p.add_argument("--threshold-mode", default="similarity",
                   choices=["similarity", "distance"],
                   help="'similarity': keep if max similarity < threshold; "
                        "'distance': keep if min normalised Hamming distance > threshold")
    p.add_argument("--hash-size", type=int, default=8,
                   help="pHash side length (8 -> 64-bit hash)")
    p.add_argument("--copy-other-splits", action="store_true",
                   help="also copy the untouched splits into --out-root")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # 0) make sure the raw dataset is available locally ---------------------- #
    try:
        data_root = ensure_dataset(
            data_root=args.data_root,
            url=args.url,
            force=args.download,
            keep_zip=args.keep_zip,
        )
    except Exception as exc:
        print(f"error: could not obtain the dataset: {exc}", file=sys.stderr)
        print(f"       manual download: {DFKI_SHARE_URL}", file=sys.stderr)
        return 1

    raw_counts = verify_layout(data_root, strict=not args.verify_only)
    if args.verify_only:
        print("\nVerification finished (nothing was balanced).")
        return 0

    args.out_root.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    reports = []
    for split in args.splits:
        split_dir = data_root / split
        if not split_dir.is_dir():
            print(f"warning: split not found, skipped: {split_dir}", file=sys.stderr)
            continue
        reports.append(
            balance_split(
                split_dir=split_dir,
                out_dir=args.out_root / split,
                target_size=args.target_size,
                strategy=args.strategy,
                threshold=args.similarity_threshold,
                hash_size=args.hash_size,
                mode=args.threshold_mode,
                seed=args.seed,
            )
        )

    if args.copy_other_splits:
        for split in SPLITS:
            src, dst = data_root / split, args.out_root / split
            if src.is_dir() and not dst.exists():
                print(f"copying untouched split: {split}")
                shutil.copytree(src, dst)

    summary = {
        "source": DFKI_SHARE_URL,
        "data_root": str(data_root),
        "out_root": str(args.out_root),
        "raw_counts": raw_counts,
        "classes": list(CLASSES),
        "similarity_threshold": args.similarity_threshold,
        "threshold_mode": args.threshold_mode,
        "hash_size": args.hash_size,
        "strategy": args.strategy,
        "seed": args.seed,
        "elapsed_sec": round(time.time() - t0, 2),
        "splits": reports,
    }
    out_json = args.out_root / "balance_report.json"
    out_json.write_text(json.dumps(summary, indent=2))

    print(f"\nDone in {summary['elapsed_sec']} s — report written to {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
