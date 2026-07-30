"""
Download and prepare the SolarFCD dataset from the DFKI Cloud (ownCloud) share.

The public share exposes a "download everything as one zip" endpoint:
    https://cloud.dfki.de/owncloud/index.php/s/<TOKEN>/download

Usage
-----
    python src/download_dataset.py --out data
    python src/download_dataset.py --out data --keep-zip
    python src/download_dataset.py --out data --verify-only

Notes
-----
* The archive is ~5 GB, so make sure you have ~11 GB free (zip + extracted).
* The download resumes automatically if it was interrupted (HTTP Range).
"""

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

import requests

SHARE_TOKEN = "SZYo97gdwY2k92Z"
BASE_URL = f"https://cloud.dfki.de/owncloud/index.php/s/{SHARE_TOKEN}"
DOWNLOAD_URL = f"{BASE_URL}/download"

EXPECTED_SPLITS = ["Train", "Valid", "Test"]
EXPECTED_CLASSES = [
    "Electrical_Fault",
    "Healthy",
    "Structural_Fault",
    "Surface_Obstruction",
]
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


# --------------------------------------------------------------------------- #
# download
# --------------------------------------------------------------------------- #
def human(n: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024.0:
            return f"{n:3.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


def download(url: str, dest: Path, chunk: int = 1 << 20) -> Path:
    """Stream the share to disk with resume support and a progress line."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    done = dest.stat().st_size if dest.exists() else 0

    headers = {"Range": f"bytes={done}-"} if done else {}
    with requests.get(url, stream=True, headers=headers, timeout=60) as r:
        if done and r.status_code == 416:
            print(f"[=] Archive already complete: {dest} ({human(done)})")
            return dest
        r.raise_for_status()

        total = int(r.headers.get("Content-Length", 0)) + done
        mode = "ab" if done and r.status_code == 206 else "wb"
        if mode == "wb":
            done = 0

        print(f"[>] Downloading {url}")
        print(f"[>] Target      {dest}  (total {human(total) if total else 'unknown'})")

        with open(dest, mode) as fh:
            for block in r.iter_content(chunk_size=chunk):
                if not block:
                    continue
                fh.write(block)
                done += len(block)
                pct = f"{done / total * 100:5.1f}%" if total else "  ?  "
                print(f"\r    {pct}  {human(done)}", end="", flush=True)
    print("\n[=] Download finished.")
    return dest


def extract(zip_path: Path, out_dir: Path) -> None:
    print(f"[>] Extracting {zip_path.name} -> {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.infolist()
        for i, m in enumerate(members, 1):
            zf.extract(m, out_dir)
            if i % 500 == 0 or i == len(members):
                print(f"\r    {i}/{len(members)} entries", end="", flush=True)
    print("\n[=] Extraction finished.")


# --------------------------------------------------------------------------- #
# layout normalisation + verification
# --------------------------------------------------------------------------- #
def find_root(out_dir: Path) -> Path:
    """
    Locate the directory that directly contains Train/ Valid/ Test/.
    ownCloud wraps the export in a folder named after the share (SolarFCD),
    and some exports add one extra nesting level.
    """
    for candidate in [out_dir, *sorted(p for p in out_dir.rglob("*") if p.is_dir())]:
        names = {p.name for p in candidate.iterdir() if p.is_dir()}
        if set(EXPECTED_SPLITS).issubset(names):
            return candidate
    raise SystemExit(
        f"[!] Could not find Train/Valid/Test under {out_dir}.\n"
        f"    Inspect the extracted folder manually and pass its path with --data-root."
    )


def count_images(split_dir: Path) -> dict:
    counts = {}
    for cls in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        counts[cls.name] = sum(
            1 for f in cls.rglob("*") if f.suffix.lower() in IMG_EXT
        )
    return counts


def verify(root: Path) -> bool:
    print(f"\n[>] Verifying dataset at {root}\n")
    ok = True
    grand_total = 0
    for split in EXPECTED_SPLITS:
        sd = root / split
        if not sd.is_dir():
            print(f"    [!] MISSING split: {split}")
            ok = False
            continue
        counts = count_images(sd)
        total = sum(counts.values())
        grand_total += total
        print(f"    {split:<6} {total:>5} images")
        for cls in EXPECTED_CLASSES:
            n = counts.get(cls)
            flag = " " if n else "!"
            print(f"      [{flag}] {cls:<22} {n if n is not None else 'MISSING':>5}")
            if not n:
                ok = False
        extra = set(counts) - set(EXPECTED_CLASSES)
        if extra:
            print(f"      [!] unexpected folders: {sorted(extra)}")
            ok = False
        print()
    print(f"    TOTAL  {grand_total} images")
    print("\n[=] Verification: " + ("PASSED" if ok else "FAILED"))
    return ok


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch + verify the SolarFCD dataset.")
    ap.add_argument("--out", default="data", help="output directory (default: data)")
    ap.add_argument("--data-root", default=None,
                    help="skip auto-detection and verify this folder directly")
    ap.add_argument("--keep-zip", action="store_true", help="do not delete the archive")
    ap.add_argument("--verify-only", action="store_true",
                    help="only check an already-downloaded copy")
    args = ap.parse_args()

    out_dir = Path(args.out).expanduser().resolve()

    if args.verify_only or args.data_root:
        root = Path(args.data_root).resolve() if args.data_root else find_root(out_dir)
        return 0 if verify(root) else 1

    zip_path = out_dir / "SolarFCD.zip"
    free = shutil.disk_usage(out_dir.parent if out_dir.exists() else Path.cwd()).free
    if free < 11 * 1024 ** 3:
        print(f"[!] Warning: only {human(free)} free; ~11 GB recommended.")

    download(DOWNLOAD_URL, zip_path)
    extract(zip_path, out_dir)
    if not args.keep_zip:
        zip_path.unlink(missing_ok=True)
        print("[=] Archive removed (use --keep-zip to retain it).")

    root = find_root(out_dir)
    print(f"\n[=] DATA_ROOT = {root}")
    print("    Use it like:  python src/train.py --data-root", root)
    return 0 if verify(root) else 1


if __name__ == "__main__":
    sys.exit(main())
