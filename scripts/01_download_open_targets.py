#!/usr/bin/env python3
"""
Script 01: Download Open Targets Platform release 26.03 Parquet datasets.

Downloads only the datasets needed for conformal drug repurposing:
  - association_by_datasource_direct  (27-source scores per target-disease pair)
  - association_overall_direct        (aggregate association score)
  - disease                           (disease metadata + therapeutic areas)
  - target                            (target/gene metadata)
  - drug_molecule                     (drug/molecule metadata)
  - drug_mechanism_of_action          (drug → target links)
  - clinical_indication               (known drug-disease indications — label source)
  - interaction                       (protein-protein interactions for target similarity)

Estimated total download: ~5–10 GB (vs ~100 GB for full platform).
"""

import os
import sys
import time
import subprocess
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────
OT_RELEASE = "26.03"
BASE_URL = f"https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/{OT_RELEASE}/output"

DATASETS = [
    "association_by_datasource_direct",
    "association_overall_direct",
    "disease",
    "target",
    "drug_molecule",
    "drug_mechanism_of_action",
    "clinical_indication",
    "interaction",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def fmt_size(nbytes: float) -> str:
    """Human-readable file size."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(nbytes) < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def fmt_duration(seconds: float) -> str:
    """Human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s"


def download_dataset(dataset: str, ds_idx: int, ds_total: int) -> None:
    """Download a single Open Targets Parquet dataset using curl."""
    import re

    url = f"{BASE_URL}/{dataset}/"
    dest = RAW_DIR / dataset
    header = f"[{ds_idx}/{ds_total}] {dataset}"

    dest.mkdir(parents=True, exist_ok=True)

    print(f"\n  ↓ {header}")
    print(f"    Fetching file listing...", end=" ", flush=True)
    t0 = time.time()

    # Get directory listing via curl
    result = subprocess.run(
        ["curl", "-s", "-L", url],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        print(f"FAILED (curl error {result.returncode})")
        return

    # Parse parquet/snappy file links from HTML directory listing
    # Open Targets uses Spark-style part-* filenames
    all_links = re.findall(r'href="([^"]+)"', result.stdout)
    parquet_files = [
        f for f in all_links
        if f.endswith(".parquet") or f.endswith(".snappy.parquet")
        or (f.startswith("part-") and not f.endswith("/"))
    ]

    # Also check for subdirectory structure (some OT datasets nest one level)
    if not parquet_files:
        subdirs = [f for f in all_links if f.endswith("/") and f not in ("../", "./")]
        for subdir in subdirs:
            sub_result = subprocess.run(
                ["curl", "-s", "-L", f"{url}{subdir}"],
                capture_output=True, text=True
            )
            sub_links = re.findall(r'href="([^"]+)"', sub_result.stdout)
            for sl in sub_links:
                if sl.endswith(".parquet") or sl.endswith(".snappy.parquet") or sl.startswith("part-"):
                    parquet_files.append(f"{subdir}{sl}")

    if not parquet_files:
        print(f"no parquet files found")
        print(f"    Links found: {all_links[:15]}")
        return

    # Check if already complete (all expected files present and non-empty)
    total_files = len(parquet_files)
    existing = []
    for fname in parquet_files:
        flat_name = fname.replace("/", "_") if "/" in fname else fname
        fp = dest / flat_name
        if fp.exists() and fp.stat().st_size > 0:
            existing.append(fp)

    if len(existing) == total_files:
        size = sum(f.stat().st_size for f in existing)
        print(f"{total_files} files — already complete ({fmt_size(size)}) — skipping")
        return

    print(f"found {total_files} files ({len(existing)} already cached)")
    downloaded_bytes = sum(f.stat().st_size for f in existing)
    skipped = len(existing)

    for i, fname in enumerate(parquet_files):
        file_url = f"{url}{fname}"
        flat_name = fname.replace("/", "_") if "/" in fname else fname
        file_dest = dest / flat_name

        if file_dest.exists() and file_dest.stat().st_size > 0:
            continue

        subprocess.run(
            ["curl", "-s", "-L", "-o", str(file_dest), file_url],
            check=True
        )

        if file_dest.exists():
            downloaded_bytes += file_dest.stat().st_size

        # Per-file progress line (overwrite in place)
        elapsed = time.time() - t0
        rate = downloaded_bytes / elapsed if elapsed > 0 else 0
        pct = (i + 1) / total_files * 100
        bar_filled = int(pct / 2.5)
        bar = "█" * bar_filled + "░" * (40 - bar_filled)
        print(
            f"\r    {bar} {pct:5.1f}%  "
            f"{i + 1}/{total_files} files  "
            f"{fmt_size(downloaded_bytes)}  "
            f"{fmt_size(rate)}/s  "
            f"{fmt_duration(elapsed)}",
            end="", flush=True
        )

    elapsed = time.time() - t0
    print()  # newline after progress bar

    n_files = len(list(dest.rglob("*.parquet")))
    if n_files > 0:
        size = sum(f.stat().st_size for f in dest.rglob("*.parquet"))
        rate = size / elapsed if elapsed > 0 else 0
        status = f"    ✓ {n_files} files, {fmt_size(size)}, {fmt_duration(elapsed)}"
        if rate > 0:
            status += f" ({fmt_size(rate)}/s)"
        if skipped > 0:
            status += f" [{skipped} already cached]"
        print(status)
    else:
        print(f"    ✗ Downloaded files but none have .parquet extension")
        print(f"    Files in {dest}: {list(dest.iterdir())[:10]}")


def check_parquet_integrity(filepath: Path) -> bool:
    """Verify a file has valid Parquet magic bytes (PAR1 at start and end)."""
    try:
        with open(filepath, "rb") as f:
            header = f.read(4)
            f.seek(-4, 2)
            footer = f.read(4)
        return header == b"PAR1" and footer == b"PAR1"
    except Exception:
        return False


def verify_downloads() -> bool:
    """Check all datasets have complete, valid parquet files."""
    print("\n── Verification ─────────────────────────────────")
    all_ok = True
    for dataset in DATASETS:
        dest = RAW_DIR / dataset
        files = list(dest.rglob("*.parquet")) if dest.exists() else []
        if not files:
            print(f"  ✗ {dataset}: MISSING")
            all_ok = False
            continue

        corrupt = [f for f in files if not check_parquet_integrity(f)]
        size = sum(f.stat().st_size for f in files)

        if corrupt:
            print(f"  ⚠ {dataset}: {len(files)} files ({fmt_size(size)}), "
                  f"{len(corrupt)} CORRUPT:")
            for cf in corrupt:
                print(f"      {cf.name} ({fmt_size(cf.stat().st_size)})")
                cf.unlink()  # delete corrupt file so re-run will fetch it
                print(f"        → deleted (will re-download on next run)")
            all_ok = False
        else:
            print(f"  ✓ {dataset}: {len(files)} files, {fmt_size(size)}")
    return all_ok


def main():
    print(f"Open Targets Platform {OT_RELEASE} — selective download")
    print(f"Destination: {RAW_DIR}")
    print(f"Datasets: {len(DATASETS)}")
    print("=" * 60)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    for i, dataset in enumerate(DATASETS, 1):
        download_dataset(dataset, ds_idx=i, ds_total=len(DATASETS))

    total_elapsed = time.time() - t_start
    print(f"\nTotal download time: {fmt_duration(total_elapsed)}")

    ok = verify_downloads()

    if ok:
        print("\n✓ All datasets downloaded successfully.")
        # Write a manifest
        manifest = RAW_DIR / "MANIFEST.txt"
        with open(manifest, "w") as f:
            f.write(f"Open Targets Platform release {OT_RELEASE}\n")
            f.write(f"Downloaded: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for dataset in DATASETS:
                dest = RAW_DIR / dataset
                n = len(list(dest.rglob("*.parquet")))
                sz = sum(fp.stat().st_size for fp in dest.rglob("*.parquet")) / 1e6
                f.write(f"{dataset}: {n} files, {sz:.0f} MB\n")
        print(f"Manifest written to {manifest}")
    else:
        print("\n✗ Some datasets missing — check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
