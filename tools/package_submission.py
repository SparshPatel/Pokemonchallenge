"""Package the submission into ``submission.tar.gz`` for Kaggle upload.

The Simulation category expects a gzipped tarball whose **top level** contains
``main.py``, ``deck.csv`` and the ``cg/`` engine package (no nested wrapper
directory). This script tars the *contents* of ``submission/`` so paths are flat.

Run from the project root::

    python tools/package_submission.py --name baseline
"""
from __future__ import annotations

import argparse
import os
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_DIR = ROOT / "submission"
OUT_DIR = ROOT / "artifacts"

REQUIRED = ["main.py", "deck.csv"]
EXPECTED_PACKAGES = ["agent", "cg"]
EXCLUDE = {"__pycache__", ".pyc", ".DS_Store"}


def _should_include(path: Path) -> bool:
    parts = set(path.parts)
    if parts & EXCLUDE:
        return False
    if path.suffix == ".pyc":
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="submission")
    args = ap.parse_args()

    missing = [f for f in REQUIRED if not (SUBMISSION_DIR / f).exists()]
    if missing:
        raise SystemExit(f"Missing required files: {missing}")

    if not (SUBMISSION_DIR / "cg").exists():
        print(
            "WARNING: submission/cg/ is missing. The package will NOT run on "
            "Kaggle until the cabt engine is included. Packaging anyway for "
            "local inspection."
        )

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"{args.name}.tar.gz"

    with tarfile.open(out_path, "w:gz") as tar:
        for item in sorted(SUBMISSION_DIR.iterdir()):
            if not _should_include(item):
                continue
            # arcname relative to submission/ => flat top level in the tar.
            tar.add(item, arcname=item.name, filter=_tar_filter)

    size_mb = out_path.stat().st_size / 1e6
    print(f"Wrote {out_path} ({size_mb:.2f} MB)")
    print("Top-level entries:")
    with tarfile.open(out_path, "r:gz") as tar:
        tops = sorted({m.name.split("/")[0] for m in tar.getmembers()})
        for t in tops:
            print(f"  {t}")


def _tar_filter(tarinfo: tarfile.TarInfo):
    base = os.path.basename(tarinfo.name)
    if base in EXCLUDE or tarinfo.name.endswith(".pyc") or "__pycache__" in tarinfo.name:
        return None
    return tarinfo


if __name__ == "__main__":
    main()
