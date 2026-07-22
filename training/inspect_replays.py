"""
inspect_replays.py
Scans a replay directory recursively, finds every ZIP file, and prints the
complete directory tree of every archive without making ANY assumptions about
its contents.
Usage:
    python inspect_replays.py "path/to/replays"
Example:
    python inspect_replays.py "../replays"
"""
from __future__ import annotations
import argparse
import os
import zipfile
from pathlib import Path
def human_size(size: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{size} B"

def print_zip_contents(zip_path: Path) -> None:
    print("=" * 120)
    print(f"ZIP : {zip_path}")
    print("=" * 120)
    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            infos = archive.infolist()
            if not infos:
                print("EMPTY ARCHIVE")
                print()
                return
            infos.sort(key=lambda x: x.filename.lower())
            total_uncompressed = 0
            total_compressed = 0
            for info in infos:
                total_uncompressed += info.file_size
                total_compressed += info.compress_size
            print(f"Entries            : {len(infos)}")
            print(f"Compressed Size    : {human_size(total_compressed)}")
            print(f"Uncompressed Size  : {human_size(total_uncompressed)}")
            print()
            print("CONTENTS")
            print("-" * 120)
            for info in infos:
                if info.is_dir():
                    typ = "DIR "
                else:
                    typ = "FILE"
                print(
                    f"{typ:<5} | "
                    f"{human_size(info.file_size):>10} | "
                    f"{human_size(info.compress_size):>10} | "
                    f"{info.filename}"
                )
            print()
    except Exception as e:
        print(f"FAILED TO READ: {e}")
        print()

def find_zip_files(root: Path):
    return sorted(root.rglob("*.zip"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "replay_root",
        type=Path,
        help="Root directory containing replay ZIPs",
    )
    args = parser.parse_args()
    replay_root = args.replay_root.resolve()
    if not replay_root.exists():
        print(f"Directory not found:\n{replay_root}")
        return
    zip_files = find_zip_files(replay_root)
    print()
    print("=" * 120)
    print("REPLAY ZIP INSPECTOR")
    print("=" * 120)
    print(f"Replay Root : {replay_root}")
    print(f"ZIP Files   : {len(zip_files)}")
    print()
    if not zip_files:
        print("No ZIP files found.")
        return
    for index, zip_path in enumerate(zip_files, start=1):
        print(f"[{index}/{len(zip_files)}]")
        print_zip_contents(zip_path)
    print("=" * 120)
    print("DONE")
    print("=" * 120)

if __name__ == "__main__":
    main()