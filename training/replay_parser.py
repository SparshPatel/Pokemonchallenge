"""
Replay parser for Pokemon Challenge.
This parser recursively scans a replay directory, finds every ZIP file,
opens every archive, discovers replay files inside, and reports how many
games were found.
No assumptions are made about ZIP names.
Example
-------
python replay_parser.py --replays ../replays
Output
------
Replay Summary
--------------
ZIP files found : 7
Games found     : 192
1.zip
    game001.json
    game002.json
    ...
submission430.zip
    game001.json
    ...
"""
from __future__ import annotations
import argparse
import sys
import zipfile
from pathlib import Path
SUPPORTED_EXTENSIONS = {
    ".json",
    ".log",
    ".txt",
}

class ReplayParser:
    def __init__(self, replay_root: Path):
        self.replay_root = replay_root
        self.zip_files: list[Path] = []
        self.games = []

    # -------------------------------------------------------------
    def discover_zip_files(self) -> None:
        self.zip_files = sorted(
            self.replay_root.rglob("*.zip")
        )

    # -------------------------------------------------------------
    def parse(self) -> None:
        self.discover_zip_files()
        if not self.zip_files:
            print("No ZIP files found.")
            return
        for zip_path in self.zip_files:
            try:
                with zipfile.ZipFile(zip_path, "r") as archive:
                    members = sorted(
                        archive.namelist()
                    )
                    game_files = []
                    for member in members:
                        if member.endswith("/"):
                            continue
                        extension = Path(member).suffix.lower()
                        if extension not in SUPPORTED_EXTENSIONS:
                            continue
                        game_files.append(member)
                    self.games.append(
                        {
                            "zip": zip_path,
                            "games": game_files,
                        }
                    )
            except zipfile.BadZipFile:
                print(
                    f"[CORRUPT] {zip_path}"
                )
            except Exception as e:
                print(
                    f"[ERROR] {zip_path}\n{e}"
                )

    # -------------------------------------------------------------
    def report(self) -> None:
        total_games = sum(
            len(x["games"])
            for x in self.games
        )
        print()
        print("=" * 60)
        print("Replay Summary")
        print("=" * 60)
        print()
        print(
            f"Replay folder : {self.replay_root}"
        )
        print(
            f"ZIP files     : {len(self.games)}"
        )
        print(
            f"Games found   : {total_games}"
        )
        print()
        for entry in self.games:
            print(entry["zip"].name)
            if not entry["games"]:
                print("    (no replay files found)")
                print()
                continue
            for game in entry["games"]:
                print(f"    {game}")
            print()

    # -------------------------------------------------------------
    def run(self):
        self.parse()
        self.report()

# ===================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Replay parser"
    )
    parser.add_argument(
        "--replays",
        required=True,
        type=Path,
        help="Root replay directory",
    )
    args = parser.parse_args()
    replay_root = args.replays
    if not replay_root.exists():
        print(
            f"Replay folder does not exist:\n{replay_root}"
        )
        sys.exit(1)
    ReplayParser(
        replay_root,
    ).run()

if __name__ == "__main__":
    main()