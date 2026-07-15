from pathlib import Path
import json
import csv

from replay_analyzer import load_directory
import analysis_engine


BASE_DIR = Path(__file__).resolve().parent.parent

REPLAY_FOLDER = BASE_DIR / "replays"
OUTPUT_FOLDER = BASE_DIR / "output"

OUTPUT_FOLDER.mkdir(exist_ok=True)


def save_summary(summary):

    with open(OUTPUT_FOLDER / "summary.json", "w") as f:
        json.dump(summary, f, indent=4)


def save_games_csv(replays):

    with open(OUTPUT_FOLDER / "games.csv", "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "file",
                "winner",
                "turns",
            ]
        )

        for replay in replays:

            writer.writerow(
                [
                    replay.filename,
                    replay.winner,
                    len(replay.turns),
                ]
            )


def main():

    print()
    print("Loading replays...")

    replays = load_directory(REPLAY_FOLDER)

    print(f"Loaded {len(replays)} games")

    results = analysis_engine.analyze(replays)

    analysis_engine.print_results(results)

    save_summary(results["summary"])
    save_games_csv(replays)

    print()
    print(f"Saved {OUTPUT_FOLDER / 'summary.json'}")
    print(f"Saved {OUTPUT_FOLDER / 'games.csv'}")
    print()

if __name__ == "__main__":
    main()