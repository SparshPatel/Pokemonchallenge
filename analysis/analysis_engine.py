from __future__ import annotations

from collections import Counter


# ==========================================================
# SUMMARY
# ==========================================================

def replay_summary(replays):

    games = len(replays)
    wins = sum(r.winner == 0 for r in replays)
    losses = games - wins

    turns = [len(r.turns) for r in replays]

    return {
        "games": games,
        "wins": wins,
        "losses": losses,
        "win_rate": round((wins / games * 100) if games else 0, 2),
        "average_turns": round(sum(turns) / games, 2) if games else 0,
        "shortest_game": min(turns) if turns else 0,
        "longest_game": max(turns) if turns else 0,
    }


# ==========================================================
# STATE DIFFERENCE
# ==========================================================

def analyse_state(replay):

    stats = Counter()

    previous = None

    for wrapper in replay.steps:

        if not isinstance(wrapper, list):
            continue

        current = None

        for record in wrapper:

            observation = record.get("observation")

            if not observation:
                continue

            state = observation.get("current")

            if state is not None:
                current = state
                break

        if current is None:
            continue

        if previous is None:
            previous = current
            continue

        p0_before = previous["players"][0]
        p0_after = current["players"][0]

        p1_before = previous["players"][1]
        p1_after = current["players"][1]

        #
        # Draws
        #

        if p0_after["handCount"] > p0_before["handCount"]:
            stats["our_draws"] += (
                p0_after["handCount"] - p0_before["handCount"]
            )

        if p1_after["handCount"] > p1_before["handCount"]:
            stats["opp_draws"] += (
                p1_after["handCount"] - p1_before["handCount"]
            )

        #
        # Bench
        #

        if len(p0_after["bench"]) > len(p0_before["bench"]):
            stats["our_bench"] += (
                len(p0_after["bench"]) - len(p0_before["bench"])
            )

        if len(p1_after["bench"]) > len(p1_before["bench"]):
            stats["opp_bench"] += (
                len(p1_after["bench"]) - len(p1_before["bench"])
            )

        #
        # Discard
        #

        if len(p0_after["discard"]) > len(p0_before["discard"]):
            stats["our_discard"] += (
                len(p0_after["discard"]) - len(p0_before["discard"])
            )

        if len(p1_after["discard"]) > len(p1_before["discard"]):
            stats["opp_discard"] += (
                len(p1_after["discard"]) - len(p1_before["discard"])
            )

        #
        # Active switches
        #

        if p0_before["active"] != p0_after["active"]:
            stats["our_active_changes"] += 1

        if p1_before["active"] != p1_after["active"]:
            stats["opp_active_changes"] += 1

        previous = current

    return stats


def build_state_report(replays):

    total = Counter()
    wins = Counter()
    losses = Counter()

    for replay in replays:

        stats = analyse_state(replay)

        total.update(stats)

        if replay.winner == 0:
            wins.update(stats)
        else:
            losses.update(stats)

    return total, wins, losses


# ==========================================================
# ACTION DETECTOR
# ==========================================================

def analyse_actions(replay):

    stats = Counter()

    previous = None

    for wrapper in replay.steps:

        if not isinstance(wrapper, list):
            continue

        current = None

        for record in wrapper:

            observation = record.get("observation")

            if not observation:
                continue

            state = observation.get("current")

            if state is not None:
                current = state
                break

        if current is None:
            continue

        if previous is None:
            previous = current
            continue

        before = previous["players"][0]
        after = current["players"][0]
                #
        # Bench additions
        #
        if len(after["bench"]) > len(before["bench"]):
            stats["bench"] += len(after["bench"]) - len(before["bench"])

        #
        # Energy attachment
        #
        if current.get("energyAttached"):
            stats["energy"] += 1

        #
        # Prize taken
        #
        before_prizes = sum(p is not None for p in before.get("prize", []))
        after_prizes = sum(p is not None for p in after.get("prize", []))

        if after_prizes < before_prizes:
            stats["prize_taken"] += before_prizes - after_prizes

        #
        # Active changed
        #
        if before["active"] != after["active"]:
            stats["active_change"] += 1

        #
        # Attack approximation
        #
        if before["active"] and after["active"]:

            b = before["active"][0]
            a = after["active"][0]

            if (
                isinstance(b, dict)
                and isinstance(a, dict)
                and b.get("serial") == a.get("serial")
                and a.get("damage", 0) > b.get("damage", 0)
            ):
                stats["attack"] += 1

        previous = current

    return stats


def build_action_report(replays):

    total = Counter()
    wins = Counter()
    losses = Counter()

    for replay in replays:

        stats = analyse_actions(replay)

        total.update(stats)

        if replay.winner == 0:
            wins.update(stats)
        else:
            losses.update(stats)

    return total, wins, losses


# ==========================================================
# PRINTING
# ==========================================================

def print_summary(summary):

    print()
    print("=" * 70)
    print("REPLAY SUMMARY")
    print("=" * 70)

    for k, v in summary.items():
        print(f"{k:<20} : {v}")

    print("=" * 70)


# ==========================================================
# PRINT HELPERS
# ==========================================================

def _print_counter(title, counter):

    print(title)

    if not counter:
        print()
        return

    for key in sorted(counter):
        print(f"{key:<24}{counter[key]}")

    print()

def print_state_report(report):

    total, wins, losses = report

    print()
    print("=" * 70)
    print("STATE DIFFERENCE REPORT")
    print("=" * 70)
    print()

    _print_counter("OVERALL", total)
    _print_counter("WINS", wins)
    _print_counter("LOSSES", losses)

    print("=" * 70)


def print_action_report(report):

    total, wins, losses = report

    print()
    print("=" * 70)
    print("ACTION REPORT")
    print("=" * 70)
    print()

    _print_counter("OVERALL", total)
    _print_counter("WINS", wins)
    _print_counter("LOSSES", losses)
    
# ==========================================================
# PUBLIC PRINT FUNCTIONS
# ==========================================================

def print_results(results):

    summary = results["summary"]
    state = results["state"]
    actions = results["actions"]

    print()
    print("=" * 70)
    print("REPLAY SUMMARY")
    print("=" * 70)

    for k, v in summary.items():
        print(f"{k:<20} : {v}")

    print("=" * 70)

    print()
    print("=" * 70)
    print("STATE DIFFERENCE REPORT")
    print("=" * 70)
    print()

    total, wins, losses = state

    _print_counter("OVERALL", total)
    _print_counter("WINS", wins)
    _print_counter("LOSSES", losses)

    print("=" * 70)

    print()
    print("=" * 70)
    print("ACTION REPORT")
    print("=" * 70)
    print()

    total, wins, losses = actions

    _print_counter("OVERALL", total)
    _print_counter("WINS", wins)
    _print_counter("LOSSES", losses)
    
# ==========================================================
# MASTER ENTRY POINT
# ==========================================================

def analyze(replays):

    return {
        "summary": replay_summary(replays),
        "state": build_state_report(replays),
        "actions": build_action_report(replays),
    }
