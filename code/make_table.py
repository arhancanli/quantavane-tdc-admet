"""Build the results table for README.md from results/*.json + the dated leaderboard file.

The would-be rank is computed by inserting our mean into the listed column and
counting how many listed entries are strictly better. Ties are reported as a
shared rank, which is what the board itself does (e.g. Clearance_Microsome has
three entries at 0.625). The rank is what we *would* hold if the result were
listed as-is; nothing has been submitted.
"""

from __future__ import annotations

import glob
import json
import os

import choose

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
LB_FILE = os.path.join(BASE, "leaderboards_2026-11-08.json")


def inner_score(rec: dict):
    """The inner-CV score of the blend this run selected.

    Newer result files store it directly. Older ones predate that field, so it is
    recovered from the stored blend candidates by matching the selected members.
    Returning None would make a round incomparable, so this must not silently
    fail: a missing score raises rather than defaulting to the other round.
    """
    if rec.get("selected_inner_mean") is not None:
        return float(rec["selected_inner_mean"])
    members = rec["selected_members"]
    for cand in rec["inner_cv"]["blend_candidates"]:
        if cand["members"] == members:
            return float(cand["inner_mean"])
    raise KeyError(f"cannot recover inner score for {rec['task']}")


COMMON_SEEDS = (101, 102, 103)


def inner_common(rec: dict) -> float:
    """The selected blend's inner score on the tuning folds every round shares.

    From 2026-11-08T18:13Z the round-3 runs of five tasks use fifteen tuning seeds
    instead of three, to make the selector less noisy. A fifteen-seed mean is not
    comparable with a three-seed mean, so the cross-round choice is made on the seeds
    the rounds have in common. Nothing is refitted: the per-fold scores are already in
    the result file. For a round that used exactly the common seeds this returns the
    same number as before, so older rounds are unaffected.
    """
    cv = rec["inner_cv"]
    seeds = list(cv["tune_seeds"])
    for cand in cv["blend_candidates"]:
        if cand["members"] == rec["selected_members"]:
            keep = [f for s, f in zip(seeds, cand["inner_folds"]) if s in COMMON_SEEDS]
            if len(keep) != len(COMMON_SEEDS):
                raise KeyError(
                    f"{rec['task']}: tuning seeds {seeds} do not contain all of "
                    f"{COMMON_SEEDS}, so this round cannot be compared with the others"
                )
            return float(sum(keep) / len(keep))
    raise KeyError(f"cannot recover inner folds for {rec['task']}")


def would_be_rank(ours: float, rows, higher_better: bool):
    """(rank, n_listed_after_insert, beaten_by) for our value against the listed column."""
    better = sum(1 for _, _, v, _ in rows if (v > ours if higher_better else v < ours))
    return better + 1, len(rows) + 1, better


def main() -> None:
    lb = json.load(open(LB_FILE))["tasks"]
    all_records = []
    for path in sorted(glob.glob(os.path.join(BASE, "results", "*.json"))):
        if path.endswith("_summary.json"):
            continue
        # *_plain.json is the lane's control arm: one fixed LightGBM config, run so
        # the registry has an A/B record. It is not a round of the model and must
        # never be a candidate for the reported number, or the table would quietly
        # pick whichever arm scored better.
        if path.endswith("_plain.json"):
            continue
        rec = json.load(open(path))
        # A tuning-only pass re-estimates an earlier round on more folds and has no
        # test score at all. It informs the choice (see choose.py); it is never a
        # candidate for the reported row.
        if rec.get("tune_only"):
            continue
        rec["_file"] = os.path.basename(path)
        all_records.append(rec)

    # A task may have been run more than once (a second round with a wider
    # candidate list). Pick the round with the better INNER-CV score, never the
    # better test score: choosing on test would make the reported number a
    # selected maximum rather than a held-out result.
    by_task: dict[str, list[dict]] = {}
    for r in all_records:
        by_task.setdefault(r["task"], []).append(r)
    records = []
    for task, rounds in by_task.items():
        if len(rounds) == 1:
            records.append(rounds[0])
            continue
        name, best, why = choose.pick([(r["_file"], r) for r in rounds],
                                      os.path.join(BASE, "results"))
        records.append(dict(best))
        rows = ", ".join(
            f"{d['file']} inner {d['inner_on_shared']:.4f} -> test {d['test_mean']:.3f}"
            + ("" if d["reported"] else " (not reported)")
            for d in why["rounds"]
        )
        print(f"<!-- {task}: {len(rounds)} rounds compared on seeds "
              f"{why['compared_on_seeds']}. {rows} -->")

    print("| Task | Metric | Our mean ± std (5 seeds) | Board top | Would-be rank | Our model |")
    print("|---|---|---|---|---|---|")
    summary = []
    for r in sorted(records, key=lambda x: x["task"]):
        task, metric = r["task"], r["metric"]
        mean, std = r["mean"], r["std"]
        if task in lb:
            rows = lb[task]["rows"]
            hb = lb[task]["higher_is_better"]
            rank, n, beaten = would_be_rank(mean, rows, hb)
            top = f"{rows[0][1]} {rows[0][2]:.3f}"
            rank_s = f"**{rank} of {n}**"
        else:
            rank, n, beaten, top, rank_s = None, None, None, "not read", "n/a"
        members = " + ".join(r["selected_members"])
        print(f"| {task} | {metric} | {mean:.3f} ± {std:.3f} | {top} | {rank_s} | {members} |")
        summary.append(
            {"task": task, "metric": metric, "mean": mean, "std": std,
             "would_be_rank": rank, "of": n, "listed_better_than_us": beaten,
             "selected_members": r["selected_members"], "wall_seconds": r["wall_seconds"]}
        )

    out = os.path.join(BASE, "results", "_summary.json")
    with open(out, "w") as fh:
        json.dump({"generated_from_leaderboards": os.path.basename(LB_FILE), "rows": summary}, fh, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
