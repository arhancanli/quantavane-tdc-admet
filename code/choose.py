"""Which round of a task gets reported. One module, so the tools cannot disagree.

This exists because they once did: `make_table.py` said BBB_Martins should report
0.923 while `make_packet.py` said 0.914, because only one of them had been taught the
comparison rule. The rule now lives here and all three tools import it.

The rule, as registered for these lanes before the runs it judges:

1. A round is reported on the strength of its INNER cross-validation score on training
   rows. Never its test score. Comparing test scores across rounds would make the
   published number a selected maximum.
2. Rounds tuned on different numbers of inner folds are not directly comparable, so
   they are compared on the folds they share.
3. Where an earlier round has been re-tuned on the later round's folds (a `tune_only`
   pass, which produces no test score), that re-estimate is used, so the comparison is
   made on the larger shared set instead of the smaller one.

Rule 3 is more measurement, not a different rule: it only ever widens the shared set.
"""

from __future__ import annotations

import glob
import json
import os

COMMON_SEEDS = (101, 102, 103)


def selected_candidate(rec: dict) -> dict:
    for cand in rec["inner_cv"]["blend_candidates"]:
        if cand["members"] == rec["selected_members"]:
            return cand
    raise KeyError(f"cannot find the selected blend of {rec['task']}")


def inner_own(rec: dict) -> float:
    """The round's inner score over its own folds."""
    if rec.get("selected_inner_mean") is not None:
        return float(rec["selected_inner_mean"])
    return float(selected_candidate(rec)["inner_mean"])


def inner_on(rec: dict, seeds) -> float | None:
    """The round's inner score restricted to `seeds`, or None if it lacks any of them."""
    have = list(rec["inner_cv"]["tune_seeds"])
    folds = selected_candidate(rec)["inner_folds"]
    keep = [f for s, f in zip(have, folds) if s in set(seeds)]
    if len(keep) != len(set(seeds)):
        return None
    return float(sum(keep) / len(keep))


def load_retunes(results_dir: str) -> dict:
    """{result file it re-tunes: the tune-only record} for every tuning-only pass."""
    out = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        if os.path.basename(path) == "_summary.json":
            continue
        try:
            rec = json.load(open(path))
        except Exception:
            continue
        if rec.get("tune_only") and rec.get("retunes"):
            out[rec["retunes"]] = rec
    return out


def pick(rounds: list[tuple[str, dict]], results_dir: str) -> tuple[str, dict, dict]:
    """Choose the round to report.

    `rounds` is [(filename, record)] for one task, excluding control arms and
    tuning-only passes. Returns (filename, record, explanation).
    """
    assert rounds, "no rounds given"
    hb = rounds[0][1]["higher_is_better"]
    retunes = load_retunes(results_dir)

    # The widest fold set every round can be scored on, counting re-tunes.
    def folds_available(name, rec):
        seeds = set(rec["inner_cv"]["tune_seeds"])
        if name in retunes:
            seeds |= set(retunes[name]["inner_cv"]["tune_seeds"])
        return seeds

    shared = set.intersection(*(folds_available(n, r) for n, r in rounds))
    if not shared:
        shared = set(COMMON_SEEDS)

    scored = []
    for name, rec in rounds:
        val = inner_on(rec, shared)
        source = name
        if val is None and name in retunes:
            val = inner_on(retunes[name], shared)
            source = f"{name} re-tuned by {retunes[name].get('task')}_retune"
        if val is None:
            val = inner_on(rec, COMMON_SEEDS)
            source = f"{name} (common seeds only)"
        if val is None:
            raise KeyError(f"{name}: cannot score on {sorted(shared)} or {COMMON_SEEDS}")
        scored.append((val, name, rec, source))

    best = max(scored, key=lambda t: t[0] if hb else -t[0])
    explanation = {
        "compared_on_seeds": sorted(shared),
        "rounds": [{"file": n, "inner_on_shared": v, "inner_on_own_folds": inner_own(r),
                    "own_seeds": r["inner_cv"]["tune_seeds"], "test_mean": r["mean"],
                    "source": src, "reported": n == best[1]}
                   for v, n, r, src in scored],
    }
    return best[1], best[2], explanation
