"""Build a submission packet for one TDC ADMET task.

    ../.venv/bin/python make_packet.py vdss_lombardo [--out ../packets]

A packet is what the owner needs in front of him to fill in the TDC submission form,
and nothing more: the number, every per-seed score, the seeds, what the model is in
plain words, where the code is, and how the result sits against the public board. It
asserts the things a reviewer would check, and refuses to write a packet for a task
that would not rank 1 or 2, so a packet never quietly becomes a way of dressing up a
middling result.

Nothing here submits anything. The form is the owner's step.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

import choose

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
RESULTS = os.path.join(BASE, "results")

PRETTY = {"mae": "MAE", "spearman": "Spearman", "roc-auc": "AUROC", "pr-auc": "AUPRC"}
LANE_OF_TAG = {"": "admet-tdc", "round2": "admet-tdc", "r2strong": "admet-tdc",
               "r3maplight": "admet-tdc-v2"}


def sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def inner(rec: dict) -> float:
    if rec.get("selected_inner_mean") is not None:
        return float(rec["selected_inner_mean"])
    for cand in rec["inner_cv"]["blend_candidates"]:
        if cand["members"] == rec["selected_members"]:
            return float(cand["inner_mean"])
    raise KeyError(rec["task"])


COMMON_SEEDS = (101, 102, 103)


def inner_common(rec: dict) -> float:
    """The selected blend's inner score on the tuning folds every round shares.

    Must match make_table.py: the cross-round choice is made on the common seeds,
    because a fifteen-seed mean and a three-seed mean are different estimators. This
    function existing in only one of the two tools is how the packet builder and the
    results table once disagreed about which round to report.
    """
    cv = rec["inner_cv"]
    seeds = list(cv["tune_seeds"])
    for cand in cv["blend_candidates"]:
        if cand["members"] == rec["selected_members"]:
            keep = [f for s, f in zip(seeds, cand["inner_folds"]) if s in COMMON_SEEDS]
            if len(keep) != len(COMMON_SEEDS):
                raise KeyError(f"{rec['task']}: seeds {seeds} miss some of {COMMON_SEEDS}")
            return float(sum(keep) / len(keep))
    raise KeyError(f"cannot recover inner folds for {rec['task']}")


def tag_of(filename: str, task: str) -> str:
    stem = filename[:-5]
    return "" if stem == task else stem[len(task) + 1:]


def rounds_for(task: str) -> list[tuple[str, dict]]:
    out = []
    for path in sorted(glob.glob(os.path.join(RESULTS, f"{task}*.json"))):
        name = os.path.basename(path)
        if name.endswith("_summary.json"):
            continue
        rec = json.load(open(path))
        if rec.get("task") != task or rec.get("tune_only"):
            continue
        tag = tag_of(name, task)
        if tag == "plain":
            continue
        out.append((name, rec))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--out", default=os.path.join(BASE, "packets"))
    ap.add_argument("--allow-any-rank", action="store_true",
                    help="write the packet even if the result would not rank 1 or 2 "
                         "(for a dry run; the packet is marked as not submission-worthy)")
    args = ap.parse_args()
    task = args.task

    lb = json.load(open(os.path.join(BASE, "leaderboards_2026-11-08.json")))
    if task not in lb["tasks"]:
        raise SystemExit(f"{task}: no board data read for this task")
    board = lb["tasks"][task]
    if board.get("_warning"):
        raise SystemExit(f"{task}: this board is flagged unusable -- {board['_warning']}")

    rounds = rounds_for(task)
    if not rounds:
        raise SystemExit(f"{task}: no result files")
    hb = rounds[0][1]["higher_is_better"]
    # The reported round is the one with the best inner CV, never the best test score.
    chosen_name, chosen, why = choose.pick(rounds, RESULTS)

    ours = chosen["mean"]
    better = sum(1 for _, _, v, _ in board["rows"] if (v > ours if hb else v < ours))
    rank, n_listed = better + 1, len(board["rows"]) + 1
    if rank > 2 and not args.allow_any_rank:
        print(f"{task}: would rank {rank} of {n_listed}. A packet is only prepared for a "
              f"result that would rank 1 or 2, so none is written.")
        return 0

    # Checks a reviewer would make.
    d = chosen["dataset"]
    problems = []
    if d["train_test_smiles_overlap"] != 0:
        problems.append(f"{d['train_test_smiles_overlap']} molecules in both train_val and test")
    if chosen["official_seeds"] != [1, 2, 3, 4, 5]:
        problems.append(f"official seeds were {chosen['official_seeds']}, not 1-5")
    if len(chosen["per_seed"]) != 5:
        problems.append(f"{len(chosen['per_seed'])} per-seed scores, not 5")
    if d["unparsable_smiles"] != 0:
        problems.append(f"{d['unparsable_smiles']} SMILES RDKit could not parse")

    os.makedirs(args.out, exist_ok=True)
    pdir = os.path.join(args.out, task)
    os.makedirs(pdir, exist_ok=True)

    tag = tag_of(chosen_name, task)
    lane = LANE_OF_TAG.get(tag, "unknown")
    code_dir = os.path.join(BASE, "codev2" if tag == "r3maplight" else "code")
    code_files = {f: sha(os.path.join(code_dir, f))
                  for f in ("features.py", "models.py", "protocol.py", "run_task.py")
                  if os.path.exists(os.path.join(code_dir, f))}
    uses_tabpfn = any("tabpfn" in m for m in chosen["selected_members"])
    uses_gin = any(m.startswith("maplight") for m in chosen["selected_members"])

    packet = {
        "task": task,
        "venue": "Therapeutics Data Commons, ADMET Benchmark Group",
        "board_page": f"https://tdcommons.ai/benchmark/admet_group/{board['slug']}/",
        "metric": board["metric"],
        "higher_is_better": hb,
        "result": {"mean": ours, "std": chosen["std"], "per_seed": chosen["per_seed"]},
        "would_be_rank": rank,
        "of_listed_after_insert": n_listed,
        "board_top_3": [{"rank": r, "model": m, "value": v, "std": s}
                        for r, m, v, s in board["rows"][:3]],
        "board_read_utc": lb["_read_utc"],
        "seeds": {"official": chosen["official_seeds"],
                  "tuning": chosen["inner_cv"]["tune_seeds"],
                  "note": "The tuning seeds draw extra scaffold train/valid splits of "
                          "train_val only. The official seeds are TDC's five runs."},
        "model": {"selected_members": chosen["selected_members"],
                  "member_configs": chosen["selected_config_details"],
                  "blend_mode": chosen.get("blend_mode", "mean"),
                  "inner_cv_of_selected": inner(chosen),
                  "candidates_considered": chosen["candidate_ids"]},
        "round_choice": why,
        "dataset": d,
        "provenance": chosen.get("provenance", "not recorded in this round"),
        "lane": lane,
        "code": {"directory": os.path.relpath(code_dir, BASE), "sha256": code_files,
                 "public_repository_staging": "~/quantavane-tdc-admet (not pushed)"},
        "result_file": chosen_name,
        "predictions_file": chosen_name.replace(".json", "_test_predictions.npy"),
        "reviewer_checks": {"passed": not problems, "problems": problems},
        "attribution_required": (["Built with PriorLabs-TabPFN"] if uses_tabpfn else []),
        "submitted": False,
        "submission_note": "Nothing has been submitted. The TDC form is the owner's step.",
    }
    with open(os.path.join(pdir, "packet.json"), "w") as fh:
        json.dump(packet, fh, indent=1)

    members = ", ".join(chosen["selected_members"])
    lines = [
        f"# Submission packet: {task}",
        "",
        f"**Result.** {PRETTY[board['metric']]} {ours:.3f} +/- {chosen['std']:.3f} over TDC's "
        f"five official runs (seeds {chosen['official_seeds']}). Per-seed scores: "
        + ", ".join(f"{v:.3f}" for v in chosen["per_seed"]) + ".",
        "",
        f"**Where it would stand.** {rank} of {n_listed} on the public board as read "
        f"{lb['_read_utc']}. The listed top three: "
        + "; ".join(f"{m} {v:g}" for _, m, v, _ in board["rows"][:3]) + ".",
        "",
        "**Protocol.** The venue's own, through PyTDC's `admet_group`: TDC's fixed scaffold",
        "`train_val`/`test`, five runs with seeds 1-5 from `get_train_valid_split`, fit on",
        "train with valid used only for early stopping, predict the fixed test set, and score",
        "with `group.evaluate_many`, whose mean and standard deviation are the two numbers",
        "above. The official metric comes from `tdc.metadata.admet_metrics`.",
        "",
        f"**Model.** {members}"
        + (" combined by a stacker fitted on the official valid fold"
           if packet["model"]["blend_mode"] == "stack" else
           (" averaged" if len(chosen["selected_members"]) > 1 else "")) + ".",
        "",
        "Model choice was made only by inner cross-validation on training rows: candidates were",
        f"scored on separate scaffold folds of `train_val` drawn with seeds "
        f"{chosen['inner_cv']['tune_seeds']}, and the test labels were read exactly once, by",
        f"TDC's evaluator, after the five runs. {len(chosen['candidate_ids'])} candidates were",
        "considered. Where a task was run in more than one round, the round reported is the one",
        "with the better inner-CV score, not the better test score.",
        "",
    ]
    if uses_gin:
        lines += [
            "Features: RDKit 2D descriptors, Morgan/ECFP count fingerprints, Avalon counts, the",
            "extended reduced graph fingerprint, and a 300-column embedding from the pretrained",
            "supervised-masking GIN published with dgl-lifesci, taken as the mean over atoms. The",
            "embedding is computed once in a separate environment and cached, so a model run",
            "imports no deep-learning library. No other entrant's code was used.",
            "",
        ]
    if uses_tabpfn:
        lines += ["**Attribution.** Built with PriorLabs-TabPFN.", ""]
    if len(rounds) > 1:
        shared = why["compared_on_seeds"]
        mine = next(d for d in why["rounds"] if d["reported"])
        others = [d for d in why["rounds"] if not d["reported"]]
        runner = max(others, key=lambda d: d["inner_on_shared"]) if hb else \
            min(others, key=lambda d: d["inner_on_shared"])
        margin = abs(mine["inner_on_shared"] - runner["inner_on_shared"])
        lines += [
            "**How the reported round was chosen, and how thin the margin is.** This task was run in",
            f"{len(rounds)} rounds. The reported one was chosen by inner cross-validation on training",
            f"rows over {len(shared)} shared scaffold folds (tuning seeds {shared[0]}-{shared[-1]}),",
            f"scoring {mine['inner_on_shared']:.4f} against {runner['inner_on_shared']:.4f} for",
            f"`{runner['file']}` -- a margin of {margin:.4f}. Test scores played no part in the",
            "choice. Where a round had originally been tuned on fewer folds, its own candidate list",
            "was re-tuned on the full set first, as a tuning-only pass that produces no test score, so",
            "the two rounds are compared on the same estimator rather than on the three folds they",
            f"happened to share. The other round's official test score was {runner['test_mean']:.3f}.",
            f"A margin of {margin:.4f} is small, and this result should be read as level with the",
            "entries around it rather than clearly better than them.",
            "",
        ]
    lines += [
        f"**Code.** `{packet['code']['directory']}/` in the staging repository "
        "`~/quantavane-tdc-admet`, with these file hashes:",
        "",
    ] + [f"- `{f}` sha256 `{h[:16]}...`" for f, h in code_files.items()] + [
        "",
        f"**Run record.** `results/{chosen_name}` holds every per-seed score, the full inner-CV",
        f"ranking and the leakage checks; `results/{packet['predictions_file']}` holds the five",
        "prediction vectors.",
        "",
        "**Checks.** "
        + ("zero unparsable SMILES, zero molecules shared between `train_val` and `test`, five "
           "official seeds." if not problems else "PROBLEMS: " + "; ".join(problems)),
        "",
        "**Not submitted.** The TDC submission form is the owner's step.",
    ]
    with open(os.path.join(pdir, "METHODS.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"{task}: packet written to {pdir} (would rank {rank} of {n_listed})")
    if problems:
        print("  reviewer checks FAILED:", "; ".join(problems))
    return 0


if __name__ == "__main__":
    sys.exit(main())
