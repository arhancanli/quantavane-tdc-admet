"""Generate the public repository README from the result files.

The table is generated, never typed, so the published numbers cannot drift from
the run records next to them. Run from the code directory:

    ../.venv/bin/python make_readme.py > ../../../quantavane-tdc-admet/README.md
"""

from __future__ import annotations

import glob
import json
import os

import choose

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
LB = json.load(open(os.path.join(BASE, "leaderboards_2026-11-08.json")))

PRETTY = {"mae": "MAE", "spearman": "Spearman", "roc-auc": "AUROC", "pr-auc": "AUPRC"}


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


def load_rounds() -> tuple[dict, dict]:
    """{task: [rounds]} for the model arm, and {task: record} for the control arm."""
    rounds: dict[str, list[dict]] = {}
    plain: dict[str, dict] = {}
    for path in sorted(glob.glob(os.path.join(BASE, "results", "*.json"))):
        name = os.path.basename(path)
        if name == "_summary.json":
            continue
        rec = json.load(open(path))
        rec["_file"] = name
        if rec.get("tune_only"):
            continue  # informs the choice, never a reported row
        if name.endswith("_plain.json"):
            plain[rec["task"]] = rec
        else:
            rounds.setdefault(rec["task"], []).append(rec)
    return rounds, plain


def pick(rs: list[dict]) -> dict:
    """The round to report, by the shared rule in choose.py. Never the best test score."""
    _name, best, _why = choose.pick([(r["_file"], r) for r in rs],
                                    os.path.join(BASE, "results"))
    return best


def rank_of(task: str, ours: float):
    t = LB["tasks"].get(task)
    if not t:
        return None, None, None
    better = sum(1 for _, _, v, _ in t["rows"] if (v > ours if t["higher_is_better"] else v < ours))
    return better + 1, len(t["rows"]) + 1, t["rows"]


def main() -> None:
    rounds, plain = load_rounds()
    chosen = {task: pick(rs) for task, rs in rounds.items()}
    uses_tabpfn = any(
        any("tabpfn" in m for m in r["selected_members"]) for r in chosen.values()
    )

    print("# QV-ADMET: inner-CV model selection on the TDC ADMET Benchmark Group")
    print()
    print("Gradient-boosted trees and a pretrained tabular model over RDKit descriptors and")
    print("fingerprints, run under the official protocol of the Therapeutics Data Commons")
    print("ADMET Benchmark Group. Every model choice is made by cross-validation inside the")
    print("training rows; the held-out test split of each task is scored once per run, by TDC's")
    print("own evaluator.")
    print()
    print("Nothing here has been submitted to the leaderboard. The ranks in the table are what")
    print("these numbers *would* hold against the public boards as read on 8 November 2026, and")
    print("they are printed so the comparison can be checked, not as a claim of a listing.")
    print()

    print("## What the code does")
    print()
    print("For one task:")
    print()
    print("1. `features.py` turns each SMILES string into RDKit 2D descriptors and several")
    print("   fingerprints (Morgan counts, feature-invariant Morgan, Avalon, MACCS keys,")
    print("   atom-pair counts, RDKit path counts). Each molecule is featurised on its own, with")
    print("   no labels and nothing from any other molecule, and the blocks are cached by a hash")
    print("   of the exact SMILES list.")
    print("2. `models.py` holds the candidate list: LightGBM at several depths and feature sets,")
    print("   ExtraTrees, CatBoost, HistGradientBoosting, and TabPFN v2 as a pretrained")
    print("   in-context learner on the descriptor-only set. On the Spearman tasks each main")
    print("   candidate is also offered on a `log1p` target.")
    print("3. `protocol.py` scores every candidate on three extra scaffold train/valid folds")
    print("   drawn from the training portion only, ranks them, picks the best top-k blend, and")
    print("   decides on held-out halves whether to stack the members or average them. Then it")
    print("   refits the chosen members on each of TDC's five official seeds and predicts the")
    print("   fixed test set. `group.evaluate_many` returns the mean and standard deviation that")
    print("   the leaderboard asks for.")
    print()
    print("The test labels are touched exactly once per task per run, by TDC's evaluator, after")
    print("the five runs are complete. No candidate, blend, hyper-parameter, feature set or")
    print("target transform is chosen with reference to them.")
    print()

    print("## How to run it")
    print()
    print("```sh")
    print("uv venv --python python3.12 .venv")
    print("uv pip install --python .venv/bin/python -r requirements.txt")
    print("uv pip install --python .venv/bin/python --no-deps PyTDC==1.1.15")
    print("code/setup_libomp.sh          # macOS without Homebrew only; see the script")
    print("code/run.sh caco2_wang        # one task; TDC downloads the splits into data/")
    print("code/run.sh vdss_lombardo --budget strong --tag strong")
    print("../.venv/bin/python code/make_table.py   # rebuild the results table")
    print("```")
    print()
    print("`PyTDC` must be installed with `--no-deps`: its `tiledbsoma` dependency pins")
    print("`pyarrow 12.0.1`, which has no Python 3.12 wheel. `setuptools` must stay below 81,")
    print("because PyTDC imports `pkg_resources`. Neither affects the benchmark path.")
    print()

    print("## Results")
    print()
    print(f"All {len(chosen)} tasks run so far, under the official five-seed protocol. \"Board top\"")
    print("is the best listed entry on that task's public page on 8 November 2026.")
    print()
    print("| Task | Metric | This code | Board top 3 | Would-be rank | Selected model |")
    print("|---|---|---|---|---|---|")
    for task in sorted(chosen):
        r = chosen[task]
        rank, n, rows = rank_of(task, r["mean"])
        top3 = "; ".join(f"{nm} {v:g}" for _, nm, v, _ in rows[:3]) if rows else "not read"
        rank_s = f"**{rank} of {n}**" if rank else "n/a"
        members = " + ".join(r["selected_members"])
        if r.get("blend_mode") == "stack":
            members += " (stacked)"
        print(f"| {task} | {PRETTY[r['metric']]} | {r['mean']:.3f} ± {r['std']:.3f} | {top3} "
              f"| {rank_s} | {members} |")
    print()

    multi = {t: rs for t, rs in rounds.items() if len(rs) > 1}
    if multi:
        print("### Tasks run in more than one round")
        print()
        print("Which round is reported above is decided by the inner-CV score of the selected")
        print("blend, before the test score is looked at. Both rounds are printed here, including")
        print("the cases where the round that lost on inner CV scored better on test, because")
        print("reporting those would make the published number a selected maximum.")
        print()
        print("Inner CV is the score on the folds the rounds were compared on, which is what the")
        print("choice was made with. Where a round was originally tuned on fewer folds, its own")
        print("candidate list was re-tuned on the full set first, as a tuning-only pass that produces")
        print("no test score, so both rounds are scored by the same estimator.")
        print()
        print("| Task | Round | Folds compared | Inner CV | Official test | Reported |")
        print("|---|---|---|---|---|---|")
        for task in sorted(multi):
            _n, _b, why = choose.pick([(r["_file"], r) for r in multi[task]],
                                      os.path.join(BASE, "results"))
            nf = len(why["compared_on_seeds"])
            for d in sorted(why["rounds"], key=lambda x: x["file"]):
                print(f"| {task} | `{d['file']}` | {nf} | {d['inner_on_shared']:.4f} | "
                      f"{d['test_mean']:.3f} | {'yes' if d['reported'] else 'no'} |")
        print()

    if plain:
        print("### One fixed config against the selection ensemble")
        print()
        print("On some tasks a single fixed LightGBM configuration (`dma-lgbm-base`, no search, no")
        print("blending) was also run under the same protocol, as a control. It is a fair question")
        print("whether the search earns its keep, and on these tasks the answer is mixed.")
        print()
        print("| Task | Metric | Selection ensemble | One fixed config | Inner CV says | Test says |")
        print("|---|---|---|---|---|---|")
        for task in sorted(plain):
            if task not in chosen:
                continue
            m, c = chosen[task], plain[task]
            hb = m["higher_is_better"]
            inner_win = "ensemble" if ((inner(m) > inner(c)) == hb) else "fixed config"
            test_win = "ensemble" if ((m["mean"] > c["mean"]) == hb) else "fixed config"
            print(f"| {task} | {PRETTY[m['metric']]} | {m['mean']:.3f} ± {m['std']:.3f} | "
                  f"{c['mean']:.3f} ± {c['std']:.3f} | {inner_win} | {test_win} |")
        print()
        print("The reported number is always the model inner CV picked, even where the control")
        print("scored better on test.")
        print()

    print("## What is in the result files")
    print()
    print("`results/<task>.json` holds the official mean and standard deviation, all five")
    print("per-seed scores, the selected members and their full configuration, the inner-CV")
    print("score of every candidate and every blend considered, the dataset sizes, the exact")
    print("package versions, and two checks: how many SMILES RDKit could not parse, and how many")
    print("molecules appear in both the training and test split. `results/<task>_test_predictions.npy`")
    print("holds the five prediction vectors. `results/<task>_plain.json` is the control arm.")
    print("`leaderboards_2026-11-08.json` is the public board data the ranks are computed against,")
    print("transcribed as printed, with the two pages whose stated metric direction contradicts")
    print("their own metric noted in the file.")
    print()

    print("## Notes on two of the tasks")
    print()
    print("- **Bioavailability_Ma**: its public board duplicates the `Pgp_Broccatelli` table from")
    print("  rank 3 down, which is open as TDC issue #394. Our run is included for completeness;")
    print("  its rank column cannot be taken at face value.")
    print("- **Clearance_Hepatocyte_AZ**: the dataset merges human and rat assays, open as TDC")
    print("  issue #405.")
    print()

    if uses_tabpfn:
        print("## Attribution")
        print()
        print("Built with PriorLabs-TabPFN.")
        print()
        print("Results whose selected model includes a `tabpfn` member use the TabPFN v2 open")
        print("weights from Prior Labs, under their licence, which requires the line above. The")
        print("checkpoints are not redistributed here; `models.py` reads them from a local path.")
        print()

    print("## Licence")
    print()
    print("MIT, see `LICENSE`. The datasets, the splits and the scoring belong to the")
    print("Therapeutics Data Commons and keep their own terms; this repository contains none of")
    print("their data, only code that asks the official `PyTDC` package for it.")
    print()
    print("Author: Arhan Canli (Quantavane).")


if __name__ == "__main__":
    main()
