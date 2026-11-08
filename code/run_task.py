"""Run the official TDC ADMET protocol for one or more tasks.

Usage (through run.sh, which sets the OpenMP library path and the nice level):
    ./run.sh caco2_wang
    ./run.sh ppbr_az vdss_lombardo
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)

import protocol  # noqa: E402  (same directory)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", nargs="+", help="TDC ADMET task names, e.g. caco2_wang")
    ap.add_argument("--data-dir", default=os.path.join(BASE, "data"))
    ap.add_argument("--out-dir", default=os.path.join(BASE, "results"))
    ap.add_argument("--cache-dir", default=os.path.join(BASE, "data", "_feature_cache"))
    ap.add_argument("--tune-seeds", type=int, nargs="+", default=list(protocol.TUNE_SEEDS))
    ap.add_argument(
        "--budget",
        choices=["lean", "full", "strong"],
        default=None,
        help="lean = LightGBM/ExtraTrees only (default); full adds CatBoost and HistGB; "
        "strong also adds the atom-pair and RDKit-path feature sets and TabPFN v2",
    )
    ap.add_argument(
        "--only-configs",
        default=None,
        help="comma-separated config ids to restrict the candidate list to, e.g. "
        "'dma-lgbm-base-log,dma-cat-base-log'. Selection is still by inner CV.",
    )
    ap.add_argument(
        "--tag",
        default=None,
        help="suffix for the result file, so a second round does not overwrite the first",
    )
    ap.add_argument(
        "--force-log",
        action="store_true",
        help="put every regression candidate on a log1p target (for a second round on a "
        "Spearman task where the first round's inner CV already favoured it)",
    )
    args = ap.parse_args()

    failures = []
    for task in args.tasks:
        try:
            protocol.run_task(
                task,
                data_dir=args.data_dir,
                out_dir=args.out_dir,
                cache_dir=args.cache_dir,
                tune_seeds=tuple(args.tune_seeds),
                budget=args.budget,
                only_configs=(
                    [c.strip() for c in args.only_configs.split(",")] if args.only_configs else None
                ),
                tag=args.tag,
                force_log=args.force_log,
            )
        except Exception as exc:  # keep the remaining tasks running
            protocol.log(f"FAILED {task}: {type(exc).__name__}: {exc}")
            import traceback

            traceback.print_exc()
            failures.append(task)
    if failures:
        protocol.log(f"failed tasks: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
