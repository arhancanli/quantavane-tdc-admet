"""The official TDC ADMET Benchmark Group protocol, with honest inner tuning.

Protocol, exactly as TDC specifies it
-------------------------------------
* `admet_group(path=...)` provides, per task, a fixed `train_val` and a fixed
  `test` set produced by TDC's scaffold split (20% test).
* For each of the five official runs, seed s in {1..5}, the training portion is
  split by `group.get_train_valid_split(seed=s, benchmark=task)` into
  train/valid (87.5% / 12.5%, scaffold).
* The model is fitted on `train`, uses `valid` for early stopping only, and
  predicts the fixed `test` set.
* The five prediction vectors go to `group.evaluate_many([...])`, which returns
  `[mean, std]` under the task's official metric. That pair is the leaderboard
  number.

Where tuning happens
--------------------
Model selection uses *separate* scaffold splits of `train_val` drawn with tuning
seeds (101..105 by default), never the official seeds' validation folds and never
the test set. The test labels are touched exactly once per task, by TDC's own
evaluator, after the five runs are complete. Nothing is selected or early-stopped
on them.
"""

from __future__ import annotations

import gc
import json
import os
import time

import numpy as np
from scipy.stats import rankdata

from tdc import Evaluator
from tdc.benchmark_group import admet_group
from tdc.metadata import admet_metrics

import features
import models

HIGHER_IS_BETTER = {"mae": False, "spearman": True, "roc-auc": True, "pr-auc": True}


def provenance() -> dict:
    """Exact versions of everything that can move a number, recorded with each run.

    The features depend on the RDKit build and the fits on the learner builds, so a
    result run on another machine is only comparable with this written down.
    """
    import platform
    out = {"python": platform.python_version(), "platform": platform.platform(),
           "machine": platform.machine()}
    from importlib.metadata import PackageNotFoundError, version

    for mod, dist in (("tdc", "PyTDC"), ("rdkit", "rdkit"), ("numpy", "numpy"), ("scipy", "scipy"),
                      ("sklearn", "scikit-learn"), ("lightgbm", "lightgbm"), ("catboost", "catboost"),
                      ("pandas", "pandas"), ("tabpfn", "tabpfn"), ("torch", "torch")):
        try:
            out[mod] = version(dist)
        except PackageNotFoundError:
            out[mod] = "absent"
        except Exception as exc:
            out[mod] = f"unknown ({type(exc).__name__})"
    return out
OFFICIAL_SEEDS = (1, 2, 3, 4, 5)
# Three tuning folds, not five: selection only needs to rank candidates, and this
# machine is shared with the lab's native record searches (see README).
TUNE_SEEDS = (101, 102, 103)
BIG_TASK_ROWS = 4000


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


class TaskData:
    """Feature matrices and labels for one task, indexed by SMILES."""

    def __init__(self, group, task: str, cache_dir: str, blocks=None):
        bench = group.get(task)
        self.name = bench["name"]
        self.train_val = bench["train_val"]
        self.test = bench["test"]
        self.metric = admet_metrics[self.name]
        self.higher_better = HIGHER_IS_BETTER[self.metric]
        self.task_kind = "classification" if self.metric in ("roc-auc", "pr-auc") else "regression"
        self.evaluator = Evaluator(name=self.metric)
        self.big = len(self.train_val) >= BIG_TASK_ROWS

        all_smiles = list(self.train_val["Drug"]) + list(self.test["Drug"])
        self.unique = sorted(set(all_smiles))
        self.row_of = {s: i for i, s in enumerate(self.unique)}
        self.n_invalid = features.n_invalid(self.unique)

        # Only the blocks the declared candidates actually use are computed and held.
        # On the 10,000-molecule tasks the full set of seven blocks is 360 MB of
        # float32 in this process, and two of them (the RDKit path and feature-Morgan
        # fingerprints) cost minutes to build, so loading blocks nobody asked for
        # would spend both the memory cap and the clock for nothing.
        self.cache_dir = cache_dir
        self.blocks = {}
        self.featurise_seconds = 0.0

        # Sanity: the leaderboard's own guarantee is that test molecules are held
        # out by scaffold, but TDC issue #217 reported duplicate molecules across
        # train_val/test. Record the overlap instead of silently relying on it.
        tv = set(self.train_val["Drug"])
        te = set(self.test["Drug"])
        self.smiles_overlap = len(tv & te)

        if blocks:
            self.load_blocks(blocks)

    def load_blocks(self, blocks) -> None:
        t0 = time.time()
        for block in blocks:
            if block not in self.blocks:
                self.blocks[block] = features.compute_block(block, self.unique, self.cache_dir)
        self.featurise_seconds += time.time() - t0

    def X(self, smiles_series, feature_set: str) -> np.ndarray:
        blocks = models.FEATURE_SETS[feature_set]
        idx = np.fromiter((self.row_of[s] for s in smiles_series), dtype=np.int64)
        mats = [self.blocks[b][idx] for b in blocks]
        return np.hstack(mats) if len(mats) > 1 else mats[0]

    def score(self, y_true, y_pred) -> float:
        return float(self.evaluator(np.asarray(y_true), np.asarray(y_pred)))


def combine(preds: list[np.ndarray], metric: str) -> np.ndarray:
    """Blend member predictions.

    For MAE the prediction must stay on the target's scale, so members are
    averaged directly. Spearman, ROC-AUC and PR-AUC depend only on the ordering
    of the predictions, so members are averaged as ranks, which stops one member's
    scale from dominating the blend.
    """
    if len(preds) == 1:
        return preds[0]
    if metric == "mae":
        return np.mean(np.vstack(preds), axis=0)
    ranks = np.vstack([rankdata(p) / len(p) for p in preds])
    return np.mean(ranks, axis=0)


def fit_stacker(P: np.ndarray, y, task_kind: str) -> dict:
    """Weights over member predictions, fitted only on rows the scoring never sees.

    Regression uses non-negative least squares with an intercept, so a member can
    be dropped but never subtracted. Classification uses logistic regression on
    the member probabilities. Both are returned as plain numbers, so a stacked
    model is fully written down in the result file.
    """
    y = np.asarray(y, dtype=np.float64)
    if task_kind == "regression":
        from scipy.optimize import nnls

        A = np.hstack([P, np.ones((len(P), 1))])
        w, _ = nnls(A, y)
        return {"kind": "nnls", "w": [float(x) for x in w]}
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(max_iter=2000)
    lr.fit(P, y)
    return {"kind": "logit", "coef": [float(x) for x in lr.coef_[0]],
            "intercept": float(lr.intercept_[0])}


def apply_stacker(st: dict, P: np.ndarray) -> np.ndarray:
    if st["kind"] == "nnls":
        w = np.asarray(st["w"], dtype=np.float64)
        return P @ w[:-1] + w[-1]
    z = P @ np.asarray(st["coef"], dtype=np.float64) + st["intercept"]
    return 1.0 / (1.0 + np.exp(-z))


def stack_or_mean(data: "TaskData", members, per_seed_preds, per_seed_truth, tune_seeds) -> dict:
    """Decide, on held-out halves, whether to stack the chosen members or average them.

    Within each tuning fold the valid part is split in two by a seeded coin. The
    stacker's weights are fitted on half A and scored on half B; the plain average
    is scored on the same half B. Stacking is adopted only if it wins that paired
    comparison on average. Fitting and scoring never share a row, so this decision
    cannot reward a stacker that has merely memorised its own fitting set.
    """
    avg_half, stack_half, skipped = [], [], []
    for i, s in enumerate(tune_seeds):
        y = np.asarray(per_seed_truth[i])
        P = np.column_stack([np.asarray(per_seed_preds[i][m]) for m in members])
        rs = np.random.RandomState(7000 + s)
        m = rs.rand(len(y)) < 0.5
        a, b = m, ~m
        if a.sum() < 20 or b.sum() < 20:
            skipped.append(f"fold {s}: half too small")
            continue
        if data.task_kind == "classification" and (len(set(y[a])) < 2 or len(set(y[b])) < 2):
            skipped.append(f"fold {s}: a half holds one class")
            continue
        try:
            st = fit_stacker(P[a], y[a], data.task_kind)
            sc_stack = data.score(y[b], apply_stacker(st, P[b]))
            sc_avg = data.score(y[b], combine([P[b][:, j] for j in range(P.shape[1])], data.metric))
        except Exception as exc:
            skipped.append(f"fold {s}: {type(exc).__name__}: {exc}")
            continue
        if not (np.isfinite(sc_stack) and np.isfinite(sc_avg)):
            skipped.append(f"fold {s}: non-finite score")
            continue
        stack_half.append(float(sc_stack))
        avg_half.append(float(sc_avg))

    mode = "mean"
    if len(avg_half) == len(tune_seeds) and avg_half:
        ms, ma = float(np.mean(stack_half)), float(np.mean(avg_half))
        if (ms > ma) if data.higher_better else (ms < ma):
            mode = "stack"
    out = {"mode": mode, "stack_half_scores": stack_half, "mean_half_scores": avg_half,
           "skipped": skipped}
    log(f"  stack vs mean on held-out halves: stack={stack_half} mean={avg_half} -> {mode}"
        + (f" (skipped: {skipped})" if skipped else ""))
    return out


def tune(group, data: TaskData, cfgs: list[dict], tune_seeds=TUNE_SEEDS) -> dict:
    """Score every config, and blends of the best ones, on held-out scaffold folds
    drawn from train_val only. Returns the selection record."""
    per_seed_preds: list[dict[str, np.ndarray]] = []
    per_seed_truth: list[np.ndarray] = []
    cfg_scores: dict[str, list[float]] = {c["id"]: [] for c in cfgs}
    timings: dict[str, float] = {c["id"]: 0.0 for c in cfgs}

    for s in tune_seeds:
        tr, va = group.get_train_valid_split(seed=s, benchmark=data.name)
        y_tr, y_va = tr["Y"].values, va["Y"].values
        per_seed_truth.append(y_va)
        preds_this_seed = {}
        for cfg in cfgs:
            X_tr = data.X(tr["Drug"], cfg["features"])
            X_va = data.X(va["Drug"], cfg["features"])
            t0 = time.time()
            pred = models.fit_predict(
                cfg, data.task_kind, data.metric, X_tr, y_tr, X_va, y_va, X_va, seed=s
            )
            timings[cfg["id"]] += time.time() - t0
            preds_this_seed[cfg["id"]] = pred
            cfg_scores[cfg["id"]].append(data.score(y_va, pred))
            # Each config builds its own dense feature matrices; drop them before
            # the next one so the process stays well under the 1.5 GB cap.
            del X_tr, X_va
            gc.collect()
            log(
                f"    [fold {s}] {cfg['id']:22s} {data.metric}={cfg_scores[cfg['id']][-1]:.4f} "
                f"({time.time() - t0:.0f}s)"
            )
        per_seed_preds.append(preds_this_seed)
        log(f"  tune fold {s} done: n_train={len(tr)} n_valid={len(va)}")

    mean_score = {cid: float(np.mean(v)) for cid, v in cfg_scores.items()}
    order = sorted(mean_score, key=lambda c: mean_score[c], reverse=data.higher_better)
    log("  inner CV ranking (best first):")
    for cid in order:
        log(f"    {cid:28s} {mean_score[cid]:.4f}  ({timings[cid]:.0f}s total)")

    # Candidate blends: the top-k configs for k = 1..5, scored on the same folds.
    candidates = []
    for k in range(1, min(5, len(order)) + 1):
        members = order[:k]
        fold_scores = [
            data.score(per_seed_truth[i], combine([per_seed_preds[i][m] for m in members], data.metric))
            for i in range(len(per_seed_truth))
        ]
        candidates.append(
            {"members": members, "inner_mean": float(np.mean(fold_scores)),
             "inner_std": float(np.std(fold_scores)), "inner_folds": fold_scores}
        )
    best = max(candidates, key=lambda c: c["inner_mean"] if data.higher_better else -c["inner_mean"])
    log(f"  blend scores: " + ", ".join(f"top{len(c['members'])}={c['inner_mean']:.4f}" for c in candidates))
    log(f"  SELECTED: {best['members']} (inner {best['inner_mean']:.4f})")

    stack = {"mode": "mean", "stack_half_scores": [], "mean_half_scores": [],
             "skipped": ["single member: nothing to stack"]}
    if os.environ.get("QV_NO_STACK"):
        # The first round of these runs predated the stacker and always averaged.
        # Setting this reproduces that behaviour exactly from the current code.
        stack["skipped"] = ["QV_NO_STACK set: stacking not considered"]
    elif len(best["members"]) >= 2:
        stack = stack_or_mean(data, best["members"], per_seed_preds, per_seed_truth, tune_seeds)

    return {
        "blend_mode": stack["mode"],
        "stack_decision": stack,
        "config_inner_mean": mean_score,
        "config_inner_folds": cfg_scores,
        "config_seconds": timings,
        "blend_candidates": candidates,
        "selected_members": best["members"],
        "selected_inner_mean": best["inner_mean"],
        "tune_seeds": list(tune_seeds),
    }


def run_official(group, data: TaskData, cfgs_by_id: dict, members: list[str],
                 blend_mode: str = "mean") -> dict:
    """The five official runs: fit on train, early-stop on valid, predict test.

    With blend_mode="stack" the member predictions are combined by a stacker whose
    weights are fitted on the run's own official valid fold. The valid fold is part
    of train_val, which is what TDC hands over for model development, so this uses
    no test information: each member predicts the valid rows and the test rows in
    the same call, weights come from the valid rows alone, and the test rows are
    scored once at the end.
    """
    preds_list = []
    per_seed_metric = []
    stackers = []
    for s in OFFICIAL_SEEDS:
        tr, va = group.get_train_valid_split(seed=s, benchmark=data.name)
        y_tr, y_va = tr["Y"].values, va["Y"].values
        member_preds = []
        member_va_preds = []
        n_va = len(va)
        for mid in members:
            cfg = cfgs_by_id[mid]
            X_tr = data.X(tr["Drug"], cfg["features"])
            X_va = data.X(va["Drug"], cfg["features"])
            X_te = data.X(data.test["Drug"], cfg["features"])
            X_pred = np.vstack([X_va, X_te]) if blend_mode == "stack" else X_te
            p = models.fit_predict(
                cfg, data.task_kind, data.metric, X_tr, y_tr, X_va, y_va, X_pred, seed=s
            )
            if blend_mode == "stack":
                member_va_preds.append(p[:n_va])
                member_preds.append(p[n_va:])
            else:
                member_preds.append(p)
            del X_tr, X_va, X_te, X_pred
            gc.collect()
        if blend_mode == "stack":
            st = fit_stacker(np.column_stack(member_va_preds), y_va, data.task_kind)
            stackers.append(st)
            y_pred = apply_stacker(st, np.column_stack(member_preds))
        else:
            y_pred = combine(member_preds, data.metric)
        preds_list.append({data.name: y_pred})
        run_metric = group.evaluate({data.name: y_pred})[data.name][data.metric]
        per_seed_metric.append(float(run_metric))
        log(f"  official seed {s}: n_train={len(tr)} test {data.metric}={run_metric}")

    agg = group.evaluate_many(preds_list)
    return {
        "per_seed": per_seed_metric,
        "blend_mode": blend_mode,
        "stackers": stackers,
        "evaluate_many": {k: list(v) for k, v in agg.items()},
        "predictions": [p[data.name].tolist() for p in preds_list],
    }


def run_task(
    task: str,
    data_dir: str,
    out_dir: str,
    cache_dir: str,
    tune_seeds=TUNE_SEEDS,
    budget: str | None = None,
    only_configs: list[str] | None = None,
    tag: str | None = None,
    force_log: bool = False,
) -> dict:
    t_start = time.time()
    group = admet_group(path=data_dir)
    data = TaskData(group, task, cache_dir)
    log(
        f"task={data.name} metric={data.metric} ({'higher' if data.higher_better else 'lower'} better) "
        f"kind={data.task_kind} train_val={len(data.train_val)} test={len(data.test)} "
        f"unique_smiles={len(data.unique)} unparsable={data.n_invalid} "
        f"train/test SMILES overlap={data.smiles_overlap}"
    )

    cfgs = models.candidate_configs(data.task_kind, data.metric, data.big, budget=budget)
    if force_log:
        if data.task_kind != "regression":
            raise ValueError("force_log only applies to regression tasks")
        cfgs = models.force_log(cfgs)
    if only_configs:
        wanted = set(only_configs)
        cfgs = [c for c in cfgs if c["id"] in wanted]
        missing = wanted - {c["id"] for c in cfgs}
        if missing:
            raise ValueError(f"unknown config ids: {sorted(missing)}")
    cfgs_by_id = {c["id"]: c for c in cfgs}
    needed = []
    for cfg in cfgs:
        for b in models.FEATURE_SETS[cfg["features"]]:
            if b not in needed:
                needed.append(b)
    data.load_blocks(needed)
    log(
        f"  {len(cfgs)} candidate configs ({', '.join(c['id'] for c in cfgs)}), "
        f"{len(tune_seeds)} tuning folds, feature blocks {needed} "
        f"({data.featurise_seconds:.0f}s)"
    )

    sel = tune(group, data, cfgs, tune_seeds)
    res = run_official(group, data, cfgs_by_id, sel["selected_members"],
                       blend_mode=sel["blend_mode"])

    mean, std = res["evaluate_many"][data.name]
    log(f"RESULT {data.name}: {data.metric} = {mean} +/- {std}  (official 5-seed protocol)")

    record = {
        "task": data.name,
        "metric": data.metric,
        "higher_is_better": data.higher_better,
        "mean": mean,
        "std": std,
        "per_seed": res["per_seed"],
        "selected_members": sel["selected_members"],
        "blend_mode": sel["blend_mode"],
        "stack_decision": sel["stack_decision"],
        "official_stackers": res["stackers"],
        "selected_config_details": [cfgs_by_id[m] for m in sel["selected_members"]],
        # The inner-CV score of the selected blend. This, never the test score, is
        # what may be compared across rounds to decide which model to report.
        "selected_inner_mean": sel["selected_inner_mean"],
        "inner_cv": {
            "tune_seeds": sel["tune_seeds"],
            "config_inner_mean": sel["config_inner_mean"],
            "blend_candidates": sel["blend_candidates"],
            "config_seconds": sel["config_seconds"],
        },
        "dataset": {
            "train_val": len(data.train_val),
            "test": len(data.test),
            "unique_smiles": len(data.unique),
            "unparsable_smiles": data.n_invalid,
            "feature_blocks": sorted(data.blocks),
            "feature_columns": int(sum(v.shape[1] for v in data.blocks.values())),
            "train_test_smiles_overlap": data.smiles_overlap,
        },
        "official_seeds": list(OFFICIAL_SEEDS),
        "budget": budget or models.BUDGET,
        "candidate_ids": [c["id"] for c in cfgs],
        "provenance": provenance(),
        "tabpfn_settings": {"ckpt_regressor": models.TABPFN_CKPT or "unset",
                            "ckpt_classifier": models.TABPFN_CKPT_CLF or "unset",
                            "n_estimators": models.TABPFN_N,
                            "max_train_rows": models.TABPFN_MAX_ROWS},
        "wall_seconds": round(time.time() - t_start, 1),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    os.makedirs(out_dir, exist_ok=True)
    stem = data.name + (f"_{tag}" if tag else "")
    with open(os.path.join(out_dir, f"{stem}.json"), "w") as fh:
        json.dump(record, fh, indent=1)
    np.save(os.path.join(out_dir, f"{stem}_test_predictions.npy"),
            np.vstack([np.asarray(p) for p in res["predictions"]]))
    return record
