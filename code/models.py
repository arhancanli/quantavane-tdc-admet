"""CPU-only model zoo for the TDC ADMET benchmark group.

Gradient-boosted trees over RDKit descriptors and fingerprints. Everything runs
on CPU; no model here uses a GPU, a pretrained network, or any external data.

A *config* is a plain dict describing one candidate: which feature blocks to use,
which learner, and whether to fit on a log-transformed target. protocol.py scores
configs by inner cross-validation on the training portion only.
"""

from __future__ import annotations

import os

import numpy as np

# LightGBM's macOS wheel links against libomp at @rpath; run.sh points
# DYLD_LIBRARY_PATH at the copy bundled inside the scikit-learn wheel so no
# Homebrew install is needed. If that did not happen, we fall back to the
# LightGBM-free part of the zoo rather than failing the run.
try:
    import lightgbm as lgb

    HAVE_LGB = True
except Exception as exc:  # pragma: no cover
    HAVE_LGB = False
    _LGB_ERR = exc

from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

N_THREADS = int(os.environ.get("ADMET_THREADS", "5"))

# Feature-block combinations. "desc+morgan" is the cheap workhorse; the wide set
# adds feature-invariant Morgan, Avalon and MACCS.
FEATURE_SETS = {
    "dm": ("desc", "morgan"),
    "dma": ("desc", "morgan", "avalon", "maccs"),
    "wide": ("desc", "morgan", "fcfp", "avalon", "maccs"),
    # Round 2 additions.
    "desc": ("desc",),  # ~210 columns: the only set small enough for TabPFN v2
    "dmap": ("desc", "morgan", "avalon", "maccs", "atompair"),
    "full": ("desc", "morgan", "fcfp", "avalon", "maccs", "atompair", "rdkfp"),
    # Round 3. The published MapLight entries describe a set of this shape: ECFP
    # counts, Avalon, the extended reduced graph, RDKit 2D descriptors, and a
    # pretrained graph-network embedding. Built here from our own code over the
    # component libraries; no entrant's code is used. 3,897 columns.
    "maplight": ("desc", "morgan", "avalon", "erg", "gin"),
    # The same set without the pretrained embedding, so inner CV can say whether the
    # embedding earns its place instead of us assuming it does. 3,597 columns.
    "mlng": ("desc", "morgan", "avalon", "erg"),
}

# TabPFN v2 (PriorLabs open weights). Any result that includes a TabPFN member
# must carry the attribution line the licence requires: "Built with
# PriorLabs-TabPFN". The checkpoint is read from a local path when TABPFN_CKPT
# names one, so a run needs no network once the weights are in place.
TABPFN_CKPT = os.environ.get("TABPFN_CKPT", "")        # v2 regressor checkpoint
TABPFN_CKPT_CLF = os.environ.get("TABPFN_CKPT_CLF", "")  # v2 classifier checkpoint
TABPFN_N = int(os.environ.get("TABPFN_N", "4"))
# TabPFN is an in-context learner: every training row sits in the forward pass,
# so cost and memory grow with the training set. Above this many rows a seeded
# subsample is used, and the cap is recorded with the result.
TABPFN_MAX_ROWS = int(os.environ.get("TABPFN_MAX_ROWS", "6000"))


def _lgb_params(task_kind: str, metric_kind: str, variant: str) -> dict:
    p = dict(
        n_estimators=1500,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=10,
        colsample_bytree=0.3,
        subsample=0.8,
        subsample_freq=1,
        reg_lambda=1.0,
        n_jobs=N_THREADS,
        verbose=-1,
    )
    if variant == "deep":
        p.update(num_leaves=63, learning_rate=0.03, colsample_bytree=0.2, min_child_samples=5)
    if variant == "shallow":
        p.update(num_leaves=15, learning_rate=0.05, colsample_bytree=0.4, min_child_samples=20)
    if task_kind == "regression":
        # L1 training for an MAE leaderboard metric, L2 otherwise.
        p["objective"] = "l1" if metric_kind == "mae" else "l2"
    else:
        p["objective"] = "binary"
    return p


def make_estimator(cfg: dict, task_kind: str, metric_kind: str, seed: int = 0):
    """Instantiate one (unfitted) learner from a config dict."""
    learner = cfg["learner"]

    if learner == "lgbm":
        if not HAVE_LGB:
            raise RuntimeError(f"LightGBM unavailable: {_LGB_ERR}")
        params = _lgb_params(task_kind, metric_kind, cfg.get("variant", "base"))
        params["random_state"] = seed
        return (lgb.LGBMRegressor if task_kind == "regression" else lgb.LGBMClassifier)(**params)

    if learner == "cat":
        params = dict(
            iterations=1200,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
            rsm=0.3,
            thread_count=N_THREADS,
            verbose=0,
            allow_writing_files=False,
            random_seed=seed,
        )
        if task_kind == "regression":
            params["loss_function"] = "MAE" if metric_kind == "mae" else "RMSE"
            return CatBoostRegressor(**params)
        params["loss_function"] = "Logloss"
        return CatBoostClassifier(**params)

    if learner == "tabpfn":
        from tabpfn import TabPFNClassifier, TabPFNRegressor

        # The checkpoint is always named explicitly. Left to itself the current
        # package downloads its newest weights (v3.5), and this lane is declared on
        # the TabPFN **v2** open weights, whose licence we hold and whose required
        # attribution line, "Built with PriorLabs-TabPFN", every result carrying a
        # TabPFN member must print. So an unset path is an error, not a default.
        ckpt = TABPFN_CKPT if task_kind == "regression" else TABPFN_CKPT_CLF
        if not ckpt or not os.path.exists(ckpt):
            raise RuntimeError(
                "TabPFN candidate asked for but the v2 checkpoint path is unset or missing: "
                f"set {'TABPFN_CKPT' if task_kind == 'regression' else 'TABPFN_CKPT_CLF'} "
                f"(got {ckpt!r}). Refusing to fall back to the package default, which would "
                "silently use different weights under a different licence."
            )
        kw = dict(device="cpu", n_estimators=TABPFN_N, random_state=seed,
                  ignore_pretraining_limits=True, model_path=ckpt)
        # Measured on this Mac: with 791 training rows, 210 features and only two
        # ensemble members, TabPFN v2 peaked at 4.3 GB and the memory watchdog killed
        # the run. It is an in-context learner, so the whole training set sits in the
        # forward pass. TABPFN_SAVE_MEMORY turns on the package's own memory-saving
        # path, which trades speed for a much smaller peak.
        if os.environ.get("TABPFN_SAVE_MEMORY"):
            kw["memory_saving_mode"] = True
        return (TabPFNRegressor if task_kind == "regression" else TabPFNClassifier)(**kw)

    if learner == "hgb":
        params = dict(
            max_iter=400,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=10,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=50,
            random_state=seed,
        )
        if task_kind == "regression":
            params["loss"] = "absolute_error" if metric_kind == "mae" else "squared_error"
            return HistGradientBoostingRegressor(**params)
        return HistGradientBoostingClassifier(**params)

    if learner == "et":
        params = dict(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=1,
            n_jobs=N_THREADS,
            random_state=seed,
        )
        return (ExtraTreesRegressor if task_kind == "regression" else ExtraTreesClassifier)(**params)

    raise ValueError(f"unknown learner {learner!r}")


def fit_predict(cfg, task_kind, metric_kind, X_tr, y_tr, X_va, y_va, X_pred, seed=0):
    """Fit one config on (X_tr, y_tr) and return predictions for X_pred.

    The validation fold is used only for early stopping, which is what TDC's
    train/valid split is for. No test-set value is ever passed to this function.
    """
    est = make_estimator(cfg, task_kind, metric_kind, seed=seed)
    learner = cfg["learner"]

    y_fit = y_tr
    y_val = y_va
    log_target = task_kind == "regression" and cfg.get("log_target", False)
    if log_target:
        y_fit = np.log1p(np.clip(y_tr, 0, None))
        y_val = np.log1p(np.clip(y_va, 0, None))

    if learner == "tabpfn":
        if len(X_tr) > TABPFN_MAX_ROWS:
            rs = np.random.RandomState(1000 + seed)
            keep = rs.choice(len(X_tr), TABPFN_MAX_ROWS, replace=False)
            X_tr, y_fit = X_tr[keep], y_fit[keep]
        est.fit(X_tr, y_fit)
    elif learner == "lgbm":
        est.fit(
            X_tr,
            y_fit,
            eval_set=[(X_va, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
    elif learner == "cat":
        est.fit(X_tr, y_fit, eval_set=(X_va, y_val), early_stopping_rounds=50, verbose=0)
    else:
        # hgb does its own internal early-stopping split; et needs no validation.
        est.fit(X_tr, y_fit)

    if task_kind == "regression":
        pred = est.predict(X_pred)
        if log_target:
            pred = np.expm1(pred)
        return np.asarray(pred, dtype=np.float64).ravel()

    proba = est.predict_proba(X_pred)
    return np.asarray(proba[:, 1], dtype=np.float64).ravel()


BUDGET = os.environ.get("ADMET_BUDGET", "lean")


def candidate_configs(task_kind: str, metric_kind: str, big: bool, budget: str | None = None) -> list[dict]:
    """The candidate list scored by inner CV.

    Two budgets, because this machine is shared with the lab's native record
    searches and runs at a load average well above its core count. Measured
    cost per fit on ~1k molecules and ~3.5k features: LightGBM a few seconds,
    ExtraTrees similar, CatBoost and HistGradientBoosting an order of magnitude
    more. So:

    * `lean` (default) is LightGBM across feature sets and depths, plus
      ExtraTrees as a decorrelated blend member.
    * `full` adds CatBoost and HistGradientBoosting, for when the machine is
      quiet or a task is worth the extra search.
    * `strong` (round 2, for a machine with real cores) adds, on top of `full`:
      the two new fingerprint blocks through the `dmap` and `full` feature sets,
      and TabPFN v2 on the descriptor-only set. TabPFN is the one candidate here
      that is not trained on this data at all: it is a pretrained in-context
      learner, which is a genuinely different inductive bias from a boosted tree
      and is why it earns a place rather than being a seventh tree variant.

    `big` trims the widest feature set on the larger datasets. `log_target` is
    offered only for Spearman metrics, where any monotone transform of the
    prediction leaves the metric unchanged, but fitting on a log scale changes
    what the squared loss emphasises on these skewed pharmacokinetic targets.
    """
    budget = budget or BUDGET
    feature_sets = ["dm", "dma"] if big else ["dm", "dma", "wide"]

    cfgs = []
    if HAVE_LGB:
        for fs in feature_sets:
            cfgs.append({"features": fs, "learner": "lgbm", "variant": "base"})
        cfgs.append({"features": "dma", "learner": "lgbm", "variant": "deep"})
        cfgs.append({"features": "dma", "learner": "lgbm", "variant": "shallow"})
    cfgs.append({"features": "dma", "learner": "et", "variant": "base"})

    if budget in ("full", "strong"):
        for fs in feature_sets:
            cfgs.append({"features": fs, "learner": "cat", "variant": "base"})
            cfgs.append({"features": fs, "learner": "hgb", "variant": "base"})

    if budget == "maplight":
        # Round 3: CatBoost and LightGBM over the MapLight-shaped set and over the
        # same set without the pretrained embedding. Nothing else, so the inner-CV
        # comparison is about the features rather than a wider model search.
        cfgs = [
            {"features": "maplight", "learner": "lgbm", "variant": "base"},
            {"features": "maplight", "learner": "lgbm", "variant": "shallow"},
            {"features": "maplight", "learner": "lgbm", "variant": "deep"},
            {"features": "maplight", "learner": "cat", "variant": "base"},
            {"features": "mlng", "learner": "lgbm", "variant": "base"},
            {"features": "mlng", "learner": "lgbm", "variant": "shallow"},
            {"features": "mlng", "learner": "lgbm", "variant": "deep"},
            {"features": "mlng", "learner": "cat", "variant": "base"},
            {"features": "mlng", "learner": "et", "variant": "base"},
        ]

    if budget == "strong":
        cfgs.append({"features": "dmap", "learner": "lgbm", "variant": "base"})
        cfgs.append({"features": "dmap", "learner": "lgbm", "variant": "shallow"})
        cfgs.append({"features": "full", "learner": "lgbm", "variant": "base"})
        cfgs.append({"features": "dmap", "learner": "cat", "variant": "base"})
        cfgs.append({"features": "dmap", "learner": "et", "variant": "base"})
        cfgs.append({"features": "desc", "learner": "tabpfn", "variant": "base"})

    if metric_kind == "spearman":
        # Test the log target on the main feature set only, rather than doubling
        # the whole list: the transform either helps on these targets or it does
        # not, and the inner folds answer that with two extra candidates.
        extra = []
        for cfg in cfgs:
            # The log target is offered on the main feature set of whichever round is
            # running. On VDss it was worth 0.22 Spearman to the plain config, so a
            # round whose candidates are all on the raw target would be crippled.
            if cfg["variant"] == "base" and cfg["features"] in ("dma", "maplight", "mlng"):
                c = dict(cfg)
                c["log_target"] = True
                extra.append(c)
        cfgs = cfgs + extra

    for cfg in cfgs:
        cfg.setdefault("log_target", False)
    return _label(cfgs)


def _label(cfgs: list[dict]) -> list[dict]:
    for cfg in cfgs:
        cfg["id"] = f"{cfg['features']}-{cfg['learner']}-{cfg['variant']}" + (
            "-log" if cfg["log_target"] else ""
        )
    return cfgs


def force_log(cfgs: list[dict]) -> list[dict]:
    """Put every candidate on a log1p target, de-duplicated.

    Used for a second round on a Spearman task where the first round's inner CV
    already showed the log target winning by a wide margin, so spending the
    remaining budget on non-log candidates is waste.
    """
    out, seen = [], set()
    for cfg in cfgs:
        c = dict(cfg)
        c["log_target"] = True
        key = (c["features"], c["learner"], c["variant"])
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return _label(out)
