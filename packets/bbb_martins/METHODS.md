# Submission packet: bbb_martins

**Result.** AUROC 0.923 +/- 0.004 over TDC's five official runs (seeds [1, 2, 3, 4, 5]). Per-seed scores: 0.919, 0.922, 0.930, 0.925, 0.920.

**Where it would stand.** 2 of 27 on the public board as read 2026-11-08 (9 tasks early, the other 13 at about 16:45 UTC). The listed top three: MiniMol 0.924; CFA 0.92; MapLight 0.916.

**Protocol.** The venue's own, through PyTDC's `admet_group`: TDC's fixed scaffold
`train_val`/`test`, five runs with seeds 1-5 from `get_train_valid_split`, fit on
train with valid used only for early stopping, predict the fixed test set, and score
with `group.evaluate_many`, whose mean and standard deviation are the two numbers
above. The official metric comes from `tdc.metadata.admet_metrics`.

**Model.** mlng-lgbm-base, mlng-lgbm-deep, mlng-lgbm-shallow, mlng-cat-base, mlng-et-base averaged.

Model choice was made only by inner cross-validation on training rows: candidates were
scored on separate scaffold folds of `train_val` drawn with seeds [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115], and the test labels were read exactly once, by
TDC's evaluator, after the five runs. 5 candidates were
considered. Where a task was run in more than one round, the round reported is the one
with the better inner-CV score, not the better test score.

**How the reported round was chosen, and how thin the margin is.** This task was run in
3 rounds. The reported one was chosen by inner cross-validation on training
rows over 3 shared scaffold folds (tuning seeds 101-103),
scoring 0.9544 against 0.9525 for
`bbb_martins_r2strong.json` -- a margin of 0.0019. Test scores played no part in the
choice. Where a round had originally been tuned on fewer folds, its own candidate list
was re-tuned on the full set first, as a tuning-only pass that produces no test score, so
the two rounds are compared on the same estimator rather than on the three folds they
happened to share. The other round's official test score was 0.917.
A margin of 0.0019 is small, and this result should be read as level with the
entries around it rather than clearly better than them.

**Code.** `codev2/` in the staging repository `~/quantavane-tdc-admet`, with these file hashes:

- `features.py` sha256 `dcf94fa93bae3077...`
- `models.py` sha256 `222bb1afadc5ee36...`
- `protocol.py` sha256 `04194c642bde8f11...`
- `run_task.py` sha256 `95b6d045ca5ad60d...`

**Run record.** `results/bbb_martins_r3maplight.json` holds every per-seed score, the full inner-CV
ranking and the leakage checks; `results/bbb_martins_r3maplight_test_predictions.npy` holds the five
prediction vectors.

**Checks.** zero unparsable SMILES, zero molecules shared between `train_val` and `test`, five official seeds.

**Not submitted.** The TDC submission form is the owner's step.
