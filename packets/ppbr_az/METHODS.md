# Submission packet: ppbr_az

**Result.** MAE 7.210 +/- 0.217 over TDC's five official runs (seeds [1, 2, 3, 4, 5]). Per-seed scores: 7.481, 7.459, 7.061, 7.088, 6.960.

**Where it would stand.** 1 of 21 on the public board as read 2026-11-08 (9 tasks early, the other 13 at about 16:45 UTC). The listed top three: Gradient Boost 7.44; MapLight + GNN 7.526; MapLight 7.66.

**Protocol.** The venue's own, through PyTDC's `admet_group`: TDC's fixed scaffold
`train_val`/`test`, five runs with seeds 1-5 from `get_train_valid_split`, fit on
train with valid used only for early stopping, predict the fixed test set, and score
with `group.evaluate_many`, whose mean and standard deviation are the two numbers
above. The official metric comes from `tdc.metadata.admet_metrics`.

**Model.** desc-tabpfn-base.

Model choice was made only by inner cross-validation on training rows: candidates were
scored on separate scaffold folds of `train_val` drawn with seeds [101, 102, 103], and the test labels were read exactly once, by
TDC's evaluator, after the five runs. 14 candidates were
considered. Where a task was run in more than one round, the round reported is the one
with the better inner-CV score, not the better test score.

**Attribution.** Built with PriorLabs-TabPFN.

**How the reported round was chosen, and how thin the margin is.** This task was run in
3 rounds. The reported one was chosen by inner cross-validation on training
rows over 3 shared scaffold folds (tuning seeds 101-103),
scoring 8.3209 against 8.8515 for
`ppbr_az.json` -- a margin of 0.5305. Test scores played no part in the
choice. Where a round had originally been tuned on fewer folds, its own candidate list
was re-tuned on the full set first, as a tuning-only pass that produces no test score, so
the two rounds are compared on the same estimator rather than on the three folds they
happened to share. The other round's official test score was 7.766.
A margin of 0.5305 is small, and this result should be read as level with the
entries around it rather than clearly better than them.

**Code.** `code/` in the staging repository `~/quantavane-tdc-admet`, with these file hashes:

- `features.py` sha256 `b805a19ce30370e7...`
- `models.py` sha256 `4190a43f10c16f38...`
- `protocol.py` sha256 `706f55b9b77d9467...`
- `run_task.py` sha256 `609192df71b9c2e9...`

**Run record.** `results/ppbr_az_r2strong.json` holds every per-seed score, the full inner-CV
ranking and the leakage checks; `results/ppbr_az_r2strong_test_predictions.npy` holds the five
prediction vectors.

**Checks.** zero unparsable SMILES, zero molecules shared between `train_val` and `test`, five official seeds.

**Not submitted.** The TDC submission form is the owner's step.
