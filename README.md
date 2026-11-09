# QV-ADMET: inner-CV model selection on the TDC ADMET Benchmark Group

Gradient-boosted trees and a pretrained tabular model over RDKit descriptors and
fingerprints, run under the official protocol of the Therapeutics Data Commons
ADMET Benchmark Group. Every model choice is made by cross-validation inside the
training rows; the held-out test split of each task is scored once per run, by TDC's
own evaluator.

Nothing here has been submitted to the leaderboard. The ranks in the table are what
these numbers *would* hold against the public boards as read on 8 November 2026, and
they are printed so the comparison can be checked, not as a claim of a listing.

## What the code does

For one task:

1. `features.py` turns each SMILES string into RDKit 2D descriptors and several
   fingerprints (Morgan counts, feature-invariant Morgan, Avalon, MACCS keys,
   atom-pair counts, RDKit path counts). Each molecule is featurised on its own, with
   no labels and nothing from any other molecule, and the blocks are cached by a hash
   of the exact SMILES list.
2. `models.py` holds the candidate list: LightGBM at several depths and feature sets,
   ExtraTrees, CatBoost, HistGradientBoosting, and TabPFN v2 as a pretrained
   in-context learner on the descriptor-only set. On the Spearman tasks each main
   candidate is also offered on a `log1p` target.
3. `protocol.py` scores every candidate on three extra scaffold train/valid folds
   drawn from the training portion only, ranks them, picks the best top-k blend, and
   decides on held-out halves whether to stack the members or average them. Then it
   refits the chosen members on each of TDC's five official seeds and predicts the
   fixed test set. `group.evaluate_many` returns the mean and standard deviation that
   the leaderboard asks for.

The test labels are touched exactly once per task per run, by TDC's evaluator, after
the five runs are complete. No candidate, blend, hyper-parameter, feature set or
target transform is chosen with reference to them.

## How to run it

```sh
uv venv --python python3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python --no-deps PyTDC==1.1.15
code/setup_libomp.sh          # macOS without Homebrew only; see the script
code/run.sh caco2_wang        # one task; TDC downloads the splits into data/
code/run.sh vdss_lombardo --budget strong --tag strong
../.venv/bin/python code/make_table.py   # rebuild the results table
```

`PyTDC` must be installed with `--no-deps`: its `tiledbsoma` dependency pins
`pyarrow 12.0.1`, which has no Python 3.12 wheel. `setuptools` must stay below 81,
because PyTDC imports `pkg_resources`. Neither affects the benchmark path.

## Results

All 17 tasks run so far, under the official five-seed protocol. "Board top"
is the best listed entry on that task's public page on 8 November 2026.

| Task | Metric | This code | Board top 3 | Would-be rank | Selected model |
|---|---|---|---|---|---|
| bbb_martins | AUROC | 0.923 ± 0.004 | MiniMol 0.924; CFA 0.92; MapLight 0.916 | **2 of 27** | mlng-lgbm-base + mlng-lgbm-deep + mlng-lgbm-shallow + mlng-cat-base + mlng-et-base |
| bioavailability_ma | AUROC | 0.725 ± 0.021 | MiniMol 0.942; MapLight + GNN 0.938; ZairaChem 0.935 | **21 of 21** | dm-lgbm-base + wide-lgbm-base |
| caco2_wang | MAE | 0.300 ± 0.013 | CaliciBoost 0.256; XG Boost 0.274; MapLight 0.276 | **9 of 25** | mlng-lgbm-deep + mlng-cat-base (stacked) |
| clearance_hepatocyte_az | Spearman | 0.460 ± 0.011 | CFA 0.536; MapLight + GNN 0.498; MapLight 0.466 | **4 of 19** | mlng-cat-base-log + mlng-lgbm-base-log + mlng-et-base-log |
| clearance_microsome_az | Spearman | 0.583 ± 0.009 | MapLight + GNN 0.63; MiniMol 0.628; MapLight 0.626 | **10 of 21** | dma-lgbm-shallow-log + wide-lgbm-base-log + dma-lgbm-deep-log |
| cyp2c9_substrate_carbonmangels | AUPRC | 0.380 ± 0.015 | MiniMol 0.474; ZairaChem 0.441; MapLight + GNN 0.437 | **13 of 21** | dm-lgbm-base + dma-lgbm-shallow |
| cyp2d6_substrate_carbonmangels | AUPRC | 0.686 ± 0.033 | ContextPred 0.736; DeepMol (AutoML) 0.731; MapLight + GNN 0.72 | **8 of 19** | dma-lgbm-shallow |
| cyp3a4_substrate_carbonmangels | AUROC | 0.644 ± 0.021 | CFA 0.667; MiniMol 0.663; CNN (DeepPurpose) 0.662 | **7 of 20** | dma-lgbm-shallow |
| dili | AUROC | 0.881 ± 0.021 | MiniMol 0.956; ZairaChem 0.925; AttrMasking 0.919 | **13 of 21** | dma-lgbm-shallow |
| half_life_obach | Spearman | 0.508 ± 0.025 | CFA 0.576; MapLight 0.562; MapLight + GNN 0.557 | **6 of 21** | dma-et-base-log + dma-lgbm-base-log + dma-et-base + dma-lgbm-base |
| herg | AUROC | 0.836 ± 0.007 | MapLight + GNN 0.88; CFA 0.875; SimGCN 0.874 | **9 of 21** | dma-lgbm-shallow + dma-lgbm-base + dm-lgbm-base |
| hia_hou | AUROC | 0.986 ± 0.002 | MiniMol 0.993; DeepMol (AutoML) 0.99; MapLight + GNN 0.989 | **5 of 21** | mlng-lgbm-base + mlng-lgbm-shallow + mlng-lgbm-deep + mlng-cat-base + mlng-et-base |
| ld50_zhu | MAE | 0.586 ± 0.004 | BaseBoosting KyQVZ6b2 0.552; ADMETrix 0.573; MiniMol 0.585 | **4 of 23** | dma-et-base + dma-lgbm-deep + dmap-lgbm-base |
| lipophilicity_astrazeneca | MAE | 0.524 ± 0.003 | MiniMol 0.456; Chemprop-RDKit 0.467; Chemprop 0.47 | **7 of 22** | dma-lgbm-shallow |
| pgp_broccatelli | AUROC | 0.910 ± 0.015 | MapLight + GNN 0.938; ZairaChem 0.935; MapLight 0.93 | **8 of 17** | dma-lgbm-deep + dma-lgbm-shallow + dma-lgbm-base + dma-et-base |
| ppbr_az | MAE | 7.766 ± 0.162 | Gradient Boost 7.44; MapLight + GNN 7.526; MapLight 7.66 | **5 of 21** | dm-lgbm-base + dma-lgbm-shallow + dma-lgbm-base |
| vdss_lombardo | Spearman | 0.702 ± 0.010 | MapLight + GNN 0.713; MapLight 0.707; CFA 0.628 | **3 of 20** | dm-lgbm-base-log + dma-lgbm-shallow-log |

### Tasks run in more than one round

Which round is reported above is decided by the inner-CV score of the selected
blend, before the test score is looked at. Both rounds are printed here, including
the cases where the round that lost on inner CV scored better on test, because
reporting those would make the published number a selected maximum.

Inner CV is the score on the folds the rounds were compared on, which is what the
choice was made with. Where a round was originally tuned on fewer folds, its own
candidate list was re-tuned on the full set first, as a tuning-only pass that produces
no test score, so both rounds are scored by the same estimator.

| Task | Round | Folds compared | Inner CV | Official test | Reported |
|---|---|---|---|---|---|
| bbb_martins | `bbb_martins.json` | 15 | 0.9439 | 0.914 | no |
| bbb_martins | `bbb_martins_r3maplight.json` | 15 | 0.9459 | 0.923 | yes |
| caco2_wang | `caco2_wang.json` | 3 | 0.3884 | 0.291 | no |
| caco2_wang | `caco2_wang_r3maplight.json` | 3 | 0.3778 | 0.300 | yes |
| clearance_hepatocyte_az | `clearance_hepatocyte_az.json` | 3 | 0.5700 | 0.454 | no |
| clearance_hepatocyte_az | `clearance_hepatocyte_az_r3maplight.json` | 3 | 0.5830 | 0.460 | yes |
| clearance_hepatocyte_az | `clearance_hepatocyte_az_round2.json` | 3 | 0.5703 | 0.455 | no |
| clearance_microsome_az | `clearance_microsome_az.json` | 3 | 0.6482 | 0.592 | no |
| clearance_microsome_az | `clearance_microsome_az_r3maplight.json` | 3 | 0.6436 | 0.601 | no |
| clearance_microsome_az | `clearance_microsome_az_round2.json` | 3 | 0.6668 | 0.583 | yes |
| half_life_obach | `half_life_obach.json` | 3 | 0.4778 | 0.508 | yes |
| half_life_obach | `half_life_obach_r3maplight.json` | 3 | 0.4773 | 0.512 | no |
| hia_hou | `hia_hou.json` | 15 | 0.9833 | 0.987 | no |
| hia_hou | `hia_hou_r3maplight.json` | 15 | 0.9842 | 0.986 | yes |
| ppbr_az | `ppbr_az.json` | 3 | 8.8515 | 7.766 | yes |
| ppbr_az | `ppbr_az_r3maplight.json` | 3 | 8.9432 | 7.712 | no |
| vdss_lombardo | `vdss_lombardo.json` | 3 | 0.6690 | 0.706 | no |
| vdss_lombardo | `vdss_lombardo_r3maplight.json` | 3 | 0.6659 | 0.709 | no |
| vdss_lombardo | `vdss_lombardo_round2.json` | 3 | 0.6883 | 0.702 | yes |

### One fixed config against the selection ensemble

On some tasks a single fixed LightGBM configuration (`dma-lgbm-base`, no search, no
blending) was also run under the same protocol, as a control. It is a fair question
whether the search earns its keep, and on these tasks the answer is mixed.

| Task | Metric | Selection ensemble | One fixed config | Inner CV says | Test says |
|---|---|---|---|---|---|
| caco2_wang | MAE | 0.300 ± 0.013 | 0.299 ± 0.013 | ensemble | fixed config |
| clearance_hepatocyte_az | Spearman | 0.460 ± 0.011 | 0.434 ± 0.027 | ensemble | ensemble |
| clearance_microsome_az | Spearman | 0.583 ± 0.009 | 0.574 ± 0.017 | ensemble | ensemble |
| dili | AUROC | 0.881 ± 0.021 | 0.879 ± 0.009 | ensemble | ensemble |
| half_life_obach | Spearman | 0.508 ± 0.025 | 0.200 ± 0.075 | ensemble | ensemble |
| herg | AUROC | 0.836 ± 0.007 | 0.844 ± 0.008 | ensemble | fixed config |
| hia_hou | AUROC | 0.986 ± 0.002 | 0.983 ± 0.003 | ensemble | ensemble |
| ld50_zhu | MAE | 0.586 ± 0.004 | 0.602 ± 0.004 | ensemble | ensemble |
| ppbr_az | MAE | 7.766 ± 0.162 | 7.904 ± 0.186 | ensemble | ensemble |
| vdss_lombardo | Spearman | 0.702 ± 0.010 | 0.484 ± 0.032 | ensemble | ensemble |

The reported number is always the model inner CV picked, even where the control
scored better on test.

## What is in the result files

`results/<task>.json` holds the official mean and standard deviation, all five
per-seed scores, the selected members and their full configuration, the inner-CV
score of every candidate and every blend considered, the dataset sizes, the exact
package versions, and two checks: how many SMILES RDKit could not parse, and how many
molecules appear in both the training and test split. `results/<task>_test_predictions.npy`
holds the five prediction vectors. `results/<task>_plain.json` is the control arm.
`leaderboards_2026-11-08.json` is the public board data the ranks are computed against,
transcribed as printed, with the two pages whose stated metric direction contradicts
their own metric noted in the file.

## Notes on two of the tasks

- **Bioavailability_Ma**: its public board duplicates the `Pgp_Broccatelli` table from
  rank 3 down, which is open as TDC issue #394. Our run is included for completeness;
  its rank column cannot be taken at face value.
- **Clearance_Hepatocyte_AZ**: the dataset merges human and rat assays, open as TDC
  issue #405.

## Licence

MIT, see `LICENSE`. The datasets, the splits and the scoring belong to the
Therapeutics Data Commons and keep their own terms; this repository contains none of
their data, only code that asks the official `PyTDC` package for it.

Author: Arhan Canli (Quantavane).
