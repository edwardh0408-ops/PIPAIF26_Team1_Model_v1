# Decision tree v1 against always saying active

On the requested test, the tree is right **65.8%** of the time. Always predicting **active** — the most common `final_outcome` — is right **58.7%** of the time. That gain shrinks once the same company cannot appear on both sides of the split.

The panel has 747 yearly snapshots of 228 companies (30 June 2021–2025), so the 30% holdout was used. It is a stratified row split, seed 42: **522 train / 225 test**. Everything, including the 5-fold search, was fit on the training rows only. **194 of the 225 test snapshots (86%)** are companies that also have rows in training.

## What v1 learned

Cross-validation maximized macro F1, so the tree has to separate insolvency and both strike-offs, not only call the majority class. The chosen tree is Gini, **depth 12**, at least 5 rows per leaf and 10 to make a split, no class weights. It grew 62 leaves. Mean macro F1 on the five training folds was **39.5%**. The best accuracy point in the same search was shallower (depth 6, at least 20 rows per leaf) and scored 64.9% cross-validated accuracy with a lower macro F1.

The first split is the number of accounts filed. **More than 8 accounts filings and the tree calls insolvency either way.** In the training rows, 25 of those 30 snapshots are insolvency. In the whole panel that is 11 of 15 companies, and they are older (median age 12.1 years, versus 4.9 years for companies with 8 filings or fewer).

Under 9 filings, the tree leans on days since the last accounts, age, how late accounts have been, strike-off warnings, and charges. Two patterns are solid enough to use as a review list:

- A **registered charge** on a licensed restaurant with few account filings often ends in insolvency: 10 of 15 such companies do, and none of those 15 are strike-offs.
- **Dormant accounts plus a strike-off warning** leans compulsory strike-off: 5 of 9 companies.

A strike-off warning on its own is a weak failure signal. At the latest snapshot, 62 of the 100 companies that stay active have had one. Sector barely matters: 494 of 747 snapshots are licensed restaurants and 233 are unlicensed restaurants and cafes. Jurisdiction, extra SIC codes, and director changes barely enter the tree. Missing accounts-lateness (174 rows with no accounts on file) was left missing so the tree could treat “no accounts” as its own path. Those rows are coded `no_accounts` on last accounts type.

Voluntary strike-off is the outcome this feature set does not catch.

## Baseline vs v1 on the 225 test snapshots

| | Always active | v1 |
|---|---:|---:|
| Accuracy | 58.7% | **65.8%** |
| Macro precision | 14.7% | **48.3%** |
| Macro recall | 25.0% | **46.8%** |
| Train accuracy | — | **80.1%** |

Train accuracy is 14 points above test accuracy, which fits a depth-12 tree with 62 leaves.

Per outcome on the test rows (precision / recall):

- **Active** (132): 73.2% / 78.8%. Baseline is 58.7% / 100%.
- **Insolvency** (56): 69.6% / 57.1%. Baseline never predicts it.
- **Compulsory strike-off** (26): 39.3% / 42.3%.
- **Voluntary strike-off** (11): 11.1% / 9.1% (1 of 11 correct).

Confusion matrix, rows true, columns predicted as active / insolvency / compulsory strike-off / voluntary strike-off:

| True outcome | Active | Insolvency | Compulsory strike-off | Voluntary strike-off |
|---|---:|---:|---:|---:|
| Active | 104 | 12 | 10 | 6 |
| Insolvency | 19 | 32 | 5 | 0 |
| Compulsory strike-off | 12 | 1 | 11 | 2 |
| Voluntary strike-off | 7 | 1 | 2 | 1 |

If insolvency and both strike-offs are collapsed to “does not stay active,” v1 precision is **66.3%** and recall is **59.1%** (93 failing snapshots). Always-active never raises that flag.

## Unseen companies

Refitting the same settings with 69 companies fully held out (230 snapshots): v1 accuracy is **52.6%**, below the **57.4%** always-active baseline. Insolvency recall is 41.4%. Compulsory strike-off recall is 3.4%. Voluntary strike-off recall is 0%. About half of snapshots that do not stay active are flagged, at 57.6% precision.

| | Always active | v1 on unseen companies |
|---|---:|---:|
| Accuracy | 57.4% | 52.6% |
| Macro precision | 14.3% | 28.1% |
| Macro recall | 25.0% | 29.4% |
| Insolvency recall | 0% | 41.4% |
| Compulsory strike-off recall | 0% | 3.4% |
| Voluntary strike-off recall | 0% | 0% |
| “Does not stay active” precision | 0% | 57.6% |
| “Does not stay active” recall | 0% | 50.0% |

Training accuracy on that company split is 83.0%. The tree still memorizes the companies it has seen. On a new company it catches about half of the snapshots that do not stay active, at 57.6% precision, and it almost never names the kind of strike-off.

## How to use v1

Use the 65.8% figure as the score the split you asked for produces. Use 52.6% as the score for a company the product has not seen. The practical use of v1 is a shortlist: long accounts history, or a charge on a licensed restaurant, for an insolvency review. It should not auto-decline a company, and it will not tell a chosen closure from a company that carries on.

The fit is in `train_decision_tree_v1.py`. Numbers are in `outputs/v1_results.json`.
