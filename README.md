# Label Correction and Spectral Features for Multiclass Dementia Detection from EEG

[![tests](https://github.com/aftab802/eeg-dementia-label-correction/actions/workflows/tests.yml/badge.svg)](https://github.com/aftab802/eeg-dementia-label-correction/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Identifier-keyed label correction, 39 graph + qEEG features and GridSearchCV-tuned XGBoost for AD/FTD/HC on OpenNeuro ds004504.

Code accompanying the manuscript **“Label Correction and Spectral Features for Multiclass Dementia Detection from EEG”** by Muhammad Aftab ur Rehman, Muhammad Awais ur Rehman, Kiran Asghar, Farwa Zahra (submitted, 2026).
Corresponding author: Muhammad Aftab ur Rehman, CYBEX – School of IT Professionals (Pvt) Limited, Faisalabad, Pakistan.

## Installation

```bash
git clone https://github.com/aftab802/eeg-dementia-label-correction.git
cd eeg-dementia-label-correction
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Python 3.10–3.13 (tested on 3.13 with TensorFlow 2.20). A GPU is optional.

## Data

OpenNeuro **ds004504** v1.0.6 — resting-state, eyes-closed EEG of 88 subjects (36 AD, 23 FTD, 29 HC), 19 channels,
500 Hz — https://openneuro.org/datasets/ds004504 (Miltiadous et al., *Data* 8(6):95, 2023,
doi [10.3390/data8060095](https://doi.org/10.3390/data8060095)). Download it, for example, with

```bash
pip install openneuro-py
openneuro-py download --dataset ds004504 --target-dir data/ds004504
```

The scripts read the denoised recordings in `derivatives/sub-XXX/eeg/` and labels from `participants.tsv`.

## Reproducing the paper

```bash
python scripts/run_label_correction.py --bids-root data/ds004504 --gsp-features path/to/original_graph_features.csv
python scripts/run_subject_vote_xgb.py --bids-root data/ds004504   # the original notebook's protocol
```
Without `--gsp-features` the eleven graph features are recomputed with `cybex_eeg/dementia/gsp.py`, an independent
reimplementation whose values will differ from the original table.

Every run writes a JSON file to `results/` with the test metrics, confusion matrix, split sizes and, for subject
data, `train_test_subject_overlap` (0 under the leakage-safe protocol).

## Reproduction status

| Result in the paper | Value | Source of the number | Re-run with this repository? |
|---|---|---|---|
| Tuned XGBoost, 39 features | 61.44 % | Original run | **Re-run: 61.50 %; nested CV 60.26 ± 5.81 %** |
| Follow-up experiments (53.41 %, 55.82 %, 60.26 %, 59.08 %, 36.36 %) | — | Original code not in the archive | Not reimplemented |
| Cost-sensitive XGBoost (notebook) | 65.91 % | Original notebook, output preserved | **Re-run: 67.05 %** |

## Points to resolve against the manuscript

* Table 3 lists 4 band-power + 24 summary + 4 ratio features (= 32) plus 11 graph features, which sums to 43, not 39.
  The code builds 28 qEEG features (6 statistics × 4 bands + 4 ratios, the band power entering as the per-band mean),
  giving 39 with the graph features.
* GridSearchCV's `best_score_` is optimistic; the script also reports a nested-CV estimate (re-run: 60.26 ± 5.81 %).

## Repository layout

```
cybex_eeg/        shared, leakage-aware library (splits, normalisation, features, models)
scripts/          one command per experiment in the paper; results are written to results/*.json
tests/            pytest suite run on synthetic data (no download needed)
original_notebooks/  the original Colab notebooks, with their printed outputs, for provenance
```

## Tests

```bash
pytest -q
```
Tests check the identifier-keyed label merge (and that a positional merge misassigns labels), the 28 qEEG and 11 graph features, the LEAD-EEG biomarkers, and that feature selection and SMOTE are fitted inside training folds only.

## Citation

See [`CITATION.cff`](CITATION.cff). Until the article is published, please cite the manuscript title above and
this repository.

## Related repositories

* [eeg-mi-leakage-benchmark](https://github.com/aftab802/eeg-mi-leakage-benchmark) — A Systematic Benchmark of Motor Imagery Classifiers Under Leakage-Safe Evaluation
* [eeg-dementia-integrity-benchmark](https://github.com/aftab802/eeg-dementia-integrity-benchmark) — Data Integrity and Feature Engineering in Small-Cohort EEG Dementia Classification
* [eeg-mi-protocol-leakage-case-study](https://github.com/aftab802/eeg-mi-protocol-leakage-case-study) — A Case Study of Evaluation-Protocol Leakage in Motor Imagery EEG Classification
* [eeg-mi-four-class-cnn](https://github.com/aftab802/eeg-mi-four-class-cnn) — Four-Class Motor Imagery Classification Using a Residual Convolutional Neural Network
* [eeg-mi-fbcsp-cnn](https://github.com/aftab802/eeg-mi-fbcsp-cnn) — CSP+CNN: Filter-Bank Common Spatial Patterns with a Convolutional Classifier for Motor Imagery Classification from EEG
* [group-aware-eegnet](https://github.com/aftab802/group-aware-eegnet) — Group-Aware EEGNet: Subject-Independent Motor Imagery Classification from EEG
* [lead-eeg](https://github.com/aftab802/lead-eeg) — LEAD-EEG: A Lightweight Ensemble Framework for Alzheimer's Detection from EEG

## License

MIT — see [LICENSE](LICENSE). The datasets are distributed by their owners under their own terms.
