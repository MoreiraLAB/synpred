# SYNpred standalone inference

This document covers the scientific standalone workflow only. The public
repository does not contain or deploy the SYNpred web interface.

## Input

Use a comma-separated CSV with the columns `Cell`, `Drug1` and `Drug2`.
The drug columns contain SMILES strings. See `standalone_example.csv`.

## Model bundle

The pretrained models are distributed separately because they are too
large for GitHub. Download the released bundle from the
[MoreiraLAB SYNpred resource page](https://www.moreiralab.com/resources/synpred/).

The expected layout is:

- `standalone_models/ML_models`: 36 `.pkl` estimators
- `standalone_models/DL_models`: 24 individual and 6 ensemble `.h5` models

Define a non-default location with:

`export SYNPRED_MODEL_DIR=/absolute/path/to/standalone_models`

## Validated environment

Python 3.8.2, NumPy 1.20.3, Pandas 1.2.5, SciPy 1.6.3,
scikit-learn 0.24.2, TensorFlow 2.3.1, h5py 2.10.0,
XGBoost 1.4.2, RDKit 2021.09.3 and Mordred 1.2.0.

Create it with `bash scripts/create_standalone_env.sh`.

## Run

`python standalone_deploy_model.py standalone_example.csv`

Output is written to
`standalone_results/standalone_example_prediction.csv`.

## Validate

`python -m unittest -v tests.test_standalone`

The repaired workflow was validated from a clean clone with a complete
1 x 1347 feature matrix, 36 ML models, 24 individual DL models and
6 ensemble models, producing finite values for all six SYNpred outputs.
