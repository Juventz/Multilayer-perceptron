## Multilayer Perceptron - Starter Kit

This repository is set up for the first part of your project: understand, preprocess, and split the breast-cancer dataset before training a model.

## First Part Requirement Alignment

Your provided file data/data.csv is handled as follows:

- 32 columns total
- label column: diagnosis (M or B)
- first column id is excluded from model features
- remaining 30 columns are numeric input features

The preparation pipeline now does exactly what the first part asks:

- raw CSV loading
- missing value imputation with feature means
- z-score normalization
- dataset split into train and validation sets
- saved processed arrays for training

### Added utilities

- utils/csv_utils.py
  - headerless CSV parsing with explicit field names
  - safe float conversion
  - numeric column detection
- utils/preprocessing.py
  - numeric feature selection
  - missing value imputation by mean
  - z-score normalization
  - target extraction
- utils/data_split.py
  - train/validation split
- src/prepare_data.py
  - CLI to run preprocessing and save outputs

### Quick start

Run from project root:

```bash
python3 -m src.prepare_data data/data.csv --target diagnosis
```

Optional arguments:

```bash
python3 -m src.prepare_data data/data.csv --target diagnosis --val-ratio 0.2 --seed 42 --save-dir data/processed
```

Generated files:

- data/processed/dataset_splits.npz
- data/processed/preprocessing_metadata.json

### Next coding step

Implement your MLP core from scratch in src/:

1. activations.py (sigmoid, tanh, relu + derivatives)
2. mlp.py (forward, backward, update)
3. train.py (training loop + loss curve)
4. predict.py (inference with saved weights)

