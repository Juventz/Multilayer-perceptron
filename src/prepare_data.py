"""
Dataset splitting program (train / validation).

This program is the mandatory first step before training the network.
It performs the following operations:

  1. Read the raw CSV file (data/data.csv) using a predefined column schema
     (the breast cancer dataset has no header row).

  2. Select numeric feature columns, excluding non-informative ones
     ('id' and 'diagnosis').

  3. Z-score normalization of each feature:
        x_normalized = (x - mean) / std
     This brings all features to the same scale (~mean 0, std 1),
     which is essential for gradient descent to converge properly.
     Without normalization, features with large values (e.g. area_mean ~654)
     would dominate features with small values (e.g. fractal_dimension_mean ~0.06).

  4. Train / validation split:
     - Train      : 80% of the data  -> used to adjust the weights
     - Validation : 20% of the data  -> used to measure generalization
     The dataset is randomly shuffled before splitting (fixed seed for
     reproducibility).

  5. Save:
     - data/processed/dataset_splits.npz          : X_train, y_train, X_val, y_val arrays
     - data/processed/preprocessing_metadata.json : means, stds, feature names
       -> These parameters must be reused identically at prediction time.

Usage
-----
    python3 -m src.prepare_data data/data.csv --target diagnosis
    python3 -m src.prepare_data data/data.csv --target diagnosis --val-ratio 0.2 --seed 42
"""

import argparse
import json
from pathlib import Path
import numpy as np

from utils.csv_utils import parse_csv_with_fieldnames
from utils.datasets import BREAST_CANCER_FIELDS
from utils.data_split import train_val_split
from utils.preprocessing import (
    build_feature_matrix,
    extract_target,
    get_numeric_features,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare data for MLP training.")
    parser.add_argument("dataset", type=str, help="Path to CSV dataset")
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target column name (example: Hogwarts House)",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Columns to exclude from numeric features",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument(
        "--save-dir",
        type=str,
        default="data/processed",
        help="Directory where processed train/val arrays are saved",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    fieldnames, rows = parse_csv_with_fieldnames(args.dataset, BREAST_CANCER_FIELDS)
    if args.target != "diagnosis":
        raise ValueError("This dataset expects --target diagnosis.")
    excluded = set(args.exclude + [args.target, "id"])

    features = get_numeric_features(fieldnames, rows, excluded_columns=excluded)
    if not features:
        raise ValueError("No numeric features found after exclusions.")

    X, means, stds = build_feature_matrix(rows, features)
    y = extract_target(rows, args.target)

    X_train, y_train, X_val, y_val = train_val_split(
        X,
        y,
        val_ratio=args.val_ratio,
        random_seed=args.seed,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    arrays_path = save_dir / "dataset_splits.npz"
    npz_payload = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
    }
    np.savez(arrays_path, **npz_payload)

    metadata_path = save_dir / "preprocessing_metadata.json"
    metadata = {
        "schema": "breast-cancer",
        "target": args.target,
        "features": features,
        "means": means.tolist(),
        "stds": stds.tolist(),
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "num_samples": int(X.shape[0]),
    }
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("Data preparation complete")
    print(f"Features selected: {len(features)}")
    print(f"Train shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Val shape:   X={X_val.shape}, y={y_val.shape}")
    print(f"Means shape: {means.shape}, Stds shape: {stds.shape}")
    print(f"Saved arrays: {arrays_path}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
