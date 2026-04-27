"""
Prediction program using a pre-trained MLP model (mandatory).

This program:
  1. Loads the saved model (weights + architecture) from a .npy file.
  2. Loads and normalizes a new CSV dataset using the same normalization
     parameters (means and stds) that were used at training time.
  3. Runs the forward pass to obtain predicted probabilities.
  4. Evaluates performance using the binary cross-entropy formula:
        E = -(1/N) * sum( y * log(p) + (1 - y) * log(1 - p) )
     where p = P(Malignant | x), y = 1 if malignant, 0 if benign.
  5. Displays predictions, binary cross-entropy, and accuracy.

For extended evaluation (precision, recall, F1, confusion matrix),
see the bonus program: src/train_bonus.py --predict

Why reuse the same normalization parameters?
--------------------------------------------
During training, each feature was standardized as x = (x - mean) / std
where mean and std were computed on the training set.
At prediction time we MUST apply the exact same transformation —
different parameters would make all predictions meaningless.

Usage
-----
    python3 -m src.predict data/data.csv
    python3 -m src.predict data/data.csv --model saved_model.npy
"""

import argparse
import json
from pathlib import Path

import numpy as np

from utils.csv_utils import parse_csv_with_fieldnames
from utils.datasets import BREAST_CANCER_FIELDS
from utils.preprocessing import build_feature_matrix
from src.layers import DenseLayer
from src.model import Network, encode_labels


def load_model(model_path: str) -> tuple[Network, list[str]]:
    """
    Restore a neural network from a .npy file saved by train.py.

    The .npy file contains a Python dictionary with:
        - 'classes' : list of class names (['B', 'M'])
        - 'layers'  : each layer's config (units, activation, initializer)
        - 'weights' : list of {'W': ..., 'b': ...} for each layer

    Args:
        model_path: Path to the .npy model file.

    Returns:
        network: Restored network with loaded weights, ready for inference.
        classes: List of class names (e.g. ['B', 'M']).
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: '{model_path}'.\n"
            "Run first: python3 -m src.train"
        )

    # allow_pickle=True is required because the .npy file contains a Python dict
    model_data: dict = np.load(model_path, allow_pickle=True).item()

    # np.load may convert lists to numpy arrays -> force to Python list
    classes: list[str] = list(model_data["classes"])

    layer_list = [
        DenseLayer(cfg["units"], cfg["activation"], cfg["weights_initializer"])
        for cfg in model_data["layers"]
    ]
    for layer, w_dict in zip(layer_list, model_data["weights"]):
        layer.W = w_dict["W"]
        layer.b = w_dict["b"]

    network = Network(layer_list)
    network.initialized = True
    return network, classes


def binary_crossentropy_eval(
    p_positive: np.ndarray, y_binary: np.ndarray
) -> float:
    """
    Compute the binary cross-entropy between predicted probabilities and true labels.

    Formula (from the subject):
        E = -(1/N) * sum_n( y_n * log(p_n) + (1 - y_n) * log(1 - p_n) )

    Args:
        p_positive: Predicted probabilities for the positive class (M), shape (n,).
        y_binary:   Binary labels: 1.0 for M, 0.0 for B, shape (n,).

    Returns:
        Scalar: mean binary cross-entropy.
    """
    p = np.clip(p_positive, 1e-15, 1.0 - 1e-15)
    return float(-np.mean(y_binary * np.log(p) + (1 - y_binary) * np.log(1 - p)))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict and evaluate with a pre-trained MLP model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dataset", type=str, help="Path to the CSV file to predict on.")
    parser.add_argument("--model",    type=str, default="saved_model.npy")
    parser.add_argument("--metadata", type=str, default="data/processed/preprocessing_metadata.json")
    return parser


def main() -> None:
    """Entry point of the mandatory prediction program."""
    parser = _build_parser()
    args = parser.parse_args()

    network, classes = load_model(args.model)
    m_idx = classes.index("M")

    meta_path = Path(args.metadata)
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: '{args.metadata}'.")

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    features: list[str] = meta["features"]
    means = np.array(meta["means"])
    stds  = np.array(meta["stds"])

    fieldnames, rows = parse_csv_with_fieldnames(args.dataset, BREAST_CANCER_FIELDS)
    X, _, _ = build_feature_matrix(rows, features, means=means, stds=stds)

    # Forward pass: each row of A is [P(Benign), P(Malignant)]
    A = network.forward(X)
    pred_indices = np.argmax(A, axis=1)
    confidences  = np.max(A, axis=1)

    # Display predictions
    print(f"\n{'#':>6}  {'Prediction':>12}  {'Confidence':>10}  P(Benign)  P(Malignant)")
    print("-" * 65)
    for i, (pred_idx, conf) in enumerate(zip(pred_indices, confidences)):
        label = classes[pred_idx]
        print(
            f"{i:>6}  {label:>12}  {conf:>10.4f}  "
            f"{A[i, 0]:>9.4f}  {A[i, 1]:>11.4f}"
        )

    # Evaluate if ground-truth labels are available
    try:
        y_true = np.array([row["diagnosis"].strip() for row in rows])
        y_binary = (y_true == "M").astype(float)
        p_malignant = A[:, m_idx]
        bce = binary_crossentropy_eval(p_malignant, y_binary)

        y_true_int, _, _ = encode_labels(y_true)
        correct = int(np.sum(pred_indices == y_true_int))
        acc = correct / len(y_true_int)

        print(f"\n{'─' * 40}")
        print(f"Binary cross-entropy loss : {bce:.4f}")
        print(f"Accuracy                  : {acc:.4f}  ({correct}/{len(y_true_int)})")
        print(f"{'─' * 40}")

    except KeyError:
        print("\n(No 'diagnosis' column found -> evaluation skipped.)")


if __name__ == "__main__":
    main()
