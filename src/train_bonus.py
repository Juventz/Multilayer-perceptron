"""
Bonus training program — all 5 bonus features in one place.

The 5 bonuses implemented
--------------------------
1. Advanced optimizer: SGD with Momentum (src/optimizers.py)
   Accumulates a velocity vector in the direction of persistent gradients,
   leading to faster convergence and less oscillation than plain SGD.

2. Early stopping (src/callbacks.py)
   Stops training automatically when val_loss (or val_accuracy) stops
   improving for `patience` consecutive epochs.
   Prevents overfitting and saves training time.

3. Multiple evaluation metrics (src/metrics.py)
   Precision, Recall, F1-score per class + confusion matrix.
   Essential for medical tasks where false negatives (missed cancers)
   have a very different cost than false positives.

4. History of metrics
   Model.fit() returns a history dict with train_loss, val_loss,
   train_accuracy, val_accuracy for every completed epoch.
   Printed as a summary table at the end.

5. Multiple learning curves on the same graph (src/model.py)
   plot_histories() overlays several training runs for easy comparison.
   Activated with --compare: trains SGD vs Momentum and overlays them.

Usage
-----
# Momentum optimizer + early stopping (recommended):
    python3 -m src.train_bonus

# Custom patience:
    python3 -m src.train_bonus --early_stopping --patience 15

# Compare SGD vs Momentum on the same graph:
    python3 -m src.train_bonus --compare

# Evaluate on a dataset after training (full report):
    python3 -m src.train_bonus --predict data/data.csv
"""

import argparse
from pathlib import Path

import numpy as np

from src.layers import DenseLayer
from src.model import Model, plot_histories, encode_labels
from src.optimizers import SGD, SGDMomentum, OPTIMIZERS
from src.metrics import (
    classification_report,
    print_confusion_matrix,
    confusion_matrix,
)
from src.callbacks import EarlyStopping
from src.predict import load_model, binary_crossentropy_eval


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bonus MLP training: momentum optimizer, early stopping, "
                    "extended metrics, model comparison.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument(
        "--data", type=str, default="data/processed/dataset_splits.npz",
        help="Path to .npz arrays from src/prepare_data.py.",
    )

    # Architecture
    parser.add_argument(
        "--layer", type=int, nargs="+", default=[24, 24], metavar="N",
        help="Hidden layer sizes. Example: --layer 24 24 24",
    )
    parser.add_argument(
        "--activation", type=str, default="sigmoid",
        choices=["sigmoid", "relu", "tanh"],
    )

    # Training hyperparameters
    parser.add_argument("--epochs",        type=int,   default=100)
    parser.add_argument("--batch_size",    type=int,   default=32)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument(
        "--loss", type=str, default="categoricalCrossentropy",
        choices=["categoricalCrossentropy"],
    )

    # Bonus 1 — Optimizer
    parser.add_argument(
        "--optimizer", type=str, default="momentum",
        choices=list(OPTIMIZERS.keys()),
        help="sgd: plain gradient descent | momentum: SGD with momentum (default)",
    )
    parser.add_argument(
        "--momentum", type=float, default=0.9,
        help="Momentum coefficient (used with --optimizer momentum).",
    )

    # Bonus 2 — Early stopping
    parser.add_argument(
        "--early_stopping", action="store_true", default=False,
        help="Stop training when val_loss stops improving.",
    )
    parser.add_argument(
        "--patience", type=int, default=10,
        help="Epochs without improvement before stopping.",
    )
    parser.add_argument(
        "--monitor", type=str, default="val_loss",
        choices=["val_loss", "val_accuracy"],
        help="Metric to watch for early stopping.",
    )

    # Bonus 5 — Compare SGD vs Momentum
    parser.add_argument(
        "--compare", action="store_true", default=False,
        help="Train SGD and Momentum on the same architecture and overlay their curves.",
    )

    # Predict after training
    parser.add_argument(
        "--predict", type=str, default=None, metavar="CSV",
        help="After training, run full evaluation (precision/recall/F1) on this CSV.",
    )

    # Output
    parser.add_argument("--save", type=str, default="saved_model.npy")

    return parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_optimizer(name: str, learning_rate: float, momentum: float):
    """Instantiate the optimizer from its name and hyperparameters."""
    if name == "sgd":
        return SGD(learning_rate=learning_rate)
    return SGDMomentum(learning_rate=learning_rate, momentum=momentum)


def build_network(layer_sizes: list[int], activation: str, output_dim: int):
    """Build a fresh Network with the given architecture."""
    model = Model()
    layers = [
        DenseLayer(size, activation=activation, weights_initializer="heUniform")
        for size in layer_sizes
    ]
    layers.append(
        DenseLayer(output_dim, activation="softmax", weights_initializer="heUniform")
    )
    return model.createNetwork(layers)


def print_history_summary(history: dict) -> None:
    """
    Bonus 4 — Print a summary table of the recorded metrics.

    Shows the first epoch, the best epoch (lowest val_loss), and the last
    epoch so the full training trajectory can be understood at a glance.
    """
    n = len(history["train_loss"])
    best_idx = int(np.argmin(history["val_loss"]))

    print("\n--- Metrics history summary ---")
    print(f"{'Epoch':>6}  {'train_loss':>10}  {'val_loss':>10}  {'train_acc':>10}  {'val_acc':>10}")
    print("-" * 54)
    for idx in sorted({0, best_idx, n - 1}):
        marker = " <- best val_loss" if idx == best_idx else ""
        print(
            f"{idx + 1:>6}  "
            f"{history['train_loss'][idx]:>10.4f}  "
            f"{history['val_loss'][idx]:>10.4f}  "
            f"{history['train_accuracy'][idx]:>10.4f}  "
            f"{history['val_accuracy'][idx]:>10.4f}"
            f"{marker}"
        )


def run_evaluation(csv_path: str, model_path: str) -> None:
    """
    Bonus 3 — Full evaluation: precision, recall, F1, confusion matrix.

    Args:
        csv_path:   Path to the labeled CSV file.
        model_path: Path to the saved .npy model.
    """
    import json
    from utils.csv_utils import parse_csv_with_fieldnames
    from utils.datasets import BREAST_CANCER_FIELDS
    from utils.preprocessing import build_feature_matrix

    network, classes = load_model(model_path)
    m_idx = classes.index("M")

    with open("data/processed/preprocessing_metadata.json", encoding="utf-8") as f:
        meta = json.load(f)

    _, rows = parse_csv_with_fieldnames(csv_path, BREAST_CANCER_FIELDS)
    X, _, _ = build_feature_matrix(
        rows, meta["features"],
        means=np.array(meta["means"]),
        stds=np.array(meta["stds"]),
    )
    A = network.forward(X)

    y_true = np.array([row["diagnosis"].strip() for row in rows])
    y_binary = (y_true == "M").astype(float)
    bce = binary_crossentropy_eval(A[:, m_idx], y_binary)

    y_true_int, _, _ = encode_labels(y_true)
    correct = int(np.sum(np.argmax(A, axis=1) == y_true_int))
    acc = correct / len(y_true_int)

    print(f"\n{'─' * 45}")
    print(f"Binary cross-entropy loss : {bce:.4f}")
    print(f"Accuracy                  : {acc:.4f}  ({correct}/{len(y_true_int)})")
    print(f"{'─' * 45}")
    print(classification_report(A, y_true_int, classes))
    print_confusion_matrix(confusion_matrix(A, y_true_int, len(classes)), classes)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # ---- Load data -------------------------------------------------------
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data not found: '{data_path}'.\n"
            "Run first: python3 -m src.prepare_data data/data.csv --target diagnosis"
        )

    arrays = np.load(data_path, allow_pickle=True)
    X_train: np.ndarray = arrays["X_train"]
    y_train: np.ndarray = arrays["y_train"]
    X_val: np.ndarray   = arrays["X_val"]
    y_val: np.ndarray   = arrays["y_val"]

    output_dim = int(len(np.unique(np.concatenate([y_train, y_val]))))

    # ---- Bonus 5: compare SGD vs Momentum --------------------------------
    if args.compare:
        _run_comparison(args, X_train, y_train, X_val, y_val, output_dim)
        return

    # ---- Single training run ---------------------------------------------
    print(f"x_train shape : {X_train.shape}")
    print(f"x_valid shape : {X_val.shape}")
    print()
    print(f"Architecture  : input({X_train.shape[1]}) -> "
          + " -> ".join(str(n) for n in args.layer)
          + f" -> output({output_dim}, softmax)")
    print(f"Hidden activation : {args.activation}")
    print(f"Optimizer         : {args.optimizer}  (lr={args.learning_rate})")
    print(f"Batch size        : {args.batch_size}")
    print(f"Epochs            : {args.epochs}")
    if args.early_stopping:
        print(f"Early stopping    : patience={args.patience}, monitor={args.monitor}")
    print()

    model = Model()
    network = build_network(args.layer, args.activation, output_dim)
    optimizer = build_optimizer(args.optimizer, args.learning_rate, args.momentum)

    # Bonus 2: early stopping
    es = None
    if args.early_stopping:
        es = EarlyStopping(
            patience=args.patience,
            restore_best_weights=True,
            monitor=args.monitor,
        )

    # Train
    history = model.fit(
        network,
        X_train, y_train,
        X_val,   y_val,
        loss=args.loss,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
        save_path=args.save,
        optimizer=optimizer,
        early_stopping=es,
    )

    # Bonus 4: history summary
    print_history_summary(history)

    # Bonus 3: extended evaluation
    if args.predict:
        run_evaluation(args.predict, args.save)


def _run_comparison(args, X_train, y_train, X_val, y_val, output_dim) -> None:
    """
    Bonus 5 — Train SGD and Momentum on the same setup and overlay their curves.

    The architecture, epochs, and batch size are identical so the comparison
    isolates the effect of the optimizer.
    """
    configs = [
        ("SGD      lr=0.01", "sgd",      0.01),
        ("Momentum lr=0.01", "momentum", 0.01),
    ]

    print("=== SGD vs Momentum comparison ===")
    print(f"Architecture : input({X_train.shape[1]}) -> "
          + " -> ".join(str(n) for n in args.layer)
          + f" -> output({output_dim}, softmax)")
    print(f"Epochs : {args.epochs}  |  Batch size : {args.batch_size}")
    print()

    histories = []
    model = Model()

    for label, opt_name, lr in configs:
        print(f"--- Training: {label} ---")
        network = build_network(args.layer, args.activation, output_dim)
        optimizer = build_optimizer(opt_name, lr, args.momentum)

        history = model.fit(
            network,
            X_train, y_train,
            X_val,   y_val,
            loss=args.loss,
            learning_rate=lr,
            batch_size=args.batch_size,
            epochs=args.epochs,
            seed=args.seed,
            save_path=None,
            optimizer=optimizer,
        )
        histories.append((label, history))
        print()

    # Bonus 5: overlay both runs on the same graph
    plot_histories(histories, save_path="comparison.png")


if __name__ == "__main__":
    main()
