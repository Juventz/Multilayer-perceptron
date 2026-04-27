"""
Training program for the Multilayer Perceptron (mandatory).

This program:
  1. Loads the data prepared by src/prepare_data.py (numpy .npz arrays).
  2. Builds the network according to the provided parameters.
  3. Trains the network using backpropagation + mini-batch gradient descent.
  4. Displays loss and accuracy metrics at every epoch.
  5. Saves the trained model to a .npy file.
  6. Displays the two learning curve graphs (loss and accuracy).

For advanced features (Adam, RMSprop, Nesterov, early stopping, precision/recall/F1),
see the bonus program: src/train_bonus.py

Usage
-----
# Default (2 hidden layers of 24, plain SGD):
    python3 -m src.train

# Custom architecture:
    python3 -m src.train --layer 24 24 24 --epochs 84 \\
        --batch_size 8 --learning_rate 0.0314

# From a JSON config file:
    python3 -m src.train --config my_config.json

JSON config file format (all keys optional):
    {
      "layer": [24, 24],
      "activation": "sigmoid",
      "epochs": 100,
      "batch_size": 32,
      "learning_rate": 0.01
    }
"""

import argparse
import json
from pathlib import Path

import numpy as np

from src.layers import DenseLayer
from src.model import Model


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an MLP on the preprocessed breast cancer dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/dataset_splits.npz",
        help="Path to the preprocessed .npz arrays (from src/prepare_data.py).",
    )
    parser.add_argument(
        "--layer",
        type=int,
        nargs="+",
        default=[24, 24],
        metavar="N",
        help="Hidden layer sizes. Example: --layer 24 24 24",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="sigmoid",
        choices=["sigmoid", "relu", "tanh"],
        help="Activation function for the hidden layers.",
    )
    parser.add_argument("--epochs",        type=int,   default=100)
    parser.add_argument("--loss",          type=str,   default="categoricalCrossentropy",
                        choices=["categoricalCrossentropy"])
    parser.add_argument("--batch_size",    type=int,   default=32)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument(
        "--save",
        type=str,
        default="saved_model.npy",
        help="Path to save the trained model (weights + architecture).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional JSON config file (CLI arguments take priority).",
    )
    return parser


def _apply_config_file(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """
    Merge a JSON config file with CLI arguments.
    Config file values are only applied when the CLI value is still the default.
    """
    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    defaults = {action.dest: action.default for action in parser._actions}
    for key, value in cfg.items():
        if key in defaults and getattr(args, key, None) == defaults.get(key):
            setattr(args, key, value)


def main() -> None:
    """Entry point of the mandatory training program."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.config:
        _apply_config_file(args, parser)

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

    # ---- Summary ---------------------------------------------------------
    print(f"x_train shape : {X_train.shape}")
    print(f"x_valid shape : {X_val.shape}")
    print()
    print(f"Architecture  : input({X_train.shape[1]}) -> "
          + " -> ".join(str(n) for n in args.layer)
          + f" -> output({output_dim}, softmax)")
    print(f"Hidden activation : {args.activation}")
    print(f"Learning rate     : {args.learning_rate}")
    print(f"Batch size        : {args.batch_size}")
    print(f"Epochs            : {args.epochs}")
    print()

    # ---- Build network ---------------------------------------------------
    model = Model()
    layer_list: list[DenseLayer] = [
        DenseLayer(size, activation=args.activation, weights_initializer="heUniform")
        for size in args.layer
    ]
    layer_list.append(
        DenseLayer(output_dim, activation="softmax", weights_initializer="heUniform")
    )
    network = model.createNetwork(layer_list)

    # ---- Train (plain SGD) -----------------------------------------------
    model.fit(
        network,
        X_train,
        y_train,
        X_val,
        y_val,
        loss=args.loss,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
