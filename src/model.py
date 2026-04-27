"""
Network container, training loop (Model), and curve comparison utility.

Overall MLP architecture
-------------------------
A Multilayer Perceptron (MLP) is a neural network organized in layers:

    [Input] -> [Hidden layer 1] -> [Hidden layer 2] -> ... -> [Output layer]

Each layer transforms its inputs: A_l = f(A_{l-1} · W_l + b_l)

Training loop (mini-batch gradient descent)
--------------------------------------------
For each epoch:
  1. Randomly shuffle the training data.
  2. Split into mini-batches of size `batch_size`.
  3. For each mini-batch:
     a. Forward pass  -> predictions A
     b. Compute initial gradient: delta = A - Y_onehot
     c. Backward pass -> update all W and b (via the chosen optimizer)
  4. Compute loss and accuracy on train + validation (for display).
  5. Check early stopping condition (if enabled).

After training:
  - Save the model (weights + architecture) to a .npy file.
  - Display learning curves (loss and accuracy).
  - Print classification report (precision, recall, F1).

Bonus utilities
---------------
  - plot_histories() : overlay multiple training runs on the same graph.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.layers import DenseLayer
from src.optimizers import SGD


class Network:
    """
    Container that links DenseLayer objects together.

    Responsibilities:
    - Initialize the weights of all layers in sequence.
    - Run the forward pass by cascading through each layer.
    - Run the backward pass in reverse order, passing the optimizer.
    """

    def __init__(self, layers: list[DenseLayer]) -> None:
        """
        Args:
            layers: Ordered list of DenseLayer objects (input -> output).
        """
        if not layers:
            raise ValueError("The network must contain at least one layer.")
        self.layers = layers
        self.initialized = False

    def initialize(self, input_dim: int, seed: int = 42) -> None:
        """
        Initialize the weights of each layer in sequence.

        The input dimension of each layer equals the number of units (output size)
        of the previous layer. For the first layer it is the number of features
        in the dataset (input_dim).

        Args:
            input_dim: Number of input features (e.g. 30 for breast cancer).
            seed: Random seed for reproducibility.
        """
        rng = np.random.default_rng(seed)
        current_dim = input_dim
        for layer in self.layers:
            layer.initialize(current_dim, rng)
            current_dim = layer.units
        self.initialized = True

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass: propagate X through all layers sequentially.

        Args:
            X: Input data of shape (n_samples, n_features).

        Returns:
            A: Output of the last layer, shape (n_samples, n_classes).
               With softmax: each row is a probability distribution.
        """
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def backward(
        self,
        delta: np.ndarray,
        learning_rate: float = 0.01,
        optimizer=None,
    ) -> None:
        """
        Backward pass: propagate the gradient through all layers in reverse
        order and update each layer's parameters.

        Args:
            delta:         Initial gradient = (A_output - Y_onehot).
            learning_rate: Used when no optimizer is provided (plain SGD).
            optimizer:     Optional optimizer (Adam, RMSprop, etc.).
        """
        for layer in reversed(self.layers):
            delta = layer.backward(delta, learning_rate, optimizer)


# ---------------------------------------------------------------------------
# Loss function and metrics
# ---------------------------------------------------------------------------

def binary_crossentropy(A: np.ndarray, Y_onehot: np.ndarray) -> float:
    """
    Compute the mean binary cross-entropy loss over a batch.

    Formula (equivalent to categorical cross-entropy for 2 classes):
        E = -(1/N) * sum( y * log(p) + (1 - y) * log(1 - p) )

    With our 2-class softmax output [P(B), P(M)]:
        p = P(M | x)  = probability of the malignant class
        y = 1 if the sample is malignant, 0 otherwise
        1 - p = P(B | x) = probability of the benign class

    Clipping to [1e-15, 1-1e-15] prevents log(0) = -inf.

    Args:
        A:        Softmax outputs, shape (n_samples, n_classes).
        Y_onehot: One-hot labels,  shape (n_samples, n_classes).

    Returns:
        Scalar: mean loss over the batch.
    """
    A_clipped = np.clip(A, 1e-15, 1.0 - 1e-15)
    return float(-np.sum(Y_onehot * np.log(A_clipped)) / A.shape[0])


def compute_accuracy(A: np.ndarray, y_int: np.ndarray) -> float:
    """
    Compute the fraction of correctly classified samples.

    Args:
        A:     Softmax outputs, shape (n_samples, n_classes).
        y_int: Integer class labels (ground truth), shape (n_samples,).

    Returns:
        Fraction of correctly classified samples (between 0 and 1).
    """
    predictions = np.argmax(A, axis=1)
    return float(np.mean(predictions == y_int))


def encode_labels(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Convert string labels to integer indices and one-hot encoding.

    Example for breast cancer: y = ['B', 'M', 'B', ...]
        classes  = ['B', 'M']   (alphabetical order)
        y_int    = [0, 1, 0, ...]
        Y_onehot = [[1,0], [0,1], [1,0], ...]

    Args:
        y: String labels, shape (n_samples,).

    Returns:
        y_int:    Integer class indices, shape (n_samples,).
        Y_onehot: One-hot matrix, shape (n_samples, n_classes).
        classes:  Sorted list of class names (index = class id).
    """
    classes: list[str] = sorted(np.unique(y).tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_int = np.array([class_to_idx[c] for c in y], dtype=int)
    Y_onehot = np.zeros((len(y), len(classes)), dtype=float)
    Y_onehot[np.arange(len(y)), y_int] = 1.0
    return y_int, Y_onehot, classes


# ---------------------------------------------------------------------------
# High-level API: Model
# ---------------------------------------------------------------------------

class Model:
    """
    High-level API to create and train a neural network.

    Example usage (Python script):
    --------------------------------
        from src.layers import DenseLayer
        from src.model import Model
        from src.optimizers import Adam
        from src.callbacks import EarlyStopping

        model = Model()
        network = model.createNetwork([
            DenseLayer(24, activation='sigmoid', weights_initializer='heUniform'),
            DenseLayer(24, activation='sigmoid', weights_initializer='heUniform'),
            DenseLayer(2,  activation='softmax', weights_initializer='heUniform'),
        ])
        model.fit(network, X_train, y_train, X_val, y_val,
                  optimizer=Adam(learning_rate=0.001),
                  epochs=100,
                  early_stopping=EarlyStopping(patience=15))

    Or via the CLI (see src/train.py):
        python3 -m src.train --layer 24 24 --optimizer adam --early_stopping
    """

    def createNetwork(self, layer_list: list[DenseLayer]) -> Network:
        """
        Wrap a list of DenseLayer objects into a Network.

        Args:
            layer_list: Ordered layers from input to output.
                        The last one must use 'softmax'.

        Returns:
            Network ready to be trained with fit().
        """
        return Network(layer_list)

    def fit(
        self,
        network: Network,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        loss: str = "categoricalCrossentropy",
        learning_rate: float = 0.01,
        batch_size: int = 32,
        epochs: int = 100,
        seed: int = 42,
        save_path: str = "saved_model.npy",
        optimizer=None,
        early_stopping=None,
    ) -> dict:
        """
        Train the network using mini-batch gradient descent and backpropagation.

        Algorithm
        ---------
        For each epoch:
          1. Randomly shuffle X_train (avoids bias from sample ordering).
          2. Split into mini-batches of size `batch_size`.
          3. For each mini-batch:
             - forward()  : compute predictions A
             - delta = A - Y_onehot  (combined softmax + cross-entropy gradient)
             - backward() : propagate gradients, update W and b via optimizer
          4. Compute metrics on the full train and val sets.
          5. Check early stopping condition.

        Args:
            network:         Network created by createNetwork().
            X_train:         Normalized feature matrix (n_train, n_features).
            y_train:         String labels for training (n_train,).
            X_val:           Normalized feature matrix (n_val, n_features).
            y_val:           String labels for validation (n_val,).
            loss:            Loss function name. Only 'categoricalCrossentropy' supported.
            learning_rate:   Used when optimizer=None (creates a plain SGD optimizer).
            batch_size:      Number of samples per mini-batch.
            epochs:          Maximum number of full passes over the training set.
            seed:            Random seed for weight initialization and shuffling.
            save_path:       Path to the .npy file for saving the model.
            optimizer:       Optimizer instance (Adam, RMSprop, etc.).
                             If None, plain SGD with `learning_rate` is used.
            early_stopping:  EarlyStopping instance. If None, training runs for
                             all `epochs` without early termination.

        Returns:
            History dictionary with keys:
                'train_loss', 'val_loss', 'train_accuracy', 'val_accuracy'
            Each key maps to a list of floats (one value per completed epoch).
        """
        if loss != "categoricalCrossentropy":
            raise ValueError(f"Loss '{loss}' not supported. Use 'categoricalCrossentropy'.")
        if network.layers[-1].activation != "softmax":
            raise ValueError("The output layer must use the 'softmax' activation.")

        # Default to plain SGD if no optimizer is provided
        if optimizer is None:
            optimizer = SGD(learning_rate)

        # Reset early stopping state for this training run
        if early_stopping is not None:
            early_stopping.reset()

        # ---- Label encoding ----------------------------------------------
        y_train_int, Y_train_onehot, classes = encode_labels(y_train)
        y_val_int,   Y_val_onehot,   _       = encode_labels(y_val)

        # ---- Weight initialization ---------------------------------------
        network.initialize(X_train.shape[1], seed)

        rng = np.random.default_rng(seed)
        n_train = X_train.shape[0]
        n_digits = len(str(epochs))

        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
        }

        # ---- Training loop -----------------------------------------------
        for epoch in range(1, epochs + 1):

            # 1. Shuffle the training data
            perm = rng.permutation(n_train)
            X_shuf = X_train[perm]
            Y_shuf = Y_train_onehot[perm]

            # 2. Mini-batch updates
            for start in range(0, n_train, batch_size):
                X_batch = X_shuf[start : start + batch_size]
                Y_batch = Y_shuf[start : start + batch_size]

                # Forward pass
                A_batch = network.forward(X_batch)

                # Combined softmax + cross-entropy gradient: delta = A - Y
                delta = A_batch - Y_batch

                # Backward pass with chosen optimizer
                network.backward(delta, learning_rate, optimizer)

            # 3. Epoch metrics
            A_train = network.forward(X_train)
            A_val   = network.forward(X_val)

            train_loss = binary_crossentropy(A_train, Y_train_onehot)
            val_loss   = binary_crossentropy(A_val,   Y_val_onehot)
            train_acc  = compute_accuracy(A_train, y_train_int)
            val_acc    = compute_accuracy(A_val,   y_val_int)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_accuracy"].append(train_acc)
            history["val_accuracy"].append(val_acc)

            # 4. Per-epoch display
            print(
                f"epoch {epoch:0{n_digits}}/{epochs} - "
                f"loss: {train_loss:.4f} - "
                f"val_loss: {val_loss:.4f} - "
                f"acc: {train_acc:.4f} - "
                f"val_acc: {val_acc:.4f}"
            )

            # 5. Early stopping check
            if early_stopping is not None:
                monitor_value = (
                    val_loss if early_stopping.monitor == "val_loss" else val_acc
                )
                if early_stopping.check(monitor_value, network, epoch):
                    break

        # ---- Save the model ----------------------------------------------
        if save_path:
            self._save_model(network, classes, save_path)

        # ---- Learning curves ---------------------------------------------
        self._plot_curves(history, len(history["train_loss"]))

        return history

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _save_model(
        self, network: Network, classes: list[str], save_path: str
    ) -> None:
        """
        Save the complete model (architecture + weights) to a .npy file.

        The file stores a Python dictionary containing:
            - 'classes' : list of class names ['B', 'M']
            - 'layers'  : configuration of each layer (units, activation, ...)
            - 'weights' : list of {'W': ..., 'b': ...} for each layer

        To reload:
            data = np.load('saved_model.npy', allow_pickle=True).item()

        Args:
            network:   Trained network.
            classes:   List of class names.
            save_path: Destination file path (e.g. 'saved_model.npy').
        """
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "classes": classes,
            "layers": [layer.get_config() for layer in network.layers],
            "weights": [
                {"W": layer.W, "b": layer.b} for layer in network.layers
            ],
        }
        np.save(save_path, model_data)
        print(f"> saving model '{save_path}' to disk...")

    def _plot_curves(self, history: dict, n_epochs: int) -> None:
        """
        Display and save the two learning curve graphs:
          1. Loss (train vs validation) over epochs.
          2. Accuracy (train vs validation) over epochs.

        Args:
            history:  Dictionary filled by fit() with per-epoch metrics.
            n_epochs: Actual number of epochs completed (may be < max if early stopping).
        """
        epoch_range = range(1, n_epochs + 1)

        fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Learning Curves", fontsize=14)

        ax_loss.plot(epoch_range, history["train_loss"],
                     label="train",      color="#2196F3", linewidth=1.5)
        ax_loss.plot(epoch_range, history["val_loss"],
                     label="validation", color="#FF9800", linewidth=1.5, linestyle="--")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Binary Cross-Entropy Loss")
        ax_loss.set_title("Loss")
        ax_loss.legend()
        ax_loss.grid(alpha=0.3)

        ax_acc.plot(epoch_range, history["train_accuracy"],
                    label="train",      color="#2196F3", linewidth=1.5)
        ax_acc.plot(epoch_range, history["val_accuracy"],
                    label="validation", color="#FF9800", linewidth=1.5, linestyle="--")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_title("Accuracy")
        ax_acc.set_ylim(0, 1.05)
        ax_acc.legend()
        ax_acc.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig("learning_curves.png", dpi=150, bbox_inches="tight")
        print("Learning curves saved to 'learning_curves.png'")
        plt.show()


# ---------------------------------------------------------------------------
# Bonus utility: compare multiple training histories on the same graph
# ---------------------------------------------------------------------------

def plot_histories(
    histories: list[tuple[str, dict]],
    save_path: str = "comparison.png",
) -> None:
    """
    Overlay multiple training histories on the same graphs for easy comparison.

    Useful for comparing:
    - Different architectures (e.g. [24,24] vs [48,48] vs [24,24,24])
    - Different optimizers (SGD vs Adam vs RMSprop)
    - Different hyperparameters (learning rates, batch sizes)
    - Effect of early stopping

    Args:
        histories:  List of (label, history_dict) tuples, where history_dict
                    is the dictionary returned by Model.fit().
                    Example: [("Adam lr=0.001", h1), ("SGD lr=0.01", h2)]
        save_path:  Path to save the comparison plot (default: 'comparison.png').

    Example:
        h1 = model.fit(net1, ...)
        h2 = model.fit(net2, ...)
        plot_histories([("Adam", h1), ("SGD", h2)])
    """
    if not histories:
        raise ValueError("histories list is empty.")

    # Use a colormap to auto-assign distinct colors
    colors = plt.cm.tab10.colors

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Model Comparison — Learning Curves", fontsize=14)

    for idx, (label, history) in enumerate(histories):
        color = colors[idx % len(colors)]
        n = len(history["train_loss"])
        epoch_range = range(1, n + 1)

        ax_loss.plot(epoch_range, history["train_loss"],
                     label=f"{label} (train)", color=color, linewidth=1.5)
        ax_loss.plot(epoch_range, history["val_loss"],
                     label=f"{label} (val)",   color=color, linewidth=1.5, linestyle="--")

        ax_acc.plot(epoch_range, history["train_accuracy"],
                    label=f"{label} (train)", color=color, linewidth=1.5)
        ax_acc.plot(epoch_range, history["val_accuracy"],
                    label=f"{label} (val)",   color=color, linewidth=1.5, linestyle="--")

    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Binary Cross-Entropy Loss")
    ax_loss.set_title("Loss Comparison")
    ax_loss.legend(fontsize=8)
    ax_loss.grid(alpha=0.3)

    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Accuracy Comparison")
    ax_acc.set_ylim(0, 1.05)
    ax_acc.legend(fontsize=8)
    ax_acc.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Comparison plot saved to '{save_path}'")
    plt.show()
