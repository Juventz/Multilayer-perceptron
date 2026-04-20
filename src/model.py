"""
Network container and training loop (Model).

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
     c. Backward pass -> update all W and b
  4. Compute loss and accuracy on train + validation (for display).
  5. Record metrics in the history dictionary.

After training:
  - Save the model (weights + architecture) to a .npy file.
  - Display learning curves (loss and accuracy).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.layers import DenseLayer


class Network:
    """
    Container that links DenseLayer objects together.

    Responsibilities:
    - Initialize the weights of all layers in sequence.
    - Run the forward pass by cascading through each layer.
    - Run the backward pass in reverse order.
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
            current_dim = layer.units  # output of this layer = input of the next
        self.initialized = True

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass: propagate X through all layers sequentially.

        Each layer receives the output (A) of the previous layer.

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

    def backward(self, delta: np.ndarray, learning_rate: float) -> None:
        """
        Backward pass: propagate the gradient through all layers in reverse
        order and update each layer's parameters.

        Args:
            delta: Initial gradient = (A_output - Y_onehot) for softmax + cross-entropy.
            learning_rate: Step size for gradient descent.
        """
        for layer in reversed(self.layers):
            delta = layer.backward(delta, learning_rate)


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
    predictions = np.argmax(A, axis=1)  # predicted class = index of max probability
    return float(np.mean(predictions == y_int))


def encode_labels(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Convert string labels to integer indices and one-hot encoding.

    Example for breast cancer: y = ['B', 'M', 'B', ...]
        classes  = ['B', 'M']   (alphabetical order)
        y_int    = [0, 1, 0, ...]
        Y_onehot = [[1,0], [0,1], [1,0], ...]

    One-hot encoding is required for the initial gradient computation
    (delta = A - Y_onehot) and for the loss calculation.

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

        model = Model()
        network = model.createNetwork([
            DenseLayer(24, activation='sigmoid', weights_initializer='heUniform'),
            DenseLayer(24, activation='sigmoid', weights_initializer='heUniform'),
            DenseLayer(2,  activation='softmax', weights_initializer='heUniform'),
        ])
        model.fit(network, X_train, y_train, X_val, y_val,
                  loss='categoricalCrossentropy', learning_rate=0.0314,
                  batch_size=8, epochs=84)

    Or via the CLI (see src/train.py):
        python3 -m src.train --layer 24 24 --epochs 84 --learning_rate 0.0314
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
             - backward() : update all W and b
          4. Compute metrics (loss and accuracy) on the full train and val sets.
          5. Print: epoch 01/70 - loss: X.XXXX - val_loss: X.XXXX

        Args:
            network:       Network created by createNetwork().
            X_train:       Normalized feature matrix (n_train, n_features).
            y_train:       String labels for training (n_train,).
            X_val:         Normalized feature matrix (n_val, n_features).
            y_val:         String labels for validation (n_val,).
            loss:          Loss function name. Only 'categoricalCrossentropy' supported.
            learning_rate: Step size for gradient descent.
            batch_size:    Number of samples per mini-batch.
            epochs:        Number of full passes over the training set.
            seed:          Random seed for weight initialization and shuffling.
            save_path:     Path to the .npy file for saving the model.

        Returns:
            History dictionary with keys:
                'train_loss', 'val_loss', 'train_accuracy', 'val_accuracy'
            Each key maps to a list of floats (one value per epoch).
        """
        if loss != "categoricalCrossentropy":
            raise ValueError(f"Loss '{loss}' not supported. Use 'categoricalCrossentropy'.")
        if network.layers[-1].activation != "softmax":
            raise ValueError("The output layer must use the 'softmax' activation.")

        # ---- Label encoding ----------------------------------------------
        # Convert 'B'/'M' strings to [1,0]/[0,1] one-hot vectors
        y_train_int, Y_train_onehot, classes = encode_labels(y_train)
        y_val_int,   Y_val_onehot,   _       = encode_labels(y_val)

        # ---- Weight initialization ---------------------------------------
        # Weight matrix dimensions are inferred from the data shape
        network.initialize(X_train.shape[1], seed)

        # Random generator for shuffling data at each epoch
        rng = np.random.default_rng(seed)
        n_train = X_train.shape[0]
        n_digits = len(str(epochs))  # for zero-padding the epoch number

        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
        }

        # ---- Training loop -----------------------------------------------
        for epoch in range(1, epochs + 1):

            # 1. Shuffle the training data (new order at each epoch)
            perm = rng.permutation(n_train)
            X_shuf = X_train[perm]
            Y_shuf = Y_train_onehot[perm]

            # 2. Mini-batch updates
            for start in range(0, n_train, batch_size):
                X_batch = X_shuf[start : start + batch_size]
                Y_batch = Y_shuf[start : start + batch_size]

                # a. Forward pass: compute predicted probabilities
                A_batch = network.forward(X_batch)

                # b. Combined softmax + cross-entropy gradient: delta = A - Y
                #    (division by n is handled inside DenseLayer.backward)
                delta = A_batch - Y_batch

                # c. Backward pass: backpropagate and update weights
                network.backward(delta, learning_rate)

            # 3. Epoch metrics (computed on the full dataset, not just the last batch)
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

            # 4. Display every epoch (format required by the subject)
            print(
                f"epoch {epoch:0{n_digits}}/{epochs} - "
                f"loss: {train_loss:.4f} - "
                f"val_loss: {val_loss:.4f} - "
                f"acc: {train_acc:.4f} - "
                f"val_acc: {val_acc:.4f}"
            )

        # ---- Save the model ----------------------------------------------
        if save_path:
            self._save_model(network, classes, save_path)

        # ---- Learning curves ---------------------------------------------
        self._plot_curves(history, epochs)

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

    def _plot_curves(self, history: dict, epochs: int) -> None:
        """
        Display and save two learning curve graphs:
          1. Loss (train vs validation) over epochs.
          2. Accuracy (train vs validation) over epochs.

        These curves help diagnose:
          - Underfitting: train loss stays high -> model too simple or lr too low.
          - Overfitting:  val loss rises while train loss falls -> model too complex.
          - Good fit:     both curves decrease and stabilize close together.

        Args:
            history: Dictionary filled by fit() with per-epoch metrics.
            epochs:  Total number of epochs (x-axis range).
        """
        epoch_range = range(1, epochs + 1)

        fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Learning Curves", fontsize=14)

        # --- Graph 1: Loss ---
        ax_loss.plot(epoch_range, history["train_loss"],
                     label="train",      color="#2196F3", linewidth=1.5)
        ax_loss.plot(epoch_range, history["val_loss"],
                     label="validation", color="#FF9800", linewidth=1.5, linestyle="--")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Binary Cross-Entropy Loss")
        ax_loss.set_title("Loss")
        ax_loss.legend()
        ax_loss.grid(alpha=0.3)

        # --- Graph 2: Accuracy ---
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
