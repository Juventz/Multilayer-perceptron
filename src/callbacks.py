"""
Training callbacks for controlling the training loop.

Callbacks are objects passed to Model.fit() that can inspect metrics at
each epoch and trigger actions (e.g. stop training, save the best model).

Available callbacks
-------------------
- EarlyStopping : stop training when a monitored metric stops improving.
"""

import numpy as np


class EarlyStopping:
    """
    Stop training when a monitored metric stops improving.

    The training loop calls `check()` at the end of every epoch. When the
    metric has not improved by more than `min_delta` for `patience` consecutive
    epochs, `check()` returns True and training stops.

    If `restore_best_weights` is True, the network's parameters are reset to
    the best weights seen during training before stopping.

    Why early stopping?
    -------------------
    Training too many epochs leads to overfitting: the model memorizes the
    training data but loses the ability to generalize to unseen samples.
    The validation loss typically starts rising while the training loss keeps
    falling. Early stopping automatically prevents this by halting as soon as
    generalization stops improving.

    Args:
        patience:             Number of epochs to wait without improvement
                              before stopping (default: 10).
        min_delta:            Minimum change to qualify as an improvement
                              (default: 1e-4).
        restore_best_weights: If True, restore weights from the epoch with
                              the best monitored metric after stopping.
        monitor:              Metric to monitor. One of: 'val_loss',
                              'val_accuracy'. (default: 'val_loss').

    Example:
        es = EarlyStopping(patience=15, restore_best_weights=True)
        model.fit(network, ..., early_stopping=es)
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        restore_best_weights: bool = True,
        monitor: str = "val_loss",
    ) -> None:
        if monitor not in ("val_loss", "val_accuracy"):
            raise ValueError(
                f"Unknown monitor '{monitor}'. Choose 'val_loss' or 'val_accuracy'."
            )
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.monitor = monitor

        # Internal state — reset at the start of each fit() call
        self._best_metric: float = float("inf") if monitor == "val_loss" else -float("inf")
        self._counter: int = 0
        self._best_weights: list | None = None
        self._best_epoch: int = 0

    def reset(self) -> None:
        """Reset internal state. Called automatically at the start of fit()."""
        self._best_metric = float("inf") if self.monitor == "val_loss" else -float("inf")
        self._counter = 0
        self._best_weights = None
        self._best_epoch = 0

    def _is_improvement(self, current: float) -> bool:
        """Return True if `current` is better than the best seen so far."""
        if self.monitor == "val_loss":
            # Lower is better
            return current < self._best_metric - self.min_delta
        else:
            # Higher is better (accuracy)
            return current > self._best_metric + self.min_delta

    def _save_weights(self, network) -> None:
        """Deep-copy all layer weights into the best-weights buffer."""
        self._best_weights = [
            (layer.W.copy(), layer.b.copy()) for layer in network.layers
        ]

    def _restore_weights(self, network) -> None:
        """Reload the best-weights buffer into the network layers."""
        if self._best_weights is not None:
            for layer, (W, b) in zip(network.layers, self._best_weights):
                layer.W = W.copy()
                layer.b = b.copy()

    def check(self, metric_value: float, network, epoch: int) -> bool:
        """
        Evaluate the metric and decide whether to stop training.

        Call this at the end of every epoch.

        Args:
            metric_value: Current epoch's monitored metric value.
            network:      The Network object (used to save/restore weights).
            epoch:        Current epoch number (for logging).

        Returns:
            True if training should stop, False otherwise.
        """
        if self._is_improvement(metric_value):
            self._best_metric = metric_value
            self._counter = 0
            self._best_epoch = epoch
            if self.restore_best_weights:
                self._save_weights(network)
        else:
            self._counter += 1

        if self._counter >= self.patience:
            if self.restore_best_weights:
                self._restore_weights(network)
                print(
                    f"\nEarly stopping triggered at epoch {epoch}. "
                    f"Restoring weights from epoch {self._best_epoch} "
                    f"({self.monitor}={self._best_metric:.4f})."
                )
            else:
                print(
                    f"\nEarly stopping triggered at epoch {epoch}. "
                    f"Best {self.monitor}={self._best_metric:.4f} "
                    f"at epoch {self._best_epoch}."
                )
            return True

        return False
