"""
Optimization algorithms for gradient descent weight updates.

Both optimizers share the same interface:
    optimizer.update(layer, dW, db)

They maintain their own internal state (velocity) keyed by the layer's
Python id, so the same optimizer object handles all layers of a network.

Available optimizers
---------------------
- SGD          : vanilla stochastic gradient descent (mandatory baseline)
- SGDMomentum  : SGD with momentum (bonus — accumulates velocity to accelerate convergence)

How momentum works
------------------
Plain SGD updates weights directly from the current gradient:
    W = W - lr * dW

SGD with Momentum adds a velocity term that accumulates past gradients:
    v = momentum * v - lr * dW
    W = W + v

The velocity v acts like a ball rolling down a hill: it accelerates in
directions where gradients consistently point the same way, and dampens
oscillations when gradients keep changing direction.
The momentum coefficient (typically 0.9) controls how much of the previous
velocity is retained at each step.
"""

import numpy as np


class SGD:
    """
    Vanilla Stochastic Gradient Descent.

    Update rule:
        W <- W - lr * dW
        b <- b - lr * db

    Args:
        learning_rate: Step size for each parameter update.
    """

    def __init__(self, learning_rate: float = 0.01) -> None:
        self.learning_rate = learning_rate

    def update(self, layer, dW: np.ndarray, db: np.ndarray) -> None:
        layer.W -= self.learning_rate * dW
        layer.b -= self.learning_rate * db


class SGDMomentum:
    """
    SGD with Momentum.

    Momentum accumulates a velocity vector in the direction of persistent
    gradients, which helps accelerate convergence and dampen oscillations.

    Update rule:
        v_W = momentum * v_W - lr * dW
        W   = W + v_W

    The velocity v_W acts as an exponentially decaying sum of past gradients,
    giving more weight to recent gradients.

    Args:
        learning_rate: Step size.
        momentum:      Fraction of the previous velocity to retain (typically 0.9).
    """

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9) -> None:
        self.learning_rate = learning_rate
        self.momentum = momentum
        self._state: dict = {}

    def update(self, layer, dW: np.ndarray, db: np.ndarray) -> None:
        lid = id(layer)
        if lid not in self._state:
            # Initialize velocity to zero on first call for this layer
            self._state[lid] = {
                "vW": np.zeros_like(layer.W),
                "vb": np.zeros_like(layer.b),
            }
        s = self._state[lid]
        # Accumulate velocity
        s["vW"] = self.momentum * s["vW"] - self.learning_rate * dW
        s["vb"] = self.momentum * s["vb"] - self.learning_rate * db
        # Apply update
        layer.W += s["vW"]
        layer.b += s["vb"]


# Registry: maps optimizer name (string) -> class
OPTIMIZERS: dict[str, type] = {
    "sgd":      SGD,
    "momentum": SGDMomentum,
}
