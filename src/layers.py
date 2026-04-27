"""
Dense (fully-connected) layer for the multilayer perceptron.

A dense layer connects EVERY input neuron to EVERY output neuron.
For a batch of n samples, the computation of one layer is:

    Z = X · W + b    # linear combination  (n x fan_out)
    A = f(Z)         # non-linear activation (n x fan_out)

    where:
        X : inputs,  shape (n, fan_in)
        W : weights, shape (fan_in, fan_out)
        b : biases,  shape (1, fan_out)  -> broadcast over n samples

During backpropagation the layer receives the gradient of the loss with
respect to its output, computes the weight gradients (dW, db), delegates
the parameter update to an optimizer, then returns the gradient with
respect to its input for the previous layer.
"""

import numpy as np

from src.activations import ACTIVATIONS, DERIVATIVES

# Supported weight initializers
VALID_INITIALIZERS = {"heUniform", "heNormal", "glorotUniform", "glorotNormal", "random"}


class DenseLayer:
    """
    Fully-connected layer with a configurable activation function.

    Main attributes
    ---------------
    units : int
        Number of neurons in this layer (output dimensionality).
    activation : str
        Activation function name ('sigmoid', 'relu', 'tanh', 'softmax').
    weights_initializer : str
        Weight initialization strategy.
    W : np.ndarray | None
        Weight matrix, shape (fan_in, units). Set by initialize().
    b : np.ndarray | None
        Bias vector, shape (1, units). Initialized to zeros.

    Cache attributes (used during backward pass)
    ---------------------------------------------
    _input : np.ndarray
        Input received during the last forward pass (needed to compute dW).
    _Z : np.ndarray
        Pre-activation from the last forward pass (needed for ReLU derivative).
    _A : np.ndarray
        Output from the last forward pass (needed for sigmoid/tanh derivatives).
    """

    def __init__(
        self,
        units: int,
        activation: str = "sigmoid",
        weights_initializer: str = "glorotUniform",
    ) -> None:
        """
        Create a dense layer.

        Args:
            units: Number of output neurons.
            activation: Activation function applied after the linear combination.
            weights_initializer: Weight initialization method (see initialize()).
        """
        if activation not in ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{activation}'. Choices: {list(ACTIVATIONS)}"
            )
        if weights_initializer not in VALID_INITIALIZERS:
            raise ValueError(
                f"Unknown initializer '{weights_initializer}'. "
                f"Choices: {sorted(VALID_INITIALIZERS)}"
            )
        self.units = units
        self.activation = activation
        self.weights_initializer = weights_initializer

        # Weights and biases: None until initialize() is called
        self.W: np.ndarray | None = None
        self.b: np.ndarray | None = None

        # Backpropagation cache
        self._input: np.ndarray | None = None   # X from forward pass
        self._Z: np.ndarray | None = None       # pre-activation
        self._A: np.ndarray | None = None       # post-activation (output)

    def initialize(self, input_dim: int, rng: np.random.Generator) -> None:
        """
        Allocate and initialize the weight matrix W and bias vector b.

        The choice of initializer significantly impacts convergence:

        - heUniform / heNormal:
            Designed for ReLU. Variance calibrated on fan_in only.
            Uniform limit:  +/- sqrt(6 / fan_in)
            Normal std dev: sqrt(2 / fan_in)

        - glorotUniform / glorotNormal (Xavier):
            Designed for sigmoid / tanh. Variance calibrated on fan_in + fan_out.
            Uniform limit:  +/- sqrt(6 / (fan_in + fan_out))
            Normal std dev: sqrt(2 / (fan_in + fan_out))

        - random:
            Standard normal distribution scaled by 0.01. Simple but may cause
            vanishing gradients if weights are too small.

        The bias b is always initialized to 0 (standard practice).

        Args:
            input_dim: Number of neurons in the previous layer (fan_in).
            rng: Numpy random generator for reproducibility.
        """
        fan_in, fan_out = input_dim, self.units

        if self.weights_initializer == "heUniform":
            limit = np.sqrt(6.0 / fan_in)
            self.W = rng.uniform(-limit, limit, (fan_in, fan_out))

        elif self.weights_initializer == "heNormal":
            self.W = rng.normal(0.0, np.sqrt(2.0 / fan_in), (fan_in, fan_out))

        elif self.weights_initializer == "glorotUniform":
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            self.W = rng.uniform(-limit, limit, (fan_in, fan_out))

        elif self.weights_initializer == "glorotNormal":
            self.W = rng.normal(0.0, np.sqrt(2.0 / (fan_in + fan_out)), (fan_in, fan_out))

        elif self.weights_initializer == "random":
            self.W = rng.standard_normal((fan_in, fan_out)) * 0.01

        # Bias shape is (1, fan_out) to broadcast correctly over the batch dimension
        self.b = np.zeros((1, fan_out))

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass: compute the layer output for a batch of inputs X.

        Steps:
            1. Z = X · W + b    (linear combination)
            2. A = f(Z)         (activation function)

        X, Z, and A are cached because they are needed during the backward pass.

        Args:
            X: Inputs of shape (n_samples, fan_in).

        Returns:
            A: Outputs of shape (n_samples, units).
        """
        self._input = X                                  # cache for dW = X.T @ dZ
        self._Z = X @ self.W + self.b                    # linear combination
        self._A = ACTIVATIONS[self.activation](self._Z)  # activation
        return self._A

    def backward(
        self,
        delta: np.ndarray,
        learning_rate: float = 0.01,
        optimizer=None,
    ) -> np.ndarray:
        """
        Backward pass: compute gradients and update W and b.

        How backpropagation works
        --------------------------
        We receive delta = dL/dA (gradient of the loss w.r.t. this layer's output).
        We compute dL/dZ = delta * f'(Z) via the chain rule.
        Then:
            dL/dW = (1/n) * X.T @ dL/dZ    (gradient of the weights)
            dL/db = (1/n) * sum(dL/dZ)     (gradient of the biases)
            dL/dX = dL/dZ @ W.T            (gradient to pass to the previous layer)

        Special case: softmax output layer + cross-entropy loss
        --------------------------------------------------------
        The combined derivative of softmax + cross-entropy is simply (A - Y).
        We therefore pass delta = (A - Y) directly and skip the f'(Z) computation
        (which would require computing the full softmax Jacobian matrix).

        Weight update
        -------------
        If an optimizer is provided, it handles the update (Adam, RMSprop, etc.).
        Otherwise, plain SGD is applied: W -= lr * dW.

        Args:
            delta:
                - Softmax output layer: delta = (A - Y_onehot), unnormalized.
                - Hidden layer: delta = dL/dA flowing back from the next layer.
            learning_rate: Used only when no optimizer is provided (plain SGD).
            optimizer:     Optional optimizer object (see src/optimizers.py).

        Returns:
            dA_prev: Gradient dL/dX to pass as `delta` to the previous layer.
        """
        n = self._input.shape[0]  # batch size for normalization

        # ---- Compute dZ = dL/dZ ------------------------------------------
        if self.activation == "softmax":
            # Combined softmax + cross-entropy gradient: dZ = (A - Y)
            dZ = delta
        elif self.activation == "relu":
            # ReLU: derivative depends on the sign of Z (pre-activation)
            dZ = delta * DERIVATIVES["relu"](self._Z)
        else:
            # sigmoid and tanh: derivative expressed using output A
            dZ = delta * DERIVATIVES[self.activation](self._A)

        # ---- Parameter gradients -----------------------------------------
        dW = self._input.T @ dZ / n
        db = np.sum(dZ, axis=0, keepdims=True) / n
        dA_prev = dZ @ self.W.T

        # ---- Parameter update --------------------------------------------
        if optimizer is not None:
            # Delegate update to the optimizer (Adam, RMSprop, etc.)
            optimizer.update(self, dW, db)
        else:
            # Plain SGD fallback
            self.W -= learning_rate * dW
            self.b -= learning_rate * db

        return dA_prev

    def get_config(self) -> dict:
        """
        Return the layer configuration as a dictionary.
        Used to save the network architecture alongside the weights.
        """
        return {
            "units": self.units,
            "activation": self.activation,
            "weights_initializer": self.weights_initializer,
        }
