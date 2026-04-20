"""
Activation functions and their derivatives for MLP layers.

In a neural network, each neuron computes:
    Z = W·X + b   (linear combination of inputs)
    A = f(Z)      (non-linear activation function applied element-wise)

Without non-linear activations, stacking multiple layers would be pointless
because the composition of linear functions is still linear. Activations allow
the network to learn complex, non-linear decision boundaries.

Derivatives are used during backpropagation to compute gradients
(how fast the loss changes with respect to the weights).
"""

import numpy as np


# ---------------------------------------------------------------------------
# Sigmoid
# ---------------------------------------------------------------------------

def sigmoid(Z: np.ndarray) -> np.ndarray:
    """
    Apply the sigmoid function element-wise.

    Formula: sigma(z) = 1 / (1 + e^(-z))

    Sigmoid squashes any value into (0, 1), making it interpretable as a
    probability. It is used in hidden layers to introduce non-linearity.

    The clip to [-500, 500] prevents numerical overflow in np.exp.

    Args:
        Z: Numpy array of any shape (pre-activation values).

    Returns:
        Array of the same shape with values in (0, 1).
    """
    return 1.0 / (1.0 + np.exp(-np.clip(Z, -500, 500)))


def sigmoid_derivative(A: np.ndarray) -> np.ndarray:
    """
    Derivative of sigmoid expressed in terms of its *output* A.

    Formula: sigma'(z) = sigma(z) * (1 - sigma(z)) = A * (1 - A)

    Using A avoids recomputing sigma(z) since it is already cached.
    Used during backpropagation to propagate the gradient.

    Args:
        A: Output of sigmoid (values in (0, 1)).

    Returns:
        Array of the same shape with local derivatives.
    """
    return A * (1.0 - A)


# ---------------------------------------------------------------------------
# Softmax
# ---------------------------------------------------------------------------

def softmax(Z: np.ndarray) -> np.ndarray:
    """
    Apply softmax row-wise (one sample per row).

    Formula: softmax(z_i) = e^(z_i) / sum_j(e^(z_j))

    Softmax converts a vector of raw scores into a probability distribution:
    all outputs are positive and sum to 1.
    It is used ONLY on the output layer of this network.

    Numerical trick: subtract the row maximum before exponentiation to avoid
    overflow (e^very_large_number -> inf). The result is identical because
    the constant cancels in the ratio.

    Args:
        Z: Array of shape (n_samples, n_classes).

    Returns:
        Array of the same shape where each row is a probability distribution.
    """
    # Numerical stability: shift by the row maximum
    Z_stable = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z_stable)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# ReLU (Rectified Linear Unit)
# ---------------------------------------------------------------------------

def relu(Z: np.ndarray) -> np.ndarray:
    """
    Apply the ReLU function: max(0, z).

    ReLU is simple and highly effective in practice. It does not saturate for
    large positive values (unlike sigmoid), which speeds up training on deep
    networks.

    Args:
        Z: Numpy array of pre-activation values.

    Returns:
        Array of the same shape with negative values set to 0.
    """
    return np.maximum(0.0, Z)


def relu_derivative(Z: np.ndarray) -> np.ndarray:
    """
    Derivative of ReLU expressed in terms of the *pre-activation* Z.

    Formula: ReLU'(z) = 1 if z > 0, else 0

    Z (not A) is used because the derivative depends on the sign of the input,
    not on the output value.

    Args:
        Z: Pre-activation values stored in cache during the forward pass.

    Returns:
        Array of the same shape with 0.0 or 1.0.
    """
    return (Z > 0).astype(float)


# ---------------------------------------------------------------------------
# Tanh
# ---------------------------------------------------------------------------

def tanh_fn(Z: np.ndarray) -> np.ndarray:
    """
    Apply the hyperbolic tangent: tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z)).

    Tanh is zero-centered (outputs in (-1, 1)), which can make it better than
    sigmoid for hidden layers because the gradients are more centered around 0.

    Args:
        Z: Numpy array of pre-activation values.

    Returns:
        Array of the same shape with values in (-1, 1).
    """
    return np.tanh(Z)


def tanh_derivative(A: np.ndarray) -> np.ndarray:
    """
    Derivative of tanh expressed in terms of its *output* A.

    Formula: tanh'(z) = 1 - tanh^2(z) = 1 - A^2

    Args:
        A: Output of tanh (values in (-1, 1)).

    Returns:
        Array of the same shape with local derivatives.
    """
    return 1.0 - A ** 2


# ---------------------------------------------------------------------------
# Registries: allow accessing functions by name (string)
# ---------------------------------------------------------------------------

#: Maps activation name -> forward function.
ACTIVATIONS: dict[str, callable] = {
    "sigmoid": sigmoid,
    "softmax": softmax,
    "relu": relu,
    "tanh": tanh_fn,
}

#: Maps activation name -> derivative function.
#: Note: sigmoid and tanh derivatives take A (post-activation);
#:       relu takes Z (pre-activation). The layer handles this distinction.
DERIVATIVES: dict[str, callable] = {
    "sigmoid": sigmoid_derivative,
    "relu": relu_derivative,
    "tanh": tanh_derivative,
    # softmax has no entry here: its gradient is computed directly
    # combined with the loss (see DenseLayer.backward).
}
