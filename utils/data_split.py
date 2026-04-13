import numpy as np


def train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.2,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split arrays into train and validation subsets."""
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of samples.")
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError("val_ratio must be in [0, 1).")

    rng = np.random.default_rng(random_seed)
    n_samples = X.shape[0]
    indices = rng.permutation(n_samples)

    val_size = int(n_samples * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    return X_train, y_train, X_val, y_val


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split arrays into train/validation/test subsets."""
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of samples.")
    if not (0.0 <= val_ratio < 1.0 and 0.0 <= test_ratio < 1.0):
        raise ValueError("val_ratio and test_ratio must be in [0, 1).")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.")

    rng = np.random.default_rng(random_seed)
    n_samples = X.shape[0]
    indices = rng.permutation(n_samples)

    test_size = int(n_samples * test_ratio)
    val_size = int(n_samples * val_ratio)

    test_indices = indices[:test_size]
    val_indices = indices[test_size : test_size + val_size]
    train_indices = indices[test_size + val_size :]

    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    return X_train, y_train, X_val, y_val, X_test, y_test
