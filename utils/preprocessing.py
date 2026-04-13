import numpy as np

from utils.csv_utils import is_numeric_column, to_float


def get_numeric_features(
    fieldnames: list[str],
    rows: list[dict[str, str]],
    excluded_columns: set[str] | None = None,
) -> list[str]:
    """
    Select numeric feature columns from a dataset.

    A column is considered numeric when all non-empty values can be converted
    to float.

    Args:
        fieldnames: Ordered column names.
        rows: Dataset rows as dictionaries.
        excluded_columns: Optional set of columns to skip (e.g. target, id).

    Returns:
        Ordered list of numeric feature names.
    """
    excluded = excluded_columns or set()
    numeric_features = []
    for field in fieldnames:
        if field in excluded:
            continue
        if is_numeric_column(rows, field):
            numeric_features.append(field)
    return numeric_features


def build_feature_matrix(
    rows: list[dict[str, str]],
    features: list[str],
    means: np.ndarray | None = None,
    stds: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a numeric feature matrix and apply z-score standardization.

    Pipeline:
    1) Read selected features from each row.
    2) Replace missing/invalid values using feature means.
    3) Standardize using z-score: (x - mean) / std.

    Reproducibility behavior:
    - Training mode: compute means/stds from current rows.
    - Inference mode: pass precomputed means/stds to reuse the exact same scaling.

    Args:
        rows: Dataset rows as dictionaries.
        features: Ordered list of feature names used as model inputs.
        means: Optional feature means to reuse.
        stds: Optional feature standard deviations to reuse.

    Returns:
        normalized: 2D array of shape (n_samples, n_features)
        means: 1D array of shape (n_features,)
        stds: 1D array of shape (n_features,)
    """
    num_samples = len(rows)
    num_features = len(features)

    matrix = np.zeros((num_samples, num_features), dtype=float)

    if means is None:
        means = np.zeros(num_features, dtype=float)
        for feature_index, feature_name in enumerate(features):
            values = []
            for row in rows:
                parsed_value = to_float(row.get(feature_name, ""))
                if parsed_value is not None:
                    values.append(parsed_value)
            if not values:
                raise ValueError(f"Feature '{feature_name}' has no numeric values.")
            means[feature_index] = float(np.mean(values))

    # Fill matrix while applying mean imputation for missing/invalid values.
    for row_index, row in enumerate(rows):
        for feature_index, feature_name in enumerate(features):
            parsed_value = to_float(row.get(feature_name, ""))
            matrix[row_index, feature_index] = (
                means[feature_index] if parsed_value is None else parsed_value
            )

    if stds is None:
        stds = np.std(matrix, axis=0)

    # Prevent division by zero for constant features.
    stds = stds.astype(float)
    stds[stds == 0.0] = 1.0

    normalized = (matrix - means) / stds
    return normalized, means, stds


def extract_target(rows: list[dict[str, str]], target_column: str) -> np.ndarray:
    """
    Extract the target column as a 1D numpy array.

    Args:
        rows: Dataset rows as dictionaries.
        target_column: Name of the label column.

    Returns:
        Array of target values with shape (n_samples,).
    """
    missing = [index for index, row in enumerate(rows) if target_column not in row]
    if missing:
        raise ValueError(f"Target column '{target_column}' missing for row index {missing[0]}.")
    return np.asarray([row[target_column] for row in rows])
