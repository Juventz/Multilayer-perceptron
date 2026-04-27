"""
Evaluation metrics for binary and multi-class classification.

Beyond accuracy, these metrics are especially important for medical classification
tasks like breast cancer detection where false negatives (missed cancers) have
very different consequences from false positives.

Metrics implemented
-------------------
- Confusion matrix : raw counts of TP, TN, FP, FN per class
- Precision        : of all predicted positives, how many are truly positive?
- Recall           : of all true positives, how many were correctly detected?
- F1 score         : harmonic mean of precision and recall
- Full report      : precision, recall, F1, support for every class

Terminology (for class M = malignant as the "positive" class)
--------------------------------------------------------------
    TP (True Positive)  : predicted M, actually M  -> correct detection
    TN (True Negative)  : predicted B, actually B  -> correct rejection
    FP (False Positive) : predicted M, actually B  -> false alarm
    FN (False Negative) : predicted B, actually M  -> missed cancer (most dangerous!)

    Precision = TP / (TP + FP)   -> how reliable are positive predictions?
    Recall    = TP / (TP + FN)   -> how many true positives are found?
    F1        = 2 * P * R / (P + R)  -> balance between precision and recall
"""

import numpy as np


def confusion_matrix(
    A: np.ndarray, y_int: np.ndarray, n_classes: int
) -> np.ndarray:
    """
    Build the confusion matrix.

    The matrix CM is indexed as CM[true_class, predicted_class].
    The diagonal contains correct predictions; off-diagonal elements are errors.

    Example for 2 classes (B=0, M=1):
        CM[0,0] = TN  (benign correctly identified)
        CM[0,1] = FP  (benign misclassified as malignant)
        CM[1,0] = FN  (malignant missed — most costly error)
        CM[1,1] = TP  (malignant correctly identified)

    Args:
        A:         Softmax output probabilities, shape (n_samples, n_classes).
        y_int:     True integer class labels, shape (n_samples,).
        n_classes: Number of classes.

    Returns:
        Integer matrix of shape (n_classes, n_classes).
    """
    predictions = np.argmax(A, axis=1)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for pred, true in zip(predictions, y_int):
        cm[true, pred] += 1
    return cm


def precision_score(cm: np.ndarray, class_idx: int) -> float:
    """
    Precision for a given class: TP / (TP + FP).

    Answers: "Of all samples predicted as this class, how many were correct?"
    High precision means few false positives.

    Args:
        cm:        Confusion matrix of shape (n_classes, n_classes).
        class_idx: Index of the class to compute precision for.

    Returns:
        Precision value in [0, 1]. Returns 0 if no predictions were made.
    """
    TP = cm[class_idx, class_idx]
    FP = cm[:, class_idx].sum() - TP
    denom = TP + FP
    return float(TP / denom) if denom > 0 else 0.0


def recall_score(cm: np.ndarray, class_idx: int) -> float:
    """
    Recall (sensitivity) for a given class: TP / (TP + FN).

    Answers: "Of all true samples of this class, how many were detected?"
    High recall means few false negatives.
    Critical for cancer detection: missing a malignant tumor is dangerous.

    Args:
        cm:        Confusion matrix of shape (n_classes, n_classes).
        class_idx: Index of the class to compute recall for.

    Returns:
        Recall value in [0, 1]. Returns 0 if no true samples exist.
    """
    TP = cm[class_idx, class_idx]
    FN = cm[class_idx, :].sum() - TP
    denom = TP + FN
    return float(TP / denom) if denom > 0 else 0.0


def f1_score(cm: np.ndarray, class_idx: int) -> float:
    """
    F1 score for a given class: 2 * P * R / (P + R).

    The F1 score is the harmonic mean of precision and recall.
    It is a single balanced metric that penalizes extreme imbalances
    between the two (e.g. high precision but very low recall).

    Args:
        cm:        Confusion matrix of shape (n_classes, n_classes).
        class_idx: Index of the class to compute F1 for.

    Returns:
        F1 value in [0, 1]. Returns 0 if both precision and recall are 0.
    """
    p = precision_score(cm, class_idx)
    r = recall_score(cm, class_idx)
    denom = p + r
    return float(2.0 * p * r / denom) if denom > 0 else 0.0


def classification_report(
    A: np.ndarray,
    y_int: np.ndarray,
    classes: list[str],
) -> str:
    """
    Generate a formatted classification report with per-class metrics.

    Shows precision, recall, F1, and support (number of true samples)
    for every class, plus the overall accuracy and macro-averaged F1.

    Args:
        A:       Softmax output probabilities, shape (n_samples, n_classes).
        y_int:   True integer class labels, shape (n_samples,).
        classes: Sorted list of class names (index = class id).

    Returns:
        Multi-line string ready to be printed.
    """
    n_classes = len(classes)
    cm = confusion_matrix(A, y_int, n_classes)

    lines = []
    lines.append(f"\n{'Class':>10}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'Support':>8}")
    lines.append("-" * 52)

    f1_values = []
    for i, cls in enumerate(classes):
        p   = precision_score(cm, i)
        r   = recall_score(cm, i)
        f1  = f1_score(cm, i)
        sup = int(cm[i, :].sum())
        f1_values.append(f1)
        lines.append(f"{cls:>10}  {p:>10.4f}  {r:>8.4f}  {f1:>8.4f}  {sup:>8d}")

    lines.append("-" * 52)
    overall_acc = float(np.mean(np.argmax(A, axis=1) == y_int))
    macro_f1    = float(np.mean(f1_values))
    lines.append(f"{'accuracy':>10}  {'':>10}  {'':>8}  {overall_acc:>8.4f}  {len(y_int):>8d}")
    lines.append(f"{'macro avg':>10}  {'':>10}  {'':>8}  {macro_f1:>8.4f}  {len(y_int):>8d}")

    return "\n".join(lines)


def print_confusion_matrix(cm: np.ndarray, classes: list[str]) -> None:
    """
    Print the confusion matrix in a readable table format.

    Args:
        cm:      Confusion matrix of shape (n_classes, n_classes).
        classes: Sorted list of class names.
    """
    col_width = max(max(len(c) for c in classes), 6) + 2
    header = " " * col_width + "".join(f"{c:>{col_width}}" for c in classes) + "  (predicted)"
    print("\nConfusion Matrix:")
    print(header)
    print("-" * len(header))
    for i, cls in enumerate(classes):
        row = f"{cls:>{col_width}}" + "".join(f"{cm[i, j]:>{col_width}}" for j in range(len(classes)))
        print(row)
    print("(actual)")
