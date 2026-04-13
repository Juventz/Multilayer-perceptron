import csv
from os import R_OK, access


def parse_csv_with_fieldnames(
    path: str,
    fieldnames: list[str],
) -> tuple[list[str], list[dict[str, str]]]:
    """
    Parse a headerless CSV by assigning explicit field names.

    The file data/data.csv contains data rows only, so we provide
    the expected column names.

    Args:
        path: Path to the CSV file.
        fieldnames: Ordered list of expected column names.

    Returns:
        A tuple of (fieldnames, rows) with one dictionary per row.
    """
    try:
        with open(path, "r", encoding="utf-8", newline="") as file:
            reader = csv.reader(file)
            rows = []
            for raw_row in reader:
                if not raw_row:
                    continue
                # Defensive check: keeps schema alignment strict and explicit.
                if len(raw_row) != len(fieldnames):
                    raise ValueError(
                        f"Row has {len(raw_row)} columns but expected {len(fieldnames)}."
                    )
                rows.append(dict(zip(fieldnames, raw_row, strict=True)))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{path} not found.") from error
    except IOError as error:
        if not access(path, R_OK):
            raise IOError(f"{path} is not readable.") from error
        raise IOError(f"{path} is not a valid CSV file.") from error
    except csv.Error as error:
        raise csv.Error(f"{path} could not be parsed.") from error

    if not rows:
        raise ValueError(f"{path} is empty.")

    return fieldnames, rows


def to_float(value: str | None) -> float | None:
    """Convert a raw string value to float, returning None for empty/invalid values."""
    if value is None:
        return None
    stripped = value.strip()
    if stripped == "":
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def is_numeric_column(rows: list[dict[str, str]], column: str) -> bool:
    """Return True only if all non-empty values in a column can be parsed as float."""
    has_numeric = False
    for row in rows:
        raw = row.get(column, "")
        if raw is None or raw.strip() == "":
            continue
        value = to_float(raw)
        if value is None:
            return False
        has_numeric = True
    return has_numeric


def get_numeric_values(rows: list[dict[str, str]], column: str) -> list[float]:
    """Extract valid numeric values from a column as floats."""
    values = []
    for row in rows:
        value = to_float(row.get(column, ""))
        if value is not None:
            values.append(value)
    return values
