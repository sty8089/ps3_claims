import hashlib
from typing import Iterable, Sequence, Union

import numpy as np
import pandas as pd


def _row_hash_to_fraction(values: Sequence) -> float:
    """Convert a sequence of values for a row into a deterministic fraction in [0, 1).

    We stringify each value, join with a separator, compute an MD5 hash and map the
    integer hash into a floating point in [0,1).
    """
    # Use '|' as separator to avoid accidental collisions from concatenation
    combined = "|".join(["" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v) for v in values])
    h = hashlib.md5(combined.encode("utf-8")).hexdigest()
    as_int = int(h, 16)
    # Map into [0,1) using 128-bit range of MD5
    return as_int / float(2 ** 128)


def create_sample_column(
    df: pd.DataFrame,
    columns: Union[str, Sequence[str]],
    training_frac: float = 0.8,
    sample_col: str = "sample",
) -> pd.DataFrame:
    """Create a reproducible sample column on a dataframe.

    The function assigns each row to 'train' or 'test' based on a deterministic
    hash of the provided column(s). If multiple columns are provided, the
    combination of their values is used to compute the hash so duplicate
    key-combinations receive the same assignment.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe. Not modified in-place; a shallow copy is returned.
    columns : str or sequence of str
        Column name or list of column names to base the split on. These columns
        must exist in `df`.
    training_frac : float, default 0.8
        Fraction of rows to assign to the training set. Must be between 0 and 1.
    sample_col : str, default 'sample'
        Name of the column to create containing values 'train' or 'test'.

    Returns
    -------
    pd.DataFrame
        A copy of `df` with an added `sample_col` column.
    """

    if isinstance(columns, str):
        cols = [columns]
    elif isinstance(columns, Iterable):
        cols = list(columns)
    else:
        raise TypeError("`columns` must be a column name or an iterable of column names")

    if not 0 <= training_frac <= 1:
        raise ValueError("`training_frac` must be between 0 and 1")

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found in dataframe: {missing}")

    out = df.copy()

    # Compute deterministic fraction for each row
    fractions = out[cols].apply(lambda row: _row_hash_to_fraction(row.values), axis=1)

    out[sample_col] = np.where(fractions < float(training_frac), "train", "test")

    return out


# keep existing name for backward compatibility; implement using the new function
def create_sample_split(df, id_column, training_frac=0.8):
    """Backward-compatible wrapper around create_sample_column.

    Parameters are the same as the old interface: id_column may be a single
    column name (string).
    """
    return create_sample_column(df, id_column, training_frac=training_frac)
