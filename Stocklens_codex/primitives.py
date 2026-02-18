"""
Pure computation helpers — no I/O, no config lookups, no pandas DataFrames at API level.
Only numpy/pandas Series in, numpy/pandas Series out.
"""

import numpy as np
import pandas as pd
from typing import List, Union


def safe_div(a: Union[pd.Series, np.ndarray, float],
             b: Union[pd.Series, np.ndarray, float],
             default: float = np.nan) -> Union[pd.Series, np.ndarray, float]:
    """Safe division handling NaN and zero."""
    is_series = isinstance(a, pd.Series) or isinstance(b, pd.Series)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where((pd.isna(b)) | (b == 0), default, a / b)
    if is_series:
        return pd.Series(result, index=a.index if isinstance(a, pd.Series) else None)
    return result


def safe_str_lower(series: pd.Series) -> pd.Series:
    """Safely convert Series to lowercase."""
    return series.fillna('').astype(str).str.lower()


def vectorized_score(series: pd.Series, bin_config: dict) -> pd.Series:
    """Apply vectorized scoring using pd.cut."""
    result = pd.cut(series, bins=bin_config['bins'], labels=bin_config['labels'], ordered=False).astype(float)
    return result.fillna(bin_config['default'])


def vectorized_string_build(n: int, conditions: List[np.ndarray], strings: List[str], separator: str = ', ') -> list:
    """
    Build strings by conditionally concatenating parts using np.where.

    Not truly vectorized (O(n * m) object-array string ops), but avoids
    an explicit Python loop over rows.

    Args:
        n: Number of rows
        conditions: List of boolean numpy arrays
        strings: List of strings to concatenate when condition is True
        separator: String to join parts

    Returns:
        List of concatenated strings
    """
    result = np.full(n, '', dtype=object)

    for cond, s in zip(conditions, strings):
        result = np.where(cond, np.where(result == '', s, result + separator + s), result)

    return result.tolist()
