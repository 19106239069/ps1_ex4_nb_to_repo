"""Functions for loading and inspecting the Titanic data."""

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]


def load_data(train_path: PathLike, test_path: PathLike) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the Kaggle training and test CSV files.

    Parameters
    ----------
    train_path, test_path:
        Paths to ``train.csv`` and ``test.csv`` respectively.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        The training and test dataframes.
    """
    train_path, test_path = Path(train_path), Path(test_path)
    if not train_path.is_file():
        raise FileNotFoundError(f"Training data not found: {train_path}")
    if not test_path.is_file():
        raise FileNotFoundError(f"Test data not found: {test_path}")
    return pd.read_csv(train_path), pd.read_csv(test_path)


def missing_data_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Summarise missing observations, missing percentages, and column types."""
    total = data.isna().sum()
    percent = data.isna().mean().mul(100)
    summary = pd.DataFrame({"Total": total, "Percent": percent, "Types": data.dtypes.astype(str)})
    return summary.T


def most_frequent_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Return each column's most frequent non-missing value and its share.

    Empty/all-missing columns receive ``None`` and a frequency of zero.
    """
    rows = []
    for column in data.columns:
        valid = data[column].dropna()
        counts = valid.value_counts()
        if counts.empty:
            item, frequency = None, 0
        else:
            item, frequency = counts.index[0], int(counts.iloc[0])
        total = int(data[column].count())
        percent = round(frequency / total * 100, 3) if total else 0.0
        rows.append({
            "Feature": column,
            "Total": total,
            "Most frequent item": item,
            "Frequency": frequency,
            "Percent from total": percent,
        })
    return pd.DataFrame(rows).set_index("Feature").T


def unique_value_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Return non-missing observation counts and distinct-value counts per column."""
    return pd.DataFrame({"Total": data.count(), "Uniques": data.nunique()}).T


def combine_train_test(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Concatenate train and test rows and label their source in a ``set`` column.

    The original ``Survived`` column identifies training rows; it is missing in
    the Kaggle test set.
    """
    combined = pd.concat([train, test], axis=0, ignore_index=True, sort=False)
    combined["set"] = np.where(combined["Survived"].isna(), "test", "train")
    return combined
