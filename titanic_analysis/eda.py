"""Reusable exploratory-data-analysis tables and plots."""

from collections.abc import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import COLOR_PALETTE


def plot_category_counts(
    data: pd.DataFrame,
    feature: str,
    hue: str | None = None,
    title: str | None = None,
    palette: list[str] | None = None,
    figsize: tuple[float, float] = (8, 4),
) -> None:
    """Plot counts for a categorical feature, optionally split by a hue column."""
    if feature not in data.columns:
        raise KeyError(f"Column not found: {feature}")
    if hue is not None and hue not in data.columns:
        raise KeyError(f"Hue column not found: {hue}")
    _, ax = plt.subplots(figsize=figsize)
    sns.countplot(data=data, x=feature, hue=hue, palette=palette or COLOR_PALETTE, ax=ax)
    ax.set_title(title or f"Passenger counts: {feature}")
    ax.grid(color="black", linestyle="-.", linewidth=0.5, axis="y", which="major", alpha=0.35)
    plt.tight_layout()
    plt.show()


def plot_categorical_features(
    data: pd.DataFrame, features: Iterable[str], hue: str | None = None
) -> None:
    """Plot one count chart per feature in ``features``."""
    for feature in features:
        plot_category_counts(data, feature, hue=hue, title=f"{feature} counts" + (f" by {hue}" if hue else ""))


def plot_numeric_distributions(
    data: pd.DataFrame,
    features: Iterable[str],
    hue: str | None = None,
    bins: int = 30,
) -> None:
    """Plot histogram distributions for one or more numeric columns."""
    for feature in features:
        _, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(data=data, x=feature, hue=hue, bins=bins, kde=False, ax=ax, multiple="layer")
        ax.set_title(f"Distribution of {feature}" + (f" by {hue}" if hue else ""))
        plt.tight_layout()
        plt.show()


def plot_two_feature_comparison(data: pd.DataFrame) -> None:
    """Show fare-interval counts and fare distributions split by survival status."""
    plot_category_counts(data, "Fare Interval", hue="Survived", title="Fare interval by survival")
    plot_numeric_distributions(data, ["Fare"], hue="Survived")


def title_survival_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Return mean survival rates grouped by normalised title and sex."""
    required = {"Titles", "Sex", "Survived"}
    missing = required.difference(data.columns)
    if missing:
        raise KeyError(f"Required columns are missing: {sorted(missing)}")
    return data[["Titles", "Sex", "Survived"]].groupby(
        ["Titles", "Sex"], as_index=False, dropna=False
    )["Survived"].mean()