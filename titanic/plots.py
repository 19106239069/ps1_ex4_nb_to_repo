import matplotlib.pyplot as plt
import seaborn as sns

from titanic.config import COLOR_LIST


def _colours(df, hue):
    """Seaborn colour arguments: one colour per hue level, or a single colour."""
    if hue:
        return {"hue": hue, "palette": COLOR_LIST[: df[hue].nunique()]}
    return {"color": COLOR_LIST[1]}


def plot_counts(df, column, hue=None):
    """Bar chart of how many rows have each value of column, optionally split by hue."""
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(x=column, data=df, ax=ax, **_colours(df, hue))
    ax.grid(color="black", linestyle="-.", linewidth=0.5, axis="y", which="major")
    ax.set_title(f"Number of passengers / {column}")
    plt.show()


def plot_distribution(df, column, hue=None):
    """Histogram of column, optionally split by hue."""
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(data=df, x=column, ax=ax, **_colours(df, hue))
    ax.set_title(f"Distribution of {column}")
    plt.show()
