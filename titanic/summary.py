import pandas as pd


def missing_data_summary(df):
    """Missing count, missing percent and dtype for each column."""
    total = df.isnull().sum()
    summary = pd.DataFrame(
        {
            "Total": total,
            "Percent": total / len(df) * 100,
            "Types": df.dtypes.astype(str),
        }
    )
    return summary.T


def most_frequent_summary(df):
    """Most frequent value in each column, how often it occurs, and its percent."""
    total = df.count()
    items, freqs = [], []
    for col in df.columns:
        counts = df[col].value_counts()
        if counts.empty:  # column is entirely missing
            items.append(0)
            freqs.append(0)
        else:
            items.append(counts.index[0])
            freqs.append(counts.iloc[0])

    summary = pd.DataFrame({"Total": total})
    summary["Most frequent item"] = items
    summary["Frequence"] = freqs
    summary["Percent from total"] = (summary["Frequence"] / total * 100).round(3)
    return summary.T


def unique_values_summary(df):
    """Number of non-missing and unique values in each column."""
    return pd.DataFrame({"Total": df.count(), "Uniques": df.nunique()}).T


def survival_rate(df, group_columns):
    """Average survival rate for each combination of group_columns."""
    return df[[*group_columns, "Survived"]].groupby(group_columns, as_index=False).mean()
