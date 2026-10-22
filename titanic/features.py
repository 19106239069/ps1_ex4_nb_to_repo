import pandas as pd

from titanic.config import (
    AGE_BINS,
    FARE_BINS,
    LARGE_FAMILY_SIZE,
    RARE_TITLES,
    SEX_MAP,
    SINGLE_FAMILY_SIZE,
    TITLE_REPLACEMENTS,
)

# Every function takes a dataframe and returns a new one with the feature added.


def add_family_size(df):
    df = df.copy()
    df["Family Size"] = df["SibSp"] + df["Parch"] + 1
    return df


def _interval(column, bins):
    """Interval number (0, 1, 2, ...) for each value; missing values get 0."""
    return pd.cut(column, bins=bins, labels=False).fillna(0)


def add_age_interval(df):
    df = df.copy()
    df["Age Interval"] = _interval(df["Age"], AGE_BINS)
    return df


def add_fare_interval(df):
    df = df.copy()
    df["Fare Interval"] = _interval(df["Fare"], FARE_BINS)
    return df


def add_sex_pclass(df):
    """Combine Sex and Pclass, e.g. "M_C3". Needs Sex before it is encoded as 0/1."""
    df = df.copy()
    df["Sex_Pclass"] = df["Sex"].str[0].str.upper() + "_C" + df["Pclass"].astype(str)
    return df


def parse_names(row):
    """Split a name into (family name, title, given name, maiden name)."""
    try:
        parts = row["Name"].split(",")
        family_name = parts[0]
        parts = parts[1].split(".")
        title = parts[0].strip() + "."
        rest = parts[1]
        if "(" in rest:
            parts = rest.split("(")
            return pd.Series([family_name, title, parts[0], parts[1].rstrip(")")])
        return pd.Series([family_name, title, rest, None])
    except Exception as ex:
        print(f"Exception: {ex}")
        return pd.Series([None, None, None, None])


def add_name_features(df):
    df = df.copy()
    df[["Family Name", "Title", "Given Name", "Maiden Name"]] = df.apply(
        parse_names, axis=1
    )
    return df


def add_family_type(df):
    """Single, Small or Large, based on Family Size."""
    df = df.copy()
    df["Family Type"] = "Small"
    df.loc[df["Family Size"] == SINGLE_FAMILY_SIZE, "Family Type"] = "Single"
    df.loc[df["Family Size"] >= LARGE_FAMILY_SIZE, "Family Type"] = "Large"
    return df


def add_aggregated_titles(df):
    """Unify variants of Miss/Mrs and group uncommon titles as "Rare". Needs Title."""
    df = df.copy()
    df["Titles"] = df["Title"].replace(TITLE_REPLACEMENTS).replace(RARE_TITLES, "Rare")
    return df


def encode_sex(df):
    """Replace "female"/"male" with 1/0 (do this after any step that needs the text)."""
    df = df.copy()
    df["Sex"] = df["Sex"].map(SEX_MAP).astype(int)
    return df


def add_all_features(df):
    """Add every engineered feature to df (does not encode Sex)."""
    df = add_family_size(df)
    df = add_age_interval(df)
    df = add_fare_interval(df)
    df = add_sex_pclass(df)
    df = add_name_features(df)
    df = add_family_type(df)
    df = add_aggregated_titles(df)
    return df
