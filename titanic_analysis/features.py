"""Feature engineering utilities used in the Titanic analysis notebook."""

import re
from typing import Any

import pandas as pd

NAME_COLUMNS = ["Family Name", "Title", "Given Name", "Maiden Name"]


def parse_name(name: Any) -> pd.Series:
    """Parse a passenger name into family name, title, given name, and maiden name.

    Malformed or missing names return four missing values rather than stopping
    feature engineering for the whole dataframe.
    """
    if not isinstance(name, str) or not name.strip():
        return pd.Series([None, None, None, None], index=NAME_COLUMNS)
    try:
        family_name, remainder = name.split(",", 1)
        title, given_part = remainder.split(".", 1)
        title = f"{title.strip()}."
        given_part = given_part.strip()
        maiden_name = None
        match = re.search(r"\(([^()]*)\)", given_part)
        if match:
            maiden_name = match.group(1).strip() or None
            given_part = (given_part[:match.start()] + given_part[match.end():]).strip()
        return pd.Series(
            [family_name.strip(), title, given_part, maiden_name], index=NAME_COLUMNS
        )
    except ValueError:
        return pd.Series([None, None, None, None], index=NAME_COLUMNS)


def add_name_features(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with four name-derived columns added from ``Name``."""
    result = data.copy()
    parsed = result["Name"].apply(parse_name)
    for column in NAME_COLUMNS:
        result[column] = parsed[column]
    return result


def add_family_size(data: pd.DataFrame) -> pd.DataFrame:
    """Add ``Family Size`` as siblings/spouses + parents/children + passenger."""
    result = data.copy()
    result["Family Size"] = result["SibSp"] + result["Parch"] + 1
    return result


def add_age_interval(data: pd.DataFrame) -> pd.DataFrame:
    """Add the notebook's age bands: 0-16, 17-32, 33-48, 49-64, and 65+. 

    Missing ages are assigned to interval 0 to preserve the original notebook's
    initialisation behaviour.
    """
    result = data.copy()
    result["Age Interval"] = pd.cut(
        result["Age"],
        bins=[-float("inf"), 16, 32, 48, 64, float("inf")],
        labels=[0, 1, 2, 3, 4],
        right=True,
    ).fillna(0).astype(int)
    return result


def add_fare_interval(data: pd.DataFrame) -> pd.DataFrame:
    """Add the notebook's four fare groups using cut points 7.91, 14.454, and 31.

    Missing fares are assigned to interval 0 to match the original notebook.
    """
    result = data.copy()
    result["Fare Interval"] = pd.cut(
        result["Fare"],
        bins=[-float("inf"), 7.91, 14.454, 31, float("inf")],
        labels=[0, 1, 2, 3],
        right=True,
    ).fillna(0).astype(int)
    return result


def add_sex_pclass(data: pd.DataFrame) -> pd.DataFrame:
    """Add a combined feature such as ``F_C1`` or ``M_C3``."""
    result = data.copy()
    result["Sex_Pclass"] = result["Sex"].astype("string").str[:1].str.upper() + "_C" + result["Pclass"].astype(str)
    return result


def add_family_type(data: pd.DataFrame) -> pd.DataFrame:
    """Add ``Family Type`` categories: Single (1), Small (2-4), or Large (5+)."""
    result = data.copy()
    result["Family Type"] = "Large"
    result.loc[result["Family Size"] == 1, "Family Type"] = "Single"
    result.loc[(result["Family Size"] > 1) & (result["Family Size"] < 5), "Family Type"] = "Small"
    return result


def add_title_column(data: pd.DataFrame) -> pd.DataFrame:
    """Create the ``Titles`` feature from the parsed ``Title`` feature."""
    result = data.copy()
    result["Titles"] = result["Title"]
    return result


def normalise_titles(data: pd.DataFrame) -> pd.DataFrame:
    """Combine equivalent and uncommon passenger titles into consistent groups."""
    result = data.copy()
    if "Titles" not in result.columns:
        result = add_title_column(result)
    result["Titles"] = result["Titles"].replace({"Mlle.": "Miss.", "Ms.": "Miss.", "Mme.": "Mrs."})
    rare_titles = ["Lady.", "the Countess.", "Capt.", "Col.", "Don.", "Dr.", "Major.", "Rev.", "Sir.", "Jonkheer.", "Dona."]
    result["Titles"] = result["Titles"].replace(rare_titles, "Rare")
    return result


def encode_sex(data: pd.DataFrame) -> pd.DataFrame:
    """Encode Titanic's ``Sex`` feature as female=1 and male=0.

    Raises
    ------
    ValueError
        If unexpected or missing values are found in ``Sex``.
    """
    result = data.copy()
    encoded = result["Sex"].map({"female": 1, "male": 0})
    if encoded.isna().any():
        unexpected = result.loc[encoded.isna(), "Sex"].drop_duplicates().tolist()
        raise ValueError(f"Unexpected or missing Sex values: {unexpected}")
    result["Sex"] = encoded.astype(int)
    return result
