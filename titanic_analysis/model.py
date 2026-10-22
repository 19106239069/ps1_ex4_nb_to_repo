"""Functions for the baseline Random Forest model and its evaluation."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from .config import MODEL_PARAMS, RANDOM_STATE, VALID_SIZE


def split_train_validation(
    data: pd.DataFrame,
    valid_size: float = VALID_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split labelled training data into train and validation dataframes."""
    if "Survived" not in data.columns:
        raise KeyError("The labelled dataframe must contain 'Survived'.")
    return train_test_split(data, test_size=valid_size, random_state=random_state, shuffle=True)


def prepare_xy(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    predictors: list[str],
    target: str = "Survived",
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Select model predictors and target vectors from train/validation frames."""
    missing = [column for column in predictors + [target] if column not in train.columns]
    if missing:
        raise KeyError(f"Required columns are missing from training data: {missing}")
    missing_valid = [column for column in predictors + [target] if column not in valid.columns]
    if missing_valid:
        raise KeyError(f"Required columns are missing from validation data: {missing_valid}")
    return train[predictors], train[target], valid[predictors], valid[target]


def create_random_forest(**overrides: object) -> RandomForestClassifier:
    """Create the notebook's baseline Random Forest classifier.

    Parameters in ``overrides`` replace the defaults from ``config.py``.
    """
    params = {**MODEL_PARAMS, **overrides}
    return RandomForestClassifier(**params)


def fit_classifier(
    classifier: RandomForestClassifier,
    features: pd.DataFrame,
    target: pd.Series,
) -> RandomForestClassifier:
    """Fit ``classifier`` and return the fitted estimator."""
    classifier.fit(features, target)
    return classifier


def predict(classifier: RandomForestClassifier, features: pd.DataFrame):
    """Generate class predictions with a fitted classifier."""
    return classifier.predict(features)


def classification_report_text(actual, predicted) -> str:
    """Return a readable precision/recall/F1 report for binary survival labels."""
    return classification_report(actual, predicted, labels=[0, 1], target_names=["Not Survived", "Survived"], zero_division=0)
