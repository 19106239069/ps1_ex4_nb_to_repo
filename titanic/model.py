from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from titanic.config import (
    PREDICTORS,
    RANDOM_STATE,
    RF_PARAMS,
    TARGET,
    TARGET_NAMES,
    VALID_SIZE,
)


def split_data(df, predictors=PREDICTORS, target=TARGET):
    """Split into train and validation. Returns train_X, train_Y, valid_X, valid_Y."""
    train, valid = train_test_split(
        df, test_size=VALID_SIZE, random_state=RANDOM_STATE, shuffle=True
    )
    return train[predictors], train[target].values, valid[predictors], valid[target].values


def train_model(train_X, train_Y, **params):
    """Fit a random forest. Pass keyword arguments to override the defaults in config."""
    model = RandomForestClassifier(**{**RF_PARAMS, **params})
    model.fit(train_X, train_Y)
    return model


def evaluate(model, X, Y):
    """Print the classification report for the model's predictions on X."""
    print(metrics.classification_report(Y, model.predict(X), target_names=TARGET_NAMES))
