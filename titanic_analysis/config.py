
"""Central configuration for the Titanic analysis project."""

from pathlib import Path

# Project root: the directory containing data/ and titanic_analysis/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

# Model validation settings
VALID_SIZE = 0.2
RANDOM_STATE = 42

# Model features and target
PREDICTORS = ["Sex", "Pclass"]
TARGET = "Survived"

# Plot settings
COLOR_PALETTE = ["#A5D7E8", "#576CBC", "#19376D", "#0b2447"]

# Random Forest parameters
MODEL_PARAMS = {
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
    "criterion": "gini",
    "n_estimators": 100,
    "verbose": 0,
}

