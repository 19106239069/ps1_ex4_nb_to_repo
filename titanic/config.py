from pathlib import Path

# Paths (anchored to the repository root, so they work from any working directory)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = PROJECT_ROOT / "train.csv"
TEST_PATH = PROJECT_ROOT / "test.csv"

# Plotting
COLOR_LIST = ["#A5D7E8", "#576CBC", "#19376D", "#0b2447"]

# Feature engineering
AGE_BINS = [-float("inf"), 16, 32, 48, 64, float("inf")]  # intervals 0-4
FARE_BINS = [-float("inf"), 7.91, 14.454, 31, float("inf")]  # intervals 0-3
SEX_MAP = {"female": 1, "male": 0}

SINGLE_FAMILY_SIZE = 1  # Family Size == 1 -> "Single"
LARGE_FAMILY_SIZE = 5  # Family Size >= 5 -> "Large", otherwise "Small"

TITLE_REPLACEMENTS = {
    "Mlle.": "Miss.",
    "Ms.": "Miss.",
    "Mme.": "Mrs.",
}
RARE_TITLES = [
    "Lady.",
    "the Countess.",
    "Capt.",
    "Col.",
    "Don.",
    "Dr.",
    "Major.",
    "Rev.",
    "Sir.",
    "Jonkheer.",
    "Dona.",
]

# Modelling
VALID_SIZE = 0.2
RANDOM_STATE = 42
PREDICTORS = ["Sex", "Pclass"]
TARGET = "Survived"
TARGET_NAMES = ["Not Survived", "Survived"]

RF_PARAMS = {
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
    "criterion": "gini",
    "n_estimators": 100,
    "verbose": False,
}
