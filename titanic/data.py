import pandas as pd

from titanic.config import TEST_PATH, TRAIN_PATH


def load_data():
    """Read the train and test csv files. Returns (train_df, test_df)."""
    return pd.read_csv(TRAIN_PATH), pd.read_csv(TEST_PATH)


def combine_train_test(train_df, test_df):
    """Stack train and test into one dataframe with a "set" column ("train"/"test")."""
    all_df = pd.concat([train_df, test_df], axis=0)
    all_df["set"] = "train"
    all_df.loc[all_df["Survived"].isna(), "set"] = "test"
    return all_df
