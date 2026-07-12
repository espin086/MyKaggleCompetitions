import config
import pandas as pd
import os


df_test = pd.read_csv(config.TEST_DATA)


def data_cleaning(df_test):
    """Take a dataframe and return a dataframe with the same columns and rows but with no missing values."""
    pass


def feature_engineering(df_test):
    """Take a dataframe and return a dataframe with the same columns and rows but with new features."""
    pass


def model(df_test):
    """Take a dataframe and return random 0 or 1 for each row and call the result 'Survived'."""
    return pd.DataFrame(
        {"PassengerId": df_test["PassengerId"], "Survived": [0] * len(df_test)}
    )


def save_submission(df_submission):
    """Save submission dataframe to csv file."""
    submission.to_csv("data/submission.csv", index=False)


def submit_to_kaggle(message):
    """Submit to kaggle using os module."""
    command = (
        f"kaggle competitions submit -c titanic -f data/submission.csv -m '{message}'"
    )
    os.system(command)


submission = model(df_test=df_test)
save_submission(df_submission=submission)
submit_to_kaggle(message="second submission")
