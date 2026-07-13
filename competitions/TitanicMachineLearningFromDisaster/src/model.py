"""Titanic survival model — v1.

A real gradient-boosted classifier with engineered features, replacing the
all-zeros baseline. Reports 5-fold cross-validated accuracy (the competition
metric), trains on the full training set, predicts the test set, and writes
submissions/submission.csv in the PassengerId,Survived format.

Run from the competition folder:  python src/model.py
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config

NUMERIC = ["Age", "Fare", "FamilySize", "IsAlone"]
CATEGORICAL = ["Pclass", "Sex", "Embarked", "Title"]

RARE_TITLES = {
    "Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev",
    "Sir", "Jonkheer", "Dona",
}
TITLE_MAP = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}


def add_features(df):
    """Engineer features shared by train and test."""
    df = df.copy()
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    title = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
    title = title.replace(TITLE_MAP)
    title = title.where(~title.isin(RARE_TITLES), "Rare")
    df["Title"] = title
    return df


def build_pipeline():
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), NUMERIC),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), CATEGORICAL),
        ]
    )
    return Pipeline([
        ("pre", pre),
        ("clf", GradientBoostingClassifier(random_state=config.RANDOM_STATE)),
    ])


def main():
    train = add_features(pd.read_csv(config.TRAINING_DATA))
    test = add_features(pd.read_csv(config.TEST_DATA))

    X = train[NUMERIC + CATEGORICAL]
    y = train[config.TARGET]

    pipe = build_pipeline()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    print(f"5-fold CV accuracy: {scores.mean():.4f} +/- {scores.std():.4f}")

    pipe.fit(X, y)
    preds = pipe.predict(test[NUMERIC + CATEGORICAL])

    submission = pd.DataFrame({
        config.ID: test[config.ID],
        config.TARGET: preds.astype(int),
    })
    submission.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"wrote {config.SUBMISSION_PATH} ({len(submission)} rows)")
    print(submission[config.TARGET].value_counts().to_dict())


if __name__ == "__main__":
    main()
