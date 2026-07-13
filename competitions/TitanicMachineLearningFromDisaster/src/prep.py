"""Titanic feature prep: enrich train/test with the proven high-ROI, target-free
derivations, then hand the enriched CSVs to the kaggle-ml-loop pipeline (whose recipe
compiler only applies generic ops — impute/encode/scale/select — so domain features
must be materialized here first).

All derivations are target-free (no use of Survived), so there is no label leakage:
  - Title      : extracted from Name, rare titles collapsed
  - FamilySize : SibSp + Parch + 1, plus IsAlone and a size bucket
  - TicketGroup: passengers sharing a ticket (counted across train+test — no target used)
  - FarePerPerson : Fare / ticket-group size
  - Deck       : first letter of Cabin (U = unknown)
  - Age        : imputed by Title-group median (medians computed on TRAIN only)

Raw high-cardinality text columns (Name, Ticket, Cabin) are dropped after extraction so
the pipeline doesn't ordinal-encode them into noise. PassengerId/Survived are preserved.

Run: python src/prep.py   ->  writes data/train_fe.csv, data/test_fe.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"


def _title(name: pd.Series) -> pd.Series:
    t = name.str.extract(r",\s*([^\.]+)\.")[0].str.strip()
    canon = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
    t = t.replace(canon)
    common = {"Mr", "Mrs", "Miss", "Master"}
    return t.where(t.isin(common), "Rare")


def build():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    both = pd.concat([train.assign(_src="tr"), test.assign(_src="te")], ignore_index=True)

    both["Title"] = _title(both["Name"])
    both["FamilySize"] = both["SibSp"] + both["Parch"] + 1
    both["IsAlone"] = (both["FamilySize"] == 1).astype(int)
    both["FamilyBucket"] = pd.cut(both["FamilySize"], [0, 1, 4, 100],
                                  labels=["alone", "small", "large"]).astype(str)
    both["TicketGroupSize"] = both.groupby("Ticket")["Ticket"].transform("count")
    both["Fare"] = both.groupby("Pclass")["Fare"].transform(lambda s: s.fillna(s.median()))
    both["FarePerPerson"] = both["Fare"] / both["TicketGroupSize"].clip(lower=1)
    both["Deck"] = both["Cabin"].fillna("U").str[0]
    both["Embarked"] = both["Embarked"].fillna(both["Embarked"].mode()[0])

    # Age imputed by Title median computed on TRAIN rows only (no leakage from test)
    title_age = both.loc[both["_src"] == "tr"].groupby("Title")["Age"].median()
    global_age = both.loc[both["_src"] == "tr", "Age"].median()
    both["Age"] = both.apply(
        lambda r: title_age.get(r["Title"], global_age) if pd.isna(r["Age"]) else r["Age"],
        axis=1).astype(float)

    keep = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked",
            "Title", "FamilySize", "IsAlone", "FamilyBucket",
            "TicketGroupSize", "FarePerPerson", "Deck"]

    tr = both[both["_src"] == "tr"].copy()
    te = both[both["_src"] == "te"].copy()
    out_tr = pd.concat([train[["PassengerId", "Survived"]].reset_index(drop=True),
                        tr[keep].reset_index(drop=True)], axis=1)
    out_te = pd.concat([test[["PassengerId"]].reset_index(drop=True),
                        te[keep].reset_index(drop=True)], axis=1)
    out_tr.to_csv(DATA / "train_fe.csv", index=False)
    out_te.to_csv(DATA / "test_fe.csv", index=False)
    print(f"train_fe.csv: {out_tr.shape} | test_fe.csv: {out_te.shape}")
    print("features:", keep)
    assert out_tr.isna().sum().sum() == 0, "unexpected NaNs in train_fe"
    assert out_te.drop(columns=[]).isna().sum().sum() == 0, "unexpected NaNs in test_fe"
    print("no NaNs — ok")


if __name__ == "__main__":
    build()
