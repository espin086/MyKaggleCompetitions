"""Config for the ROGII - Wellbore Geology Prediction competition."""

SLUG = "rogii-wellbore-geology-prediction"

# Data lives per-well, not as flat train/test CSVs. Each well is an 8-char hash
# with a horizontal_well.csv (+ typewell.csv, + .png for train).
DATA_DIR = "data"
TRAIN_DIR = "data/train"
TEST_DIR = "data/test"
SAMPLE_SUBMISSION = "data/sample_submission.csv"

SUBMISSION_PATH = "submissions/submission.csv"
SUBMISSIONS_LOG = "SUBMISSIONS.md"

TARGET = "tvt"          # True Vertical Thickness (ft), predicted for the evaluation zone
ID = "id"               # {WELLNAME}_{row_index}, e.g. 000d7d20_1442
METRIC = "RMSE"         # lower is better; public LB leader ~4.86

RANDOM_STATE = 42
