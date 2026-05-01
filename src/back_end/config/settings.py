"""
Project-wide configuration: paths, constants, and default hyperparameters.
"""

from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[3]

DATA_RAW_DIR    = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR      = ROOT_DIR / "models" / "saved_models"
LOGS_DIR        = ROOT_DIR / "models" / "logs"
REPORTS_DIR     = ROOT_DIR / "reports"
FIGURES_DIR     = REPORTS_DIR / "figures"

RAW_CSV_PATH              = DATA_RAW_DIR / "export.csv"
LABELS_CSV_PATH           = DATA_RAW_DIR / "RES2-6-9-labels.csv"
FEATURES_PARQUET_PATH     = DATA_PROCESSED_DIR / "features.parquet"
CLASSIFICATION_MODEL_PATH = MODELS_DIR / "classification_model.pkl"
REGRESSION_MODEL_PATH     = MODELS_DIR / "regression_model.pkl"
METER_ID_COL = "id"
TIMESTAMP_COL = "horodate"
VALUE_COL     = "valeur"          # electricity consumption in Wh

# Half-hourly resolution → 48 readings per day
READINGS_PER_DAY = 48

# Season month ranges (Northern Hemisphere)
WINTER_MONTHS = [11, 12, 1, 2, 3]
SUMMER_MONTHS = [6, 7, 8]

# Occupancy threshold: a day is considered "occupied" if total daily
# consumption exceeds this value (Wh).  Corresponds to the bare minimum
# for a permanently occupied dwelling (lighting + standby loads).
OCCUPANCY_DAY_THRESHOLD_WH = 500
CLASSIFICATION_PARAMS = {
    "n_estimators":  200,
    "max_depth":     6,
    "learning_rate": 0.05,
    "random_state":  42,
}

REGRESSION_PARAMS = {
    "n_estimators":  200,
    "max_depth":     6,
    "learning_rate": 0.05,
    "random_state":  42,
}

TEST_SIZE    = 0.2
RANDOM_STATE = 42
