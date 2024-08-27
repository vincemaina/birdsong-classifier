import joblib
from typing import Self

SAVED_MODEL_PATH = "./xgboost_model.pkl"


class XGBoostModel:
    """Model that deals with tabular data extracted from an audio file."""

    def __init__(self):
        pass

    def load_model(self):
        return joblib.load(SAVED_MODEL_PATH)

    def save_model(self, model: Self):
        joblib.dump(model, SAVED_MODEL_PATH)
