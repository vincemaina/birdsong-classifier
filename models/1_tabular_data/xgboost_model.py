import os
import joblib
import noisereduce as nr
import librosa
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from data import xeno_canto_api
from multiprocessing import Pool


SAVED_MODEL_PATH = "models/1_tabular_data/xgboost_model.pkl"
SAVED_LABEL_ENCODER_PATH = "models/1_tabular_data/xgboost_label_encoder.pkl"


def extract_features(y, sr):
    """Extract required features for this model."""

    y = nr.reduce_noise(y, sr)  # Apply spectral gate
    y = librosa.util.normalize(y)  # Normalize signal

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_flatness = librosa.feature.spectral_flatness(y=y)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y=y)
    rms = librosa.feature.rms(y=y)

    return {
        # Centroid
        "mean_centroid": np.mean(spectral_centroid),
        "std_centroid": np.std(spectral_centroid),
        # Bandwidth
        "mean_bandwidth": np.mean(spectral_bandwidth),
        "std_bandwidth": np.std(spectral_bandwidth),
        # Contrast
        "mean_contrast": np.mean(spectral_contrast),
        "std_contrast": np.std(spectral_contrast),
        # Flatness
        "mean_flatness": np.mean(spectral_flatness),
        "std_flatness": np.std(spectral_flatness),
        # Rolloff
        "mean_rolloff": np.mean(spectral_rolloff),
        "std_rolloff": np.std(spectral_rolloff),
        # Zero_crossing_rate
        "mean_zcr": np.mean(zcr),
        "std_zcr": np.std(zcr),
        # RMS
        "mean_rms": np.mean(rms),
        "std_rms": np.std(rms),
    }


def extract_features_by_id(file_id):
    """Extract features for the given file id."""

    print(f"Extract features for {file_id}...")
    y, sr = xeno_canto_api.client.load_recording(file_id)

    return extract_features(y, sr)


def bulk_extract_features(file_ids: list, n_processes=8):
    """Get extracted features for a list of file ids."""

    p = Pool(n_processes)
    features = p.map(extract_features_by_id, file_ids)
    df = pd.DataFrame(features)

    return df


class XGBoostModel:
    """Model that deals with tabular data extracted from an audio file."""

    def __init__(self):
        """Construct new XGBoostModel model."""

        self._model = None
        self._label_encoder = None

    def load(self) -> XGBClassifier:
        """Returns saved model if exists, else None."""

        # If model already loaded
        if self._model is not None:
            return self._model

        if not os.path.exists(SAVED_MODEL_PATH):
            raise Exception("No model exists.")

        self._model = joblib.load(SAVED_MODEL_PATH)
        self._label_encoder = joblib.load(SAVED_LABEL_ENCODER_PATH)

        return self._model, self._label_encoder

    def save(self, model: XGBClassifier, label_encoder: LabelEncoder):
        """Save model and label encoder to local directory."""

        joblib.dump(model, SAVED_MODEL_PATH)
        self._model = model

        joblib.dump(label_encoder, SAVED_LABEL_ENCODER_PATH)
        self._label_encoder = label_encoder

    def train(self, file_ids, labels, seed=1):
        """Train the model on the provided dataset."""

        feature_lists = bulk_extract_features(file_ids)

        print(feature_lists)
        #
        # X = df["features"]
        # y = df["name"]
        #
        # # Split training and test data
        # X_train, X_test, y_train, y_test = train_test_split(
        #     X, y, test_size=0.2, random_state=seed
        # )
        #
        # le = LabelEncoder().fit(y)
        # y_train = le.transform(y_train)
        # y_test = le.transform(y_test)
        #
        # model = XGBClassifier(objective="multi:softprob")
        # model.fit(X_train, y_train)
        #
        # self.evaluate(X_test, y_test)
        # self.save(model, le)

    def evaluate(self, X, y):
        """Evaluate model accuracy."""

        # TODO: Add methods for evaluating accuracy here.

    def predict(self, y, sr):
        """Predict class for given audio."""

        features = extract_features(y, sr)
        model = self.load()
        probabilities = model.predict_proba(features)

        return probabilities


if __name__ == "__main__":
    from data.selected_birds import birds

    ids = []
    labels = []

    for bird in birds:
        genus, subspecies = bird
        print("\ngenus:", genus)
        print("subspecies:", subspecies)
        data = xeno_canto_api.client.query(genus, subspecies)
        for recording in data["recordings"][:3]:
            ids.append(recording["id"])
            labels.append(genus + subspecies)

    model = XGBoostModel()
    model.train(file_ids=ids, labels=labels)

    # import random
    #
    # genus, subspecies = random.choice(birds)
    #
    # print("genus:", genus)
    # print("subspecies:", subspecies)
    #
    # data = xeno_canto_api.client.query(genus, subspecies)
    #
    # recording_id = data["recordings"][0]["id"]
    #
    # y, sr = xeno_canto_api.client.load_recording(recording_id)
    # print(y)
    #
    # model = XGBoostModel()
    #
    # features = model.extract_features(y=y, sr=sr)
    #
    # print(features)
