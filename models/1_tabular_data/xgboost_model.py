import joblib
import os
import noisereduce as nr
import librosa
import numpy as np


SAVED_MODEL_PATH = "models/1_tabular_data/xgboost_model.pkl"


class XGBoostModel:
    """Model that deals with tabular data extracted from an audio file."""

    def __init__(self):
        """Construct new XGBoostModel model."""

        self._model = self.load_model()

    def load_model(self):
        """Returns saved model if exists, else None."""

        if not os.path.exists(SAVED_MODEL_PATH):
            return None
        return joblib.load(SAVED_MODEL_PATH)

    def save_model(self, model):
        """Save model to local directory."""

        joblib.dump(model, SAVED_MODEL_PATH)

    def extract_features(self, y, sr):
        """Extract required features for this model."""

        y = nr.reduce_noise(y, sr)  # Apply spectral gate
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

    def train_model(self):
        """Train the model on the provided dataset."""

        pass

    def predict(self):
        pass


if __name__ == "__main__":
    from data.xeno_canto_api import client

    y, sr = client.load_recording(900826)
    print(y)

    model = XGBoostModel()

    features = model.extract_features(y=y, sr=sr)

    print("Extracted features:", features)
