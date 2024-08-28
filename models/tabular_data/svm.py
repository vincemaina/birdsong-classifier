"""
Support vector model for classifying birdsbased based on tabular data extracted from recordings.
"""

import os
import numpy as np
import pandas as pd
import optuna
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, top_k_accuracy_score
from warnings import simplefilter

simplefilter("ignore", category=FutureWarning)


SEED = 1  # For random states
SAVED_MODEL_PATH = "models/tabular_data/"
MODEL_NAME = "svm_model.pkl"
LE_NAME = "svm_le.pkl"
SCALER_NAME = "svm_scaler.pkl"


def evaluate_model(X_test, y_test, model, le) -> None:
    """Evaluate the SVM models accuracy."""

    # Predict probabilities for the test set
    y_proba = model.predict_proba(X_test)

    # Display predictions against actual labels
    df = pd.DataFrame(
        {
            "Predictions": le.inverse_transform(np.argmax(y_proba, axis=1)),
            "Actual": le.inverse_transform(y_test),
        }
    )
    df["Correct"] = df["Predictions"] == df["Actual"]
    print(df.sample(20))

    # Display classification report
    cf = classification_report(
        le.inverse_transform(y_test),
        le.inverse_transform(np.argmax(y_proba, axis=1)),
        labels=np.unique(le.inverse_transform(np.argmax(y_proba, axis=1))),
    )
    print("\nClassification report:\n\n", cf)

    # Display top_k_accuracy_score
    for k in [1, 2, 3]:
        top_k_accuracy = top_k_accuracy_score(y_test, y_proba, k=k)
        print(f"Top-{k} Accuracy: {top_k_accuracy * 100 :.2f}%")


class SVM:
    """Represents an SVM classifier model."""

    def __init__(self) -> None:
        """Construct new SVM instance."""

        print("New SVM instance.")

        self._model = None
        self._label_encoder = None
        self._scaler = None

    def save_model(self, model, le, scaler):
        """Save model to disk."""

        joblib.dump(model, SAVED_MODEL_PATH + MODEL_NAME)
        self._model = model

        joblib.dump(le, SAVED_MODEL_PATH + LE_NAME)
        self._label_encoder = le

        joblib.dump(scaler, SAVED_MODEL_PATH + SCALER_NAME)
        self._scaler = scaler

    def load_model(self):
        """Load model from disk."""

        # If model already loaded
        if self._model is not None:
            return self._model, self._label_encoder, self._scaler

        if not os.path.exists(SAVED_MODEL_PATH + MODEL_NAME):
            raise Exception("No model exists.")

        self._model = joblib.load(SAVED_MODEL_PATH + MODEL_NAME)
        self._label_encoder = joblib.load(SAVED_MODEL_PATH + LE_NAME)
        self._scaler = joblib.load(SAVED_MODEL_PATH + SCALER_NAME)

        return self._model, self._label_encoder, self._scaler

    def train(self, X, y, tuning=True):
        """Train the model on the provided dataset."""

        # Encode as they are strings
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # Split training and test data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=SEED
        )

        # Initialize SVM model with RBF kernel
        svm_model = SVC(kernel="rbf", probability=True)

        # Calibrate to output probabilities
        calibrated_svm = CalibratedClassifierCV(svm_model, cv=5)
        calibrated_svm.fit(X_train, y_train)

        # Evaluate accuracy of the new model
        evaluate_model(X_test, y_test, calibrated_svm, le)

        if not tuning:
            return

        # Hyper parameter tuning

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        def objective(trial):
            if trial.number % 10 == 0:
                print("Trial:", trial.number)

            # Suggest values for hyperparameters
            C = trial.suggest_loguniform("C", 1e-3, 1e2)
            gamma = trial.suggest_loguniform("gamma", 1e-4, 1e1)
            kernel = trial.suggest_categorical(
                "kernel", ["rbf", "linear", "poly", "sigmoid"]
            )

            # Create model
            model = SVC(C=C, gamma=gamma, kernel=kernel)

            # Perform cross-validation and return mean accuracy
            accuracy = cross_val_score(model, X_train, y_train, cv=5).mean()

            return accuracy

        # Create a study and optimize
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=100)

        # Print best hyperparameters
        print("Best hyperparameters:", study.best_params)
        print("Best cross-validation accuracy:", study.best_value)

        # Train the final model on the entire training set with the best hyperparameters
        best_model = SVC(**study.best_params)

        # Calibrate to output probabilities
        calibrated_svm = CalibratedClassifierCV(best_model, cv=5)
        calibrated_svm.fit(X_train, y_train)

        evaluate_model(X_test, y_test, calibrated_svm, le)

        self.save_model(calibrated_svm, le, scaler)

    def evaluate(self, X_test, y_test):
        """Evaluate the current model."""

        model, le, scaler = self.load_model()
        X_test = scaler.transform(X_test)  # type: ignore
        y_test = le.transform(y_test)  # type: ignore
        evaluate_model(X_test, y_test, model, le)

    def predict_proba(self, X):
        """Make a prediction."""

        model, le, scaler = self.load_model()

        return model.predict_proba(X), le


if __name__ == "__main__":
    # Debug

    import pandas as pd
    from data.selected_birds import birds
    from data.xeno_canto_api import client
    from helpers.time import time_to_seconds
    from data.tabular_data import bulk_extract_features, extract_features_by_id

    mode = "Evaluate"

    if mode == "Train":
        model = SVM()

        ids = []
        labels = []

        for bird in birds:
            genus, subspecies = bird
            print("\ngenus:", genus)
            print("subspecies:", subspecies)
            data = client.query(genus, subspecies)
            print("Number of recordings:", data["numRecordings"])
            recordings = [
                i for i in data["recordings"] if time_to_seconds(i["length"]) < 300
            ]
            for recording in data["recordings"][:100]:
                ids.append(recording["id"])
                labels.append(genus + subspecies)

        X = bulk_extract_features(ids)

        y = pd.DataFrame({"name": labels})["name"]

        model.train(X, y)

    elif mode == "Evaluate":
        print("Evaulating accuracy of the current model.")

        model = SVM()

        ids = []
        labels = []

        for bird in birds:
            genus, subspecies = bird
            print("\ngenus:", genus)
            print("subspecies:", subspecies)
            data = client.query(genus, subspecies)
            print("Number of recordings:", data["numRecordings"])
            recordings = [
                i for i in data["recordings"] if time_to_seconds(i["length"]) < 300
            ]
            for recording in data["recordings"][95:100]:
                ids.append(recording["id"])
                labels.append(genus + subspecies)

        X = bulk_extract_features(ids)

        y = pd.DataFrame({"name": labels})["name"]

        print(X.head())

        model.evaluate(X, y)

    elif mode == "Predict":
        import sys

        print("Predict class of something.")

        file_id = sys.argv[1]
        features = extract_features_by_id(file_id)

        df = pd.DataFrame(features, index=[0])  # type: ignore

        model = SVM()
        y_pred, le = model.predict_proba(df)
        print(y_pred)
        print(le.inverse_transform(np.argmax(y_pred, axis=1)))  # type: ignore

    else:
        raise Exception("Invalid mode given.")
