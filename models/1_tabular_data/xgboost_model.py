import os
import joblib
import numpy as np
import pandas as pd
import optuna
from warnings import simplefilter
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import top_k_accuracy_score, accuracy_score, log_loss
from sklearn.metrics import classification_report
from data import xeno_canto_api, tabular_data
from helpers.time import time_to_seconds

simplefilter("ignore", category=FutureWarning)


SAVED_MODEL_PATH = "models/1_tabular_data/xgboost_model.pkl"
SAVED_LABEL_ENCODER_PATH = "models/1_tabular_data/xgboost_label_encoder.pkl"


def evaluate(X_test, y_test, model: XGBClassifier, le):
    """Evaluate model accuracy."""

    y_pred = model.predict_proba(X_test)

    print("\nNum predictions:", len(y_pred))
    print("Num labels:", len(y_test))

    df = pd.DataFrame(
        {
            "Predictions": le.inverse_transform(np.argmax(y_pred, axis=1)),
            "True": le.inverse_transform(y_test),
            "Correct": np.argmax(y_pred, axis=1) == y_test,
            "Confidence": np.max(y_pred, axis=1),
        }
    )

    print(df.sample(20))

    print("\nAccuracy score:", accuracy_score(y_test, np.argmax(y_pred, axis=1)))
    print("Log loss:", log_loss(y_test, y_pred, labels=np.arange(10)))

    k = 3
    top_k_score = top_k_accuracy_score(y_test, y_pred, k=k, labels=np.arange(10))
    print(f"Top {k} accuracy score:", top_k_score, end="\n\n")

    cr = classification_report(
        le.inverse_transform(y_test), le.inverse_transform(np.argmax(y_pred, axis=1))
    )
    print(cr)

    return top_k_score


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

    def train(self, X, y, seed=1):
        """Train the model on the provided dataset."""

        print(y.head())

        # Split training and test data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )

        le = LabelEncoder()
        le = le.fit(y)
        y_train = le.transform(y_train)
        y_test = le.transform(y_test)

        model = XGBClassifier(objective="multi:softprob")
        model.fit(X_train, y_train)

        print("Finished training!")

        evaluate(X_test, y_test, model, le)

        # Hyperparameter optimization

        def objective(trial):
            if trial.number % 10 == 0:
                print("Trial:", trial.number)

            # Define hyperparameters to tune
            param = {
                "objective": "multi:softprob",
                "n_estimators": trial.suggest_int("n_estimators", 10, 500),
                "max_depth": trial.suggest_int("max_depth", 1, 20),
                "learning_rate": trial.suggest_loguniform("learning_rate", 0.001, 0.5),
            }

            # Train the model
            model = XGBClassifier(**param)
            model.fit(X_train, y_train)

            # Predict probabilities
            probabilities = model.predict_proba(X_test)

            return top_k_accuracy_score(
                y_test, probabilities, k=3, labels=np.arange(10)
            )

        optuna.logging.set_verbosity(optuna.logging.ERROR)

        # Create a study and optimize
        study = optuna.create_study(direction="maximize")  # Minimize log loss
        study.optimize(objective, n_trials=100)  # type: ignore

        # Get the best parameters
        best_params = study.best_params
        print("Best Parameters:", best_params)
        print("Number of finished trials:", len(study.trials))
        print("Best score: %.4f" % study.best_value)

        # Get the best model
        best_model = XGBClassifier(**best_params)
        best_model.fit(X_train, y_train)

        evaluate(X_test, y_test, best_model, le)

        self.save(best_model, le)

    def predict_proba(self, y, sr):
        """Predict class for given audio."""

        features = tabular_data.extract_features(y, sr)

        df = pd.DataFrame(features, index=[0])  # type: ignore

        print(df.head(), end="\n\n")

        model, le = self.load()
        probabilities = model.predict_proba(df)[0]

        prob_map = {}

        for i in range(len(probabilities)):
            bird_name = le.inverse_transform([i])[0]
            if bird_name in prob_map:
                raise Exception("Duplicate.")
            prob_map[bird_name] = float(probabilities[i])

        sorted_map = sorted(prob_map.items(), key=lambda x: x[1], reverse=True)

        for i in sorted_map:
            print(i)

        return probabilities

    def evaluate(self, X_test, y_test):
        """Evaluate the current model."""

        model, le = self.load()
        y_test = le.transform(y_test)
        evaluate(X_test, y_test, model, le)


if __name__ == "__main__":
    from data.selected_birds import birds

    mode = "evaluate"

    if mode == "train":
        ids = []
        labels = []

        for bird in birds:
            genus, subspecies = bird
            print("\ngenus:", genus)
            print("subspecies:", subspecies)
            data = xeno_canto_api.client.query(genus, subspecies)
            print("Number of recordings:", data["numRecordings"])
            recordings = [
                i for i in data["recordings"] if time_to_seconds(i["length"]) < 300
            ]
            for recording in data["recordings"][:50]:
                ids.append(recording["id"])
                labels.append(genus + subspecies)

        model = XGBoostModel()

        PATH = "models/1_tabular_data/features.pkl.xz"

        X = tabular_data.bulk_extract_features(ids)

        y = pd.DataFrame({"name": labels})["name"]

        model.train(X, y)

    elif mode == "predict":
        import random

        genus, subspecies = random.choice(birds)

        print("genus:", genus)
        print("subspecies:", subspecies)

        data = xeno_canto_api.client.query(genus, subspecies)

        print("Number of recordings:", data["numRecordings"])

        recording_id = data["recordings"][-1]["id"]

        y, sr = xeno_canto_api.client.load_recording(recording_id)

        model = XGBoostModel()

        prediction = model.predict_proba(y, sr)

    elif mode == "evaluate":
        print("Evaluate model performance.")

        model = XGBoostModel()

        ids = []
        labels = []

        for bird in birds:
            genus, subspecies = bird
            print("\ngenus:", genus)
            print("subspecies:", subspecies)
            data = xeno_canto_api.client.query(genus, subspecies)
            print("Number of recordings:", data["numRecordings"])
            recordings = [
                i for i in data["recordings"] if time_to_seconds(i["length"]) < 300
            ]
            for recording in data["recordings"][50:80]:
                ids.append(recording["id"])
                labels.append(genus + subspecies)

        model = XGBoostModel()

        PATH = "models/1_tabular_data/features.pkl.xz"

        X = tabular_data.bulk_extract_features(ids)

        y = pd.DataFrame({"name": labels})["name"]

        model.evaluate(X, y)

    else:
        raise Exception("Invalid mode selected.")
