from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical  # type: ignore


def encode_labels(y):
    le = LabelEncoder()
    y_transformed = le.fit_transform(y)
    y_categorised = to_categorical(y_transformed)
    return y_categorised
