import librosa
import os
import noisereduce as nr
import numpy as np
from data import xeno_canto_api
import pandas as pd
import pickle as pkl

from multiprocessing import Pool


CACHE_DIR = "data/tabular_data/cache"


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
    
    file_path = CACHE_DIR + f"/{file_id}.pkl"
    
    # Check if we have already extracted features for this file
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            print(f"Load features for {file_id} from cache...")
            return pkl.load(f)

    print(f"Extract features for {file_id}...")
    y, sr = xeno_canto_api.client.load_recording(file_id)
    print(f"Loaded recording for {file_id}...")
    feature = extract_features(y, sr)
    
    # Save to cache
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(file_path, "wb") as f:
        pkl.dump(feature, f)
    
    return feature


def bulk_extract_features(file_ids: list, n_processes=8):
    """Get extracted features for a list of file ids."""

    p = Pool(n_processes)
    features = p.map(extract_features_by_id, file_ids)
        
    df = pd.DataFrame(features)

    return df