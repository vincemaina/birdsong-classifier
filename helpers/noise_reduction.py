from numpy import ndarray
import noisereduce as nr


def apply_spectral_gate(y, sr) -> ndarray:
    """Apply spectral gating to reduce noise"""
    y = nr.reduce_noise(y, sr)
    return y


if __name__ == "__main__":
    import librosa

    # Load audio file
    y, sr = librosa.load("data/bird_recordings/25745")

    # Apply spectral gating to reduce noise
    y = apply_spectral_gate(y, sr)
