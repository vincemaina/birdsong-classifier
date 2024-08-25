import librosa
import numpy as np

def get_spectogram(y: np.ndarray, sr: float) -> np.ndarray:
    """Returns melspectrogram for input audio."""
    
    return librosa.feature.melspectrogram(y=y, sr=sr)

def get_fft(y, sr):
    """Returns a FFT for the input audio."""
       
    # Perform the FFT
    fft_result = np.fft.fft(y)
    
    # Get the frequency values corresponding to the FFT result
    frequencies = np.fft.fftfreq(len(fft_result), 1/sr)
    
    # Since the FFT result is symmetrical, we take only the positive half
    magnitude = np.abs(fft_result)
    frequencies = frequencies[:len(frequencies)//2]
    magnitude = magnitude[:len(magnitude)//2]
    
    # Normalize the magnitudes to a range of 0 to 1
    magnitude = magnitude / np.max(magnitude)
    
    return magnitude, frequencies

def get_spectral_centroid(y, sr):
    return librosa.feature.spectral_centroid(y=y, sr=sr)

def get_spectral_bandwidth(y, sr):
    return librosa.feature.spectral_bandwidth(y=y, sr=sr)

def get_mfcc(y, sr):
    return librosa.feature.mfcc(y=y, sr=sr)

def get_stft(y):
    return librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

def get_frequency_content(y):
    # Compute the STFT with fixed n_fft
    N_FFT = 1024
    D = librosa.stft(y, n_fft=N_FFT)

    # Compute the magnitude (absolute value)
    magnitude = np.abs(D)

    # Compute the average amplitude for each frequency bin
    average_amplitude = np.mean(magnitude, axis=1)
    
    # Normalize the average amplitudes so the maximum amplitude is 1
    normalized_amplitude = average_amplitude / np.max(average_amplitude)
    
    return {
        'average_amplitudes': normalized_amplitude,
        'stft': D
    }

def get_features(y, sr):
    return {
        'spectrogram': get_spectogram(y, sr),
        'fft': get_fft(y, sr),
        'spectral_centroid': get_spectral_centroid(y, sr),
        'spectral_bandwidth': get_spectral_bandwidth(y, sr),
        'mfcc': get_mfcc,
        'stft': get_stft(y),
        'avg_frequency_amplitudes': get_frequency_content(y)
    }
