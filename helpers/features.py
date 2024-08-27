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


def get_spectral_centroids(y, sr):
    """Indicates where the center of mass of the spectrum is located."""
    
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
    
    return {
        'values': spectral_centroids,
        'mean': np.mean(spectral_centroids),
        'variance': np.std(spectral_centroids)
    }


def get_spectral_bandwidths(y, sr):
    """Indicates range of spectral mass."""
    
    spectral_bandwidths = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    
    return {
        'values': spectral_bandwidths,
        'mean': np.mean(spectral_bandwidths),
        'variance': np.std(spectral_bandwidths)
    }
    
    
def get_spectral_flatnesses(y, sr):
    """Indicates noisy-ness of signal."""
    
    spectral_flatnesses = librosa.feature.spectral_flatness(y=y)
    
    return {
        'values': spectral_flatnesses,
        'mean': np.mean(spectral_flatnesses),
        'variance': np.std(spectral_flatnesses)
    }
    
    
def get_spectral_constrasts(y, sr):
    """Indicates constrast between peaks and troughs."""
    
    spectral_contrasts = librosa.feature.spectral_contrast(y=y, sr=sr)
    
    return {
        'values': spectral_contrasts,
        'mean': np.mean(spectral_contrasts),
        'variance': np.std(spectral_contrasts)
    }


def get_zero_crossing_rates(y):
    """Indicates how often a signal changes sign."""
    
    zero_crossing_rates = librosa.feature.zero_crossing_rate(y)
    
    return {
        'values': zero_crossing_rates,
        'mean': np.mean(zero_crossing_rates),
        'variance': np.std(zero_crossing_rates)
    }
    

def get_chromas(y, sr):
    """Indicates the energy of different pitches."""
    
    chromas = librosa.feature.chroma_stft(y=y, sr=sr)
    
    return {
        'values': chromas,
        'mean': np.mean(chromas),
        'variance': np.std(chromas)
    }
    

def get_rms(y):
    """Indicates the energy of the signal."""
    
    rms = librosa.feature.rms(y=y)
    
    return {
        'values': rms,
        'mean': np.mean(rms),
        'variance': np.std(rms)
    }
    




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
        'spectral_centroid': get_spectral_centroids(y, sr),
        'spectral_bandwidth': get_spectral_bandwidths(y, sr),
        'mfcc': get_mfcc,
        'stft': get_stft(y),
        'avg_frequency_amplitudes': get_frequency_content(y)
    }
