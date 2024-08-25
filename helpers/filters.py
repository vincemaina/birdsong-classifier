from scipy.signal import butter, filtfilt

DEFAULT_CUTOFF_FREQUENCY = 150

def butter_highpass(cutoff, sr, order=5):
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def highpass_filter(data, sr, cutoff=DEFAULT_CUTOFF_FREQUENCY, order=5):
    b, a = butter_highpass(cutoff, sr, order=order)
    y_filtered = filtfilt(b, a, data)
    return y_filtered
