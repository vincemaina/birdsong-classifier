import librosa

SAMPLE_RATE = 16000

def load_audio(id):
    return librosa.load(f"data/bird_recordings/{id}", mono=True, sr=SAMPLE_RATE)

def normalize_audio(y):
    return librosa.util.normalize(y)
