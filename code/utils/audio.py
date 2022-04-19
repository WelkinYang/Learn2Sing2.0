import librosa
from scipy.io import wavfile
import soundfile as sf
import sys

def load_wav(wav_path, sr=16000):
    audio = librosa.core.load(wav_path, sr=sr)[0]
    return audio

def save_wav(wav, path, hparams, norm=False):
    if norm:
        wav *= 32767 / max(0.01, np.max(np.abs(wav)))
        wavfile.write(path, hparams.sample_rate, wav.astype(np.int16))
    else:
        sf.write(path, wav, hparams.sample_rate)

