import numpy as np
import librosa

def extract_mfcc(audio, sr, n_mfcc=13, hop_length=512, n_fft=2048):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc,
                                hop_length=hop_length, n_fft=n_fft)
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
    return mfcc.T  # (time, features)
