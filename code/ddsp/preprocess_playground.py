import librosa as li
from ddsp.get_audio_descriptors import spectral_centroid, spectral_flatness, temporal_centroid
from ddsp.core import extract_loudness, extract_pitch
import numpy as np
import os

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    sampling_rate = 16000
    block_size = 256
    signal_length = 64000
    x, sr = li.load('D:/Music/Datasets/nsynth-valid-flute/audio/flute_acoustic_002-077-075.wav', sr=sampling_rate)
    N = (signal_length - len(x) % signal_length) % signal_length
    x = np.pad(x, (0, N))

    print(x.shape)
    print(N)
    
    pitch = extract_pitch(x, sampling_rate, block_size)
    loudness = extract_loudness(x, sampling_rate, block_size)
    mfcc = li.feature.mfcc(y=x[:-1], sr=sampling_rate, n_mfcc=30, n_fft=1024, hop_length=256, n_mels=128, fmin=20, fmax=8000)
    mfcc = np.transpose(mfcc, (1, 0))
    spec_centroid = spectral_centroid(x, sampling_rate, 256, 128)
    spec_flatness = spectral_flatness(x, 256, 128)
    tempo_centroid = temporal_centroid(x, sampling_rate, 128, 64)
    timbre_descriptors = np.array([spec_centroid, spec_flatness, tempo_centroid])
    print(pitch.shape)
    print(loudness.shape)
    print(mfcc.shape)

    x = x.reshape(-1, signal_length)
    pitch = pitch.reshape(x.shape[0], -1)
    loudness = loudness.reshape(x.shape[0], -1)
    mfcc = np.expand_dims(mfcc, axis=0)
    timbre_descriptors = np.expand_dims(timbre_descriptors, axis=0)

    print(x.shape)
    print(pitch.shape)
    print(loudness.shape)
    print(mfcc.shape)
    print(timbre_descriptors.shape)

if __name__ == "__main__":
    main()