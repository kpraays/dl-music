import numpy as np
import librosa

def amplitude_envelope(signal, window_size, hop_size): 
    return np.array(
        [max(signal[i:i+window_size]) for i in range(0, len(signal), hop_size)])

def temporal_centroid(signal, sample_rate, window_size, hop_size):
    envelope = amplitude_envelope(signal, window_size, hop_size)
    window_per_second = sample_rate / hop_size
    return np.sum(np.arange(1, len(envelope)+1) * envelope) / np.sum(envelope) / window_per_second

def spectral_centroid(signal, sample_rate, window_size, hop_size):
    time_variant_sc = librosa.feature.spectral_centroid(
        y=signal, sr=sample_rate, n_fft=window_size, hop_length=hop_size,
        window='hann', center=True)
    return np.median(time_variant_sc)

def spectral_flatness(signal, window_size, hop_size):
    time_variant_sf = librosa.feature.spectral_flatness(
        y=signal, n_fft=window_size, hop_length=hop_size,
        window='hann', center=True)
    return np.median(time_variant_sf)
