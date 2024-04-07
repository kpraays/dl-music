import numpy as np
import librosa
from scipy.signal import butter, sosfilt, hilbert

def amplitude_envelope(signal, window_size, hop_size):
    return np.array(
        [max(signal[i:i+window_size]) for i in range(0, len(signal), hop_size)])

def temporal_centroid(signal, sample_rate, window_size, hop_size):
    # envelope = amplitude_envelope(signal, window_size, hop_size)
    # window_per_second = sample_rate / hop_size
    # return np.sum(np.arange(1, len(envelope)+1) * envelope) / np.sum(envelope) / window_per_second
    n_samples = signal.shape[0]
    sos = butter(2, 10, "lowpass", fs=sample_rate, output="sos")
    filtered_signal = sosfilt(sos, np.abs(signal))
    temporal_centroid = np.sum(np.arange(1, n_samples+1) * filtered_signal) \
        / np.sum(filtered_signal) / sample_rate
    return temporal_centroid

def spectral_centroid(signals, sample_rate, window_size, hop_size):
    time_variant_sc = librosa.feature.spectral_centroid(
        y=signals, sr=sample_rate, n_fft=window_size, hop_length=hop_size,
        window='hann', center=True)
    time_variant_sc[time_variant_sc < 5] = np.nan
    return np.nanmedian(time_variant_sc, axis=-1) if signals.ndim > 1 \
        else np.nanmedian(time_variant_sc)

def spectral_flatness(signals, window_size, hop_size):
    time_variant_sf = librosa.feature.spectral_flatness(
        y=signals, n_fft=window_size, hop_length=hop_size,
        window='hann', center=True)
    time_variant_sf[np.logical_or(time_variant_sf < 0.00001, time_variant_sf > 0.9999)] = np.nan
    return np.nanmedian(time_variant_sf, axis=-1) if signals.ndim > 1 \
        else np.nanmedian(time_variant_sf)

def get_all_descriptors(signal, sample_rate, window_size, hop_size):
    S = np.abs(librosa.magphase(librosa.stft(
        y=signal, n_fft=window_size, hop_length=hop_size,
        window='hann', center=True)))
    time_variant_sc = librosa.feature.spectral_centroid(S=S)
    time_variant_sf = librosa.feature.spectral_flatness(S=S)
    time_variant_sc[time_variant_sc < 5] = np.nan
    time_variant_sf[np.logical_or(time_variant_sf < 0.00001, time_variant_sf > 0.99995)] = np.nan
    s_centroid = np.nanmedian(time_variant_sc)
    s_flatness = np.nanmedian(time_variant_sf)
    t_centroid = temporal_centroid(signal, sample_rate, window_size, hop_size)
    return np.array([s_centroid, s_flatness, t_centroid])
    

def temporal_centroid_batch(signals, sample_rate):
    n_channels, n_samples = signals.shape

    # hilbert method from Timbre Toolbox (Peeters et al. 2011)
    # sos = butter(3, 5, "lowpass", fs=sample_rate, output="sos")
    # filtered_signals = sosfilt(sos, np.abs(hilbert(signals)))

    # filtering method from Tarjano and Pereira (2022)
    # https://www.sciencedirect.com/science/article/pii/S1051200421002682
    sos = butter(2, 10, "lowpass", fs=sample_rate, output="sos")
    filtered_signals = sosfilt(sos, np.abs(signals))
    temporal_centroid = np.sum(np.arange(1, n_samples+1) * filtered_signals, axis=-1) \
        / np.sum(filtered_signals, axis=-1) / sample_rate
    return np.expand_dims(temporal_centroid, -1)