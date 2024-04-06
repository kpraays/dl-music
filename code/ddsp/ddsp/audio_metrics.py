import numpy as np
from librosa import hz_to_midi

def signal_to_noise_ratio(ori_signal, rec_signal):
    return 20*np.log10(np.linalg.norm(ori_signal) / \
                       np.linalg.norm(ori_signal - rec_signal))

def pitch_distance(ori_pitch, rec_pitch):
    return np.linalg.norm(hz_to_midi(rec_pitch) - hz_to_midi(ori_pitch), ord=1) / ori_pitch.size

def loudness_distance(ori_loudness, rec_loudness):
    return np.linalg.norm(rec_loudness - ori_loudness, ord=1) / \
        ori_loudness.size

def timbre_distance(ori_timbre, rec_timbre):
    return np.linalg.norm(rec_timbre - ori_timbre, ord=1) / \
        ori_timbre.size
