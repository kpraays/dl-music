import librosa as li
import numpy as np
from ddsp.core import extract_loudness, extract_pitch
import torch
import torchaudio
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

sampling_rate = 16000
block_size = 256
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sig, sr = li.load('./string_acoustic_048-060-100.wav', sr=sampling_rate)

pitch = extract_pitch(sig, sampling_rate, block_size)
loudness = extract_loudness(sig, sampling_rate, block_size)
mfcc = li.feature.mfcc(y=sig[:-1], sr=sampling_rate, n_mfcc=30, n_fft=1024, hop_length=256, n_mels=128, fmin=20, fmax=8000)
mfcc = np.transpose(mfcc, (1, 0))

sig = sig.reshape(-1, sig.shape[0])
pitch = pitch.reshape(sig.shape[0], -1)
loudness = loudness.reshape(sig.shape[0], -1)
mfcc = np.expand_dims(mfcc, axis=0)

sig = torch.from_numpy(sig).to(device)
pitch = torch.from_numpy(pitch).to(torch.float32).unsqueeze(-1).to(device)
loudness = torch.from_numpy(loudness).to(torch.float32).unsqueeze(-1).to(device)
mfcc = torch.from_numpy(mfcc).to(device)

model = torch.jit.load("./export/ddsp_mytraining_pretrained.ts")

audio = model(pitch, loudness, mfcc)
audio = audio.squeeze(-1)
torchaudio.save(
    "./demo_audio.wav", audio.detach(), sampling_rate,
    encoding="PCM_S", bits_per_sample=16)
