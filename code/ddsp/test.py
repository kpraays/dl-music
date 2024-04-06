import numpy as np
import torch
from ddsp.audio_metrics import signal_to_noise_ratio, pitch_distance, loudness_distance, timbre_distance
from librosa import hz_to_midi
from ddsp.model import DDSP
from ddsp.core import extract_loudness, extract_pitch
import yaml
from preprocess import Dataset
from tqdm import tqdm
from ddsp.get_audio_descriptors import get_all_descriptors
import io
import sys
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open("config.yaml", "r") as config:
    config = yaml.safe_load(config)

sampling_rate = config["preprocess"]["sampling_rate"]
block_size = config["preprocess"]["block_size"]

model = torch.jit.load(config["test"]["pretrained_model_location"]).to(device)
model.eval() # enable the evaluation mode

dataset = Dataset(config["test"]["data_dir"])

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    drop_last=True,
)

snr_list = []
pitch_dist_list = []
loudness_dist_list = []
ori_s_centroid = []
ori_s_flatness = []
ori_t_centroid = []
rec_s_centroid = []
rec_s_flatness = []
rec_t_centroid = []

t = tqdm(total=len(dataloader))
for signal, pitch, loudness, mfcc, timbre, source in dataloader:
    # prepare original information
    ori_signal = signal.squeeze().detach().numpy()
    ori_pitch = pitch.squeeze().detach().numpy()
    ori_loudness = loudness.squeeze().detach().numpy()
    ori_timbre = timbre.squeeze().detach().numpy()

    # prepare data for sending to the model
    signal = signal.to(device)
    pitch = pitch.unsqueeze(-1).to(device)
    loudness = loudness.unsqueeze(-1).to(device)
    mfcc = mfcc.to(device)
    timbre = timbre.to(device)
    timbre = timbre.unsqueeze(1).repeat(1, pitch.size(1), 1)
    source = source.to(device)

    with torch.no_grad():
        y = model(pitch, loudness, mfcc, timbre, source).squeeze(-1)
    
        rec_signal = y.squeeze().detach().cpu().numpy()
        sys.stdout = io.StringIO()
        rec_pitch = extract_pitch(rec_signal, sampling_rate, block_size)
        sys.stdout = sys.__stdout__
        rec_loudness = extract_loudness(rec_signal, sampling_rate, block_size)
        rec_timbre = get_all_descriptors(rec_signal, sampling_rate, 256, 128)
        
        snr = signal_to_noise_ratio(ori_signal, rec_signal)
        pitch_dist = pitch_distance(ori_pitch, rec_pitch)
        loudness_dist = loudness_distance(ori_loudness, rec_loudness)
        timbre_dist = timbre_distance(ori_timbre, rec_timbre)

        snr_list.append(snr)
        pitch_dist_list.append(pitch_dist)
        loudness_dist_list.append(loudness_dist)
        ori_s_centroid.append(ori_timbre[0])
        ori_s_flatness.append(ori_timbre[1])
        ori_t_centroid.append(ori_timbre[2])
        rec_s_centroid.append(rec_timbre[0])
        rec_s_flatness.append(rec_timbre[1])
        rec_t_centroid.append(rec_timbre[2])

        t.update()

t.close()

get_l1_dist = lambda ori_val, rec_val: np.abs(rec_val - ori_val)

mean_snr = np.array(snr_list).mean()
mean_pitch_dist = np.array(pitch_dist_list).mean()
mean_loudness_dist = np.array(loudness_dist_list).mean()
mean_s_centroid_dist = get_l1_dist(hz_to_midi(np.array(ori_s_centroid)), hz_to_midi(np.array(rec_s_centroid))).mean()
mean_s_flatness_dist = get_l1_dist(np.array(ori_s_flatness), np.array(rec_s_flatness)).mean()
mean_t_centroid_dist = get_l1_dist(np.array(ori_t_centroid), np.array(rec_t_centroid)).mean()

print(f"SNR = {mean_snr:.4f}",
      f"L1 Pitch Distance = {mean_pitch_dist:.4f} (note)",
      f"L1 Loudness Distance = {mean_loudness_dist:.4f}",
      f"L1 Spectral Centroid Distance = {mean_s_centroid_dist:.4f} (note)",
      f"L1 Spectral Flatness Distance = {mean_s_flatness_dist:.4f}",
      f"L1 Temporal Centroid Distance = {mean_t_centroid_dist:.4f} (sec)",
      sep="\n")
