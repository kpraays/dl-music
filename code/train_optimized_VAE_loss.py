import torch
from torch.utils.tensorboard import SummaryWriter
import yaml
from ddsp.model import DDSP
from effortless_config import Config
from os import path
from preprocess import Dataset
from tqdm import tqdm
from ddsp.get_audio_descriptors import spectral_centroid, spectral_flatness, temporal_centroid, temporal_centroid_batch, get_all_descriptors
from ddsp.core import multiscale_fft, safe_log, mean_std_loudness
import soundfile as sf
from einops import rearrange
from ddsp.utils import get_scheduler
import numpy as np
import pandas as pd
import torch.nn.functional as F

def spectral_distance(x, y, scales, eps=1e-8):
    """
    Computes the multiscale spectral distance between two waveforms.
    
    Parameters:
    - x, y: Input waveforms.
    - scales: A list of scales (window sizes) to compute the STFT.
    - eps: A small value to avoid division by zero.
    
    Returns:
    - Multiscale spectral distance between x and y.
    """
    distance = 0.0
    for scale in scales:
        hop_length = scale // 4
    
        X = torch.stft(x, n_fft=scale, hop_length=hop_length, window = torch.hann_window(scale).to(signal), return_complex=True)
        Y = torch.stft(y, n_fft=scale, hop_length=hop_length, window = torch.hann_window(scale).to(signal),return_complex=True)
        
        magnitude_X = torch.abs(X)
        magnitude_Y = torch.abs(Y)
        difference =  magnitude_X - magnitude_Y
        
        frobenius_norm = torch.norm(difference, p='fro')
        l1_norm = torch.norm(difference, p=1)
        
        scale_distance = frobenius_norm / (torch.norm(X, p='fro') + eps)
        scale_distance += torch.log(l1_norm + eps)
        
        distance += scale_distance
    return distance

class args(Config):
    CONFIG = "config.yaml"
    NAME = "debug"
    ROOT = "runs"
    STEPS = 500000
    BATCH = 16
    START_LR = 1e-3
    STOP_LR = 1e-4
    DECAY_OVER = 400000


args.parse_args()

with open(args.CONFIG, "r") as config:
    config = yaml.safe_load(config)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = DDSP(**config["model"]).to(device)

dataset = Dataset(config["preprocess"]["out_dir"])
# dataset_valid = Dataset(config["validation"]["out_dir"])

dataloader = torch.utils.data.DataLoader(
    dataset,
    args.BATCH,
    True,
    drop_last=True,
)

mean_loudness, std_loudness = mean_std_loudness(dataloader)
config["data"]["mean_loudness"] = mean_loudness
config["data"]["std_loudness"] = std_loudness

writer = SummaryWriter(path.join(args.ROOT, args.NAME), flush_secs=20)

with open(path.join(args.ROOT, args.NAME, "config.yaml"), "w") as out_config:
    yaml.safe_dump(config, out_config)

opt = torch.optim.Adam(model.parameters(), lr=args.START_LR)

schedule = get_scheduler(
    len(dataloader),
    args.START_LR,
    args.STOP_LR,
    args.DECAY_OVER,
)

# scheduler = torch.optim.lr_scheduler.LambdaLR(opt, schedule)

best_loss = float("inf")
mean_loss = 0
n_element = 0
step = 0
epochs = int(np.ceil(args.STEPS / len(dataloader)))

timbre_loss_factor = 1
loss_array = []
for e in tqdm(range(epochs)):
    for signal, pitch, loudness, mfcc, timbre, source in dataloader:
        signal = signal.to(device)
        pitch = pitch.unsqueeze(-1).to(device)
        loudness = loudness.unsqueeze(-1).to(device)
        mfcc = mfcc.to(device)
        timbre = timbre.to(device)
        ori_timbre = timbre.detach().cpu().numpy()
        timbre = timbre.unsqueeze(1).repeat(1, pitch.size(1), 1)
        source = source.to(device)
        
        loudness = (loudness - mean_loudness) / std_loudness
        # mask = source!=2
        # for value in source[mask]:
        #     print(value)
        # print("first batch")
        # filter based on source here - 0, 1 (not giving synthetic)
        y,logvar,mu = model(pitch, loudness, mfcc, timbre, source)
        y = y.squeeze(-1)
        rec_signals = y.detach().cpu().numpy()
        
        # parallel version, way slower than non-parallel version
        # rec_spec_centroid = spectral_centroid(rec_signals, config["preprocess"]["sampling_rate"], 256, 128)
        # rec_spec_flatness = spectral_flatness(rec_signals, 256, 128)
        # rec_t_centroid = temporal_centroid_batch(rec_signals, config["preprocess"]["sampling_rate"])
        # rec_timbre = np.concatenate((rec_spec_centroid, rec_spec_flatness, rec_t_centroid), axis=-1)
        # rec_timbre = torch.from_numpy(rec_timbre).to(device)

        rec_timbre = np.zeros_like(ori_timbre)
        for i in range(rec_signals.shape[0]):
            # rec_sig = rec_signals[i]
            # rec_spec_centroid = spectral_centroid(rec_sig, config["preprocess"]["sampling_rate"], 256, 128)
            # rec_spec_flatness = spectral_flatness(rec_sig, 256, 128)
            # rec_tempo_centroid = temporal_centroid(rec_sig, config["preprocess"]["sampling_rate"], 128, 64)
            # rec_timbre[i] = np.array([rec_spec_centroid, rec_spec_flatness, rec_tempo_centroid])
            rec_timbre[i] = get_all_descriptors(rec_signals[i], config["preprocess"]["sampling_rate"], 256, 128)
        # rec_timbre = torch.from_numpy(rec_timbre).to(device)

        ori_erb_spec_centroid = 1000/(24.7*4.37) * np.log(4.37*ori_timbre[:, 0]/1000 + 1 + 1e-7)
        rec_erb_spec_centroid = 1000/(24.7*4.37) * np.log(4.37*rec_timbre[:, 0]/1000 + 1 + 1e-7)
        ori_spec_flatness = ori_timbre[:, 1]
        rec_spec_flatness = rec_timbre[:, 1]
        ori_temporal_centroid = ori_timbre[:, 2]
        rec_temporal_centroid = rec_timbre[:, 2]
        timbre_loss = np.abs(rec_erb_spec_centroid - ori_erb_spec_centroid).mean() \
            + np.abs(rec_spec_flatness - ori_spec_flatness).mean() \
            + np.abs(rec_temporal_centroid - ori_temporal_centroid).mean()

        # ori_stft = multiscale_fft(
        #     signal,
        #     config["train"]["scales"],
        #     config["train"]["overlap"],
        # )
        # rec_stft = multiscale_fft(
        #     y,
        #     config["train"]["scales"],
        #     config["train"]["overlap"],
        # )

        loss = 0
        # for s_x, s_y in zip(ori_stft, rec_stft):
        #     lin_loss = (s_x - s_y).abs().mean()
        #     log_loss = (safe_log(s_x) - safe_log(s_y)).abs().mean()
        #     loss = loss + lin_loss + log_loss

        spectral_dist = spectral_distance(signal, y, config["train"]["scales"])
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = spectral_dist/16 + 0.5 * kl_divergence/16

        loss = loss + timbre_loss_factor * timbre_loss
        loss_array.append(loss.item())

        opt.zero_grad()
        loss.backward()
        opt.step()

        writer.add_scalar("loss", loss.item(), step)

        step += 1

        n_element += 1
        mean_loss += (loss.item() - mean_loss) / n_element

    if not e % 10:
        writer.add_scalar("lr", schedule(e), e)
        writer.add_scalar("reverb_decay", model.reverb.decay.item(), e)
        writer.add_scalar("reverb_wet", model.reverb.wet.item(), e)
        # scheduler.step()
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(
                model.state_dict(),
                path.join(args.ROOT, args.NAME, "state.pth"),
            )

        mean_loss = 0
        n_element = 0

        audio = torch.cat([signal, y], -1).reshape(-1).detach().cpu().numpy()

        sf.write(
            path.join(args.ROOT, args.NAME, f"eval_{e:06d}.wav"),
            audio,
            config["preprocess"]["sampling_rate"],
        )
    
# print(loss_array)
df = pd.DataFrame(loss_array)
     
# saving the dataframe
df.to_csv("./loss.csv", sep=',',index=False)