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

dataset_train = Dataset(config["preprocess"]["out_dir"])
dataset_validation = Dataset(config["validation"]["data_dir"])

dataloader_train = torch.utils.data.DataLoader(
    dataset_train,
    args.BATCH,
    True,
    drop_last=True,
)

dataloader_valid = torch.utils.data.DataLoader(
    dataset_validation,
    1,
    False,
    drop_last=False,
)

mean_loudness, std_loudness = mean_std_loudness(dataloader_train)
config["data"]["mean_loudness"] = mean_loudness
config["data"]["std_loudness"] = std_loudness

writer = SummaryWriter(path.join(args.ROOT, args.NAME), flush_secs=20)

with open(path.join(args.ROOT, args.NAME, "config.yaml"), "w") as out_config:
    yaml.safe_dump(config, out_config)

opt = torch.optim.Adam(model.parameters(), lr=args.START_LR)

schedule = get_scheduler(
    len(dataloader_train),
    args.START_LR,
    args.STOP_LR,
    args.DECAY_OVER,
)

scheduler = torch.optim.lr_scheduler.LambdaLR(opt, schedule)

best_loss = float("inf")
train_loss = 0
n_batch = 0
step = 0
epochs = int(np.ceil(args.STEPS / len(dataloader_train)))

timbre_loss_factor = 1

for e in tqdm(range(epochs)):
    # training phase
    model.train()
    for signal, pitch, loudness, mfcc, timbre, source in dataloader_train:
        signal = signal.to(device)
        pitch = pitch.unsqueeze(-1).to(device)
        loudness = loudness.unsqueeze(-1).to(device)
        mfcc = mfcc.to(device)
        timbre = timbre.to(device)
        ori_timbre = timbre.detach().cpu().numpy()
        timbre = timbre.unsqueeze(1).repeat(1, pitch.size(1), 1)
        source = source.to(device)

        loudness = (loudness - mean_loudness) / std_loudness

        y, mu, logvar = model(pitch, loudness, mfcc, timbre, source)
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
            rec_timbre[i] = get_all_descriptors(rec_signals[i], config["preprocess"]["sampling_rate"], 256, 128)
        
        ori_erb_spec_centroid = 1000/(24.7*4.37) * np.log(4.37*ori_timbre[:, 0]/1000 + 1 + 1e-7)
        rec_erb_spec_centroid = 1000/(24.7*4.37) * np.log(4.37*rec_timbre[:, 0]/1000 + 1 + 1e-7)
        ori_spec_flatness = ori_timbre[:, 1]
        rec_spec_flatness = rec_timbre[:, 1]
        ori_temporal_centroid = ori_timbre[:, 2]
        rec_temporal_centroid = rec_timbre[:, 2]
        timbre_loss = np.abs(rec_erb_spec_centroid - ori_erb_spec_centroid).mean() \
            + np.abs(rec_spec_flatness - ori_spec_flatness).mean() \
            + np.abs(rec_temporal_centroid - ori_temporal_centroid).mean()

        ori_stft = multiscale_fft(
            signal,
            config["train"]["scales"],
            config["train"]["overlap"],
        )
        rec_stft = multiscale_fft(
            y,
            config["train"]["scales"],
            config["train"]["overlap"],
        )

        loss = 0
        for s_x, s_y in zip(ori_stft, rec_stft):
            lin_loss = (s_x - s_y).abs().mean()
            log_loss = (safe_log(s_x) - safe_log(s_y)).abs().mean()
            loss = loss + lin_loss + log_loss

        kl_divergence = 1/2 * torch.sum(torch.pow(mu, 2) + torch.pow(torch.exp(logvar), 2) - 2 * logvar - 1)

        loss = loss + timbre_loss_factor * timbre_loss + kl_divergence

        opt.zero_grad()
        loss.backward()
        opt.step()

        writer.add_scalar("loss/train", loss.item() / args.BATCH, step)

        step += 1

        n_batch += 1
        train_loss += loss.item()

    # validation phase
    model.eval()
    valid_loss = 0
    for signal, pitch, loudness, mfcc, timbre, source in dataloader_valid:
        signal = signal.to(device)
        pitch = pitch.unsqueeze(-1).to(device)
        loudness = loudness.unsqueeze(-1).to(device)
        mfcc = mfcc.to(device)
        timbre = timbre.to(device)
        ori_timbre = timbre.detach().cpu().numpy()
        timbre = timbre.unsqueeze(1).repeat(1, pitch.size(1), 1)
        source = source.to(device)

        with torch.no_grad():
            y, mu, logvar = model(pitch, loudness, mfcc, timbre, source)
            y = y.squeeze(-1)

        rec_signals = y.detach().cpu().numpy()
        rec_timbre = np.zeros_like(ori_timbre)

        for i in range(rec_signals.shape[0]):
            rec_timbre[i] = get_all_descriptors(rec_signals[i], config["preprocess"]["sampling_rate"], 256, 128)

        ori_erb_spec_centroid = 1000/(24.7*4.37) * np.log(4.37*ori_timbre[:, 0]/1000 + 1 + 1e-7)
        rec_erb_spec_centroid = 1000/(24.7*4.37) * np.log(4.37*rec_timbre[:, 0]/1000 + 1 + 1e-7)
        ori_spec_flatness = ori_timbre[:, 1]
        rec_spec_flatness = rec_timbre[:, 1]
        ori_temporal_centroid = ori_timbre[:, 2]
        rec_temporal_centroid = rec_timbre[:, 2]
        timbre_loss = np.abs(rec_erb_spec_centroid - ori_erb_spec_centroid).mean() \
            + np.abs(rec_spec_flatness - ori_spec_flatness).mean() \
            + np.abs(rec_temporal_centroid - ori_temporal_centroid).mean()

        ori_stft = multiscale_fft(
            signal,
            config["train"]["scales"],
            config["train"]["overlap"],
        )
        rec_stft = multiscale_fft(
            y,
            config["train"]["scales"],
            config["train"]["overlap"],
        )

        loss = 0
        for s_x, s_y in zip(ori_stft, rec_stft):
            lin_loss = (s_x - s_y).abs().mean()
            log_loss = (safe_log(s_x) - safe_log(s_y)).abs().mean()
            loss = loss + lin_loss + log_loss

        kl_divergence = 1/2 * torch.sum(torch.pow(mu, 2) + torch.pow(torch.exp(logvar), 2) - 2 * logvar - 1)

        valid_loss += loss + timbre_loss_factor * timbre_loss + kl_divergence

    writer.add_scalar("loss/valid", valid_loss.item() / len(dataloader_valid), step)

    writer.add_scalar("lr", schedule(e), e)
    writer.add_scalar("reverb_decay", model.reverb.decay.item(), e)
    writer.add_scalar("reverb_wet", model.reverb.wet.item(), e)
    scheduler.step()
    if train_loss / args.BATCH / n_batch < best_loss:
        best_loss = train_loss / args.BATCH / n_batch
        torch.save(
            model.state_dict(),
            path.join(args.ROOT, args.NAME, "state.pth"),
        )

    train_loss = 0
    n_batch = 0
