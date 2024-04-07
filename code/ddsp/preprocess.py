import yaml
import pathlib
import librosa as li
from ddsp.get_audio_descriptors import get_all_descriptors, spectral_centroid, spectral_flatness, temporal_centroid
from ddsp.core import extract_loudness, extract_pitch
from effortless_config import Config
import numpy as np
from tqdm import tqdm
import numpy as np
from os import makedirs, path
import torch
from scipy.io import wavfile
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def get_files(data_location, extension, **kwargs):
    return list(pathlib.Path(data_location).rglob(f"*.{extension}"))


def preprocess(f, sampling_rate, block_size, signal_length, oneshot, **kwargs):
    x, sr = li.load(f, sr=sampling_rate)
    N = (signal_length - len(x) % signal_length) % signal_length
    x = np.pad(x, (0, N))

    if oneshot:
        x = x[..., :signal_length]

    pitch = extract_pitch(x, sampling_rate, block_size)
    loudness = extract_loudness(x, sampling_rate, block_size)
    mfcc = li.feature.mfcc(y=x[:-1], sr=sampling_rate, n_mfcc=30, n_fft=1024, hop_length=256, n_mels=128, fmin=20, fmax=8000)
    mfcc = np.transpose(mfcc, (1, 0))
    timbre_descriptors = get_all_descriptors(x, sampling_rate, 256, 128)

    x = x.reshape(-1, signal_length)
    pitch = pitch.reshape(x.shape[0], -1)
    loudness = loudness.reshape(x.shape[0], -1)
    mfcc = np.expand_dims(mfcc, axis=0)
    timbre_descriptors = np.expand_dims(timbre_descriptors, axis=0)

    return x, pitch, loudness, mfcc, timbre_descriptors


class Dataset(torch.utils.data.Dataset):
    def __init__(self, out_dir):
        super().__init__()
        self.signals = np.load(path.join(out_dir, "signals.npy"))
        self.pitchs = np.load(path.join(out_dir, "pitchs.npy"))
        self.loudness = np.load(path.join(out_dir, "loudness.npy"))
        self.mfcc = np.load(path.join(out_dir, "mfcc.npy"))
        self.timbres = np.load(path.join(out_dir, "timbres.npy"))
        self.sources = np.load(path.join(out_dir, "sources.npy"))
        
    def __len__(self):
        return self.signals.shape[0]

    def __getitem__(self, idx):
        signal = torch.from_numpy(self.signals[idx])
        pitch = torch.from_numpy(self.pitchs[idx])
        loudness = torch.from_numpy(self.loudness[idx])
        mfcc = torch.from_numpy(self.mfcc[idx])
        timbre = torch.from_numpy(self.timbres[idx])
        source = torch.from_numpy(self.sources[idx])

        return signal, pitch, loudness, mfcc, timbre, source


def main():
    class args(Config):
        CONFIG = "config.yaml"

    args.parse_args()
    with open(args.CONFIG, "r") as config:
        config = yaml.safe_load(config)

    files = get_files(**config["data"])
    pb = tqdm(files)

    signals = []
    pitchs = []
    loudness = []
    mfccs = []
    timbres = []
    sources = []

    for f in pb:
        pb.set_description(str(f))
        filename = str(f)[str(f).replace("\\", "/").rfind("/")+1:]
        sound_source = -1
        if "acoustic" in filename:
            sound_source = 0
        elif "electronic" in filename:
            sound_source = 1
        elif "synthetic"in filename:
            sound_source = 2

        x, p, l, mfcc, timbre = preprocess(f, **config["preprocess"])
        signals.append(x)
        pitchs.append(p)
        loudness.append(l)
        mfccs.append(mfcc)
        timbres.append(timbre)
        sources.append(sound_source)

    signals = np.concatenate(signals, 0).astype(np.float32)
    pitchs = np.concatenate(pitchs, 0).astype(np.float32)
    loudness = np.concatenate(loudness, 0).astype(np.float32)
    mfccs = np.concatenate(mfccs, 0).astype(np.float32)
    timbres = np.concatenate(timbres, 0).astype(np.float32)
    sources = np.expand_dims(np.array(sources), axis=-1).astype(np.float32)

    out_dir = config["preprocess"]["out_dir"]
    makedirs(out_dir, exist_ok=True)

    np.save(path.join(out_dir, "signals.npy"), signals)
    np.save(path.join(out_dir, "pitchs.npy"), pitchs)
    np.save(path.join(out_dir, "loudness.npy"), loudness)
    np.save(path.join(out_dir, "mfcc.npy"), mfccs)
    np.save(path.join(out_dir, "timbres.npy"), timbres)
    np.save(path.join(out_dir, "sources.npy"), sources)


if __name__ == "__main__":
    main()