import torch
import torch.nn as nn
from .core import mlp, gru, scale_function, remove_above_nyquist, upsample
from .core import harmonic_synth, amp_to_impulse_response, fft_convolve
from .core import resample
import math


class Reverb(nn.Module):
    def __init__(self, length, sampling_rate, initial_wet=0, initial_decay=5):
        super().__init__()
        self.length = length
        self.sampling_rate = sampling_rate

        self.noise = nn.Parameter((torch.rand(length) * 2 - 1).unsqueeze(-1))
        self.decay = nn.Parameter(torch.tensor(float(initial_decay)))
        self.wet = nn.Parameter(torch.tensor(float(initial_wet)))

        t = torch.arange(self.length) / self.sampling_rate
        t = t.reshape(1, -1, 1)
        self.register_buffer("t", t)

    def build_impulse(self):
        t = torch.exp(-nn.functional.softplus(-self.decay) * self.t * 500)
        noise = self.noise * t
        impulse = noise * torch.sigmoid(self.wet)
        impulse[:, 0] = 1
        return impulse

    def forward(self, x):
        lenx = x.shape[1]
        impulse = self.build_impulse()
        impulse = nn.functional.pad(impulse, (0, 0, 0, lenx - self.length))

        x = fft_convolve(x.squeeze(-1), impulse.squeeze(-1)).unsqueeze(-1)

        return x


class Encoder(nn.Module):
    def __init__(self, hidden_size=512, out_size=16):
        super().__init__()
        self.layer_norm = nn.LayerNorm(30)
        self.gru = nn.GRU(30, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, out_size)
        
    def forward(self, x):
        out = self.layer_norm(x)
        out, _ = self.gru(out)
        out = self.fc(out)
        return out
# change Standard encoder to VAE
class VAEncoder(nn.Module):
    def __init__(self, hidden_size=512, latent_dim=16,dropout_rate=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(33)
        self.gru = nn.GRU(33, hidden_size, batch_first=True)
        # Output both mean and log variance for latent space
        
        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size, latent_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.sig_mu = nn.LeakyReLU()
        self.sig_logvar = nn.LeakyReLU()

    def forward(self, x):
        out = self.layer_norm(x)
        out, _ = self.gru(out)
        # Separate outputs for mean and log variance
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        mu = self.dropout(mu)
        logvar = self.dropout(logvar)
        mu = self.sig_mu(mu)
        logvar = self.sig_logvar(logvar)
        return mu, logvar
class VAEncoder_dropout(nn.Module):
    def __init__(self, hidden_size=512, latent_dim=20, dropout_rate=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(33)
        self.gru = nn.GRU(33, hidden_size, batch_first=True)
        # Adding dropout layer
        # self.dropout = nn.Dropout(dropout_rate)
        # Output both mean and log variance for latent space with dropout
        self.fc_mu = nn.Sequential(
            nn.Linear(hidden_size, latent_dim),
            # nn.Dropout(dropout_rate),
            nn.LeakyReLU() # Add dropout after linear transformation
        )
        self.fc_logvar = nn.Sequential(
            nn.Linear(hidden_size, latent_dim),
            # nn.Dropout(dropout_rate),  # Add dropout after linear transformation
            nn.LeakyReLU()
        )

    def forward(self, x):
        # Normalize the input
        normalized_x = self.layer_norm(x)
        
        # Pass the input through the GRU layer
        # GRU returns output and a final hidden state, but we only need the hidden state
        _, hidden = self.gru(normalized_x)
        
        # The hidden state output by GRU layers has dimensions [num_layers * num_directions, batch, hidden_size]
        # For a single layer, single direction GRU, we reshape this to [batch, hidden_size] for further processing
        # hidden = hidden.squeeze(0)
        
        # Apply dropout and linear transformation to get the mean and log variance
        # Since we are using nn.Sequential for fc_mu and fc_logvar, we can directly call them on the hidden state
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        
        return mu, logvar

class DDSP(nn.Module):
    def __init__(self, hidden_size, n_harmonic, n_bands, sampling_rate,
                 block_size):
        super().__init__()
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))

        self.encoder = VAEncoder(512,20,dropout_rate=0.1)
        self.in_mlps = nn.ModuleList([mlp(1, hidden_size, 3)] * 2 + [mlp(20, hidden_size, 3)])
        self.gru = gru(3, hidden_size)
        self.out_mlp = mlp(hidden_size + 2, hidden_size, 3)

        self.proj_matrices = nn.ModuleList([
            nn.Linear(hidden_size, n_harmonic + 1),
            nn.Linear(hidden_size, n_bands),
        ])

        self.reverb = Reverb(sampling_rate, sampling_rate)

        self.register_buffer("cache_gru", torch.zeros(1, 1, hidden_size))
        self.register_buffer("phase", torch.zeros(1))
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    def forward(self, pitch, loudness, mfcc, timbre, source):
        ##standard encoder
        # latent_z = self.encoder(mfcc)
        ##standard encoder

        ##VAE
        mu, logvar = self.encoder(mfcc)  # Use VAEEncoder here
        # print(mu.size())
        # print(logvar.size())
        # print(mfcc.size())
        # print(timbre.size())
        # print(torch.concat([mfcc,timbre],-1).size())
        
        latent_z = self.reparameterize(mu, logvar)  # Sample latent vector
        ##VAE
        # timbre = timbre.unsqueeze(1).repeat(1, 250, 1)
        # print(self.in_mlps[0](pitch).size())
        # print(self.in_mlps[1](loudness).size())
        # print(self.in_mlps[2](latent_z).size())

        # print(pitch.size())
        # print(loudness.size())
        # print(latent_z.size())
        hidden = torch.cat([
            self.in_mlps[0](pitch),
            self.in_mlps[1](loudness),
            self.in_mlps[2](latent_z),
        ], -1)
        hidden = torch.cat([self.gru(hidden)[0], pitch, loudness], -1)
        hidden = self.out_mlp(hidden)

        # harmonic part
        param = scale_function(self.proj_matrices[0](hidden))
        print("projec_mat[0]hidden", self.proj_matrices[0](hidden).size())
        print("param", param.size())
        total_amp = param[..., :1]
        print("total_amp",total_amp.size())
        amplitudes = param[..., 1:]
        print("amplitude[...,:1]", amplitudes.size())
        amplitudes = remove_above_nyquist(
            amplitudes,
            pitch,
            self.sampling_rate,
        )
        print("amplitude_remove_nyquist", amplitudes.size())
        amplitudes /= amplitudes.sum(-1, keepdim=True)
        amplitudes *= total_amp

        amplitudes = upsample(amplitudes, self.block_size)
        print("upsample_amplitude",amplitudes.size())
        pitch = upsample(pitch, self.block_size)
        print("upsample_pitch",pitch.size())
        harmonic = harmonic_synth(pitch, amplitudes, self.sampling_rate)
        print("harmonic size",harmonic.size())
        # noise part
        param = scale_function(self.proj_matrices[1](hidden) - 5)
        print("projecmat[1]",self.proj_matrices[1](hidden).size())
        impulse = amp_to_impulse_response(param, self.block_size)
        print("impulse",impulse.size())
        noise = torch.rand(
            impulse.shape[0],
            impulse.shape[1],
            self.block_size,
        ).to(impulse) * 2 - 1
        print("noise",noise.size())
        noise = fft_convolve(noise, impulse).contiguous()
        print("fft noise", noise.size())
        noise = noise.reshape(noise.shape[0], -1, 1)
        print("noise reshape", noise.size())
        signal = harmonic + noise
        print("signal size", signal.size())
        #reverb part
        signal = self.reverb(signal)
        print("reverb signal", signal.size())
        # return signal,logvar,mu
        return signal
        # return signal

    def realtime_forward(self, pitch, loudness):
        hidden = torch.cat([
            self.in_mlps[0](pitch),
            self.in_mlps[1](loudness),
        ], -1)

        gru_out, cache = self.gru(hidden, self.cache_gru)
        self.cache_gru.copy_(cache)

        hidden = torch.cat([gru_out, pitch, loudness], -1)
        hidden = self.out_mlp(hidden)

        # harmonic part
        param = scale_function(self.proj_matrices[0](hidden))

        total_amp = param[..., :1]
        amplitudes = param[..., 1:]

        amplitudes = remove_above_nyquist(
            amplitudes,
            pitch,
            self.sampling_rate,
        )
        amplitudes /= amplitudes.sum(-1, keepdim=True)
        amplitudes *= total_amp

        amplitudes = upsample(amplitudes, self.block_size)
        pitch = upsample(pitch, self.block_size)

        n_harmonic = amplitudes.shape[-1]
        omega = torch.cumsum(2 * math.pi * pitch / self.sampling_rate, 1)

        omega = omega + self.phase
        self.phase.copy_(omega[0, -1, 0] % (2 * math.pi))

        omegas = omega * torch.arange(1, n_harmonic + 1).to(omega)

        harmonic = (torch.sin(omegas) * amplitudes).sum(-1, keepdim=True)

        # noise part
        param = scale_function(self.proj_matrices[1](hidden) - 5)

        impulse = amp_to_impulse_response(param, self.block_size)
        noise = torch.rand(
            impulse.shape[0],
            impulse.shape[1],
            self.block_size,
        ).to(impulse) * 2 - 1

        noise = fft_convolve(noise, impulse).contiguous()
        noise = noise.reshape(noise.shape[0], -1, 1)

        signal = harmonic + noise

        return signal