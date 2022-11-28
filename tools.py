""" from https://github.com/NVIDIA/tacotron2 """

import torch
import numpy as np
from scipy.io.wavfile import read
from scipy.io.wavfile import write

import audio.stft as stft
import audio.hparams_audio as hparams
from audio.audio_processing import griffin_lim
import pyworld as pw

import torch.nn as nn
import torch.nn.functional as F

from sklearn import preprocessing
import librosa


_stft = stft.TacotronSTFT(
    hparams.filter_length, hparams.hop_length, hparams.win_length,
    hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
    hparams.mel_fmax)


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def get_mel(filename):
    audio, sampling_rate = load_wav_to_torch(filename)
    if sampling_rate != _stft.sampling_rate:
        raise ValueError("{} {} SR doesn't match target {} SR".format(
            sampling_rate, _stft.sampling_rate))
    audio_norm = audio / hparams.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    melspec = _stft.mel_spectrogram(audio_norm)
    melspec = torch.squeeze(melspec, 0)
    # melspec = torch.from_numpy(_normalize(melspec.numpy()))

    return melspec


def get_mel_from_wav(audio):
    sampling_rate = hparams.sampling_rate
    if sampling_rate != _stft.sampling_rate:
        raise ValueError("{} {} SR doesn't match target {} SR".format(
            sampling_rate, _stft.sampling_rate))
    audio_norm = audio / hparams.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    melspec = _stft.mel_spectrogram(audio_norm)
    melspec = torch.squeeze(melspec, 0)

    return melspec

def get_energy(filename):
    audio, sampling_rate = load_wav_to_torch(filename)
    if sampling_rate != _stft.sampling_rate:
        raise ValueError("{} {} SR doesn't match target {} SR".format(
            sampling_rate, _stft.sampling_rate))
    
    magnitudes, phases = _stft.stft_fn.transform(audio.unsqueeze(0))
    magnitudes = magnitudes.squeeze(0)
    e = torch.norm(magnitudes, dim=0)
    return e

def get_pitch(filename):
    audio, sampling_rate = load_wav_to_torch(filename)
    if sampling_rate != _stft.sampling_rate:
        raise ValueError("{} {} SR doesn't match target {} SR".format(
            sampling_rate, _stft.sampling_rate))
    audio = audio.to(torch.float64).numpy()
    
    f0, timeaxis = pw.dio(
        audio,
        sampling_rate,
        frame_period=_stft.hop_length / _stft.sampling_rate * 1000,
    )
    
    return f0

def extract_feature(filename):
    audio, sampling_rate = load_wav_to_torch(filename)
    if sampling_rate != _stft.sampling_rate:
        raise ValueError("{} {} SR doesn't match target {} SR".format(
            sampling_rate, _stft.sampling_rate))
    audio = audio.numpy()
    
    pre_emphasis = 0.97
    audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
    
    mfccs = librosa.feature.mfcc(y=audio, sr=sampling_rate, n_mfcc=40, n_mels=40,
                                 power=2.0, window = 'hamming', fmin = 0.,
                                 fmax = 8000., hop_length=_stft.hop_length,
                                 n_fft=1024, win_length=1024, center=True)
    return mfccs

#defining model
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim_1, hidden_dim_2, out_dim=1):
        super(MLP, self).__init__()
        
        self.in_dim = in_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.out_dim = out_dim
        
        ## 1st hidden layer
        self.linear_1 = nn.Linear(self.in_dim, self.hidden_dim_1)
        self.linear_1.weight.detach().normal_(0.0, 0.1)
        self.linear_1.bias.detach().zero_()
        self.linear_1_bn = nn.BatchNorm1d(self.hidden_dim_1,momentum=0.6)
        
        ## 2nd hidden layer
        self.linear_2 = nn.Linear(self.hidden_dim_1, self.hidden_dim_2)
        self.linear_2.weight.detach().normal_(0.0, 0.1)
        self.linear_2.bias.detach().zero_()
        self.linear_2_bn = nn.BatchNorm1d(self.hidden_dim_2,momentum=0.6)
        
        ## Out layer
        self.linear_out = nn.Linear(self.hidden_dim_2, self.out_dim)
        self.linear_out.weight.detach().normal_(0.0, 0.1)
        self.linear_out.bias.detach().zero_()
        
    def forward(self, x):
        out = self.linear_1(x)
        out = self.linear_1_bn(out)
        out = F.relu(out)
        
        out = self.linear_2(out)
        out = self.linear_2_bn(out)
        out = F.relu(out)
        out = F.dropout(out, p=0.175, training=self.training)
        
        out = self.linear_out(out)
        out = nn.Tanh()(out)
        return out.flatten()

mlp = MLP(40, 260, 180, 1)
mlp.load_state_dict(torch.load('mlp.pt', map_location=torch.device('cpu')))
mlp.eval()

def get_emotion(filename):
    X = [extract_feature(filename)]
    X = np.transpose(np.concatenate(X, axis=1))
    X = preprocessing.normalize(X, norm='l2', axis=1, copy=False)
    y_pred = mlp(torch.tensor(X))
    return y_pred.flatten().detach()


def inv_mel_spec(mel, out_filename, griffin_iters=60):
    mel = torch.stack([mel])
    # mel = torch.stack([torch.from_numpy(_denormalize(mel.numpy()))])
    mel_decompress = _stft.spectral_de_normalize(mel)
    mel_decompress = mel_decompress.transpose(1, 2).data.cpu()
    spec_from_mel_scaling = 1000
    spec_from_mel = torch.mm(mel_decompress[0], _stft.mel_basis)
    spec_from_mel = spec_from_mel.transpose(0, 1).unsqueeze(0)
    spec_from_mel = spec_from_mel * spec_from_mel_scaling

    audio = griffin_lim(torch.autograd.Variable(
        spec_from_mel[:, :, :-1]), _stft.stft_fn, griffin_iters)

    audio = audio.squeeze()
    audio = audio.cpu().numpy()
    audio_path = out_filename
    write(audio_path, hparams.sampling_rate, audio)
