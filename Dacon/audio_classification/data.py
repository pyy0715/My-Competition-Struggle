import numpy as np
import torch
import librosa
from audiomentations import Compose, AddGaussianNoise, Shift, TimeStretch, PitchShift

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, hparams, csv, transform=None):
        self.hparams = hparams
        self.csv = csv.reset_index(drop=True)
        self.aug = Compose([AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
                            TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
                            PitchShift(min_semitones=-4,
                                       max_semitones=4, p=0.5),
                            Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5)])
        self.transform = transform

    def __len__(self):
        return len(self.csv)
    
    def generate_mel(self, audio, sr, n_fft, win_length, hop_length, n_mels):
        S = librosa.feature.melspectrogram(y=audio,
                                           sr=sr,
                                           n_fft=n_fft,
                                           win_length=win_length,
                                           hop_length=hop_length,
                                           n_mels=n_mels)
        S = librosa.power_to_db(S, ref=np.max)
        S = np.clip((S - self.hparams.min_level_db) / -
                    self.hparams.min_level_db, 0, 1)
        return S
    
    def features_extractor(self, audio_path):
        audio = np.load(audio_path)
        audio = librosa.util.fix_length(audio, self.hparams.min_length)
        
        if self.transform:
            audio = self.aug(audio, sample_rate=self.hparams.sr)

        mel = []
        for n_fft, win_length in zip(self.hparams.n_fft, self.hparams.win_length):
            S = self.generate_mel(audio,
                                  self.hparams.sr,
                                  n_fft, win_length, self.hparams.hop_length,
                                  self.hparams.n_mels)
            mel.append(S)
        return np.array(mel)

    def __getitem__(self, index):
        path = self.csv.iloc[index, -1]
        label = self.csv.iloc[index, 1]
        
        mel = self.features_extractor(path)
        return (
            torch.tensor(mel, dtype=torch.float),
            torch.tensor(label, dtype=torch.long)
        )
