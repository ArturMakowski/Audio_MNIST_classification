import os 

from torch.utils.data import Dataset
import torchaudio
import pandas as pd

class AudioMNISTDataset(Dataset):
    def __init__(self, audio_dir_path):
        self.audio_dir_path = audio_dir_path

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass

    def _get_audio_sample_path(self, index):
        pass

    def _get_audio_sample_label(self, index):
        pass