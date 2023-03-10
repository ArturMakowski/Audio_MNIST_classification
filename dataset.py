import os 

import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd

class AudioMNISTDataset(Dataset):

    """Audio MNIST dataset.
    Args:   
        audio_dir_path (str): Path to the directory containing the audio files.    
    """

    def __init__(self, audio_dir_path, transformation, 
                 num_samples_per_clip, 
                 device):

        """Initializes the dataset."""

        self.audio_dir_path = audio_dir_path
        self.device = device
        self.transformation = transformation.to(self.device)
        self.num_samples_per_clip = num_samples_per_clip
        self.file_list = []
        self.label_list = []
        for root, _, files in os.walk(self.audio_dir_path):
            for file in files:
                if file.endswith(".wav"):
                    self.file_list.append(os.path.join(root, file))
                    self.label_list.append(int(file.split("_")[0]))

    def __len__(self):

        """Returns the total number of audio files."""

        return len(self.file_list)

    def __getitem__(self, index):

        """Returns a tuple (signal, label, file_path)"""

        signal, sr = torchaudio.load(self.file_list[index])
        signal = signal.to(self.device)
        signal = self._right_zero_pad(signal)
        signal_transformed = self.transformation(signal)
        return signal,signal_transformed, sr, self.label_list[index], self.file_list[index]
    
    def _right_zero_pad(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples_per_clip:
            num_missing_samples = self.num_samples_per_clip - length_signal
            signal = torch.nn.functional.pad(signal, (0, num_missing_samples))
        return signal
    


def plot_distribution_of_audio_lengths(audio_mnist):
    audio_file_lengths = []
    sample_rates = []

    for i in range(len(audio_mnist)):
        signal, _, sr, _, _ = audio_mnist[i]
        audio_file_lengths.append(signal.shape[1])
        sample_rates.append(sr)

    print(f"Max audio file length: {max(audio_file_lengths)} samples and {max(audio_file_lengths)/48000} seconds")
    fig = plt.figure(figsize=(13, 10))
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.hist(audio_file_lengths, bins=100)
    ax1.set_title("Distribution of audio file lengths")
    ax1.set_xlabel("Audio file length")
    ax1.set_ylabel("Number of audio files")

    # ax2 = fig.add_subplot(3, 1, 2)
    # ax2.hist(sample_rates, bins=100)
    # ax2.set_title("Distribution of sample rates")
    # ax2.set_xlabel("Sample rate")
    # ax2.set_ylabel("Number of audio files")


if __name__ == "__main__":

    # Test the dataset

    AUDIO_DIR_PATH = "/home/armak/Python_projects_WSL/Audio_MNIST_classification/data"
    SAMPLE_RATE = 48000
    NUM_SAMPLES = 48000

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    print(f'Using {device} device')

    mel_spectrogram_transformation = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    audio_mnist = AudioMNISTDataset(AUDIO_DIR_PATH, 
                                    mel_spectrogram_transformation, 
                                    NUM_SAMPLES, 
                                    device)

    print(f"There are {len(audio_mnist)} samples in the dataset.")

    signal, signal_transformed, sr, label, file_path = audio_mnist[0]

    print(f'{signal.shape}, {signal_transformed.shape}, {label}, {file_path}')

    # Plot the distribution of the audio file lengths and sample rates
    # plot_distribution_of_audio_lengths(audio_mnist)

