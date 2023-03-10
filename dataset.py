import os 

import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torchaudio
import pandas as pd

class AudioMNISTDataset(Dataset):

    """Audio MNIST dataset.
    Args:   
        audio_dir_path (str): Path to the directory containing the audio files.    
    """

    def __init__(self, audio_dir_path):

        """Initializes the dataset."""

        self.audio_dir_path = audio_dir_path
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
        return signal, self.label_list[index], self.file_list[index]
    
def plot_distribution_of_audio_file_lengths(audio_mnist):
    audio_file_lengths = []
    for i in range(len(audio_mnist)):
        signal, _, _ = audio_mnist[i]
        audio_file_lengths.append(signal.shape[1])
    plt.hist(audio_file_lengths, bins=100)
    plt.title("Distribution of audio file lengths")
    plt.xlabel("Audio file length")
    plt.ylabel("Number of audio files")
    plt.show()

if __name__ == "__main__":

    # Test the dataset

    audio_dir_path = "/home/armak/Python_projects_WSL/Audio_MNIST_classification/data"

    audio_mnist = AudioMNISTDataset(audio_dir_path)

    print(f"There are {len(audio_mnist)} samples in the dataset.")

    signal, label, file_path = audio_mnist[0]

    print(f'{signal.shape}, {label}, {file_path}')

    # Plot the distribution of the audio file lengths
    # plot_distribution_of_audio_file_lengths(audio_mnist)
