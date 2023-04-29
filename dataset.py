import os 

import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import torchaudio
from torch.utils.data import DataLoader

class AudioMNISTDataset(Dataset):

    """Audio MNIST dataset.
    Args:   
        audio_dir_path (str): Path to the directory containing the audio files.    
    """

    def __init__(self, 
                 audio_dir_path, 
                 transformation, 
                 num_samples_per_clip, 
                 device,
                 train_set=True):

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
            if train_set:
                if len(self.file_list) == 24000:
                    break
            else:
                if len(self.file_list) == 6000:
                    break  
        # if train_set:
        #     self.file_list = self.file_list[:int(len(self.file_list)*0.8)]
        #     self.label_list = self.label_list[:int(len(self.file_list)*0.8)]
        # else:
        #     self.file_list = self.file_list[int(len(self.file_list)*0.8):]
        #     self.label_list = self.label_list[int(len(self.file_list)*0.8):]

    def __len__(self):

        """Returns the total number of audio files."""

        return len(self.file_list)

    def __getitem__(self, index):

        """Returns a tuple (signal_transformed(melSpec), label, file_path) for a given index."""

        signal, sr = torchaudio.load(self.file_list[index]) # type: ignore
        signal = signal.to(self.device)
        signal = self._right_zero_pad(signal)
        signal_transformed = self.transformation(signal)
        return signal_transformed, self.label_list[index], self.file_list[index]
    
    def _right_zero_pad(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples_per_clip:
            num_missing_samples = self.num_samples_per_clip - length_signal
            signal = torch.nn.functional.pad(signal, (0, num_missing_samples))
        return signal
    


def plot_distribution_of_audio_lengths(audio_mnist):
    audio_file_lengths = []

    for i in range(len(audio_mnist)):
        signal, _, _ = audio_mnist[i]
        audio_file_lengths.append(signal.shape[1])


    print(f"Max audio file length: {max(audio_file_lengths)} samples and {max(audio_file_lengths)/48000} seconds")
    fig = plt.figure(figsize=(13, 10))
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.hist(audio_file_lengths, bins=100)
    ax1.set_title("Distribution of audio file lengths")
    ax1.set_xlabel("Audio file length")
    ax1.set_ylabel("Number of audio files")


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

    audio_mnist_train = AudioMNISTDataset(AUDIO_DIR_PATH, 
                                    mel_spectrogram_transformation, 
                                    NUM_SAMPLES, 
                                    device,
                                    train_set=True)
    
    audio_mnist_test = AudioMNISTDataset(AUDIO_DIR_PATH, 
                                    mel_spectrogram_transformation, 
                                    NUM_SAMPLES, 
                                    device,
                                    train_set=False)
                                         

    print(f"There are {len(audio_mnist_train)} samples in the train dataset.")
    print(f"There are {len(audio_mnist_test)} samples in the test dataset.")

    signal_transformed, label, file_path = audio_mnist_train[0]

    print(f'{signal_transformed.shape}, {label}, {file_path}')

    # train_dataloader = DataLoader(audio_mnist_train, batch_size=128, shuffle=True)

    # Plot the distribution of the audio file lengths and sample rates
    # plot_distribution_of_audio_lengths(audio_mnist)

