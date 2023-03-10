import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader

from dataset import AudioMNISTDataset
from model import AudioMNISTModel

def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        for input, target in data_loader:
            input, target = input.to(device), target.to(device)

            # calculate loss
            prediction = model(input)
            loss = loss_fn(prediction, target)

            # backpropagate error and update weights
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        print(f"loss: {loss.item()}")
        print("---------------------------")
    print("Finished training")

if __name__ == "__main__":

    BATCH_SIZE = 128
    EPOCHS = 10
    LEARNING_RATE = 0.001

    AUDIO_DIR_PATH = "/home/armak/Python_projects_WSL/Audio_MNIST_classification/data"
    SAMPLE_RATE = 48000
    NUM_SAMPLES = 48000

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    mel_spectrogram_transformation = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64)

    audio_mnist_dataset = AudioMNISTDataset(AUDIO_DIR_PATH, 
                                    mel_spectrogram_transformation, 
                                    NUM_SAMPLES, 
                                    device)
    
    train_dataloader = DataLoader(audio_mnist_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # construct model and assign it to device
    audio_mnist_model = AudioMNISTModel().to(device)
    print(audio_mnist_model)

    # initialise loss funtion + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(audio_mnist_model.parameters(),
                                 lr=LEARNING_RATE)

    # train model
    train(audio_mnist_model, train_dataloader, loss_fn, optimiser, device, EPOCHS)

    # save model
    torch.save(audio_mnist_model.state_dict(), "AudioMNISTModel.pth")
    print("Trained CNN saved at AudioMNISTModel.pth")
