import torch
import torchaudio

from dataset import AudioMNISTDataset
from model import AudioMNISTModel

def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected

if __name__ == "__main__":

    class_mapping = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    SAMPLE_RATE = 48000
    NUM_SAMPLES = 48000
    device = "cpu"
    AUDIO_DIR_PATH = "/home/armak/Python_projects_WSL/Audio_MNIST_classification/data"

    # load back the model
    cnn = AudioMNISTModel()
    state_dict = torch.load("AudioMNISTModel.pth")
    cnn.load_state_dict(state_dict)

    # load audio mnist dataset
    mel_spectrogram_transformation = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    audio_mnist_dataset = AudioMNISTDataset(AUDIO_DIR_PATH, 
                                    mel_spectrogram_transformation, 
                                    NUM_SAMPLES, 
                                    device)


    # get a sample from the audio mnist dataset for inference
    input, target = audio_mnist_dataset[0][1], audio_mnist_dataset[0][3] # [batch size, num_channels, fr, time]
    input.unsqueeze_(0)

    # make an inference
    predicted, expected = predict(cnn, input, target,
                                  class_mapping)
    print(f"Predicted: '{predicted}', expected: '{expected}'")
