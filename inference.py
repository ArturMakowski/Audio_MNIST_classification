import torch
import torchaudio
import numpy as np

from dataset import AudioMNISTDataset
from model import AudioMNISTModel
from sklearn.metrics import classification_report


def predict(model, input, target):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        predicted = predictions[0].argmax(0)
        expected = target
    return predicted, expected


def accuracy(y_pred, y_true):
    correct = np.sum(np.array(y_pred) == np.array(y_true))
    acc = (correct / len(y_pred)) * 100
    return acc


if __name__ == "__main__":

    class_mapping = ["zero", "one", "two", "three",
                     "four", "five", "six", "seven", "eight", "nine"]
    SAMPLE_RATE = 48000
    NUM_SAMPLES = 48000
    device = "cpu"
    AUDIO_DIR_PATH = "/home/armak/Python_projects_WSL/Audio_MNIST_classification/data"

    # load back the model
    audio_classifier = AudioMNISTModel()
    state_dict = torch.load(
        "/home/armak/Python_projects_WSL/Audio_MNIST_classification/AudioMNISTModel.pth")
    audio_classifier.load_state_dict(state_dict)

    # load audio mnist dataset
    mel_spectrogram_transformation = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    audio_mnist_dataset_test = AudioMNISTDataset(AUDIO_DIR_PATH,
                                                 mel_spectrogram_transformation,
                                                 NUM_SAMPLES,
                                                 device,
                                                 train_set=False)

    # get a sample from the audio mnist dataset for inference
    y_pred = []
    y_true = []
    for i in range(len(audio_mnist_dataset_test)):
        predicted, expected = predict(audio_classifier,
                                      audio_mnist_dataset_test[i][0].unsqueeze_(
                                          0),
                                      audio_mnist_dataset_test[i][1])
        y_pred.append(predicted)
        y_true.append(expected)
        # print(f"Predicted: '{predicted}', expected: '{expected}'")

    # calculate accuracy
    acc = accuracy(y_pred, y_true)
    print(f"Accuracy: {acc:.2f} %")

    # print classification report
    print(classification_report(y_true, y_pred, target_names=class_mapping))
