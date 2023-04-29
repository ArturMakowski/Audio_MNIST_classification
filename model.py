from torch import nn
from torchsummary import summary

class AudioMNISTModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, 
                    out_channels=16, 
                    kernel_size=3,
                    stride=1, 
                    padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, 
                    out_channels=32, 
                    kernel_size=3,
                    stride=1, 
                    padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, 
                    out_channels=64, 
                    kernel_size=3,
                    stride=1, 
                    padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, 
                    out_channels=128, 
                    kernel_size=3,
                    stride=1, 
                    padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128*5*7, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        x = self.conv1(input)                                                           
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.softmax(x)
        return x
    
if __name__ == "__main__":
    model = AudioMNISTModel()
    summary(model, (1, 64, 94))