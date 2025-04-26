import torch
import torch.nn as nn
import torch.nn.functional as F


torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.get_device_name(0)


class SSVEPDNN(nn.Module):
    def __init__(self, num_classes=40, channels=9, samples=250, subbands=3):
        super(SSVEPDNN, self).__init__()
        # [batch, subbands, channels, time]
        # Subband combination layer
        self.subband_combination = nn.Conv2d(
            subbands, 1, kernel_size=(1, 1), bias=False
        )
        # Channel combination layer
        self.channel_combination = nn.Conv2d(1, 120, kernel_size=(channels, 1))
        # First dropout
        self.drop1 = nn.Dropout(0.1)
        # Third layer - Time convolution
        self.third_conv = nn.Conv2d(120, 120, kernel_size=(1, 2), stride=(1, 2))
        # Second droput
        self.drop2 = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        # 4th conv - FIR filtering
        self.fourth_conv = nn.Conv2d(120, 120, kernel_size=(1, 10), padding="same")
        self.drop3 = nn.Dropout(0.95)

        # Fully connected layer - Classifier
        self.fc = nn.Linear(120 * (samples // 2), num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        with torch.no_grad():
            self.subband_combination.weight.fill_(1.0)

    def forward(self, x):
        # x shape: [batch, subbands, channels, time]
        x = self.subband_combination(x)  # [batch, 1, channels, time]
        x = self.channel_combination(x)  # [batch, 120, 1, time]
        x = self.drop1(x)
        x = self.third_conv(x)  # [batch, 120, 1, time/2]
        x = self.drop2(x)
        x = self.relu(x)
        x = self.fourth_conv(x)  # [batch, 120, 1, time/2]
        x = self.drop3(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)  # [batch, num_classes]
        output = F.softmax(x, dim=1)
        return output
