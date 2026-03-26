import torch
import torch.nn as nn
import torch.nn.functional as F

class SMALLDNN(nn.Module):
    """Small SSVEP Deep Neural Network with subband processing."""
    def __init__(self, num_classes=40, channels=9, samples=250, subbands=3, n_filters=64, dropout_rate=0.5):
        super(SMALLDNN, self).__init__()
        # [batch, subbands, channels, time]
        # Subband combination layer
        self.subband_combination = nn.Conv2d(
            subbands, 1, kernel_size=(1, 1), bias=False
        )
        # "Time" convolution
        self.time_conv = nn.Conv2d(1, n_filters, kernel_size=(1, 10), stride=(1, 2))
        # First dropout
        self.drop1 = nn.Dropout(dropout_rate)
        # Signals combination layer
        self.channel_combination = nn.Conv2d(n_filters, n_filters, kernel_size=(channels, 1))
        # Second dropout
        self.drop2 = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

        # Fully connected layer - Classifier
        # Temporal size after Conv2d(kernel=(1,10), stride=(1,2), padding=0):
        # L_out = floor((L_in - 10)/2) + 1
        reduced_samples = ((samples - 10) // 2) + 1
        self.fc = nn.Linear(n_filters * reduced_samples, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        with torch.no_grad():
            self.subband_combination.weight.fill_(1.0)
            for m in self.modules():
                if isinstance(m, nn.Conv2d) and m != self.subband_combination:
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x shape: [batch, subbands, channels, time]
        x = self.subband_combination(x)  # [batch, 1, channels, time]
        x = self.time_conv(x)  # [batch, 120, 1, time/2]
        x = self.drop1(x)
        x = self.channel_combination(x)  # [batch, 120, 1, time]
        x = self.drop2(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)  # [batch, num_classes]
        return x