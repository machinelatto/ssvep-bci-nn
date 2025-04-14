import torch
import torch.nn as nn
import torch.nn.functional as F


torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.get_device_name(0)


# Define o modelo com conv1d
class CNNModel1d(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(CNNModel1d, self).__init__()

        self.cnn = nn.Sequential(
            # Camada 1
            nn.Conv1d(n_channels, 64, kernel_size=8),  # entrada
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            # Camada 2
            nn.Conv1d(64, 32, kernel_size=8),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            # Camada 3
            nn.Conv1d(32, 32, kernel_size=8),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            # Camada 4
            nn.Conv1d(32, 16, kernel_size=8),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Linear(144, 250),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(250, 125),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(125, n_classes),
        )

    def forward(self, x):
        # Camadas convolucionais
        x = self.cnn(x)
        # Flatten
        x = x.view(x.size(0), -1)
        # Fully connected
        x = self.fc(x)
        # log softmax para pegar as probabilidades
        output = F.log_softmax(x, dim=1)
        return output
