import torch
import torch.nn as nn
import torch.nn.functional as F


torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.get_device_name(0)


# From https://github.com/YuDongPan/DL_Classifier/blob/main/Utils/Constraint.py
class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)


# Define o modelo com conv1d
class CompactCNN(nn.Module):
    def __init__(
        self,
        nb_classes=4,
        Chans=9,
        Samples=250,
        dropoutRate=0.5,
        kernLength=250,
        F1=96,
        D=1,
        F2=96,
    ):
        super(CompactCNN, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernLength), padding="same", bias=False),
            nn.BatchNorm2d(F1),
            Conv2dWithConstraint(
                F1,
                F1 * D,
                (Chans, 1),
                groups=F1,
                padding="valid",
                bias=False,
                max_norm=1,
            ),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropoutRate),
        )

        # Block 2
        self.block2 = nn.Sequential(
            # Separable conv in pytorch
            nn.Conv2d(
                F1 * D, F1 * D, (1, 16), groups=F1 * D, padding="same", bias=False
            ),  # Depthwise
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False, padding="same"),  # Pointwise
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropoutRate),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(F2 * (Samples // (4 * 8)), nb_classes),
        )

    def forward(self, x):
        # Camadas convolucionais
        x = self.block1(x)
        x = self.block2(x)
        # Flatten
        x = x.view(x.size(0), -1)
        # Fully connected
        x = self.classifier(x)
        # softmax para pegar as probabilidades
        output = F.softmax(x, dim=1)
        return output
