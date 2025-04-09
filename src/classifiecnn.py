import torch.nn as nn

class Classifiercnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # [3, 64, 64] -> [16, 64, 64]
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> [16, 32, 32]
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # -> [32, 32, 32]
            nn.ReLU(),
            nn.MaxPool2d(2)  # -> [32, 16, 16]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x