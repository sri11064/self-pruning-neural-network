import torch.nn as nn
from models.prunable_linear import PrunableLinear

class PrunableNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.fc1 = PrunableLinear(32*32*3, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_all_gates(self):
        return [
            self.fc1.get_gates(),
            self.fc2.get_gates(),
            self.fc3.get_gates()
        ]