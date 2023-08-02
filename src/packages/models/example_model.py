import torch.nn as nn
import torch.nn.functional as F


class ExampleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))


if __name__ == "__main__":
    import torch

    model = ExampleModel()

    inp = torch.randn(1, 1, 28, 28)

    out = model(inp)
