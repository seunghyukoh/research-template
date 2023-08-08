import torch.nn as nn
import torch.nn.functional as F


class ExampleModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.ModuleList([nn.Linear(512, 100), nn.Linear(100, 10)])

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, target=None):
        for i in range(len(self.layers)):
            x = F.relu(self.layers[i](x))

        prob = self.softmax(x)

        if target is None:
            return prob

        loss = F.cross_entropy(x, target)

        return x, loss


if __name__ == "__main__":
    import torch

    model = ExampleModel()

    batch_size = 32

    inp = torch.randn(batch_size, 512)
    label = torch.randint(0, 10, (batch_size,))

    logit, loss = model(inp, label)

    print(loss.item())
