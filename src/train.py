from torch.utils.data import DataLoader

from packages.data_utils import ExampleDataset
from packages.models import ExampleModel


def train():
    dataset = ExampleDataset()
    data_loader = DataLoader(dataset, batch_size=32)

    model = ExampleModel()

    for idx, (data, label) in enumerate(data_loader):
        out = model(data)


if __name__ == "__main__":
    train()
