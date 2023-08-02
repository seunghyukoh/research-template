import torch
from torch.utils.data import Dataset


class ExampleDataset(Dataset):
    def __init__(self):
        super().__init__()

        self.data = torch.rand(100, 1, 28, 28)
        self.labels = torch.randint(0, 10, (100,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


if __name__ == "__main__":
    dataset = ExampleDataset()
