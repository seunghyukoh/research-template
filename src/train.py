from accelerate import Accelerator
from torch.optim import Adam
from torch.utils.data import DataLoader

from packages.data_utils import ExampleDataset
from packages.models import ExampleModel


def train():
    accelerator = Accelerator()

    dataset = ExampleDataset()
    dataloader = DataLoader(dataset, batch_size=32)

    model = ExampleModel()

    optimizer = Adam(model.parameters(), lr=1e-3)

    model, dataloader, optimizer = accelerator.prepare(model, dataloader, optimizer)

    for idx, (data, label) in enumerate(dataloader):
        out = model(data)

        loss = out - label  # TODO: Replace with actual loss function

        accelerator.backward(loss)

        optimizer.step()
        optimizer.zero_grad()

    accelerator.end_training()
    accelerator.save_state("checkpoints/accelerator_state.pt")


if __name__ == "__main__":
    train()
