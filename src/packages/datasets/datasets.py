from ..utils import generate_unique_id
from .base import BaseSFTDataset


class GSM8KDataset(BaseSFTDataset):
    DATASET_PATH = "openai/gsm8k"
    DATASET_KWARGS = {"name": "main"}

    @property
    def validation(self):
        # GSM8K does not have a validation set
        return self.dataset["test"]

    @property
    def test(self):
        return self.dataset["test"]

    def set_uid(self):
        self.dataset = self.dataset.map(
            lambda x: {
                "uid": generate_unique_id(
                    {
                        "answer": x["answer"],
                        "question": x["question"],
                    }
                )
            },
            num_proc=self.num_workers,
            desc="Setting UID",
        )

    def preprocess_dataset(self):
        self.dataset = self.dataset.map(
            lambda x: {
                "messages": [
                    {"role": "user", "content": x["question"]},
                    {"role": "assistant", "content": x["answer"]},
                ]
            },
            num_proc=self.num_workers,
            desc="Preprocessing dataset",
        )


# Implement datasets here
