import logging
from abc import ABC, abstractmethod

import torch
from transformers import PreTrainedTokenizer

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BaseSFTDataset(ABC):
    DATASET_PATH = None
    DATASET_KWARGS = {}

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        dataset_path: str | None = None,
        max_train_samples: int = -1,
        max_validation_samples: int = -1,
        max_test_samples: int = -1,
        cleanup_cache_files: bool = False,
        only_inputs: bool = False,
        num_workers: int = 8,
        shuffle_seed: int = None,
        **kwargs,
    ):
        if dataset_path is None:
            self.dataset = load_dataset(path=self.DATASET_PATH, **self.DATASET_KWARGS)
        else:
            self.dataset = load_from_disk(dataset_path)

        if cleanup_cache_files:
            self.dataset.cleanup_cache_files()

        self.num_workers = num_workers
        self.preprocess_dataset()

        if shuffle_seed is not None:
            logger.log(logging.INFO, f"Shuffling dataset with seed {shuffle_seed}")
            self.dataset = self.dataset.shuffle(seed=shuffle_seed)

        # Limit the number of samples
        assert isinstance(max_train_samples, int)
        assert isinstance(max_validation_samples, int)
        assert isinstance(max_test_samples, int)

        if max_train_samples > 0:
            max_train_samples = min(max_train_samples, len(self.train))
            self.dataset["train"] = self.train.select(range(max_train_samples))
        if max_validation_samples > 0:
            max_validation_samples = min(max_validation_samples, len(self.validation))
            self.dataset["validation"] = self.validation.select(
                range(max_validation_samples)
            )
        if max_test_samples > 0:
            max_test_samples = min(max_test_samples, len(self.test))
            self.dataset["test"] = self.test.select(range(max_test_samples))

        # Set UID
        self.set_uid()

        self.tokenizer = tokenizer
        self.has_chat_template = tokenizer.chat_template is not None

        assert self.has_chat_template, "Chat template is not set"
        self.tokenize(only_inputs=only_inputs)

    @property
    def train(self):
        return self.dataset["train"]

    @property
    def validation(self):
        return self.dataset["validation"]

    @property
    def test(self):
        return self.dataset["test"]

    @abstractmethod
    def set_uid(self):
        pass

    @abstractmethod
    def preprocess_dataset(self):
        pass

    def tokenize(self, only_inputs=False):
        def _subroutine(sample):
            messages = sample["messages"]

            encodings = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=only_inputs,
            )

            return {
                "input_ids": encodings.input_ids[0],
                "attention_mask": encodings.attention_mask[0],
                "labels": encodings.input_ids.clone()[0] if not only_inputs else None,
            }

        self.dataset = self.dataset.map(
            _subroutine,
            batched=False,
            num_proc=self.num_workers,
            desc="Tokenizing",
        )


class RandomInputDataset:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        input_length: int,
        label_length: int,
        max_train_samples: int = -1,
        max_validation_samples: int = -1,
        max_test_samples: int = -1,
        cleanup_cache_files: bool = False,
        num_workers: int = 8,
        shuffle_seed: int = None,
        **kwargs,
    ):
        self.tokenizer = tokenizer

        self.input_length = input_length
        self.label_length = label_length

        self.max_train_samples = max_train_samples
        self.max_validation_samples = max_validation_samples
        self.max_test_samples = max_test_samples

        self.num_workers = num_workers
        self.shuffle_seed = shuffle_seed

        self._make_dataset()

        if cleanup_cache_files:
            self.dataset.cleanup_cache_files()

        if shuffle_seed is not None:
            logger.log(logging.INFO, f"Shuffling dataset with seed {shuffle_seed}")
            self.dataset = self.dataset.shuffle(seed=shuffle_seed)

        self._set_uid()

    @property
    def train(self):
        return self.dataset["train"]

    @property
    def validation(self):
        return self.dataset["validation"]

    @property
    def test(self):
        return self.dataset["test"]

    def _make_dataset(self):
        assert self.input_length > 0
        assert self.label_length >= 0

        only_inputs = self.label_length == 0

        self._make_inputs()
        if not only_inputs:
            self._make_labels()

    def _make_inputs(self):
        train_inputs = self._make_input(self.max_train_samples)
        validation_inputs = self._make_input(self.max_validation_samples)
        test_inputs = self._make_input(self.max_test_samples)

        train = Dataset.from_dict(
            {
                "input_ids": [x.clone() for x in train_inputs],
                "attention_mask": [torch.ones_like(x) for x in train_inputs],
            }
        )
        validation = Dataset.from_dict(
            {
                "input_ids": [x.clone() for x in validation_inputs],
                "attention_mask": [torch.ones_like(x) for x in validation_inputs],
            }
        )
        test = Dataset.from_dict(
            {
                "input_ids": [x.clone() for x in test_inputs],
                "attention_mask": [torch.ones_like(x) for x in test_inputs],
            }
        )

        self.dataset = DatasetDict(
            {
                "train": train,
                "validation": validation,
                "test": test,
            }
        )

    def _make_input(self, num_samples):
        return torch.randint(
            0, self.tokenizer.vocab_size, (num_samples, self.input_length)
        )

    def _make_labels(self):
        assert self.dataset is not None

        def subroutine(sample):
            input_ids = sample["input_ids"]
            labels = input_ids.clone()
            labels[: -self.label_length] = -100

            return {
                "labels": labels,
            }

        self.dataset = self.dataset.map(
            subroutine,
            batched=False,
            num_proc=self.num_workers,
            desc="Making labels...",
        )

    def _set_uid(self):
        import uuid

        for split in ["train", "validation", "test"]:
            self.dataset[split] = self.dataset[split].map(
                lambda x: {
                    "uid": str(uuid.uuid4()),
                    **x,
                },
                batched=False,
                num_proc=self.num_workers,
                desc="Setting UID...",
            )
