import os

import datasets

DEFAULT_CACHE_DIR = "./.cache"


def load_datasets(path: str = "", cache_dir: str = DEFAULT_CACHE_DIR):
    if os.path.exists(path):
        datasets.load_from_disk(path)
    else:
        datasets.load_dataset(path, cache_dir=cache_dir)

    print("Dataset loaded.")

    # TODO


if __name__ == "__main__":
    if os.getcwd().endswith("src"):
        """Warning!!! This is a naive implementation.
        If you want to use this script in other directory, please modify this part.
        """
        os.chdir("..")

    # For example
    load_datasets("awettig/Pile-Wikipedia-0.5B-6K-opt")
    load_datasets("awettig/Pile-Github-0.5B-6K-opt")
