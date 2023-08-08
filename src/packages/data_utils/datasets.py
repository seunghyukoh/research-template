import os

import datasets


def load_raw_datasets(data_args, model_args):
    dataset_name = data_args.dataset_name
    dataset_config_name = data_args.dataset_config_name
    cache_dir = model_args.cache_dir if model_args.cache_dir else None
    use_auth_token = True if model_args.use_auth_token else None
    validation_split_percentage = data_args.validation_split_percentage

    is_dataset_ready = data_args.dataset_name is not None
    if is_dataset_ready:
        datasets = _load_from_repository(
            dataset_name,
            dataset_config_name,
            cache_dir,
            use_auth_token,
            validation_split_percentage,
        )
    else:
        train_file = data_args.train_file
        validation_file = data_args.validation_file
        keep_linebreaks = data_args.keep_linebreaks
        datasets = _load_from_file(
            train_file,
            validation_file,
            keep_linebreaks,
            cache_dir,
            use_auth_token,
            validation_split_percentage,
            **data_args,
        )

    return datasets


def _load_from_repository(
    dataset_name,
    dataset_config_name,
    cache_dir,
    use_auth_token,
    validation_split_percentage,
):
    raw_datasets = datasets.load_dataset(
        dataset_name,
        dataset_config_name,
        cache_dir=cache_dir,
        use_auth_token=use_auth_token,
    )

    have_validation_split = "validation" in raw_datasets.keys()

    if not have_validation_split:
        raw_datasets["train"] = datasets.load_dataset(
            dataset_name,
            dataset_config_name,
            split=f"train[{validation_split_percentage}%:]",
            cache_dir=cache_dir,
            use_auth_token=use_auth_token,
        )
        raw_datasets["validation"] = datasets.load_dataset(
            dataset_name,
            dataset_config_name,
            split=f"train[:{validation_split_percentage}%]",
            cache_dir=cache_dir,
            use_auth_token=use_auth_token,
        )


def _load_from_file(
    train_file,
    validation_file,
    keep_linebreaks,
    cache_dir,
    use_auth_token,
    validation_split_percentage,
    **kwargs,
):
    data_files = {}
    dataset_args = {}
    if train_file is not None:
        data_files["train"] = train_file
    if validation_file is not None:
        data_files["validation"] = validation_file

    extension = (
        train_file.split(".")[-1]
        if train_file is not None
        else validation_file.split(".")[-1]
    )

    if extension == "txt":
        extension
        dataset_args["keep_linebreaks"] = keep_linebreaks

    raw_datasets = datasets.load_dataset(
        extension,
        data_files=data_files,
        cache_dir=cache_dir,
        use_auth_token=use_auth_token,
        **kwargs,
    )

    have_validation_split = "validation" in raw_datasets.keys()

    if not have_validation_split:
        raw_datasets["train"] = datasets.load_dataset(
            extension,
            data_files=data_files,
            split=f"train[{validation_split_percentage}%:]",
            cache_dir=cache_dir,
            use_auth_token=use_auth_token,
            **kwargs,
        )
        raw_datasets["validation"] = datasets.load_dataset(
            extension,
            data_files=data_files,
            split=f"train[:{validation_split_percentage}%]",
            cache_dir=cache_dir,
            use_auth_token=use_auth_token,
            **kwargs,
        )


def load_preprocessed_datasets(data_args, model_args):
    assert data_args.preprocessed_train_datasets is not None
    assert data_args.preprocessed_validation_datasets is not None

    cache_dir = model_args.cache_dir if model_args.cache_dir else None
    use_auth_token = True if model_args.use_auth_token else None

    dataset_dict = {}
    for train_file in data_args.preprocessed_train_datasets:
        name = os.path.basename(train_file).split(".")[0]
        if os.path.exists(train_file):
            data = datasets.load_from_disk(train_file)
        else:
            data = datasets.load_dataset(
                train_file,
                split="train",
                cache_dir=cache_dir,
                use_auth_token=use_auth_token,
            )

        dataset_dict[f"train-{name}"] = data
        print(f"Loaded {train_file} training data, {len(data)} examples")

    for valid_file in data_args.preprocessed_validation_datasets:
        name = os.path.basename(valid_file).split(".")[0]
        if os.path.exists(valid_file):
            data = datasets.load_from_disk(valid_file)
        else:
            data = datasets.load_dataset(
                valid_file,
                split="test",
                cache_dir=cache_dir,
                use_auth_token=use_auth_token,
            )

        dataset_dict[f"validation-{name}"] = data
        print(f"Loaded {valid_file} validation data, {len(data)} examples")

    train_data = []
    for key in dataset_dict.keys():
        if key.startswith("train"):
            train_data.append(dataset_dict[key])

    dataset_dict["train"] = datasets.concatenate_datasets(train_data)

    lm_datasets = datasets.dataset_dict.DatasetDict(dataset_dict)
    return lm_datasets
