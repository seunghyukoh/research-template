import hashlib

import yaml


def hash_config(config: dict) -> str:
    return str(int(hashlib.sha256(str(config).encode()).hexdigest(), 16) % (2**63))


def build_experiment(
    name: str,
    description: str,
    runs: list,
    num_workers: int = 8,
    resume: bool = True,
    skip_killed: bool = True,
    skip_crashed: bool = True,
    skip_failed: bool = True,
    tags: list = None,
    resources: dict = None,
    shared_args: dict = None,
    shared_custom_args: dict = None,
    output_path: str = None,
) -> dict:
    """Build an experiment configuration with GPU resource support.

    Args:
        name: Experiment name
        description: Experiment description
        runs: List of run configurations. Each run can have:
            - command: Command to execute
            - args: Hydra arguments
            - custom_args: Custom Hydra arguments
            - resources: Resource requirements (e.g., {"num_gpus": 1.0})
        num_workers: Number of CPU workers
        resume: Whether to resume runs
        skip_killed: Whether to skip killed runs
        skip_crashed: Whether to skip crashed runs
        skip_failed: Whether to skip failed runs
        tags: List of tags for the experiment
        resources: Global resource requirements for all runs (used if run doesn't specify resources)
        shared_args: Shared Hydra arguments for all runs
        shared_custom_args: Shared custom Hydra arguments for all runs
        output_path: Path to save the YAML file (optional)

    Returns:
        Experiment configuration dict
    """
    tags = tags or []
    shared_args = shared_args or {}
    shared_custom_args = shared_custom_args or {}

    # Generate run IDs for runs that don't have them
    for run in runs:
        if "args" not in run:
            run["args"] = {}
        if "logging" not in run["args"]:
            run["args"]["logging"] = {}
        if "run_id" not in run["args"]["logging"]:
            run_id = f"run_{hash_config(run)}"
            run["args"]["logging"]["run_id"] = run_id

    experiment_config = dict(
        name=name,
        description=description,
        tags=tags,
        num_workers=num_workers,
        resume=resume,
        skip_killed=skip_killed,
        skip_crashed=skip_crashed,
        skip_failed=skip_failed,
        resources=resources,
        shared_args=shared_args,
        shared_custom_args=shared_custom_args,
        runs=runs,
    )

    if output_path:
        yaml.dump(experiment_config, open(output_path, "w"))

    return experiment_config


def build_experiment_001_demo_sft():
    name = "001-demo-sft"
    description = "Demo SFT experiment"
    tags = ["sft", "debug"]
    num_workers = 8
    resume = True
    skip_killed = True
    skip_crashed = True
    skip_failed = True
    resources = dict(
        num_gpus=1.0,  # Request 1 full GPU per run
    )
    # Arguments
    batch_size = 64

    shared_args = dict(
        logging=dict(
            exp_id="${name}",
            resume="${ifelse:${resume}, auto, never}",
        ),
        # debug=dict(
        #     dry_run=True,
        # ),
        training=dict(
            per_device_train_batch_size=batch_size,
        ),
    )
    shared_custom_args = dict(
        logging=dict(
            tags="${tags}",
        ),
    )

    runs = []

    for learning_rate in [1e-4, 1e-5, 1e-6]:
        run_config = dict(
            command="python run_sft.py",
            args=dict(
                logging=dict(
                    run_name=f"lr_{learning_rate}",
                ),
                training=dict(
                    learning_rate=learning_rate,
                ),
            ),
        )

        runs.append(run_config)

    build_experiment(
        name=name,
        description=description,
        tags=tags,
        num_workers=num_workers,
        resume=resume,
        skip_killed=skip_killed,
        skip_crashed=skip_crashed,
        skip_failed=skip_failed,
        resources=resources,
        shared_args=shared_args,
        shared_custom_args=shared_custom_args,
        runs=runs,
        output_path="experiments/001-demo-sft.yaml",
    )


if __name__ == "__main__":
    build_experiment_001_demo_sft()
