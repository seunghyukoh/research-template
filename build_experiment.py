import hashlib

import yaml


def hash_config(config: dict) -> str:
    return str(int(hashlib.sha256(str(config).encode()).hexdigest(), 16) % (2**63))


def build_experiment_001_demo_sft():
    name = "001-demo-sft"
    description = "Demo SFT experiment"
    tags = ["sft", "debug"]
    num_workers = 8
    resume = True
    skip_killed = True
    skip_crashed = True
    skip_failed = True

    # Arguments
    batch_size = 64

    shared_args = dict(
        logging=dict(
            exp_id="${name}",
            resume="${ifelse:${resume}, auto, never}",
        ),
        debug=dict(
            dry_run=True,
        ),
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

        run_id = f"run_{hash_config(run_config)}"
        run_config["args"]["logging"]["run_id"] = run_id

        runs.append(run_config)

    yaml.dump(
        dict(
            name=name,
            description=description,
            tags=tags,
            num_workers=num_workers,
            resume=resume,
            skip_killed=skip_killed,
            skip_crashed=skip_crashed,
            skip_failed=skip_failed,
            shared_args=shared_args,
            shared_custom_args=shared_custom_args,
            runs=runs,
        ),
        open("experiments/001-demo-sft.yaml", "w"),
    )


if __name__ == "__main__":
    build_experiment_001_demo_sft()
