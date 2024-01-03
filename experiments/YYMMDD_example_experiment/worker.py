import os
from datetime import datetime
import json
from enum import IntEnum, auto

FLAG_DIR = "./flags"


class ExperimentState(IntEnum):
    PENDING = auto()
    RUNNING = auto()
    FINISHED = auto()
    FAILED = auto()

    def __str__(self):
        return self.value


def list_flags():
    return os.listdir(FLAG_DIR)


def flag_path(flag_name):
    return os.path.join(FLAG_DIR, flag_name)


def read_flag(flag_name):
    with open(flag_path(flag_name), "r") as f:
        return json.load(f)


def make_flag(value, now):
    assert value is not None, "Flag value must be specified"

    flag_name = value["name"]
    assert flag_name is not None, "Flag name must be specified"
    assert now is not None

    value.update(dict(created_at=now.isoformat()))

    with open(flag_path(f"{flag_name}.json"), "w") as f:
        json.dump(value, f)


def update_flag(value, now):
    assert value is not None, "Flag value must be specified"

    flag_name = value["name"]
    assert flag_name is not None, "Flag name must be specified"
    assert now is not None

    value.update(dict(updated_at=now.isoformat()))

    with open(flag_path(f"{flag_name}.json"), "w") as f:
        json.dump(value, f)


def process(value):
    import time

    time.sleep(10)

    value.update(dict(state=ExperimentState.FINISHED))

    update_flag(value, datetime.now())


def worker(worker_id=0):
    while True:
        all_flags = list_flags()

        for flag in all_flags:
            experiment_details = read_flag(flag)

            if experiment_details["state"] == ExperimentState.PENDING:
                now = datetime.now()
                experiment_details.update(dict(state=ExperimentState.RUNNING))
                update_flag(experiment_details, now)

                print(
                    f"[Worker {worker_id}] Starting experiment {experiment_details['name']}"
                )

                process(experiment_details)

                print(
                    f"[Worker {worker_id}] Finished experiment {experiment_details['name']}"
                )


if __name__ == "__main__":
    now = datetime.now()

    configs = [
        {"name": f"experiment{i}", "state": ExperimentState.PENDING} for i in range(5)
    ]

    for config in configs:
        make_flag(config, now)

    import multiprocessing
    import time

    for i in range(2):
        multiprocessing.Process(target=worker, args=(i,)).start()
        time.sleep(1)
