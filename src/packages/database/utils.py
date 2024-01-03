import sqlite3
import uuid

from dataclasses import dataclass
from errors import ExperimentNotFoundError


@dataclass
class Experiment:
    DEFAULT_STATE = "waiting"

    id: str
    experiment_name: str
    run_name: str
    experiment_file: str
    config_file: str
    state: str
    created_at: str
    updated_at: str = None

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.validate()

    def validate(self):
        assert self.experiment_name is not None, "experiment_name cannot be None"
        assert self.run_name is not None, "run_name cannot be None"
        assert self.experiment_file is not None, "experiment_file cannot be None"
        assert self.config_file is not None, "config_file cannot be None"
        assert self.state is not None, "state cannot be None"
        assert self.state in [
            "waiting",
            "ongoing",
            "finished",
            "failed",
        ], "state must be one of 'waiting', 'ongoing', 'finished', 'failed'"
        assert self.created_at is not None, "created_at cannot be None"

    @staticmethod
    def create(experiment_name, run_name, experiment_file, config_file):
        return Experiment(
            id=str(uuid.uuid4()),
            experiment_name=experiment_name,
            run_name=run_name,
            experiment_file=experiment_file,
            config_file=config_file,
            state=Experiment.DEFAULT_STATE,
            created_at="",  # TODO: Add created_at
        )

    @staticmethod
    def from_db_to_entity(raw_experiment):
        return Experiment(
            id=raw_experiment[0],
            experiment_name=raw_experiment[1],
            run_name=raw_experiment[2],
            experiment_file=raw_experiment[3],
            config_file=raw_experiment[4],
            state=raw_experiment[5],
            created_at=raw_experiment[6],
            updated_at=raw_experiment[7],
        )

    def to_db(self):
        self.validate()
        return dict(
            id=self.id,
            experiment_name=self.experiment_name,
            run_name=self.run_name,
            experiment_file=self.experiment_file,
            config_file=self.config_file,
            state=self.state,
            created_at=self.created_at,
            updated_at="",  # TODO: Add updated_at
        )

    def update_state(self, state):
        if self.state == "waiting":
            assert state in [
                "ongoing",
                "failed",
            ], "state must be one of 'ongoing', 'failed'"
        elif self.state == "ongoing":
            assert state in [
                "finished",
                "failed",
            ], "state must be one of 'finished', 'failed'"
        elif self.state == "finished":
            raise Exception("Cannot update state of a finished experiment")

        self.state = state


def create(experiment):
    connection = sqlite3.connect("./database.sqlite")

    cursor = connection.cursor()
    experiment = Experiment.create(**experiment)

    query = """INSERT INTO Experiments VALUES(
    :id,
    :experiment_name,
    :run_name,
    :experiment_file,
    :config_file,
    :state,
    (datetime('now', 'localtime')),
    NULL
);"""

    cursor.execute(query, experiment.to_db())

    connection.commit()

    return experiment.id


def find_all():
    connection = sqlite3.connect("./database.sqlite")

    cursor = connection.cursor()

    cursor.execute(
        """SELECT * FROM Experiments""",
    )

    experiments = cursor.fetchall()

    return [Experiment.from_db_to_entity(experiment) for experiment in experiments]


def find_by_id(experiment_id):
    connection = sqlite3.connect("./database.sqlite")

    cursor = connection.cursor()

    cursor.execute(
        """SELECT * FROM Experiments WHERE id = ?""",
        (experiment_id,),
    )

    experiment = cursor.fetchone()

    if experiment is None:
        raise ExperimentNotFoundError(experiment_id)

    return Experiment.from_db_to_entity(experiment)


def find_many_by_experiment_name(experiment_name):
    connection = sqlite3.connect("./database.sqlite")

    cursor = connection.cursor()

    cursor.execute(
        """SELECT * FROM Experiments WHERE experiment_name = ?""",
        (experiment_name,),
    )

    experiments = cursor.fetchall()

    return [Experiment.from_db_to_entity(experiment) for experiment in experiments]


def find_many_by_run_name(run_name):
    connection = sqlite3.connect("./database.sqlite")

    cursor = connection.cursor()

    cursor.execute(
        """SELECT * FROM Experiments WHERE run_name = ?""",
        (run_name,),
    )

    experiments = cursor.fetchall()

    return [Experiment.from_db_to_entity(experiment) for experiment in experiments]


def find_many_by_state(state):
    connection = sqlite3.connect("./database.sqlite")

    cursor = connection.cursor()

    cursor.execute(
        """SELECT * FROM Experiments WHERE state = ?""",
        (state,),
    )

    experiments = cursor.fetchall()

    return [Experiment.from_db_to_entity(experiment) for experiment in experiments]


def update_state_by_id(experiment_id, state):
    experiment = find_by_id(experiment_id)
    experiment.update_state(state)
    experiment.to_db()

    # TODO: Save this experiment entity to the database

    connection = sqlite3.connect("./database.sqlite")

    cursor = connection.cursor()

    cursor.execute(
        """UPDATE Experiments SET state = ? WHERE id = ?""",
        (state, experiment_id),
    )

    connection.commit()


def delete_by_id(experiment_id):
    connection = sqlite3.connect("./database.sqlite")

    cursor = connection.cursor()

    cursor.execute(
        """DELETE FROM Experiments WHERE id = ?""",
        (experiment_id,),
    )

    connection.commit()


if __name__ == "__main__":
    experiment = dict(
        experiment_name="test_experiment",
        run_name="test_run",
        experiment_file="test.py",
        config_file="test.yaml",
    )

    create(experiment)
    create(experiment)

    all_experiments = find_all()

    try:
        print()
        print("Find by id")
        experiment = find_by_id(all_experiments[0].id)
        print(experiment)

        print()
        print("Update state by id")
        update_state_by_id(all_experiments[0].id, "ongoing")

        print()
        print("Find many by experiment name")
        experiments = find_many_by_experiment_name("test_experiment")
        print(experiments)

        print()
        print("Find many by run name")
        experiments = find_many_by_run_name("test_run")
        print(experiments)

        print()
        print("Find many by state=waiting")
        experiments = find_many_by_state("waiting")
        print(experiments)

        print()
        print("Find many by state=ongoing")
        experiments = find_many_by_state("ongoing")
        print(experiments)
    finally:
        print()
        print("Delete All")
        for experiment_id in [experiment.id for experiment in all_experiments]:
            print(f"Deleting experiment {experiment_id}")
            delete_by_id(experiment_id)
