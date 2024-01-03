# TODO: Add docstrings to all errors


class ExperimentNotFoundError(Exception):
    def __init__(self, experiment_id):
        self.experiment_id = experiment_id

    def __str__(self):
        return f"Experiment with id {self.experiment_id} not found"


class ExperimentAlreadyExistsError(Exception):
    pass


class ExperimentNotRunningError(Exception):
    pass


class ExperimentNotFinishedError(Exception):
    pass
