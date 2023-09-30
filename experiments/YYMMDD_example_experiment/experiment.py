import os
import sys


def get_workspace():
    from dotenv import load_dotenv

    load_dotenv()

    workspace_name = os.getenv("WORKSPACE_NAME")

    workspace_dir_path = os.path.abspath(__file__).split(workspace_name)[0]
    workspace_path = os.path.join(workspace_dir_path, workspace_name)

    return workspace_path


def cd_to_root():
    workspace = get_workspace()
    os.chdir(workspace)


cd_to_root()
sys.path.append("./src")


### End of snippet ###
from packages.experiments import WandBExperiment
from packages.utils import tracker_init


class ExampleExperiment(WandBExperiment):
    def update_args(self):
        model_args, data_args, training_args, experimental_args = self.config

        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.experimental_args = experimental_args

    def run(self):
        with tracker_init(name=self.run_name, config=self.config_dict) as tracker:
            self.tracker = tracker
            return self._run()

    def _run(self):
        self.tracker.log({"accuracy": 0.2}, step=0)
        self.tracker.log({"accuracy": 0.3}, step=1)
        self.tracker.log({"accuracy": 0.4}, step=2)


if __name__ == "__main__":
    from packages.args import parse_args

    config, config_dict, run_name = parse_args()

    experiment = ExampleExperiment(
        name="YYMMDD_example_experiment",
        run_name=run_name,
        config=config,
        config_dict=config_dict,
    )

    experiment.run()
