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

from experiment import Experiment

from args import parse_args

if __name__ == "__main__":
    config, config_dict, run_name = parse_args()

    experiment = Experiment(
        run_name=run_name,
        config=config,
        config_dict=config_dict,
        use_wandb=True,
    )

    experiment.run()
