import os
import shutil
from pathlib import Path

import yaml

paths = os.listdir("outputs")
for path in paths:
    print(path)

    output_path = Path("outputs") / path

    if output_path.is_dir():
        status_file = output_path / ".hydra" / "status.yaml"
        if not status_file.exists():
            shutil.rmtree(output_path)
            continue

        with open(status_file, "r") as f:
            status = yaml.safe_load(f)
            if status["status"] != "success":
                shutil.rmtree(output_path)
