import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

from wandb import Api

load_dotenv()

WANDB_ENTITY = os.getenv("WANDB_ENTITY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT")

api = Api()

stable_runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}", filters={"group": "stable"})
stable_run_ids = set([run.id for run in stable_runs])

paths = os.listdir("outputs")
unstable_run_ids = set(paths) - stable_run_ids

if len(unstable_run_ids) == 0:
    print("No unstable runs found")
    sys.exit(0)

print(f"We found {len(unstable_run_ids)} unstable runs")
print("First 5 directories to be removed:")
for i, path in enumerate(list(unstable_run_ids)[:5]):
    print(f"  {path}")

confirmation = input("\nDo you want to proceed with removing these directories? (y/N): ")
if confirmation.lower() != "y":
    print("Aborted.")
    sys.exit(0)

print("Removing unstable runs...")

for path in list(unstable_run_ids):
    print(f"Removing output: {path}")
    shutil.rmtree(Path("outputs") / path)
