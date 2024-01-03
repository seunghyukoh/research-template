import json
import subprocess

import torch

NUM_VISIBLE_DEVICES = torch.cuda.device_count()
DEFAULT_ATTRIBUTES = (
    "index",
    "uuid",
    "name",
    "timestamp",
    "memory.total",
    "memory.free",
    "memory.used",
    "utilization.gpu",
    "utilization.memory",
)


def get_devices(total_processes=1, index=0):
    if not torch.cuda.is_available() or NUM_VISIBLE_DEVICES == 0:
        raise RuntimeError("CUDA is not available.")

    num_usable_devices = NUM_VISIBLE_DEVICES // total_processes

    devices = list(range(index * num_usable_devices, (index + 1) * num_usable_devices))

    return devices


def get_gpu_info(nvidia_smi_path="nvidia-smi", keys=DEFAULT_ATTRIBUTES, no_units=True):
    nu_opt = "" if not no_units else ",nounits"
    cmd = "%s --query-gpu=%s --format=csv,noheader%s" % (
        nvidia_smi_path,
        ",".join(keys),
        nu_opt,
    )

    output = subprocess.check_output(cmd, shell=True)
    lines = output.decode().split("\n")
    lines = [line.strip() for line in lines if line.strip() != ""]

    return [{k: v for k, v in zip(keys, line.split(", "))} for line in lines]


get_gpu_info()
