import torch

NUM_VISIBLE_DEVICES = torch.cuda.device_count()


def get_devices(total_processes=1, index=0):
    if not torch.cuda.is_available() or NUM_VISIBLE_DEVICES == 0:
        raise RuntimeError("CUDA is not available.")

    num_usable_devices = NUM_VISIBLE_DEVICES // total_processes

    devices = list(range(index * num_usable_devices, (index + 1) * num_usable_devices))

    return devices
