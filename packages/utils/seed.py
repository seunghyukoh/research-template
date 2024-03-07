import random

import numpy as np
import torch


def random_seeder(seed):
    """Fix randomness"""
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
