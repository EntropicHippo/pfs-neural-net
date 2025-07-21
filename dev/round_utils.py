# round_utils.py
import numpy as np
import torch
from config import Config

def smart_round(time_spent: np.ndarray,
                completion: np.ndarray,
                class_info: np.ndarray,
                cfg: Config
                ) -> tuple:
    """
    Round time allocations to multiples of class requirements.
    """
    class_req = class_info[:, 0]
    data = np.round(time_spent / class_req) * class_req
    min_req = class_req.min()
    n_fibers, n_classes = data.shape

    # Greedy trimming
    for i in range(n_fibers):
        while data[i].sum() > cfg.total_time and (data[i] > 0).any():
            eligible = np.where(data[i] > 0)[0]
            c = eligible[np.argmax(completion[eligible])]
            data[i, c] -= class_req[c]

    # Greedy filling
    for i in range(n_fibers):
        while True:
            leftover = cfg.total_time - data[i].sum()
            if leftover < min_req:
                break
            eligible = np.where(class_req <= leftover)[0]
            c = eligible[np.argmin(completion[eligible])]
            data[i, c] += class_req[c]

    updated_time = torch.tensor(data.flatten(), dtype=torch.float)
    updated_fiber_time = data.sum(axis=1)
    updated_n_counts = data / class_req
    updated_completion = updated_n_counts.sum(axis=0) / (class_info[:, 1] / cfg.n_fields)
    updated_utility = updated_completion.min()
    return data, updated_utility, updated_time, updated_fiber_time, updated_completion
