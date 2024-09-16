#!/usr/bin/env python

"""
Implements diverse, frequently used functions.
"""

### IMPORTS ###
import const
import torch
import os
import numpy
import random
import numpy as np
import const


### FUNCTIONS ###

def set_gpu() -> torch.device:
    """
    Set the device (GPU or CPU) for training and inference

    Returns
    -------
    device : torch.device
        The device 
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} for training and inference")
    
    return device


def set_seed(seed: int = const.SEED) -> None:
    """
    Set random seeds for reproducability of results

    Parameters
    ----------
    seed : int, optional
        Random seed number.
    """

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

DEVICE = set_gpu()


