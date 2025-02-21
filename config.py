import torch
import numpy as np
import random
import logging
import os
import sys
import time
import torch.backends.cudnn as cudnn

def set_seeds(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

set_seeds(42)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
         logging.FileHandler('training.log'),
         logging.StreamHandler()
    ]
)

# External hyperparameters dictionary for easy configuration.
HYPERPARAMS = {
    "learning_rate_initial": 3e-4,
    "learning_rate_final": 1e-4,
    "reward_penalty_invalid": -2,
    "reward_penalty_game_over": -100,
    "corner_weight": 0.01,  # bonus weight factor for corner heuristic (I set it because having the highest pieces in the corners is a good strategy based on personal experience)
    "entropy_coef_initial": 0.1,
    "entropy_coef_final": 0.05
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}") 