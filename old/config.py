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

# Configure logging to be compatible with Windows command prompt
# and prevent encoding errors with non-ASCII characters
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
         logging.FileHandler('training.log', encoding='utf-8'),
         logging.StreamHandler(sys.stdout)  # Explicitly use stdout
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

# H100 optimization settings
def setup_h100_optimizations():
    """Configure optimizations for NVIDIA H100 GPUs."""
    if torch.cuda.is_available():
        # Check if the GPU is an H100
        gpu_name = torch.cuda.get_device_name(0)
        if 'H100' in gpu_name:
            print(f"Detected NVIDIA H100 GPU: {gpu_name}")
            
            # Enable TF32 precision (specific to NVIDIA Ampere and later GPUs)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("Enabled TF32 precision for faster matrix multiplications")
            
            # Set optimal CUDA settings for H100
            torch.backends.cudnn.benchmark = True
            print("Enabled cuDNN benchmark mode for optimized convolutions")
            
            # Configure memory allocator
            torch.cuda.empty_cache()
            torch.cuda.memory.empty_cache()
            print("Configured memory settings for H100")
            
            return True
    
    return False

# Call this function to set up H100 optimizations
h100_available = setup_h100_optimizations()