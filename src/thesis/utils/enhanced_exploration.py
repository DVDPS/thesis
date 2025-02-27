import os
import torch
import torch.optim as optim
import numpy as np
import logging
from game2048 import Game2048
from agent import PPOAgent
from training import train

# Add safe globals for model loading
torch.serialization.add_safe_globals([np._core.multiarray.scalar, np.dtype])

def resume_with_exploration(
    agent, 
    optimizer, 
    checkpoint_path=None,
    epochs=500, 
    entropy_coef=0.2,
    exploration_noise=1.5,
    min_exploration_noise=0.15,
    learning_rate=0.0005,
    output_dir="checkpoints/high_exploration"
):
    """
    Resume training with increased exploration parameters to help the agent 
    break out of local optima.
    
    Args:
        agent: The PPO agent to train
        optimizer: The optimizer to use
        checkpoint_path: Path to the checkpoint to load (if not already loaded)
        epochs: Number of additional epochs to train
        entropy_coef: Entropy coefficient for the loss function
        exploration_noise: Initial exploration noise value
        min_exploration_noise: Minimum exploration noise value
        learning_rate: Learning rate for the optimizer
        output_dir: Directory to save output models
    """
    # Create environment
    env = Game2048()
    
    # Load checkpoint if provided and agent is not already loaded
    if checkpoint_path and os.path.exists(checkpoint_path):
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        agent.load_state_dict(checkpoint['model_state_dict'])
        
        # Get training info from checkpoint
        start_epoch = checkpoint.get('epoch', 0)
        best_running_reward = checkpoint.get('running_reward', float('-inf'))
        logging.info(f"Loaded checkpoint: epoch={start_epoch}, reward={best_running_reward:.2f}")
    else:
        start_epoch = 0
        best_running_reward = float('-inf')
    
    # Set exploration parameters
    logging.info(f"Setting exploration parameters: noise={exploration_noise}, min_noise={min_exploration_noise}")
    original_noise = agent.exploration_noise
    original_min_noise = agent.min_exploration_noise
    
    agent.exploration_noise = exploration_noise
    agent.min_exploration_noise = min_exploration_noise
    
    # Update learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info(f"Starting enhanced exploration training for {epochs} epochs")
    logging.info(f"Models will be saved to {output_dir}")
    
    # Train with enhanced exploration
    train(
        agent, 
        env, 
        optimizer,
        epochs=epochs,
        mini_batch_size=64,
        ppo_epochs=6,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        entropy_coef=entropy_coef,
        max_grad_norm=0.5,
        steps_per_update=400,
        start_epoch=0,  # Reset epoch counter for this run
        best_running_reward=best_running_reward,
        checkpoint_dir=output_dir
    )
    
    # Restore original exploration values
    agent.exploration_noise = original_noise
    agent.min_exploration_noise = original_min_noise
    
    logging.info(f"Enhanced exploration training complete! Models saved to {output_dir}")
    return output_dir

def balanced_exploration(
    agent, 
    optimizer, 
    checkpoint_path=None,
    epochs=500, 
    entropy_coef=0.15,
    exploration_noise=1.0,
    min_exploration_noise=0.1,
    learning_rate=0.0008,
    output_dir="checkpoints/balanced_exploration"
):
    """
    Resume training with balanced exploration parameters to help the agent 
    explore without being too aggressive or too timid.
    
    Args:
        agent: The PPO agent to train
        optimizer: The optimizer to use
        checkpoint_path: Path to the checkpoint to load (if not already loaded)
        epochs: Number of additional epochs to train
        entropy_coef: Entropy coefficient for the loss function
        exploration_noise: Initial exploration noise value
        min_exploration_noise: Minimum exploration noise value
        learning_rate: Learning rate for the optimizer
        output_dir: Directory to save output models
    """
    # Create environment
    env = Game2048()
    
    # Load checkpoint if provided and agent is not already loaded
    if checkpoint_path and os.path.exists(checkpoint_path):
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        agent.load_state_dict(checkpoint['model_state_dict'])
        
        # Get training info from checkpoint
        start_epoch = checkpoint.get('epoch', 0)
        best_running_reward = checkpoint.get('running_reward', float('-inf'))
        logging.info(f"Loaded checkpoint: epoch={start_epoch}, reward={best_running_reward:.2f}")
    else:
        start_epoch = 0
        best_running_reward = float('-inf')
    
    # Set exploration parameters
    logging.info(f"Setting balanced exploration parameters: noise={exploration_noise}, min_noise={min_exploration_noise}")
    original_noise = agent.exploration_noise
    original_min_noise = agent.min_exploration_noise
    
    agent.exploration_noise = exploration_noise
    agent.min_exploration_noise = min_exploration_noise
    
    # Update learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info(f"Starting balanced exploration training for {epochs} epochs")
    logging.info(f"Models will be saved to {output_dir}")
    
    # Train with enhanced exploration
    train(
        agent, 
        env, 
        optimizer,
        epochs=epochs,
        mini_batch_size=96,
        ppo_epochs=6,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        entropy_coef=entropy_coef,
        max_grad_norm=0.5,
        steps_per_update=400,
        start_epoch=0,  # Reset epoch counter for this run
        best_running_reward=best_running_reward,
        checkpoint_dir=output_dir
    )
    
    # Restore original exploration values
    agent.exploration_noise = original_noise
    agent.min_exploration_noise = original_min_noise
    
    logging.info(f"Balanced exploration training complete! Models saved to {output_dir}")
    return output_dir 