import numpy as np
import torch
import math
from tqdm import tqdm
import time
import os

from src.thesis.environment.parallel_game2048 import ParallelGame2048
from agents.parallel_cnn_agent import ParallelCNNAgent

def get_gpu_memory_usage():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3  # Convert to GB
    return 0

def train_parallel_cnn_agent(
    num_episodes=100000,
    num_envs=256,  # Increased from 64 to 256 for more parallelism
    batch_size=32768,  # Increased batch size for H100
    update_interval=32,  # More frequent updates
    device=None
):
    """Train a CNN agent using parallel environments"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Enable cuDNN benchmarking for better performance
    torch.backends.cudnn.benchmark = True
    
    print(f"Starting Parallel CNN Training for H100...")
    print(f"Using device: {device}")
    
    # Initialize parallel game environment
    parallel_game = ParallelGame2048(num_envs=num_envs)
    
    # Initialize agent with larger buffer size for H100
    agent = ParallelCNNAgent(
        device=device,
        buffer_size=1000000,  # Increased buffer size
        batch_size=batch_size
    )
    
    # Training loop
    episode_rewards = []
    episode_scores = []
    episode_steps = []
    episode_max_tiles = []
    
    # Progress bar
    pbar = tqdm(range(num_episodes), desc="Training")
    
    for episode in pbar:
        # Reset environments
        states = parallel_game.reset_all()
        
        # Initialize episode variables
        episode_reward = 0
        episode_score = 0
        episode_step = 0
        episode_max_tile = 0
        
        # Run episode
        while True:
            # Get actions from agent
            actions = agent.select_actions(states)
            
            # Step environments
            next_states, rewards, dones, infos = parallel_game.step(actions)
            
            # Update episode statistics
            episode_reward += np.mean(rewards)
            episode_score += np.mean(infos['scores'])
            episode_step += 1
            episode_max_tile = max(episode_max_tile, np.max(next_states))
            
            # Store experiences
            for i in range(num_envs):
                agent.store_experience(states[i], rewards[i], next_states[i], dones[i])
            
            # Update network if enough experiences
            if len(agent.replay_buffer) >= batch_size:
                loss = agent.update_batch(num_batches=1)
            
            # Update states
            states = next_states
            
            # Check if all environments are done
            if np.all(dones):
                break
        
        # Record episode statistics
        episode_rewards.append(episode_reward)
        episode_scores.append(episode_score)
        episode_steps.append(episode_step)
        episode_max_tiles.append(episode_max_tile)
        
        # Update progress bar
        pbar.set_postfix({
            'reward': f"{episode_reward:.2f}",
            'score': f"{episode_score:.2f}",
            'steps': episode_step,
            'max_tile': episode_max_tile
        })
        
        # Save best model
        if episode > 0 and episode % 1000 == 0:
            agent.save('best_parallel_cnn_model.pth')
    
    return agent

if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    print("Starting Parallel CNN Training for H100...")
    
    # Adjust these based on your H100 performance
    trained_agent = train_parallel_cnn_agent(
        num_episodes=100000,
        num_envs=256,           # Process 256 games in parallel
        batch_size=32768,       # Large batch size for H100
        update_interval=32      # Update less frequently with larger batches
    )
    
    print("\nTraining complete and best model saved to 'best_parallel_cnn_model.pth'")