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
        dones = np.zeros(num_envs, dtype=bool)
        
        # Initialize episode variables
        episode_reward = 0
        episode_score = 0
        episode_step = 0
        episode_max_tile = 0
        
        # Run episode
        while True:
            # Select actions for active environments
            actions = np.zeros(num_envs, dtype=np.int32)
            for i in range(num_envs):
                if dones[i]:
                    continue
                    
                if np.random.rand() < agent.epsilon:
                    # Random action
                    valid_moves = parallel_game.get_valid_moves(i)
                    if valid_moves:
                        actions[i] = np.random.choice(valid_moves)
                    else:
                        dones[i] = True
                        continue
                else:
                    # Greedy action selection
                    action_values = agent.batch_evaluate_actions(states[np.newaxis, i], parallel_game, env_idx=i)
                    if action_values is not None and len(action_values) > 0 and len(action_values[0]) > 0:
                        try:
                            # Find best action based on value
                            best_action, _, _, best_value = max(action_values[0], key=lambda x: x[3])
                            actions[i] = best_action
                        except (IndexError, TypeError):
                            # If there's an error with the tuple format, choose random
                            valid_moves = parallel_game.get_valid_moves(i)
                            if valid_moves:
                                actions[i] = np.random.choice(valid_moves)
                            else:
                                dones[i] = True
                                continue
                    else:
                        # No valid moves, mark as done
                        dones[i] = True
                        continue
            
            # Step environments
            next_states, rewards, new_dones, infos = parallel_game.step(actions)
            
            # Convert PyTorch tensors to NumPy arrays for statistics
            rewards_np = rewards.cpu().numpy() if isinstance(rewards, torch.Tensor) else rewards
            scores_np = infos['scores'].cpu().numpy() if isinstance(infos['scores'], torch.Tensor) else infos['scores']
            next_states_np = next_states.cpu().numpy() if isinstance(next_states, torch.Tensor) else next_states
            dones_np = new_dones.cpu().numpy() if isinstance(new_dones, torch.Tensor) else new_dones
            
            # Update episode statistics
            episode_reward += np.mean(rewards_np)
            episode_score += np.mean(scores_np)
            episode_step += 1
            episode_max_tile = max(episode_max_tile, np.max(next_states_np))
            
            # Store experiences
            for i in range(num_envs):
                if not dones[i]:
                    agent.store_experience(
                        states[i].cpu().numpy() if isinstance(states[i], torch.Tensor) else states[i],
                        rewards_np[i],
                        next_states_np[i],
                        dones_np[i]
                    )
            
            # Update network if enough experiences
            if len(agent.replay_buffer) >= batch_size:
                loss = agent.update_batch(num_batches=1)
            
            # Update states and done flags
            states = next_states
            dones = dones_np  # Use NumPy array for dones
            
            # Check if all environments are done
            if np.all(dones):
                break
        
        # Record episode statistics
        episode_rewards.append(episode_reward)
        episode_scores.append(episode_score)
        episode_steps.append(episode_step)
        episode_max_tiles.append(episode_max_tile)
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Update progress bar
        pbar.set_postfix({
            'reward': f"{episode_reward:.2f}",
            'score': f"{episode_score:.2f}",
            'steps': episode_step,
            'max_tile': episode_max_tile,
            'epsilon': f"{agent.epsilon:.3f}"
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