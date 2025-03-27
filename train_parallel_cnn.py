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

def train_parallel_cnn_agent(num_episodes=100000, num_envs=64, epsilon_start=0.5, 
                             batch_size=16384, update_interval=64):
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize agent with large batch size for H100
    agent = ParallelCNNAgent(device, buffer_size=500000, batch_size=batch_size)
    
    # Initialize parallel environments
    parallel_game = ParallelGame2048(num_envs=num_envs, seed=42)
    
    # Print initial GPU memory usage
    print(f"Initial GPU Memory Usage: {get_gpu_memory_usage():.2f} GB")
    print(f"Initial replay buffer size: {len(agent.replay_buffer)}")
    
    # Training statistics
    best_score = 0
    best_max_tile = 0
    episode_rewards = []
    episode_scores = []
    episode_max_tiles = []
    episode_steps = []
    episode_losses = []
    
    # Maximum tile tracking
    max_tile_count = {}
    
    # Progressive training parameters
    initial_update_freq = 100
    final_update_freq = 1000
    min_epsilon = 0.1
    
    # Enhanced milestone rewards
    milestone_rewards = {
        512: 50,
        1024: 150,
        2048: 400,
        4096: 1000,
        8192: 2500
    }
    
    # Batch processing parameters
    max_steps_per_episode = 2000
    min_experiences = 10000
    max_batches = 16
    
    # Track processing time
    start_time = time.time()
    total_steps = 0
    
    # Training loop with progress bar
    for episode in tqdm(range(0, num_episodes, num_envs)):
        # Track per-episode statistics for each environment
        env_rewards = [0] * num_envs
        env_scores = [0] * num_envs
        env_max_tiles = [0] * num_envs
        env_steps = [0] * num_envs
        
        # Reset all environments
        states = parallel_game.reset_all()
        dones = np.zeros(num_envs, dtype=bool)
        
        # Track tiles across all environments
        highest_tiles = np.zeros(num_envs)
        
        # Dynamic epsilon decay (exponential)
        current_epsilon = max(min_epsilon, epsilon_start * (0.9999 ** episode))
        
        # Smooth target network update frequency
        progress = min(1.0, episode / 10000)
        agent.target_update_frequency = int(initial_update_freq + progress * (final_update_freq - initial_update_freq))
        
        # Episode loop
        for step in range(max_steps_per_episode):
            # For active environments, choose actions
            actions = np.zeros(num_envs, dtype=np.int32)
            
            # Choose epsilon-greedy actions
            for i in range(num_envs):
                if dones[i]:
                    continue
                    
                if np.random.rand() < current_epsilon:
                    # Random action
                    valid_moves = parallel_game.get_valid_moves(i)
                    if valid_moves:
                        actions[i] = np.random.choice(valid_moves)
                    else:
                        dones[i] = True
                        continue
                else:
                    # Greedy action selection
                    action_values = agent.batch_evaluate_actions(states[np.newaxis, i], parallel_game)
                    if action_values[0]:
                        # Find best action
                        best_action, _, best_value = max(action_values[0], key=lambda x: x[2])
                        actions[i] = best_action
                    else:
                        dones[i] = True
                        continue
            
            # Take a step in all environments
            prev_states = states.copy()
            next_states, rewards, new_dones, infos = parallel_game.step(actions)
            
            # Update statistics and store experiences
            for i in range(num_envs):
                if dones[i]:
                    continue
                    
                # Update done status
                if new_dones[i]:
                    dones[i] = True
                
                # Calculate additional rewards
                base_reward = rewards[i].item() / 100.0
                env_rewards[i] += base_reward
                
                # Track score
                env_scores[i] = infos["scores"][i].item()
                
                # Track maximum tile
                current_max_tile = np.max(next_states[i])
                if current_max_tile > env_max_tiles[i]:
                    env_max_tiles[i] = current_max_tile
                
                # Add milestone rewards
                if current_max_tile in milestone_rewards and current_max_tile > highest_tiles[i]:
                    env_rewards[i] += milestone_rewards[current_max_tile]
                    highest_tiles[i] = current_max_tile
                
                # Update global max tile count
                max_tile = int(current_max_tile)
                max_tile_count[max_tile] = max_tile_count.get(max_tile, 0) + 1
                
                # Store experience
                agent.store_experience(
                    prev_states[i],
                    base_reward,
                    next_states[i],
                    new_dones[i]
                )
                
                # Count steps
                env_steps[i] += 1
                total_steps += 1
            
            # Check if all environments are done
            if np.all(dones):
                break
                
            # Periodically update the model with large batches
            if (total_steps % update_interval == 0) and (len(agent.replay_buffer) >= min_experiences):
                num_batches = min(max_batches, len(agent.replay_buffer) // batch_size)
                if num_batches > 0:
                    batch_loss = agent.update_batch(num_batches=num_batches)
                    episode_losses.append(batch_loss)
        
        # Record episode statistics
        for i in range(num_envs):
            if env_steps[i] > 0:  # Only count environments that actually did something
                episode_rewards.append(env_rewards[i])
                episode_scores.append(env_scores[i])
                episode_max_tiles.append(env_max_tiles[i])
                episode_steps.append(env_steps[i])
                
                # Update best score/tile
                if env_scores[i] > best_score or (env_scores[i] == best_score and env_max_tiles[i] > best_max_tile):
                    best_score = env_scores[i]
                    best_max_tile = env_max_tiles[i]
                    agent.save("best_parallel_model.pth")
        
        # Log progress every 10 episodes
        if episode % (10 * num_envs) < num_envs:
            # Calculate statistics
            recent_rewards = episode_rewards[-num_envs * 10:]
            recent_scores = episode_scores[-num_envs * 10:]
            recent_max_tiles = episode_max_tiles[-num_envs * 10:]
            recent_steps = episode_steps[-num_envs * 10:]
            recent_losses = episode_losses[-10:] if episode_losses else [0]
            
            # Get max tile distribution
            sorted_tile_counts = sorted(max_tile_count.items(), key=lambda x: x[0], reverse=True)
            
            elapsed_time = time.time() - start_time
            steps_per_second = total_steps / elapsed_time if elapsed_time > 0 else 0
            
            print(f"\nEpisode {episode}/{num_episodes}")
            print(f"Last {len(recent_rewards)} Environments Statistics:")
            print(f"  Average Reward: {np.mean(recent_rewards):.2f}")
            print(f"  Average Score: {np.mean(recent_scores):.2f}")
            print(f"  Average Max Tile: {np.mean(recent_max_tiles):.2f}")
            print(f"  Average Steps: {np.mean(recent_steps):.2f}")
            print(f"  Average Loss: {np.mean(recent_losses):.6f}")
            print(f"\nOverall Statistics:")
            print(f"  Best Score: {best_score}")
            print(f"  Best Max Tile: {best_max_tile}")
            print(f"  Total Steps: {total_steps}")
            print(f"  Steps/Second: {steps_per_second:.2f}")
            print(f"\nMax Tile Distribution:")
            for tile, count in sorted_tile_counts[:5]:
                print(f"  {tile}: {count} times")
            print(f"\nTraining Info:")
            print(f"  Buffer Size: {len(agent.replay_buffer)}")
            print(f"  Epsilon: {current_epsilon:.3f}")
            print(f"  GPU Memory Usage: {get_gpu_memory_usage():.2f} GB")
            print(f"  Target Network Update Frequency: {agent.target_update_frequency}")
            print(f"  Elapsed Time: {elapsed_time:.2f} seconds")
            print("-" * 50)
    
    # Final statistics
    print("\nTraining Complete!")
    print(f"\nFinal Statistics:")
    print(f"  Best Score: {best_score}")
    print(f"  Best Max Tile: {best_max_tile}")
    print(f"  Total Steps: {total_steps}")
    print(f"  Total Episodes: {min(num_episodes, len(episode_rewards))}")
    print(f"  GPU Memory Peak: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print(f"  Training Time: {time.time() - start_time:.2f} seconds")
    
    return agent

if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    print("Starting Parallel CNN Training for H100...")
    
    # Adjust these based on your H100 performance
    trained_agent = train_parallel_cnn_agent(
        num_episodes=100000,
        num_envs=64,           # Process 64 games in parallel
        epsilon_start=0.5,
        batch_size=16384,      # Large batch size for H100
        update_interval=64     # Update less frequently with larger batches
    )
    
    print("\nTraining complete and best model saved to 'best_parallel_model.pth'")