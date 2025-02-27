import os
import numpy as np
import torch
import torch.nn.functional as F
import logging
from ..config import device
from ..environment.game2048 import preprocess_state, preprocess_state_onehot, Game2048
from ..agents.base_agent import PPOAgent
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from collections import deque
import random

# Simple replacement for TrainingStats
class TrainingStats:
    def __init__(self):
        self.epoch_rewards = []
        self.running_rewards = []
        self.policy_losses = []
        self.value_losses = []
        self.entropies = []
        self.max_tiles = []
    
    def update(self, epoch_reward, running_reward, policy_loss, value_loss, entropy, max_tile):
        self.epoch_rewards.append(epoch_reward)
        self.running_rewards.append(running_reward)
        self.policy_losses.append(policy_loss)
        self.value_losses.append(value_loss)
        self.entropies.append(entropy)
        self.max_tiles.append(max_tile)
    
    def plot(self, filename="training_stats.png"):
        if not self.epoch_rewards:
            return  # No data to plot
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot rewards
        axes[0, 0].plot(self.epoch_rewards, alpha=0.6, label="Episode Reward")
        axes[0, 0].plot(self.running_rewards, label="Running Reward")
        axes[0, 0].set_title("Rewards")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].legend()
        
        # Plot max tiles
        axes[0, 1].plot(self.max_tiles)
        axes[0, 1].set_title("Max Tile Achieved")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Max Tile")
        
        # Plot losses
        axes[1, 0].plot(self.policy_losses, label="Policy Loss")
        axes[1, 0].plot(self.value_losses, label="Value Loss")
        axes[1, 0].set_title("Losses")
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("Loss")
        axes[1, 0].legend()
        
        # Plot entropy
        axes[1, 1].plot(self.entropies)
        axes[1, 1].set_title("Entropy")
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].set_ylabel("Entropy")
        
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

def compute_advantages_vectorized(rewards, values, gamma: float = 0.99, lam: float = 0.95) -> np.ndarray:
    rewards = np.array(rewards, dtype=np.float32)
    values = np.array(values, dtype=np.float32)
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0
    for t in reversed(range(T)):
        next_value = values[t+1] if t+1 < T else 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages

def collect_trajectories(agent: PPOAgent, env: Game2048, min_steps: int = 500) -> dict:
    """
    Collect trajectories from the environment using the current policy.
    Optimized for better sampling efficiency and more complete game experiences.
    """
    traj_states = []
    traj_actions = []
    traj_rewards = []
    traj_log_probs = []
    traj_values = []
    terminal_states = []
    total_steps = 0
    max_tile_overall = 0
    total_score = 0
    episode_count = 0
    max_episodes = 20  # Cap the number of episodes to avoid very long collection times
    
    # Pre-allocate buffers for faster append operations
    states_buffer = []
    actions_buffer = []
    rewards_buffer = []
    log_probs_buffer = []
    values_buffer = []
    
    # Run episodes until we have enough steps or reach the max episode count
    while total_steps < min_steps and episode_count < max_episodes:
        episode_count += 1
        state = env.reset()
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_log_probs = []
        episode_values = []
        done = False
        episode_step = 0
        max_episode_steps = 300  # Prevent excessively long episodes
        
        while not done and episode_step < max_episode_steps:
            # Process state with one-hot encoding
            state_proc = preprocess_state_onehot(state)
            state_tensor = torch.tensor(state_proc, dtype=torch.float, device=device).unsqueeze(0)
            
            # Get valid moves and create action mask
            valid_moves = env.get_possible_moves()
            if not valid_moves:
                done = True
                episode_states.append(state_proc)
                break
                
            # Create action mask to restrict to valid moves
            action_mask = torch.full((1, 4), float('-inf'), device=device)
            action_mask[0, valid_moves] = 0
            
            # Get action from policy
            with torch.no_grad():
                logits, value = agent(state_tensor)
                logits = logits + action_mask
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                
            # Execute action in environment
            next_state, reward, done, info = env.step(action.item())
            
            # Store transition
            episode_states.append(state_proc)
            episode_actions.append(action.item())
            episode_rewards.append(reward)
            episode_log_probs.append(log_prob.item())
            episode_values.append(value.item())
            
            # Update state
            state = next_state
            max_tile_overall = max(max_tile_overall, int(np.max(state)))
            total_score += info.get('merge_score', 0)
            episode_step += 1
            total_steps += 1
            
        # Process the final state if episode ended
        if done:
            final_state_proc = preprocess_state_onehot(state)
            # Only append final state if it's different from the last one
            if len(episode_states) == 0 or not np.array_equal(episode_states[-1], final_state_proc):
                episode_states.append(final_state_proc)
            terminal_states.append(final_state_proc)
        
        # Add complete episode to trajectory buffers
        states_buffer.extend(episode_states)
        actions_buffer.extend(episode_actions)
        rewards_buffer.extend(episode_rewards)
        log_probs_buffer.extend(episode_log_probs)
        values_buffer.extend(episode_values)
        
        # Ensure we have a minimum number of complete episodes
        if episode_count >= 5 and total_steps >= min_steps:
            break
    
    # Convert buffer lists to final trajectory lists
    traj_states = np.array(states_buffer)
    traj_actions = np.array(actions_buffer)
    traj_rewards = np.array(rewards_buffer)
    traj_log_probs = np.array(log_probs_buffer)
    traj_values = np.array(values_buffer)
    
    logging.debug(f"Collected {len(traj_states)} steps from {episode_count} episodes. Max tile: {max_tile_overall}")
    
    return {
        'states': traj_states,
        'actions': traj_actions,
        'rewards': traj_rewards,
        'log_probs': traj_log_probs,
        'values': traj_values,
        'max_tile': max_tile_overall,
        'terminal_states': np.array(terminal_states) if terminal_states else np.zeros((1, 16, 4, 4)),
        'total_score': total_score,
        'episode_count': episode_count
    }

def save_checkpoint(agent: PPOAgent, optimizer, epoch: int, running_reward: float, max_tile: int, filename: str) -> None:
    # Ensure the directory exists (using the directory portion of the filename)
    checkpoint_dir = os.path.dirname(filename)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Resolve an absolute path to avoid path issues on Windows.
    abs_filename = os.path.abspath(filename)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': agent.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'running_reward': running_reward,
        'max_tile': max_tile,
    }, abs_filename)

def train(agent: PPOAgent, env: Game2048, optimizer, epochs: int = 1000, 
          mini_batch_size: int = 64, ppo_epochs: int = 8,
          clip_param: float = 0.2, gamma: float = 0.99, lam: float = 0.95, 
          entropy_coef: float = 0.8, max_grad_norm: float = 0.5,
          steps_per_update: int = 500, 
          start_epoch: int = 0,
          best_running_reward: float = float('-inf'),
          checkpoint_dir: str = "checkpoints") -> None:
    
    # Initialize best_score with the loaded value
    best_score = best_running_reward
    running_reward = best_running_reward
    stats = TrainingStats()
    
    logging.info("Starting training...")
    logging.info(f"Training for {epochs} epochs")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize training parameters with more effective decay schedule
    initial_lr = 5e-4
    final_lr = 5e-5  # Lower final learning rate for fine-tuning
    min_entropy = 0.05
    
    block_best_running_reward = -float('inf')
    block_best_epoch_info = None
    block_best_score = 0
    block_best_tile = 0
    
    # Use a longer cosine annealing period for better optimization
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=final_lr)
    
    # Use GradScaler for mixed precision training
    scaler = torch.amp.GradScaler(device='cuda')
    
    # Keep track of the highest tile achieved for curriculum learning
    highest_tile_achieved = 0
    best_total_score = 0
    
    # Start from the loaded epoch
    for epoch in range(start_epoch, epochs):
        progress = epoch / epochs
        # Update the learning rate via the scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        agent.update_exploration(progress)
        
        # Enhanced entropy coefficient decay schedule
        # Maintain higher entropy for longer for better exploration
        initial_entropy_coef = entropy_coef
        final_entropy_coef = min_entropy
        # Extend decay period to 50% of training
        decay_fraction = 0.5  
        decay_epochs = decay_fraction * epochs
        if epoch < decay_epochs:
            current_entropy_coef = initial_entropy_coef - ((initial_entropy_coef - final_entropy_coef) * (epoch / decay_epochs))
        else:
            current_entropy_coef = final_entropy_coef
        
        # Collect trajectories
        trajectory = collect_trajectories(agent, env, min_steps=steps_per_update)
        if len(trajectory['states']) < 8:
            continue
        
        # Update highest tile achieved for tracking progress
        current_max_tile = trajectory['max_tile']
        current_total_score = trajectory['total_score']
        
        # Update batch best stats
        block_best_score = max(block_best_score, current_total_score)
        block_best_tile = max(block_best_tile, current_max_tile)
        
        # Log significant milestones (only log when we achieve new records)
        if current_max_tile > highest_tile_achieved:
            highest_tile_achieved = current_max_tile
            logging.info(f"[MILESTONE] Achieved new highest tile {highest_tile_achieved} at epoch {epoch}")
            # Only save milestone model if it's a significant improvement (128, 256, 512, 1024, 2048)
            if highest_tile_achieved in [128, 256, 512, 1024, 2048]:
                milestone_path = os.path.join(checkpoint_dir, f"model_tile_{highest_tile_achieved}.pt")
                save_checkpoint(agent, optimizer, epoch, running_reward, highest_tile_achieved, milestone_path)
                logging.info(f"Saved milestone model for tile {highest_tile_achieved} at {milestone_path}")
                
        if current_total_score > best_total_score:
            old_best = best_total_score
            best_total_score = current_total_score
            # Only log if significant improvement (>10% increase)
            if best_total_score > old_best * 1.1 or best_total_score > 10000:
                logging.info(f"[HIGH SCORE] New high score: {best_total_score} at epoch {epoch} (previous: {old_best})")
        
        # Process trajectory data
        states = torch.tensor(trajectory['states'], dtype=torch.float, device=device)
        actions = torch.tensor(trajectory['actions'], dtype=torch.long, device=device)
        old_log_probs = torch.tensor(trajectory['log_probs'], dtype=torch.float, device=device)
        rewards = trajectory['rewards']
        values = trajectory['values']
        
        # Compute advantages
        advantages = compute_advantages_vectorized(rewards, values, gamma, lam)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float, device=device)
        returns_tensor = torch.tensor(np.array(values) + advantages, dtype=torch.float, device=device)
        
        # Training phase
        agent.train()
        num_samples = len(advantages)
        indices = np.arange(num_samples)
        
        # Track loss metrics
        epoch_value_loss = 0
        epoch_policy_loss = 0
        epoch_entropy = 0
        update_count = 0
        
        # PPO update loop
        for _ in range(ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, num_samples, mini_batch_size):
                end = start + mini_batch_size
                batch_idx = indices[start:end]
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages_tensor[batch_idx]
                batch_returns = returns_tensor[batch_idx]
                
                # Mixed precision forward pass
                with torch.amp.autocast(device_type='cuda'):
                    policy_logits, value_preds = agent(batch_states, training=True)
                    dist = torch.distributions.Categorical(logits=policy_logits)
                    new_log_probs = dist.log_prob(batch_actions)
                    entropy = dist.entropy().mean()
                    
                    # Calculate PPO losses
                    ratio = torch.exp(new_log_probs - batch_old_log_probs)
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * batch_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # Enhanced value loss with huber loss for stability
                    value_loss = F.huber_loss(value_preds.view(-1), batch_returns, delta=1.0)
                    
                    # Combined loss with adaptive entropy coefficient
                    loss = policy_loss + 0.5 * value_loss - current_entropy_coef * entropy
                
                # Optimizer step with gradient scaling
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                # Clip gradients for stability
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                
                # Track metrics
                epoch_value_loss += value_loss.item()
                epoch_policy_loss += policy_loss.item()
                epoch_entropy += entropy.item()
                update_count += 1
        
        # Average losses
        if update_count > 0:
            epoch_value_loss /= update_count
            epoch_policy_loss /= update_count
            epoch_entropy /= update_count
        
        # Calculate reward stats
        episode_reward = np.sum(rewards)
        episode_length = len(trajectory['states'])
        max_tile = trajectory['max_tile']
        total_score = trajectory['total_score']
        
        # Update running reward with adaptive coefficient
        # Use a smaller coefficient for smoother tracking (0.03 instead of 0.05)
        running_reward = 0.03 * episode_reward + (1 - 0.03) * running_reward
        
        # Update training statistics
        stats.update(episode_reward, max_tile, episode_length, running_reward,
                    epoch_policy_loss, epoch_value_loss, epoch_entropy, 
                    total_score)

        # Track best performance within the current block
        if running_reward > block_best_running_reward:
            block_best_running_reward = running_reward
            block_best_epoch_info = {
                "epoch": epoch,
                "running_reward": running_reward,
                "episode_reward": episode_reward,
                "max_tile": max_tile,
                "episode_length": episode_length,
                "policy_loss": epoch_policy_loss,
                "value_loss": epoch_value_loss,
                "entropy": epoch_entropy,
                "learning_rate": current_lr,
                "trajectory": trajectory['states'],
                "terminal_states": trajectory['terminal_states'],
                "total_score": total_score,
            }

        # Log progress every 20 epochs
        if (epoch + 1) % 20 == 0:
            # Calculate the starting epoch number for this block (using 1-indexed epochs)
            start_epoch_block = (epoch + 1) - 20 + 1
            logging.info(f"\n----- TRAINING PROGRESS: EPOCHS {start_epoch_block} to {epoch + 1} -----")
            logging.info(f"PARAMETERS: entropy={current_entropy_coef:.4f}, lr={current_lr:.6f}")
            logging.info(f"GLOBAL BEST: highest_tile={highest_tile_achieved}, best_score={best_total_score}")
            logging.info(f"BATCH BEST: highest_tile={block_best_tile}, best_score={block_best_score}")
            logging.info(f"STATS: avg_reward={np.mean(stats.recent_rewards):.2f}, current_best_tile={stats.best_max_tile}")
            
            if block_best_epoch_info:
                logging.info("Best epoch in this batch:")
                info = block_best_epoch_info
                logging.info(f"  Epoch {info['epoch']}: reward={info['running_reward']:.2f}, "
                            f"max_tile={info['max_tile']}, score={info['total_score']}, "
                            f"steps={info['episode_length']}")
                logging.info(f"  Metrics: policy_loss={info['policy_loss']:.6f}, "
                            f"value_loss={info['value_loss']:.6f}, entropy={info['entropy']:.6f}")
                logging.info("-" * 30)
                # Plot the terminal (final) board states for the best epoch in this block.
                plot_board_trajectory(info['terminal_states'], f"board_trajectory_epoch_{info['epoch']}.png")
                
            # Reset batch tracking
            block_best_running_reward = -float('inf')
            block_best_epoch_info = None
            block_best_score = 0
            block_best_tile = 0
                
            # Save intermediate models every 100 epochs
            if (epoch + 1) % 100 == 0:
                save_checkpoint(agent, optimizer, epoch, running_reward, max_tile, 
                                os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pt"))
        
        # Save best model based on running reward
        if running_reward > best_score:
            best_score = running_reward
            save_checkpoint(agent, optimizer, epoch, running_reward, max_tile, os.path.join(checkpoint_dir, "best_model.pt"))
    
    # Print final training stats and ensure highest_tile_achieved is reflected in stats
    stats.best_max_tile = max(stats.best_max_tile, highest_tile_achieved)
    stats.print_summary()
    logging.info(f"TRAINING COMPLETE! Final model saved as '{checkpoint_dir}/best_model.pt'")
    logging.info(f"HIGHEST TILE: {highest_tile_achieved}")
    logging.info(f"TRAINING PROGRESS: Plot saved as 'training_progress.png'")

def onehot_to_board(onehot_state):
    """Convert a one-hot encoded state back to regular board format."""
    board = np.zeros((4, 4), dtype=np.int32)
    for i in range(onehot_state.shape[0]):  # iterate through channels
        if i == 0:  # skip the empty tile channel
            continue
        # Where this channel has 1s, put 2^i in the board
        board += (onehot_state[i] * (2 ** i)).astype(np.int32)
    return board

def plot_board_trajectory(boards, filename):
    """
    Plots a series of 2048 board states from an episode.
    If the episode has more than 10 moves, 10 states are sampled uniformly.
    Each board is displayed using imshow with annotated cell values.
    """
    # Convert one-hot boards back to regular format
    regular_boards = [onehot_to_board(board) for board in boards]
    
    # Filter out boards that look like initial boards
    filtered_boards = [board for board in regular_boards if np.count_nonzero(board) >= 3]
    if len(filtered_boards) == 0:
        filtered_boards = regular_boards
        
    num_boards = len(filtered_boards)
    sample_count = min(10, num_boards)
    sample_indices = np.linspace(0, num_boards - 1, sample_count, dtype=int)
    boards_to_plot = [filtered_boards[i] for i in sample_indices]
    
    n = len(boards_to_plot)
    fig, axs = plt.subplots(1, n, figsize=(n * 3, 3))
    if n == 1:
        axs = [axs]
        
    for ax, board in zip(axs, boards_to_plot):
        # Define colors for the tiles
        colors = [
            "#F5DEB3",  # 0: empty cell, very light brown
            "#FFFFFF",  # 1: tile 2
            "#FFFFFF",  # 2: tile 4
            "#FAFAD2",  # 3: tile 8
            "#FFEFD5",  # 4: tile 16
            "#FFEC8B",  # 5: tile 32
            "#FFD700",  # 6: tile 64
            "#FFC107",  # 7: tile 128
            "#FFB300",  # 8: tile 256
            "#FFA000",  # 9: tile 512
            "#FF9800",  # 10: tile 1024
            "#FFEB3B"   # 11: tile 2048
        ]
        
        # Create color map
        cmap_custom = ListedColormap(colors)
        bounds = np.arange(-0.5, len(colors) + 0.5, 1)
        norm = BoundaryNorm(bounds, cmap_custom.N)
        
        # Convert board values to log2 scale for color mapping
        display_board = np.zeros_like(board)
        mask = board > 0
        display_board[mask] = np.log2(board[mask]).astype(int)
        
        im = ax.imshow(display_board, cmap=cmap_custom, norm=norm, interpolation='nearest')
        
        # Annotate each cell with its actual value
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                val = board[i, j]
                ax.text(j, i, str(val), va='center', ha='center', color='black', fontsize=12)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

class PrioritizedReplayBuffer:
    """Replay buffer with prioritized experience replay"""
    def __init__(self, capacity=10000, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        self.beta_increment = beta_increment
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.high_value_buffer = []  # Special buffer for high-value states
        
    def add(self, state, action, reward, next_state, done, max_tile):
        """Add experience to buffer with max priority"""
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
            
        # Set max priority for new experience
        max_priority = np.max(self.priorities) if self.buffer else 1.0
        self.priorities[self.position] = max_priority
        
        # If this state has a high-value tile, also add to high-value buffer
        if max_tile >= 256:  # Consider 256+ as high value
            self.high_value_buffer.append((state, action, reward, next_state, done))
            # Keep high_value_buffer from growing too large
            if len(self.high_value_buffer) > self.capacity // 5:
                self.high_value_buffer.pop(0)
                
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size, high_value_ratio=0.3):
        """Sample batch with priority, optionally including high-value states"""
        high_value_count = min(int(batch_size * high_value_ratio), len(self.high_value_buffer))
        regular_count = batch_size - high_value_count
        
        # Sample from regular buffer based on priorities
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]
            
        # Convert priorities to probabilities
        probabilities = priorities ** self.alpha
        probabilities /= np.sum(probabilities)
        
        # Sample indices based on probabilities
        indices = np.random.choice(len(probabilities), regular_count, p=probabilities)
        
        # Calculate importance sampling weights
        weights = (len(probabilities) * probabilities[indices]) ** (-self.beta)
        weights /= np.max(weights)  # Normalize
        
        # Get samples from regular buffer
        regular_samples = [self.buffer[idx] for idx in indices]
        
        # Add high-value samples if available
        high_value_samples = []
        if high_value_count > 0 and self.high_value_buffer:
            high_value_samples = random.sample(self.high_value_buffer, min(high_value_count, len(self.high_value_buffer)))
            # For simplicity, give high-value samples weight 1.0
            weights = np.append(weights, np.ones(len(high_value_samples)))
            
        # Combine samples
        samples = regular_samples + high_value_samples
        
        # Increment beta for next sampling
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Return samples and weights
        batch = list(map(list, zip(*samples)))
        return batch, weights, indices
        
    def update_priorities(self, indices, priorities):
        """Update priorities for sampled indices"""
        for idx, priority in zip(indices, priorities):
            if idx < len(self.priorities):
                self.priorities[idx] = priority

def sanitize_logits(logits):
    """
    Handle NaN values in logits to prevent distribution errors.
    
    Args:
        logits: The model output logits tensor
        
    Returns:
        Sanitized logits with NaN values replaced
    """
    # Check if we have any NaN values
    if torch.isnan(logits).any():
        # Replace NaN values with a very negative number (effectively zero probability)
        logits = torch.where(torch.isnan(logits), 
                             torch.tensor(-1e7, device=logits.device, dtype=logits.dtype),
                             logits)
    
    # Clip extreme values to improve numerical stability
    logits = torch.clamp(logits, min=-1e7, max=1e7)
    
    return logits

def improved_train(agent, env, optimizer, epochs=2000, 
                  batch_size=128, replay_buffer_size=50000,
                  gamma=0.99, target_update_freq=10,
                  prioritized_replay=True, curriculum_phases=True,
                  optimized_buffer=None, use_fast_preprocessing=True):
    """
    Improved training function with experience replay and curriculum learning
    """
    # Import optimized preprocessing if enabled
    if use_fast_preprocessing:
        from performance_optimization import fast_preprocess_state_onehot
        # Pre-allocate buffer for state preprocessing
        import numpy as np
        state_buffer = np.zeros((16, 4, 4), dtype=np.float32)
        next_state_buffer = np.zeros((16, 4, 4), dtype=np.float32)

    # Initialize replay buffer - use optimized if provided
    if prioritized_replay:
        if optimized_buffer is not None:
            logging.info("Using optimized prioritized replay buffer")
            replay_buffer = optimized_buffer
        else:
            replay_buffer = PrioritizedReplayBuffer(capacity=replay_buffer_size)
    else:
        replay_buffer = deque(maxlen=replay_buffer_size)
        
    # Initialize target network for more stable learning
    target_agent = type(agent)(
        board_size=agent.board_size,
        hidden_dim=256,
        input_channels=agent.input_channels
    ).to(device)
    target_agent.load_state_dict(agent.state_dict())
    target_agent.eval()
    
    # Tracking metrics
    running_reward = -float('inf')
    best_reward = -float('inf')
    best_max_tile = 0
    episode_rewards = []
    max_tiles = []
    
    # Curriculum phases config
    if curriculum_phases:
        # Define phases based on max tile targets
        phases = [
            {"target": 64, "episodes": 100, "explore_bonus": 1.5},
            {"target": 128, "episodes": 200, "explore_bonus": 1.2},
            {"target": 256, "episodes": 400, "explore_bonus": 1.0},
            {"target": 512, "episodes": 800, "explore_bonus": 0.8},
            {"target": 1024, "episodes": 1000, "explore_bonus": 0.5}
        ]
        current_phase = 0
        phase_episodes = 0
    
    # Main training loop
    total_steps = 0
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in range(epochs):
        # Update phase if using curriculum learning
        if curriculum_phases and current_phase < len(phases):
            phase = phases[current_phase]
            phase_episodes += 1
            
            # Move to next phase if conditions met
            if phase_episodes >= phase["episodes"] or best_max_tile >= phase["target"]:
                if current_phase < len(phases) - 1:
                    current_phase += 1
                    phase_episodes = 0
                    logging.info(f"Moving to curriculum phase {current_phase + 1}: target={phases[current_phase]['target']}")
            
            # Apply phase-specific exploration bonus
            agent.exploration_noise = max(
                agent.min_exploration_noise, 
                agent.min_exploration_noise * phases[current_phase]["explore_bonus"]
            )
        else:
            # Default exploration update
            progress = epoch / epochs
            agent.update_exploration(progress)
        
        # Reset environment
        state = env.reset()
        
        # Use optimized preprocessing if enabled
        if use_fast_preprocessing:
            state_proc = fast_preprocess_state_onehot(state, state_buffer)
        else:
            state_proc = preprocess_state_onehot(state)
            
        episode_reward = 0
        max_tile = 0
        done = False
        
        # Collect experience for one episode
        while not done:
            # Get valid moves and create action mask
            valid_moves = env.get_possible_moves()
            if not valid_moves:
                break
                
            # Get action from policy with epsilon-greedy exploration
            state_tensor = torch.tensor(state_proc, dtype=torch.float, device=device).unsqueeze(0)
            with torch.no_grad():
                # Create action mask
                action_mask = torch.full((1, 4), float('-inf'), device=device)
                action_mask[0, valid_moves] = 0
                
                # Get action probabilities and value
                logits, value = agent(state_tensor)
                logits = logits + action_mask
                
                # Sanitize logits to prevent NaN issues
                logits = sanitize_logits(logits)
                
                # Sample action with error handling
                try:
                    # Action from policy
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample().item()
                except (ValueError, RuntimeError) as e:
                    # Fallback to a random valid action if distribution creation fails
                    logging.warning(f"Distribution error: {e}, using random valid action")
                    action = random.choice(valid_moves)
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            
            # Use optimized preprocessing for next state if enabled
            if use_fast_preprocessing:
                next_state_proc = fast_preprocess_state_onehot(next_state, next_state_buffer)
            else:
                next_state_proc = preprocess_state_onehot(next_state)
            
            # Track max tile
            tile = info.get('max_tile', 0)
            max_tile = max(max_tile, tile)
            
            # Store transition in replay buffer
            experience = (state_proc, action, reward, next_state_proc, done)
            if prioritized_replay:
                # Pass the tile value to the replay buffer for high-value state tracking
                replay_buffer.add(state_proc, action, reward, next_state_proc, done, tile)
            else:
                replay_buffer.append(experience)
                
            # Update state and accumulate reward
            state = next_state
            state_proc = next_state_proc
            episode_reward += reward
            total_steps += 1
            
            # Train on a batch from replay buffer
            if len(replay_buffer) >= batch_size:
                # Sample from replay buffer
                if prioritized_replay:
                    batch, weights, indices = replay_buffer.sample(batch_size)
                    states, actions, rewards, next_states, dones = batch
                    weights = torch.tensor(weights, dtype=torch.float, device=device)
                else:
                    batch = random.sample(replay_buffer, batch_size)
                    states, actions, rewards, next_states, dones = zip(*batch)
                    weights = torch.ones(batch_size, device=device)
                
                # Convert to tensors
                states_tensor = torch.tensor(np.array(states), dtype=torch.float, device=device)
                actions_tensor = torch.tensor(actions, dtype=torch.long, device=device)
                rewards_tensor = torch.tensor(rewards, dtype=torch.float, device=device)
                next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float, device=device)
                dones_tensor = torch.tensor(dones, dtype=torch.float, device=device)
                
                # Use mixed precision training if available
                if hasattr(torch.cuda, 'amp') and torch.cuda.is_available():
                    with torch.amp.autocast(device_type='cuda'):
                        # Get current Q values
                        logits, values = agent(states_tensor)
                        
                        # Sanitize logits
                        logits = sanitize_logits(logits)
                        
                        # Get next Q values from target network
                        with torch.no_grad():
                            next_logits, next_values = target_agent(next_states_tensor)
                            next_logits = sanitize_logits(next_logits)
                            
                            # Handle NaN values in next_values
                            if torch.isnan(next_values).any():
                                next_values = torch.where(torch.isnan(next_values), 
                                                        torch.zeros_like(next_values), 
                                                        next_values)
                            
                        # Compute TD targets
                        td_targets = rewards_tensor + gamma * next_values.squeeze() * (1 - dones_tensor)
                        
                        # Handle NaN values in values
                        if torch.isnan(values).any():
                            values = torch.where(torch.isnan(values), 
                                                torch.zeros_like(values), 
                                                values)
                        
                        # Compute value loss
                        value_loss = F.mse_loss(values.squeeze(), td_targets, reduction='none')
                        value_loss = (value_loss * weights).mean()
                        
                        # Compute policy loss (simplified PPO) with error handling
                        try:
                            dist = torch.distributions.Categorical(logits=logits)
                            log_probs = dist.log_prob(actions_tensor)
                            entropy = dist.entropy().mean()
                            
                            # Simple policy gradient loss
                            advantages = (td_targets - values.squeeze()).detach()
                            policy_loss = -(log_probs * advantages).mean()
                            
                            # Combined loss with entropy bonus
                            loss = value_loss + policy_loss - 0.01 * entropy
                        except (ValueError, RuntimeError) as e:
                            logging.warning(f"Error in loss computation: {e}, using value loss only")
                            loss = value_loss
                else:
                    # Get current Q values
                    logits, values = agent(states_tensor)
                    
                    # Sanitize logits
                    logits = sanitize_logits(logits)
                    
                    # Get next Q values from target network
                    with torch.no_grad():
                        next_logits, next_values = target_agent(next_states_tensor)
                        next_logits = sanitize_logits(next_logits)
                        
                        # Handle NaN values in next_values
                        if torch.isnan(next_values).any():
                            next_values = torch.where(torch.isnan(next_values), 
                                                    torch.zeros_like(next_values), 
                                                    next_values)
                        
                    # Compute TD targets
                    td_targets = rewards_tensor + gamma * next_values.squeeze() * (1 - dones_tensor)
                    
                    # Handle NaN values in values
                    if torch.isnan(values).any():
                        values = torch.where(torch.isnan(values), 
                                            torch.zeros_like(values), 
                                            values)
                    
                    # Compute value loss
                    value_loss = F.mse_loss(values.squeeze(), td_targets, reduction='none')
                    value_loss = (value_loss * weights).mean()
                    
                    # Compute policy loss (simplified PPO) with error handling
                    try:
                        dist = torch.distributions.Categorical(logits=logits)
                        log_probs = dist.log_prob(actions_tensor)
                        entropy = dist.entropy().mean()
                        
                        # Simple policy gradient loss
                        advantages = (td_targets - values.squeeze()).detach()
                        policy_loss = -(log_probs * advantages).mean()
                        
                        # Combined loss with entropy bonus
                        loss = value_loss + policy_loss - 0.01 * entropy
                    except (ValueError, RuntimeError) as e:
                        logging.warning(f"Error in loss computation: {e}, using value loss only")
                        loss = value_loss
                
                # Compute TD errors for prioritization
                td_errors = torch.abs(values.squeeze() - td_targets).detach().cpu().numpy()
                
                # Check for NaN in td_errors
                if np.isnan(td_errors).any():
                    td_errors = np.where(np.isnan(td_errors), np.ones_like(td_errors), td_errors)
                
                # Optimize with safety checks
                optimizer.zero_grad()
                
                try:
                    loss.backward()
                    
                    # Check for NaN gradients
                    has_nan_grad = False
                    for param in agent.parameters():
                        if param.grad is not None and torch.isnan(param.grad).any():
                            has_nan_grad = True
                            param.grad = torch.zeros_like(param.grad)  # Zero out NaN gradients
                    
                    if has_nan_grad:
                        logging.warning("NaN gradients detected and zeroed")
                        
                    # Clip gradients for stability (use a smaller value)
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                    optimizer.step()
                except RuntimeError as e:
                    logging.error(f"Error in backward pass: {e}")
                    # Skip this optimization step
                    
                # Update priorities in buffer
                if prioritized_replay:
                    replay_buffer.update_priorities(indices, td_errors + 1e-6)  # Small constant for stability
                    
        # Update target network periodically
        if epoch % target_update_freq == 0:
            target_agent.load_state_dict(agent.state_dict())
            
        # Track statistics
        episode_rewards.append(episode_reward)
        max_tiles.append(max_tile)
        running_reward = 0.05 * episode_reward + (1 - 0.05) * (running_reward if running_reward != -float('inf') else episode_reward)
        
        # Log progress
        if (epoch + 1) % 20 == 0:
            logging.info(f"Epoch {epoch+1}/{epochs} | " 
                        f"Reward: {episode_reward:.1f} | "
                        f"Running reward: {running_reward:.1f} | "
                        f"Max tile: {max_tile} | "
                        f"Exploration: {agent.exploration_noise:.2f}")
            
        # Save best model
        if running_reward > best_reward:
            best_reward = running_reward
            best_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'running_reward': running_reward,
                'max_tile': max_tile,
            }, best_path)
            logging.info(f"New best model saved with reward {best_reward:.1f}")
            
        # Save milestone models for high tiles
        if max_tile > best_max_tile:
            best_max_tile = max_tile
            if best_max_tile in [256, 512, 1024, 2048]:
                milestone_path = os.path.join(checkpoint_dir, f"model_tile_{best_max_tile}.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': agent.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'running_reward': running_reward,
                    'max_tile': best_max_tile,
                }, milestone_path)
                logging.info(f"New milestone model saved for tile {best_max_tile}")
                
        # Clear GPU cache periodically to prevent memory fragmentation
        if torch.cuda.is_available() and epoch % 50 == 0:
            torch.cuda.empty_cache()
                
    # Return training statistics
    return {
        'episode_rewards': episode_rewards,
        'max_tiles': max_tiles,
        'best_reward': best_reward,
        'best_max_tile': best_max_tile
    }
