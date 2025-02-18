import os
import numpy as np
import torch
import torch.nn.functional as F
import logging
from config import device
from training_stats import TrainingStats
from game2048 import preprocess_state, Game2048
from agent import PPOAgent
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

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
    traj_states = []
    traj_actions = []
    traj_rewards = []
    traj_log_probs = []
    traj_values = []
    terminal_states = []
    total_steps = 0
    max_tile_overall = 0

    while total_steps < min_steps:
        state = env.reset()
        done = False
        while not done:
            state_proc = preprocess_state(state)
            state_tensor = torch.tensor(state_proc, dtype=torch.float, device=device).unsqueeze(0)
            valid_moves = env.get_possible_moves()
            if not valid_moves:
                # If no valid moves, mark the episode as finished and save the current state
                done = True
                traj_states.append(state_proc)
                break
            action_mask = torch.full((1, 4), float('-inf'), device=device)
            action_mask[0, valid_moves] = 0
            with torch.no_grad():
                logits, value = agent(state_tensor)
                logits = logits + action_mask
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            next_state, reward, done, info = env.step(action.item())
            traj_states.append(state_proc)
            traj_actions.append(action.item())
            traj_rewards.append(reward)
            traj_log_probs.append(log_prob.item())
            traj_values.append(value.item())
            total_steps += 1
            state = next_state
            max_tile_overall = max(max_tile_overall, int(np.max(state)))
            if done:
                final_state_proc = preprocess_state(state)
                # Append the final state if it is not already the last state in the trajectory
                if len(traj_states) == 0 or not np.array_equal(traj_states[-1], final_state_proc):
                    traj_states.append(final_state_proc)
                terminal_states.append(final_state_proc)
                break

    return {
        'states': np.array(traj_states),
        'actions': np.array(traj_actions),
        'rewards': np.array(traj_rewards),
        'log_probs': np.array(traj_log_probs),
        'values': np.array(traj_values),
        'max_tile': max_tile_overall,
        'terminal_states': np.array(terminal_states)
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
    # Removed logging for checkpoint save.

def train(agent: PPOAgent, env: Game2048, optimizer, epochs: int = 1000, mini_batch_size: int = 64, ppo_epochs: int = 8,
          clip_param: float = 0.2, gamma: float = 0.99, lam: float = 0.95, entropy_coef: float = 0.8, max_grad_norm: float = 0.5,
          steps_per_update: int = 500) -> None:
    
    best_score = -float('inf')
    running_reward = 0
    stats = TrainingStats()
    
    logging.info("Starting training...")
    logging.info(f"Training for {epochs} epochs")
    os.makedirs("checkpoints", exist_ok=True)
    
    initial_lr = 3e-4
    final_lr = 1e-4
    min_entropy = 0.05

    block_best_running_reward = -float('inf')
    block_best_epoch_info = None

    # Initialize a cosine annealing scheduler for learning rate.
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=final_lr)

    for epoch in range(epochs):
        progress = epoch / epochs
        # Update the learning rate via the scheduler.
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        agent.update_exploration(progress)
        
        # Entropy coefficient decay (could also be managed by a scheduler)
        initial_entropy_coef = 0.1
        final_entropy_coef = min_entropy
        decay_fraction = 0.3
        decay_epochs = decay_fraction * epochs
        if epoch < decay_epochs:
            current_entropy_coef = initial_entropy_coef - ((initial_entropy_coef - final_entropy_coef) / decay_epochs) * epoch
        else:
            current_entropy_coef = final_entropy_coef
        
        trajectory = collect_trajectories(agent, env, min_steps=steps_per_update)
        if len(trajectory['states']) < 8:
            continue
        
        states = torch.tensor(trajectory['states'], dtype=torch.float, device=device)
        actions = torch.tensor(trajectory['actions'], dtype=torch.long, device=device)
        old_log_probs = torch.tensor(trajectory['log_probs'], dtype=torch.float, device=device)
        rewards = trajectory['rewards']
        values = trajectory['values']
        
        advantages = compute_advantages_vectorized(rewards, values, gamma, lam)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float, device=device)
        returns_tensor = torch.tensor(np.array(values) + advantages, dtype=torch.float, device=device)
        
        agent.train()
        num_samples = len(advantages)
        indices = np.arange(num_samples)
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
                
                policy_logits, value_preds = agent(batch_states, training=True)
                dist = torch.distributions.Categorical(logits=policy_logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(value_preds.view(-1), batch_returns)
                loss = policy_loss + 0.5 * value_loss - current_entropy_coef * entropy
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()
        
        episode_reward = np.sum(rewards)
        episode_length = len(trajectory['states'])
        max_tile = trajectory['max_tile']
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
        stats.update(episode_reward, max_tile, episode_length, running_reward,
                     policy_loss.item(), value_loss.item(), entropy.item())

        if running_reward > block_best_running_reward:
            block_best_running_reward = running_reward
            block_best_epoch_info = {
                "epoch": epoch,
                "running_reward": running_reward,
                "episode_reward": episode_reward,
                "max_tile": max_tile,
                "episode_length": episode_length,
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "entropy": entropy.item(),
                "learning_rate": current_lr,
                "trajectory": trajectory['states'],
                "terminal_states": trajectory['terminal_states']
            }

        if (epoch + 1) % 50 == 0:
            # Calculate the starting epoch number for this block (using 1-indexed epochs)
            start_epoch = (epoch + 1) - 50 + 1
            logging.info(f"\nComputed epochs {start_epoch} to {epoch + 1}.")
            logging.info("Best Epoch Info for this block:")
            info = block_best_epoch_info
            logging.info(f"Epoch {info['epoch']} with Running Reward: {info['running_reward']:.2f}, "
                         f"Episode Reward: {info['episode_reward']:.2f}, Max Tile: {info['max_tile']}, "
                         f"Episode Length: {info['episode_length']}, Policy Loss: {info['policy_loss']:.6f}, "
                         f"Value Loss: {info['value_loss']:.6f}, Entropy: {info['entropy']:.6f}, "
                         f"Learning Rate: {info['learning_rate']:.6f}")
            logging.info("-" * 30)
            # Plot the terminal (final) board states for the best epoch in this block.
            plot_board_trajectory(info['terminal_states'], f"board_trajectory_epoch_{info['epoch']}.png")
            block_best_running_reward = -float('inf')
            block_best_epoch_info = None
        
        if running_reward > best_score:
            best_score = running_reward
            save_checkpoint(agent, optimizer, epoch, running_reward, max_tile, os.path.join("checkpoints", "best_model.pt"))
    
    stats.print_summary()
    logging.info("Training completed! Final model saved as 'checkpoints/best_model.pt'")
    logging.info("Training progress plot saved as 'training_progress.png'")

def plot_board_trajectory(boards, filename):
    """
    Plots a series of 2048 board states from an episode.
    If the episode has more than 10 moves, 10 states are sampled uniformly.
    Each board is displayed using imshow with annotated cell values.
    """
    # Filter out boards that look like initial boards (e.g., those with less than 3 nonzero entries)
    filtered_boards = [board for board in boards if np.count_nonzero(board) >= 3]
    if len(filtered_boards) == 0:
        filtered_boards = boards
    num_boards = len(filtered_boards)
    sample_count = min(10, num_boards)
    sample_indices = np.linspace(0, num_boards - 1, sample_count, dtype=int)
    boards_to_plot = [filtered_boards[i] for i in sample_indices]
    n = len(boards_to_plot)
    fig, axs = plt.subplots(1, n, figsize=(n * 3, 3))
    if n == 1:
        axs = [axs]
    for ax, board in zip(axs, boards_to_plot):
        # Convert board values to integers since they are log2 scaled (e.g., 2->1, 4->2, etc.)
        int_board = board.astype(int)

        # Define discrete colors: index 0 (empty) is very light brown, 1 and 2 (tiles 2 and 4) are white,
        # and from index 3 (tile 8) up to index 11 (tile 2048) colors gradually darken, with 2048 as yellow
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
        cmap_custom = ListedColormap(colors)
        bounds = np.arange(-0.5, len(colors) + 0.5, 1)
        norm = BoundaryNorm(bounds, cmap_custom.N)
        im = ax.imshow(int_board, cmap=cmap_custom, norm=norm, interpolation='nearest')

        # Annotate each cell with its original value (convert log2 back to tile value)
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                display_val = int(2 ** board[i, j]) if board[i, j] > 0 else 0
                ax.text(j, i, str(display_val), va='center', ha='center', color='black', fontsize=12)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close() 