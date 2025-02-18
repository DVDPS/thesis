import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
import time
import logging
import os
from datetime import datetime
from typing import Tuple, List, Optional
import numpy.typing as npt
import sys  # Needed for command-line flags.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

# ======= Reproducibility & Logging Setup =======

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ======= Environment: Game2048 =======

class Game2048:
    def __init__(self, size: int = 4, seed: Optional[int] = None):
        self.size = size
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.reset()
        self.previous_max_tile = 0
        # Factor for the corner heuristic bonus.
        self.corner_factor = 0.01

    def reset(self) -> npt.NDArray[np.int32]:
        self.board = np.zeros((self.size, self.size), dtype=np.int32)
        self.score = 0
        self.previous_max_tile = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board.copy()

    def add_random_tile(self) -> None:
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            cell = random.choice(empty_cells)
            self.board[cell] = 2 if random.random() < 0.8 else 4

    def get_possible_moves(self) -> List[int]:
        moves = []
        for action in range(4):
            temp_board = self.board.copy()
            _, _, changed = self._move(temp_board, action, test_only=True)
            if changed:
                moves.append(action)
        return moves

    def _merge_row(self, row: npt.NDArray[np.int32]) -> Tuple[npt.NDArray[np.int32], int, bool]:
        """Merge a single row; returns new row, score gained, and a flag if changed."""
        original = row.copy()
        row = row[row != 0]
        score = 0
        changed = False
        if len(row) == 0:
            return np.zeros(self.size, dtype=np.int32), 0, False

        i = 0
        while i < len(row) - 1:
            if row[i] == row[i + 1]:
                row[i] *= 2
                score += row[i]
                row = np.delete(row, i + 1)
                changed = True
            i += 1

        if not changed and len(row) != len(original):
            changed = True

        row = np.pad(row, (0, self.size - len(row)), 'constant')
        return row, score, changed

    def _move(self, board: npt.NDArray[np.int32], action: int, test_only: bool = False) -> Tuple[npt.NDArray[np.int32], int, bool]:
        """
        Executes a move on a given board. action: 0=up, 1=right, 2=down, 3=left.
        Returns new board, score gained, and a flag whether the board changed.
        """
        rotated = np.rot90(board.copy(), k=action)
        total_score = 0
        changed = False
        for i in range(self.size):
            new_row, score, row_changed = self._merge_row(rotated[i])
            if row_changed:
                changed = True
            rotated[i] = new_row
            total_score += score
        new_board = np.rot90(rotated, k=-action)
        return new_board, total_score, changed

    def corner_heuristic(self) -> float:
        """
        Compute a bonus based on the positioning of high-value tiles in the top-left corner.
        Higher weights in the top-left encourage keeping large tiles there.
        """
        # Define a weight matrix for the top-left corner strategy.
        weights = np.array([
            [4.0, 3.0, 2.0, 1.0],
            [3.0, 2.0, 1.0, 0.5],
            [2.0, 1.0, 0.5, 0.25],
            [1.0, 0.5, 0.25, 0.1]
        ])
        # Compute the log2 of board tiles; leave zeros as zeros.
        board_log = self.board.astype(np.float32)
        mask = board_log > 0
        board_log[mask] = np.log2(board_log[mask])
        # Multiply elementwise by the weights.
        return np.sum(board_log * weights)

    def step(self, action: int) -> Tuple[npt.NDArray[np.int32], float, bool, dict]:
        """Execute an action, update state and return (state, reward, done, info)."""
        old_board = self.board.copy()
        old_empty = np.sum(old_board == 0)
        new_board, score_gain, valid_move = self._move(self.board, action)

        if valid_move:
            self.board = new_board
            self.score += score_gain
            self.add_random_tile()

        new_empty = np.sum(self.board == 0)
        new_max_tile = int(np.max(self.board))
        
        # Base reward components.
        merge_reward = score_gain / 100.0
        max_tile_reward = 0.0
        if new_max_tile > self.previous_max_tile:
            max_tile_reward = np.log2(new_max_tile) * 20
            self.previous_max_tile = new_max_tile
        empty_bonus = 0.2 * new_empty
        
        reward = merge_reward + max_tile_reward + empty_bonus
        if not valid_move:
            reward -= 2  # penalty for invalid move
        if self.is_game_over():
            reward -= 100  # game over penalty
        
        # Apply the corner heuristic bonus.
        corner_bonus = self.corner_heuristic() * self.corner_factor
        reward += corner_bonus
        
        info = {
            'score': self.score,
            'max_tile': new_max_tile,
            'valid_move': valid_move,
            'empty_cells': new_empty,
            'merge_reward': merge_reward,
            'max_tile_reward': max_tile_reward,
            'empty_bonus': empty_bonus,
            'corner_bonus': corner_bonus
        }
        return self.board.copy(), reward, self.is_game_over(), info

    def is_game_over(self) -> bool:
        return len(self.get_possible_moves()) == 0

# ======= Preprocessing =======

def preprocess_state(state: npt.NDArray[np.int32]) -> npt.NDArray[np.float32]:
    """Convert board state to log2 scale; zeros remain zeros."""
    state = state.astype(np.float32)
    mask = state > 0
    state[mask] = np.log2(state[mask])
    return state

# ======= PPO Agent =======

class PPOAgent(nn.Module):
    def __init__(self, board_size: int = 4, hidden_dim: int = 256):
        super(PPOAgent, self).__init__()
        self.board_size = board_size

        self.input_norm = nn.LayerNorm([1, board_size, board_size])
        self.conv1 = nn.Conv2d(1, 64, kernel_size=2, stride=1, padding='same')
        self.conv2 = nn.Conv2d(64, 128, kernel_size=2, stride=1, padding='same')
        self.conv3 = nn.Conv2d(128, 128, kernel_size=2, stride=1, padding='same')
        self.ln1 = nn.GroupNorm(8, 64)
        self.ln2 = nn.GroupNorm(8, 128)
        self.ln3 = nn.GroupNorm(8, 128)
        self.shortcut = nn.Conv2d(1, 128, kernel_size=1)
        self.fc1 = nn.Linear(128 * board_size * board_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout = nn.Dropout(0.1)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 4)
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        self.apply(self._init_weights)
        self.to(device)

        self.exploration_noise = 1.0
        self.min_exploration_noise = 0.01

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x, training: bool = False):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float, device=device)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = x.float()
        x = self.input_norm(x)
        identity = self.shortcut(x)
        x = F.relu(self.ln1(self.conv1(x)))
        x = F.relu(self.ln2(self.conv2(x)))
        x = F.relu(self.ln3(self.conv3(x))) + identity
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        if training and self.training:
            noise = self.exploration_noise * torch.randn_like(x)
            x = x + noise
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value

    def update_exploration(self, progress: float) -> None:
        self.exploration_noise = max(self.min_exploration_noise, 1.0 * (1 - progress))

# ======= Advantage Computation =======

def compute_advantages_vectorized(rewards: List[float], values: List[float],
                                  gamma: float = 0.99, lam: float = 0.95) -> np.ndarray:
    rewards = np.array(rewards, dtype=np.float32)
    values = np.array(values, dtype=np.float32)
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0
    for t in reversed(range(T)):
        next_value = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages

# ======= Trajectory Collection =======

def collect_trajectories(agent: PPOAgent, env: Game2048, min_steps: int = 500) -> dict:
    traj_states = []
    traj_actions = []
    traj_rewards = []
    traj_log_probs = []
    traj_values = []
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
                break

    return {
        'states': np.array(traj_states),
        'actions': np.array(traj_actions),
        'rewards': np.array(traj_rewards),
        'log_probs': np.array(traj_log_probs),
        'values': np.array(traj_values),
        'max_tile': max_tile_overall
    }

# ======= Training Statistics =======

class TrainingStats:
    def __init__(self, window_size: int = 100):
        self.rewards = []
        self.max_tiles = []
        self.episode_lengths = []
        self.running_rewards = []
        self.policy_losses = []
        self.value_losses = []
        self.entropies = []
        self.window_size = window_size
        self.recent_rewards = deque(maxlen=window_size)
        self.start_time = time.time()

    def update(self, episode_reward: float, max_tile: int, episode_length: int,
               running_reward: float, policy_loss: float, value_loss: float, entropy: float) -> None:
        self.rewards.append(episode_reward)
        self.max_tiles.append(max_tile)
        self.episode_lengths.append(episode_length)
        self.running_rewards.append(running_reward)
        self.policy_losses.append(policy_loss)
        self.value_losses.append(value_loss)
        self.entropies.append(entropy)
        self.recent_rewards.append(episode_reward)

    def plot_progress(self, epoch: int) -> None:
        plt.clf()
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        ax1.plot(self.rewards, label='Episode Reward', alpha=0.6)
        ax1.plot(self.running_rewards, label='Running Reward', linewidth=2)
        ax1.set_title('Rewards over Time')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.legend()
        ax2.plot(self.max_tiles)
        ax2.set_title('Maximum Tile Achieved')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Max Tile')
        ax3.plot(self.policy_losses, label='Policy Loss', alpha=0.6)
        ax3.plot(self.value_losses, label='Value Loss', alpha=0.6)
        ax3.set_title('Training Losses')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Loss')
        ax3.legend()
        ax4.plot(self.episode_lengths, label='Episode Length', color='purple')
        ax4_twin = ax4.twinx()
        ax4_twin.plot(self.entropies, label='Entropy', color='orange', alpha=0.6)
        ax4.set_title('Episode Length and Entropy')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Episode Length')
        ax4_twin.set_ylabel('Entropy')
        plt.tight_layout()
        plt.savefig('training_progress.png')
        plt.close()

    def print_summary(self) -> None:
        training_time = time.time() - self.start_time
        hours, rem = divmod(training_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print("\n====== Training Summary ======")
        print(f"Training Duration: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        print(f"Best Reward: {max(self.rewards):.2f}")
        print(f"Average Reward (last {self.window_size} episodes): {np.mean(self.recent_rewards):.2f}")
        print(f"Best Max Tile: {max(self.max_tiles)}")
        print(f"Average Max Tile (last {self.window_size} episodes): {np.mean(self.max_tiles[-self.window_size:]):.2f}")
        print(f"Average Episode Length: {np.mean(self.episode_lengths):.2f}")
        print(f"Final Policy Loss: {self.policy_losses[-1]:.6f}")
        print(f"Final Value Loss: {self.value_losses[-1]:.6f}")
        print(f"Final Entropy: {self.entropies[-1]:.6f}")
        print(f"Final Running Reward: {self.running_rewards[-1]:.2f}")
        print("============================")

# ======= Checkpoint Utility =======

def save_checkpoint(agent: PPOAgent, optimizer: torch.optim.Optimizer, epoch: int,
                    running_reward: float, max_tile: int, filename: str) -> None:
    torch.save({
        'epoch': epoch,
        'model_state_dict': agent.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'running_reward': running_reward,
        'max_tile': max_tile,
    }, filename)
    # Removed logging for checkpoint save.

# ======= Training Loop =======

def train(agent: PPOAgent, env: Game2048, optimizer: torch.optim.Optimizer,
          epochs: int = 1000, mini_batch_size: int = 64, ppo_epochs: int = 8,
          clip_param: float = 0.2, gamma: float = 0.99, lam: float = 0.95,
          entropy_coef: float = 0.05, max_grad_norm: float = 0.5,
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

    for epoch in range(epochs):
        progress = epoch / epochs
        current_lr = final_lr + 0.5 * (initial_lr - final_lr) * (1 + np.cos(progress * np.pi))
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        agent.update_exploration(progress)
        
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
                "learning_rate": current_lr
            }

        if (epoch + 1) % 100 == 0:
            logging.info(f"\nComputed epochs {epoch - 99} to {epoch}.")
            logging.info("Best Epoch Info for this block:")
            info = block_best_epoch_info
            logging.info(f"Epoch {info['epoch']} with Running Reward: {info['running_reward']:.2f}, "
                         f"Episode Reward: {info['episode_reward']:.2f}, Max Tile: {info['max_tile']}, "
                         f"Episode Length: {info['episode_length']}, Policy Loss: {info['policy_loss']:.6f}, "
                         f"Value Loss: {info['value_loss']:.6f}, Entropy: {info['entropy']:.6f}, "
                         f"Learning Rate: {info['learning_rate']:.6f}")
            logging.info("-" * 30)
            # Reset block best variables for the next block.
            block_best_running_reward = -float('inf')
            block_best_epoch_info = None
        
        if running_reward > best_score:
            best_score = running_reward
            save_checkpoint(agent, optimizer, epoch, running_reward, max_tile, os.path.join("checkpoints", "best_model.pt"))
    
    stats.print_summary()
    logging.info("Training completed! Final model saved as 'checkpoints/best_model.pt'")
    logging.info("Training progress plot saved as 'training_progress.png'")

# ======= Test Utilities =======

def test_merge_row():
    print("Testing _merge_row with sample rows:")
    test_rows = [
        np.array([2, 2, 2, 2], dtype=np.int32),
        np.array([2, 2, 4, 4], dtype=np.int32),
        np.array([0, 0, 0, 0], dtype=np.int32),
        np.array([2, 0, 2, 4], dtype=np.int32)
    ]
    game = Game2048()
    for row in test_rows:
        merged_row, score, changed = game._merge_row(row)
        print("Original row:", row.tolist())
        print("Merged row:  ", merged_row.tolist())
        print("Score gained:", score, "| Changed:", changed)
        print("-" * 40)

def test_game_over_condition():
    print("Testing game over condition:")
    board1 = np.array([
        [2, 2, 4, 8],
        [16, 32, 64, 128],
        [2, 2, 4, 8],
        [16, 32, 64, 128]
    ], dtype=np.int32)
    
    board2 = np.array([
        [2, 4, 2, 4],
        [4, 2, 4, 2],
        [2, 4, 2, 4],
        [4, 2, 2, 2]
    ], dtype=np.int32)
    
    game = Game2048()
    game.board = board1.copy()
    print("Case 1: Board with mergeable tiles")
    print(game.board)
    print("is_game_over:", game.is_game_over())
    print("-" * 40)
    
    game.board = board2.copy()
    print("Case 2: Board without mergeable tiles")
    print(game.board)
    print("is_game_over:", game.is_game_over())
    print("-" * 40)

# ======= Main =======

if __name__ == "__main__":
    if "--test-merge" in sys.argv:
        test_merge_row()
        sys.exit(0)
    if "--test-gameover" in sys.argv:
        test_game_over_condition()
        sys.exit(0)

    env = Game2048()
    agent = PPOAgent()
    optimizer = optim.Adam(agent.parameters(), lr=3e-4, eps=1e-5, weight_decay=1e-4)
    
    train(agent, env, optimizer,
          epochs=15000,
          mini_batch_size=128,
          ppo_epochs=8,
          clip_param=0.2,
          gamma=0.99,
          lam=0.95,
          entropy_coef=0.8,
          max_grad_norm=0.5,
          steps_per_update=500)
