import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

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