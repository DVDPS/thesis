import re
import matplotlib.pyplot as plt

# Regular expression pattern to match the best epoch info lines.
# The pattern matches:
#   Epoch <epoch> with Running Reward: <value>, Episode Reward: <value>, Max Tile: <value>,
#   Episode Length: <value>, Policy Loss: <value>, Value Loss: <value>, Entropy: <value>, Learning Rate: <value>
pattern = re.compile(
    r"Epoch (\d+) with Running Reward: ([\d\.]+), Episode Reward: ([\d\.]+), Max Tile: (\d+), "
    r"Episode Length: (\d+), Policy Loss: ([\-\d\.]+), Value Loss: ([\d\.]+), Entropy: ([\d\.]+), "
    r"Learning Rate: ([\d\.]+)"
)

# Initialize lists for each metric.
epochs = []
running_rewards = []
episode_rewards = []
max_tiles = []
episode_lengths = []
policy_losses = []
value_losses = []
entropies = []
learning_rates = []

# Open and parse the training.log file.
with open("training.log", "r") as log_file:
    for line in log_file:
        match = pattern.search(line)
        if match:
            epoch_num = int(match.group(1))
            running_reward = float(match.group(2))
            episode_reward = float(match.group(3))
            max_tile = int(match.group(4))
            episode_length = int(match.group(5))
            policy_loss = float(match.group(6))
            value_loss = float(match.group(7))
            entropy = float(match.group(8))
            learning_rate = float(match.group(9))

            epochs.append(epoch_num)
            running_rewards.append(running_reward)
            episode_rewards.append(episode_reward)
            max_tiles.append(max_tile)
            episode_lengths.append(episode_length)
            policy_losses.append(policy_loss)
            value_losses.append(value_loss)
            entropies.append(entropy)
            learning_rates.append(learning_rate)

# Create plots for the different metrics.
plt.figure(figsize=(16, 12))

# Plot running reward vs. epoch.
plt.subplot(3, 2, 1)
plt.plot(epochs, running_rewards, marker='o', linestyle='-')
plt.xlabel("Epoch")
plt.ylabel("Running Reward")
plt.title("Running Reward vs. Epoch")

# Plot episode reward vs. epoch.
plt.subplot(3, 2, 2)
plt.plot(epochs, episode_rewards, marker='o', linestyle='-', color="orange")
plt.xlabel("Epoch")
plt.ylabel("Episode Reward")
plt.title("Episode Reward vs. Epoch")

# Plot policy loss and value loss.
plt.subplot(3, 2, 3)
plt.plot(epochs, policy_losses, marker='o', linestyle='-', color="green", label="Policy Loss")
plt.plot(epochs, value_losses, marker='o', linestyle='-', color="red", label="Value Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Losses vs. Epoch")
plt.legend()

# Plot learning rate vs. epoch.
plt.subplot(3, 2, 4)
plt.plot(epochs, learning_rates, marker='o', linestyle='-', color="purple")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Learning Rate vs. Epoch")

# Plot max tile vs. epoch.
plt.subplot(3, 2, 5)
plt.plot(epochs, max_tiles, marker='o', linestyle='-', color="brown")
plt.xlabel("Epoch")
plt.ylabel("Max Tile")
plt.title("Max Tile vs. Epoch")

# Hide the 6th subplot (unused).
plt.subplot(3, 2, 6)
plt.axis('off')

plt.tight_layout()
plt.savefig("training_metrics.png")
plt.show() 