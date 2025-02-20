# 2048 Game AI with PPO

This project implements an AI agent that learns to play the 2048 game using Proximal Policy Optimization (PPO), a state-of-the-art reinforcement learning algorithm. The implementation includes several modern deep learning techniques and optimizations.

## Project Structure

### Key Components

- **agent.py**: Implements the PPO agent using PyTorch, featuring:
  - Convolutional neural network for board state processing
  - Separate policy and value heads
  - Optimistic value initialization
  - Exploration noise mechanism

- **game2048.py**: Complete 2048 game implementation with:
  - Efficient board manipulation using NumPy
  - Custom reward shaping
  - State preprocessing utilities
  - Game mechanics validation

- **training.py**: PPO training implementation including:
  - Experience collection
  - Advantage estimation
  - Policy updates with clipping
  - Mixed precision training support

## Training Visualization

The training process generates several visualization artifacts:
- Training metrics plots (`training_progress.png`):
  - Running reward vs. episode
  - Maximum tile achieved
  - Policy and value losses
  - Entropy and learning rate

- Board state trajectories for each training block:
  - Visualizes game states at regular intervals
  - Shows progression of agent's strategy
  - Helps identify learning patterns

- Detailed training logs (`training.log`):
  - Episode rewards and running averages
  - Maximum tiles achieved
  - Loss values and learning metrics
  - Training configuration details

## Implementation Details

### State Representation
The game state is represented as a 16-channel tensor where:
- Channel 0: Empty tiles
- Channel 1-15: Tiles with values 2^1 through 2^15

This representation allows the network to:
- Easily identify empty spaces
- Recognize tile values independently
- Process spatial relationships effectively

### Reward Structure
The reward function includes:
- Merge rewards (sum of merged tile values)
  - Immediate feedback for successful merges
  - Scaled to encourage larger merges
- Empty tile bonus
  - Encourages maintaining free spaces
  - Helps prevent board lockup
- Monotonicity bonus
  - Rewards organized board states
  - Encourages strategic tile placement
- Penalties for:
  - Invalid moves (-2)
  - Game over (-100)

### PPO Configuration
Core PPO parameters:
- Clip parameter: 0.3
- GAE parameter (λ): 0.95
- Discount factor (γ): 0.99
- Mini-batch size: 32
- PPO epochs: 12

Training optimizations:
- Learning rate annealing
- Entropy coefficient decay
- Gradient clipping
- Mixed precision training

## Requirements and Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended)

### Dependencies

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/2048-ppo.git
cd 2048-ppo
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```
