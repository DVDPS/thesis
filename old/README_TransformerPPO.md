# Transformer-based PPO for 2048 Game

This extension implements a Transformer-based architecture for Proximal Policy Optimization (PPO) to play the 2048 game. The Transformer architecture leverages self-attention mechanisms to better capture spatial patterns and long-term dependencies in the game state.

## Key Advantages

1. **Spatial Understanding**: The Transformer architecture excels at capturing spatial relationships between tiles on the board.

2. **Global Context**: Self-attention mechanisms allow the agent to consider the entire board state when making decisions, rather than focusing on local patterns.

3. **Sample Efficiency**: PPO tends to be more sample-efficient than DQN for complex tasks like 2048, where the state space is large.

4. **Stable Training**: The implementation includes various stability improvements like learning rate scheduling, gradient clipping, and early stopping.

## Architecture Details

The architecture consists of several key components:

- **Board Embedding**: Converts the one-hot encoded board state into embeddings suitable for the Transformer.

- **Positional Encoding**: Adds positional information to help the model understand the spatial layout of the board.

- **Transformer Blocks**: Multi-head self-attention layers that process the board representation.

- **Global Attention Pooling**: Aggregates board information into a single context vector using a learnable query token.

- **Policy and Value Heads**: Separate networks for policy (action selection) and value (state evaluation) functions.

## How to Train

You can train the Transformer-based PPO agent using the provided script:

```bash
python -m src.thesis.train_transformer_ppo \
    --total-timesteps 1000000 \
    --embed-dim 256 \
    --num-heads 4 \
    --num-layers 4 \
    --learning-rate 3e-4 \
    --mixed-precision \
    --output-dir transformer_ppo_results
```

### Key Training Parameters

- `--embed-dim`: Embedding dimension for the Transformer (default: 256)
- `--num-heads`: Number of attention heads (default: 4)
- `--num-layers`: Number of Transformer layers (default: 4)
- `--learning-rate`: Learning rate (default: 3e-4)
- `--mixed-precision`: Use mixed precision training for faster computation on supported GPUs
- `--use-data-parallel`: Use data parallelism for training on multiple GPUs

### Advanced Parameters

- `--gamma`: Discount factor (default: 0.995)
- `--clip-ratio`: PPO clip ratio (default: 0.2)
- `--vf-coef`: Value function coefficient (default: 0.5)
- `--ent-coef`: Entropy coefficient (default: 0.01)
- `--gae-lambda`: GAE lambda parameter (default: 0.95)
- `--update-epochs`: Number of PPO update epochs (default: 4)

## Monitoring Training Progress

Training progress is logged using TensorBoard. You can monitor the training by running:

```bash
tensorboard --logdir transformer_ppo_results/tensorboard
```

This will show metrics including:
- Episode rewards
- Maximum tile achieved
- Policy and value losses
- Entropy
- Learning rate

## Using a Trained Agent

To use a trained agent, you can load the saved model:

```python
from src.thesis.agents.transformer_ppo_agent import TransformerPPOAgent
from src.thesis.environment.game2048 import Game2048, preprocess_state_onehot

# Initialize the agent with the same parameters used during training
agent = TransformerPPOAgent(
    board_size=4,
    embed_dim=256,
    num_heads=4,
    num_layers=4
)

# Load the saved model
agent.load("transformer_ppo_results/best_model.pt")

# Play a game
env = Game2048()
state = env.reset()
done = False

while not done:
    # Process state
    state_proc = preprocess_state_onehot(state)
    
    # Get valid moves and select action
    valid_moves = env.get_possible_moves()
    if not valid_moves:
        break
        
    action, _, _ = agent.get_action(state_proc, valid_moves, deterministic=True)
    
    # Execute action
    state, reward, done, _ = env.step(action)
    
    # Print or visualize the board as needed
    print(state)
```

## Performance Comparison

The Transformer-based PPO architecture is expected to outperform standard CNN-based PPO and DQN approaches for the 2048 game for several reasons:

1. Better spatial understanding of the board state
2. Ability to capture long-term dependencies between tiles
3. Global context through self-attention mechanisms
4. Stable policy updates through PPO's clipped objective

## Requirements

- PyTorch 1.8+
- NumPy
- TensorBoard
- tqdm
- matplotlib

## Tuning Recommendations

For best performance, consider the following tuning recommendations:

1. **Embedding Dimension**: Higher values (512+) may capture more complex patterns but require more compute.
2. **Number of Heads**: 4-8 heads typically work well for the 2048 board size.
3. **Number of Layers**: 4-6 layers provide good depth without excessive computation.
4. **Learning Rate**: Start with 3e-4 and adjust based on loss curves.
5. **Entropy Coefficient**: Increase for more exploration (0.02-0.05) if the agent gets stuck.

## Advanced Usage

For advanced scenarios, you can modify the agent architecture in `src/thesis/agents/transformer_ppo_agent.py`. Key areas that might benefit from customization:

- Modifying the board embedding approach
- Changing the positional encoding scheme
- Adjusting the attention mechanism
- Adding more skip connections
- Experimenting with different layer normalization approaches 