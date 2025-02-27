# Reinforcement Learning for 2048 Game

This project uses reinforcement learning with PPO (Proximal Policy Optimization) to train an agent to play the 2048 game.

## Updated Structure

The code has been consolidated for better organization:

```
.
├── agent.py                 # PPO Agent implementation
├── config.py                # Configuration and hyperparameters
├── game2048.py              # 2048 game environment
├── main.py                  # Main entry point with CLI options
├── test_utils.py            # Testing utilities
├── training.py              # Core training loop
├── training_stats.py        # Training statistics tracking
└── utils/                   # Utility modules
    ├── curriculum_learning.py  # Curriculum learning implementation
    ├── enhanced_exploration.py # High exploration training
    └── visualizations.py       # Visualization tools
```

## Command Line Interface

The `main.py` script provides a comprehensive command-line interface for different training approaches.

### Windows Virtual Environment Usage

If you're using a Windows virtual environment, use the following command format:

```
.venv\Scripts\python main.py [options]
```

### Basic Options

- `--epochs N`: Number of training epochs (default: 7000)
- `--checkpoint PATH`: Path to a checkpoint to resume from
- `--output-dir DIR`: Directory to save models (default: "checkpoints")

### Training Modes

- `--high-exploration`: Use high exploration parameters
- `--curriculum`: Use curriculum learning focused on high-value tiles

### High Exploration Parameters

- `--entropy VALUE`: Entropy coefficient for exploration (default: 0.2)
- `--exploration-noise VALUE`: Exploration noise factor (default: 1.5)
- `--min-exploration-noise VALUE`: Minimum exploration noise (default: 0.15)

### Curriculum Learning Parameters

- `--target-tiles TILES`: Target tiles for curriculum learning, comma-separated (default: "256,512")
- `--curriculum-epochs N`: Number of epochs for curriculum fine-tuning (default: 200)

### Visualization

- `--visualize`: Generate visualizations after training
- `--visualize-only`: Only generate visualizations without training

### Test Options

- `--test-merge`: Run merge row test
- `--test-gameover`: Run game over condition test

## Training Strategies

### 1. Standard Training

Windows:
```
.venv\Scripts\python main.py --epochs 5000
```

This runs the default PPO training with standard exploration parameters.

### 2. High Exploration

If your agent is stuck at a plateau (e.g., consistently reaching tile 512 but no further):

Windows:
```
.venv\Scripts\python main.py --high-exploration --epochs 500 --entropy 0.2 --exploration-noise 1.5
```

This increases exploration to help the agent discover new strategies.

### 3. Curriculum Learning

To focus training on specific high-value tile situations:

Windows:
```
.venv\Scripts\python main.py --curriculum --target-tiles 256,512 --curriculum-epochs 200
```

This generates synthetic board states with high-value tiles and trains the agent to handle them effectively.

### 4. Combined Approach

For best results, you can combine approaches:

1. First train with standard parameters until plateauing
2. Then use high exploration to break through the plateau:
   ```
   .venv\Scripts\python main.py --high-exploration --checkpoint checkpoints/best_model.pt
   ```
3. Finally, use curriculum learning to further improve:
   ```
   .venv\Scripts\python main.py --curriculum --checkpoint checkpoints/high_exploration_XXX/best_model.pt
   ```

## Generating Visualizations

To visualize how an agent plays:

```
.venv\Scripts\python main.py --visualize-only --checkpoint checkpoints/best_model.pt
```

This will generate:
- Game playthrough animations
- Action heatmaps
- Value grids
- Board trajectories

## Monitoring Progress

When training, look for:
- Tile achievement milestones (128, 256, 512, 1024, 2048)
- Score improvements over time
- Entropy values in training logs
- Board organization patterns

## Tips for Breaking Through Plateaus

1. **Increase Exploration**: When progress stalls, use the high exploration mode to discover new strategies
2. **Curriculum Training**: Focus on specific situations with high-value tiles to improve critical merging decisions
3. **Adjust Learning Rate**: Lower learning rates help with fine-tuning behaviors
4. **Monitor for Patterns**: Watch for board organization patterns (e.g., keeping high values in corners)
5. **Be Patient**: Breaking through from 512 to 1024 and 2048 tiles can take significant training time

## Windows-Specific Troubleshooting

If you encounter import errors or other issues when running with `.venv\Scripts\python`:

1. Make sure you're running commands from the project root directory
2. If modules still can't be found, try installing them with `.venv\Scripts\pip install`
3. For visualization problems, ensure matplotlib and imageio are installed:
   ```
   .venv\Scripts\pip install matplotlib imageio
   ```
4. Path separators: Use backslashes (`\`) for Windows file paths in command line arguments 