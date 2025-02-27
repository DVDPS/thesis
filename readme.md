# 2048 Game AI with Reinforcement Learning

This project implements an AI agent that learns to play the 2048 game using Proximal Policy Optimization (PPO) and other reinforcement learning techniques. The implementation includes several modern deep learning approaches and optimizations.

## Project Organization

The project has been reorganized into a clean, modular Python package structure:

```
thesis/
├── src/thesis/                # Main package
│   ├── agents/                # Agent implementations
│   │   ├── base_agent.py      # PPO agent base classes
│   │   ├── enhanced_agent.py  # Enhanced agent with improvements
│   │   └── simplified_agent.py # Simplified agent implementation
│   ├── environment/           # Game environment
│   │   ├── game2048.py        # 2048 game implementation
│   │   └── improved_reward.py # Enhanced reward functions
│   ├── training/              # Training implementations
│   │   ├── training.py        # Standard PPO training
│   │   └── simplified_training.py # Simplified training loop
│   ├── utils/                 # Utility modules
│   │   ├── curriculum_learning.py # Curriculum learning implementation
│   │   ├── enhanced_exploration.py # Exploration techniques
│   │   └── visualizations.py  # Visualization tools
│   ├── config.py              # Configuration settings
│   └── main.py                # Unified entry point
├── checkpoints/               # Model checkpoints
├── visualizations/            # Generated visualizations
├── setup.py                   # Package setup
├── requirements.txt           # Dependencies
├── train.bat                  # Windows training script
└── train.sh                   # Unix training script
```

## Features

- **Multiple Agent Architectures**:
  - Standard PPO agent with convolutional layers
  - Enhanced agent with residual connections and batch normalization
  - Simplified agent with stable learning properties

- **Advanced Training Methods**:
  - Standard PPO training
  - Simplified stable training
  - Balanced exploration training
  - Curriculum learning for high tile values

- **Optimizations**:
  - Dynamic batch sizing
  - Exploration noise management
  - Optimistic value initialization
  - Reward function shaping

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/thesis.git
cd thesis
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Unix/Linux
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install in development mode:
```bash
pip install -e .
```

## Usage

The project provides a unified command-line interface through `train.bat` (Windows) or `train.sh` (Unix):

### Basic Usage

```bash
# Windows
train.bat [mode]

# Unix
./train.sh [mode]
```

Available modes:
- `standard`: Standard PPO training
- `simplified`: Simplified stable training
- `enhanced`: Enhanced agent training
- `balanced`: Balanced exploration training
- `curriculum`: Curriculum learning for high-value tiles
- `evaluate`: Evaluate a trained model

### Examples

#### Training a standard agent:
```bash
train.bat standard --epochs 3000 --batch-size 64
```

#### Training with balanced exploration:
```bash
train.bat balanced --epochs 2000 --dynamic-batch --min-batch-size 16
```

#### Curriculum learning (after initial training):
```bash
train.bat curriculum --checkpoint checkpoints/enhanced/best_model.pt
```

#### Evaluating a model:
```bash
train.bat evaluate checkpoints/enhanced/best_model.pt --games 50
```

## Advanced Options

The main script supports various advanced options:

- `--dynamic-batch`: Enable dynamic batch size scheduling
- `--min-batch-size`: Minimum batch size for dynamic scheduling
- `--exploration`: Override initial exploration noise
- `--min-exploration`: Override minimum exploration noise
- `--curriculum`: Enable curriculum learning
- `--curriculum-epochs`: Set curriculum learning epochs

See all options with:
```bash
python -m src.thesis.main --help
```

## Cleaning Up

The repository includes a cleanup script to remove old/unnecessary files:

```bash
python cleanup.py
```

## License

This project is for educational and research purposes.

## Acknowledgments

This project is inspired by various reinforcement learning techniques and 2048 game implementations.
