"""2048 Game AI with Reinforcement Learning."""

__version__ = "0.1.0"

# Import key components for convenient access
from .environment.game2048 import Game2048, preprocess_state_onehot
# Removed import for improved_reward as it doesn't exist
# from .environment.improved_reward import apply_improved_reward
# Removed import for base_agent as it doesn't exist
# from .agents.base_agent import PPOAgent
from .agents.enhanced_agent import EnhancedAgent
from .agents.dqn_agent import DQNAgent
from .config import set_seeds, device 