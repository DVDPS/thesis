"""2048 Game AI with Reinforcement Learning."""

__version__ = "0.1.0"

# Import key components for convenient access
from .environment.game2048 import Game2048, preprocess_state_onehot
from .environment.improved_reward import apply_improved_reward
from .agents.base_agent import PPOAgent
from .agents.enhanced_agent import EnhancedAgent
from .config import set_seeds, device 