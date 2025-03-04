"""
Evaluation utilities for 2048 agents.
"""

from .evaluation import evaluate_agent
from .mcts_evaluation import main as evaluate_mcts
from .analyze_results import analyze_multiple_results
from .comprehensive_eval import run_comprehensive_evaluation 