import argparse
import logging
import os
import sys

def setup_logging(log_file="evaluation.log"):
    """
    Set up logging configuration.
    
    Args:
        log_file: Path to the log file
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def parse_args(args=None):
    """
    Parse command-line arguments.
    
    Args:
        args: Command-line arguments (defaults to sys.argv)
        
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="2048 MCTS Evaluation System"
    )
    
    # General options
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed (default: 42)")
    parser.add_argument("--checkpoint", type=str, default=None, 
                        help="Path to checkpoint to resume from")
    parser.add_argument("--output-dir", type=str, default="evaluation_results", 
                        help="Directory to save results (default: evaluation_results)")
    parser.add_argument("--hidden-dim", type=int, default=256, 
                        help="Hidden dimension size (default: 256)")
    
    # Evaluation options
    parser.add_argument("--games", type=int, default=10, 
                        help="Number of games to play in evaluation (default: 10)")
    parser.add_argument("--render", action="store_true",
                        help="Render the games during evaluation")
    parser.add_argument("--max-steps", type=int, default=1000,
                        help="Maximum steps per game (default: 1000)")
    parser.add_argument("--save-trajectories", action="store_true",
                        help="Save full game trajectories for analysis")
    
    # MCTS options
    parser.add_argument("--mcts-simulations", type=int, default=200, 
                        help="Number of MCTS simulations (default: 200)")
    parser.add_argument("--mcts-temperature", type=float, default=0.5, 
                        help="Temperature for MCTS action selection (default: 0.5)")
    parser.add_argument("--compare", action="store_true", 
                        help="Compare regular agent with MCTS-enhanced version")
    
    return parser.parse_args(args) 