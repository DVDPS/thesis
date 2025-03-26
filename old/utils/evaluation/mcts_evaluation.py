"""
Evaluation script for MCTS enhanced agents.
This module provides functionality to compare MCTS-enhanced agents with regular agents.
"""

import torch
import numpy as np
import logging
import time
import os
from ...environment.game2048 import Game2048, preprocess_state_onehot
from ...config import device
from ...agents.dqn_agent import DQNAgent
from ..mcts import wrap_agent_with_mcts
from .evaluation import evaluate_agent

def main(checkpoint_path, num_games=5, mcts_simulations=50):
    """
    Main function to run the MCTS evaluation.
    
    Args:
        checkpoint_path: Path to model checkpoint
        num_games: Number of games to play
        mcts_simulations: Number of MCTS simulations
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Load model
    logging.info(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    agent = DQNAgent()
    agent.load_state_dict(checkpoint['model_state_dict'])
    agent.eval()
    
    # Create MCTS-enhanced version
    logging.info(f"Creating MCTS agent with {mcts_simulations} simulations")
    mcts_agent = wrap_agent_with_mcts(agent, num_simulations=mcts_simulations, temperature=0.5)
    
    # Evaluate regular agent
    logging.info("\nEvaluating regular agent...")
    regular_results = evaluate_agent(agent, num_games=num_games)
    
    # Evaluate MCTS agent
    logging.info("\nEvaluating MCTS agent...")
    mcts_results = evaluate_agent(mcts_agent, num_games=num_games)
    
    # Print comparison
    logging.info("\n" + "=" * 50)
    logging.info("COMPARISON RESULTS:")
    logging.info(f"Average Max Tile: Regular = {regular_results['avg_max_tile']:.1f}, MCTS = {mcts_results['avg_max_tile']:.1f}")
    logging.info(f"Average Score: Regular = {regular_results['avg_score']:.1f}, MCTS = {mcts_results['avg_score']:.1f}")
    logging.info(f"Best Max Tile: Regular = {regular_results['max_tile_reached']}, MCTS = {mcts_results['max_tile_reached']}")
    
    logging.info("\nRegular Agent Tile Distribution:")
    for tile, count in sorted(regular_results['tile_counts'].items()):
        logging.info(f"  {tile}: {count} games ({count/num_games*100:.1f}%)")
        
    logging.info("\nMCTS Agent Tile Distribution:")
    for tile, count in sorted(mcts_results['tile_counts'].items()):
        logging.info(f"  {tile}: {count} games ({count/num_games*100:.1f}%)")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate MCTS enhancement of 2048 agents")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--games", type=int, default=5, help="Number of games to play")
    parser.add_argument("--simulations", type=int, default=50, help="Number of MCTS simulations")
    
    args = parser.parse_args()
    main(args.checkpoint, args.games, args.simulations) 